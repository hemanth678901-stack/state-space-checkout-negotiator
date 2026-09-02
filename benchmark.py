"""
Comparative Benchmark Engine for Checkout Negotiation Strategies.

Simulates 1,000 identical customer checkout sessions across three distinct strategies:
  1. Rule-Based Baseline: Static heuristic (concede if abandoning, hold otherwise).
  2. Unbounded Agent (Simulated Naive LLM): Stochastic concessions that leak margin under pressure.
  3. Bounded MCTS Negotiator: State-space bounded search with UCT optimization.

Computes:
  - Conversion Rate (%)
  - Average Converted Discount (%)
  - Net Revenue Captured ($)
  - Estimated Net Profit / Margin Retained ($)
  - Margin Leakage Incidents (discounts > 25% or unnecessary erosion)
  - Average Decision Latency (ms)

Exports results to `benchmark_results.json` and renders an ASCII comparison table.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from typing import Any

from negotiator.actions import Action
from negotiator.config import NegotiatorConfig
from negotiator.mcts.tree import MCTSEngine
from negotiator.opponent_model import OpponentModel
from negotiator.state import (
    UserIntent,
    UserResponse,
    classify_cart_value,
    make_initial_state,
    transition_intent,
)


@dataclass
class CustomerProfile:
    """Represents a customer visiting the checkout."""

    customer_id: int
    initial_intent: UserIntent
    cart_value: float
    price_sensitivity: float  # [0.5, 1.5] scaling factor on discount appeal


@dataclass
class SessionResult:
    """Outcome of a single negotiation session."""

    customer_id: int
    converted: bool
    final_discount_pct: int
    base_cart_value: float
    gross_revenue: float
    net_revenue: float
    profit_margin: float
    turns_taken: int
    margin_leakage: bool
    latency_ms_total: float


# ═══════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT & SIMULATED USER REACTION
# ═══════════════════════════════════════════════════════════════════════════


class CustomerSimulator:
    """
    Ground-truth customer behavior model for benchmark evaluation.
    Evaluates offers based on customer intent, cart value, and discount.
    """

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.opponent_priors = OpponentModel()

    def respond(
        self,
        intent: UserIntent,
        cart_value: float,
        action: Action,
        current_discount_pct: int,
        price_sensitivity: float,
    ) -> UserResponse:
        """Simulate real user response to the retailer's action."""
        # Base probabilities from true behavioral distribution
        p_acc, p_rej, p_cnt = self.opponent_priors.response_probs(
            state=make_initial_state(intent, classify_cart_value(cart_value)),
            action=action,
        )

        # Discount sensitivity boost: each +5% discount boosts acceptance
        discount_factor = (current_discount_pct / 100.0) * price_sensitivity
        p_acc_adjusted = min(0.95, p_acc + discount_factor * 0.4)
        remaining = 1.0 - p_acc_adjusted
        p_rej_adjusted = remaining * (p_rej / (p_rej + p_cnt + 1e-6))

        draw = self.rng.random()
        if draw < p_acc_adjusted:
            return UserResponse.ACCEPT
        elif draw < p_acc_adjusted + p_rej_adjusted:
            return UserResponse.REJECT
        else:
            return UserResponse.COUNTER


# ═══════════════════════════════════════════════════════════════════════════
#  STRATEGIES
# ═══════════════════════════════════════════════════════════════


class RuleBasedAgent:
    """
    Heuristic baseline:
    - If user is ABANDONING or RETURNING, concede 5% up to 15%.
    - If user is BROWSING or CONSIDERING, cross-sell.
    - If user is READY_TO_BUY, always HOLD.
    """

    def choose_action(
        self, intent: UserIntent, discount_pct: int, last_resp: UserResponse | None
    ) -> Action:
        if (
            intent in (UserIntent.ABANDONING, UserIntent.RETURNING)
            and discount_pct < 15
        ):
            return Action.CONCEDE
        elif intent == UserIntent.READY_TO_BUY:
            return Action.HOLD
        elif (
            intent == UserIntent.CONSIDERING
            and last_resp == UserResponse.COUNTER
            and discount_pct < 10
        ):
            return Action.CONCEDE
        else:
            return Action.CROSS_SELL


class UnboundedNaiveLLMAgent:
    """
    Simulated Naive LLM Agent (Free-form generator without formal verification):
    - Susceptible to bargaining pressure (hallucinates higher discounts).
    - Can leak margin beyond 25% (up to 40%).
    - Fails to protect margin on READY_TO_BUY users.
    """

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def choose_action(
        self,
        intent: UserIntent,
        discount_pct: int,
        turn: int,
        last_resp: UserResponse | None,
    ) -> tuple[Action, int]:
        # Under repeated counter/rejection, naive LLM panics and offers large concessions
        if (
            last_resp in (UserResponse.REJECT, UserResponse.COUNTER)
            or intent == UserIntent.ABANDONING
        ):
            # 70% chance of conceding 5%, 10%, or 15%
            concession = self.rng.choice([5, 10, 15])
            new_discount = discount_pct + concession
            # Naive LLM lacks hard mask: leaks up to 45%
            return Action.CONCEDE, min(45, new_discount)

        if intent == UserIntent.READY_TO_BUY and self.rng.random() < 0.35:
            # Unnecessary concession to ready buyers
            return Action.CONCEDE, discount_pct + 5

        # Fallback to hold or cross-sell
        return (
            Action.CROSS_SELL if self.rng.random() < 0.5 else Action.HOLD
        ), discount_pct


class MCTSAgentWrapper:
    """Our bounded MCTS search policy."""

    def __init__(self, config: NegotiatorConfig) -> None:
        self.config = config

    def choose_action(self, state, seed: int) -> Action:
        engine = MCTSEngine(config=self.config, seed=seed)
        return engine.search(state)


# ═══════════════════════════════════════════════════════════════════════════
#  BENCHMARK RUNNER
# ═══════════════════════════════════════════════════════════════════════════


def generate_customer_cohort(n: int = 1000, seed: int = 42) -> list[CustomerProfile]:
    """Generate 1,000 diverse customer profiles."""
    rng = random.Random(seed)
    intents = [
        UserIntent.BROWSING,
        UserIntent.CONSIDERING,
        UserIntent.READY_TO_BUY,
        UserIntent.ABANDONING,
        UserIntent.RETURNING,
    ]
    # Realistic weights: 25% Browsing, 35% Considering, 20% Ready, 15% Abandoning, 5% Returning
    intent_weights = [0.25, 0.35, 0.20, 0.15, 0.05]

    customers = []
    for cid in range(n):
        intent = rng.choices(intents, weights=intent_weights, k=1)[0]
        # Cart value log-normal between $20 and $900
        val = min(1200.0, max(25.0, rng.lognormvariate(4.8, 0.7)))
        price_sens = rng.uniform(0.7, 1.3)
        customers.append(CustomerProfile(cid, intent, round(val, 2), price_sens))

    return customers


def run_benchmark_for_strategy(
    strategy_name: str,
    cohort: list[CustomerProfile],
    config: NegotiatorConfig,
    base_seed: int = 100,
) -> list[SessionResult]:
    """Run all 1,000 customers through a specific strategy."""
    results: list[SessionResult] = []

    mcts_agent = MCTSAgentWrapper(config)
    rule_agent = RuleBasedAgent()

    for idx, cust in enumerate(cohort):
        rng = random.Random(base_seed + cust.customer_id)
        sim = CustomerSimulator(rng)
        naive_agent = UnboundedNaiveLLMAgent(rng)

        current_intent = cust.initial_intent
        current_discount = 0
        converted = False
        turns = 0
        added_cross_sell_value = 0.0
        margin_leaked = False
        total_latency_ms = 0.0

        # State initialization for MCTS
        cart_bin = classify_cart_value(cust.cart_value, config)
        state = make_initial_state(current_intent, cart_bin, config)
        last_response = None

        for turn in range(config.T_MAX):
            turns += 1
            t0 = time.perf_counter()

            # ── Select Action by Strategy ─────────────────────────
            if strategy_name == "MCTS_Bounded":
                action = mcts_agent.choose_action(
                    state, seed=base_seed + idx * 10 + turn
                )
                if action == Action.CONCEDE:
                    current_discount = min(
                        config.DISCOUNT_MAX,
                        current_discount + config.DISCOUNT_STEP_SIZE,
                    )
            elif strategy_name == "Rule_Based":
                action = rule_agent.choose_action(
                    current_intent, current_discount, last_response
                )
                if action == Action.CONCEDE:
                    current_discount = min(
                        15, current_discount + config.DISCOUNT_STEP_SIZE
                    )
            elif strategy_name == "Unbounded_Naive_LLM":
                action, current_discount = naive_agent.choose_action(
                    current_intent, current_discount, turn, last_response
                )
                if current_discount > 25:
                    margin_leaked = True
            else:
                raise ValueError(f"Unknown strategy: {strategy_name}")

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            total_latency_ms += elapsed_ms

            if action == Action.CROSS_SELL:
                # Add cross-sell value
                added_cross_sell_value += cust.cart_value * 0.25

            # ── Customer Reaction ─────────────────────────────────
            resp = sim.respond(
                current_intent,
                cust.cart_value,
                action,
                current_discount,
                cust.price_sensitivity,
            )
            last_response = resp

            if resp == UserResponse.ACCEPT:
                converted = True
                break
            elif resp == UserResponse.REJECT:
                current_intent = transition_intent(current_intent, resp, rng.random())
            else:  # COUNTER
                current_intent = transition_intent(current_intent, resp, rng.random())

            # Update MCTS state
            state = make_initial_state(current_intent, cart_bin, config)

        # ── Compute Economics ─────────────────────────────────────
        if converted:
            gross = cust.cart_value + added_cross_sell_value
            net = (
                cust.cart_value * (1.0 - current_discount / 100.0)
                + added_cross_sell_value
            )
            # Assume base product margin is 50%, cross-sell margin is 30%
            base_profit = (cust.cart_value * 0.50) - (
                cust.cart_value * (current_discount / 100.0)
            )
            cs_profit = added_cross_sell_value * 0.30
            profit = max(0.0, base_profit + cs_profit)
        else:
            gross = 0.0
            net = 0.0
            profit = 0.0

        if current_discount > 25:
            margin_leaked = True

        results.append(
            SessionResult(
                customer_id=cust.customer_id,
                converted=converted,
                final_discount_pct=current_discount if converted else 0,
                base_cart_value=cust.cart_value,
                gross_revenue=round(gross, 2),
                net_revenue=round(net, 2),
                profit_margin=round(profit, 2),
                turns_taken=turns,
                margin_leakage=margin_leaked,
                latency_ms_total=round(total_latency_ms, 2),
            )
        )

    return results


def summarize_results(name: str, results: list[SessionResult]) -> dict[str, Any]:
    """Compute aggregate KPI metrics for a strategy run."""
    n = len(results)
    conversions = sum(1 for r in results if r.converted)
    conversion_rate = (conversions / n) * 100.0

    converted_discounts = [r.final_discount_pct for r in results if r.converted]
    avg_discount = (
        (sum(converted_discounts) / len(converted_discounts))
        if converted_discounts
        else 0.0
    )

    total_net_rev = sum(r.net_revenue for r in results)
    total_profit = sum(r.profit_margin for r in results)
    leakage_incidents = sum(1 for r in results if r.margin_leakage)

    all_latencies = [
        r.latency_ms_total / r.turns_taken for r in results if r.turns_taken > 0
    ]
    avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0.0

    return {
        "strategy": name,
        "total_sessions": n,
        "conversions": conversions,
        "conversion_rate_pct": round(conversion_rate, 2),
        "avg_converted_discount_pct": round(avg_discount, 2),
        "net_revenue_captured_usd": round(total_net_rev, 2),
        "net_profit_captured_usd": round(total_profit, 2),
        "margin_leakage_incidents": leakage_incidents,
        "avg_decision_latency_ms": round(avg_latency, 2),
    }


def main():
    print("=" * 80)
    print("  CHECKOUT NEGOTIATOR — 1,000 SESSION COMPARATIVE BENCHMARK")
    print("=" * 80)
    print("Generating 1,000 diverse customer cohort profiles...")
    cohort = generate_customer_cohort(n=1000, seed=42)

    config = NegotiatorConfig(BURN_IN_ACTIVE=False, MCTS_ITERATIONS=300)

    strategies = ["Rule_Based", "Unbounded_Naive_LLM", "MCTS_Bounded"]
    summaries = []
    raw_exports = {}

    for strat in strategies:
        print(f"Running simulation for strategy: {strat} ...")
        t0 = time.perf_counter()
        session_results = run_benchmark_for_strategy(strat, cohort, config)
        elapsed = time.perf_counter() - t0
        summary = summarize_results(strat, session_results)
        summaries.append(summary)
        raw_exports[strat] = {
            "summary": summary,
            "session_sample": [asdict(r) for r in session_results[:10]],
        }
        print(
            f"  Completed in {elapsed:.2f}s | Conv: {summary['conversion_rate_pct']}% | Net Rev: ${summary['net_revenue_captured_usd']:,.2f}"
        )

    # ── Render ASCII Table ──────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(
        f"{'STRATEGY':<22} | {'CONV %':<8} | {'AVG DISC %':<11} | {'NET REVENUE':<14} | {'NET PROFIT':<13} | {'LEAKS':<6} | {'LATENCY':<8}"
    )
    print("-" * 90)
    for s in summaries:
        print(
            f"{s['strategy']:<22} | "
            f"{s['conversion_rate_pct']:>6.2f}% | "
            f"{s['avg_converted_discount_pct']:>9.2f}% | "
            f"${s['net_revenue_captured_usd']:>12,.2f} | "
            f"${s['net_profit_captured_usd']:>11,.2f} | "
            f"{s['margin_leakage_incidents']:>5} | "
            f"{s['avg_decision_latency_ms']:>6.2f}ms"
        )
    print("=" * 90)

    # Export to JSON
    export_path = "benchmark_results.json"
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(raw_exports, f, indent=2)
    print(f"\n[OK] Benchmark results exported successfully to '{export_path}'.\n")


if __name__ == "__main__":
    main()
