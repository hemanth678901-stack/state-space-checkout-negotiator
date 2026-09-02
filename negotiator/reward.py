"""
Reward function for the Checkout Negotiator MCTS.

Three components:
  1. Step reward   — immediate signal per (state, action, response).
  2. Terminal reward — end-of-episode signal.
  3. Normalization  — maps raw reward to [0, 1] for UCB1 balance.

Cross-sell valuation uses **margin** (flat 30 %), not sticker price,
to prevent the MCTS from prioritising high-revenue / low-profit items.
"""

from __future__ import annotations

from .actions import Action
from .config import DEFAULT_CONFIG, NegotiatorConfig
from .state import State, UserResponse


def step_reward(
    state: State,
    action: Action,
    response: UserResponse,
    config: NegotiatorConfig = DEFAULT_CONFIG,
) -> float:
    """
    Immediate reward for a single (state, action, response) transition.

    R_step = α · 𝟙[accept] · V · (1 − Δ')
           − β · Δ' · V · 𝟙[concede]
           + γ_cs · 𝟙[cross_sell] · 𝟙[¬reject] · V_cs

    where  Δ' = Δ + 0.05 · 𝟙[concede]
           V_cs = V · cross_sell_margin  (margin, not sticker)
    """
    V = state.cart_midpoint(config)
    delta = state.discount_pct / 100.0
    is_concede = action == Action.CONCEDE
    delta_prime = delta + 0.05 if is_concede else delta

    # ── Conversion revenue ───────────────────────────────────
    conversion = 0.0
    if response == UserResponse.ACCEPT:
        conversion = config.ALPHA * V * (1.0 - delta_prime)

    # ── Margin erosion cost ──────────────────────────────────
    margin_cost = 0.0
    if is_concede:
        margin_cost = config.BETA * delta_prime * V

    # ── Cross-sell uplift (margin-based) ─────────────────────
    cross_sell = 0.0
    if action == Action.CROSS_SELL and response != UserResponse.REJECT:
        v_cs = V * config.CROSS_SELL_MARGIN  # 30 % of cart value
        cross_sell = config.GAMMA_CS * v_cs

    return conversion - margin_cost + cross_sell


def terminal_reward(
    state: State,
    config: NegotiatorConfig = DEFAULT_CONFIG,
) -> float:
    """
    End-of-episode reward based on final outcome.

      • ACCEPT  →  V · (1 − Δ)   (net revenue after discount)
      • REJECT  → −λ · V          (opportunity cost of lost sale)
      • Other   →  0              (turn limit / offer pending)
    """
    V = state.cart_midpoint(config)
    delta = state.discount_pct / 100.0

    last = state.last_response
    if last == UserResponse.ACCEPT:
        return V * (1.0 - delta)
    if last == UserResponse.REJECT:
        return -config.LAMBDA_ABANDON * V
    return 0.0


def normalize_reward(
    raw_reward: float,
    config: NegotiatorConfig = DEFAULT_CONFIG,
) -> float:
    """
    Map raw cumulative reward to [0, 1] for UCB1 numerical stability.

    Uses conservative bounds from config so that no plausible rollout
    can produce a value outside the range.
    """
    r_range = config.R_MAX - config.R_MIN
    if r_range == 0:
        return 0.5  # Degenerate — should never happen in production.
    clamped = max(config.R_MIN, min(config.R_MAX, raw_reward))
    return (clamped - config.R_MIN) / r_range
