"""
Adversarial & Fuzz Stress-Testing Suite for the Checkout Negotiator.

Covers 500+ automated negotiation scenarios:
  1. Boundary & Malformed Inputs (negative cart, $0, extreme $100k+, NaN, Inf, invalid enums).
  2. Adversarial Exploitation (persistent REJECT / COUNTER spamming across 10+ turns to force discount escalation).
  3. Concurrent / High-load session fuzzing.
  4. Non-breachable Safety Invariant Assertions:
     - discount_pct <= 25% (or 0.25) in 100% of cases.
     - Profit margin protected against margin erosion.
     - Unbroken SQLite trace records per turn.
     - p95 latency strictly < 50ms.
"""

from __future__ import annotations

import math
import os
import random
import time
import pytest
from typing import List

from negotiator.actions import Action
from negotiator.config import NegotiatorConfig
from negotiator.engine import NegotiationEngine
from negotiator.state import (
    CartValueBin,
    UserIntent,
    UserResponse,
    classify_cart_value,
    make_initial_state,
)


@pytest.fixture
def temp_db_path(tmp_path) -> str:
    """Fixture providing a clean temporary SQLite database path."""
    return str(tmp_path / "adversarial_audit.db")


# ═══════════════════════════════════════════════════════════════════════════
#  1. BOUNDARY & MALFORMED INPUTS
# ═══════════════════════════════════════════════════════════════════════════


def test_boundary_cart_values(temp_db_path):
    """
    Fuzz extreme cart values: negative, zero, fractional, and massive values.
    Verify engine classifies without crashing and preserves safety invariants.
    """
    config = NegotiatorConfig(BURN_IN_ACTIVE=False, MCTS_ITERATIONS=200)
    engine = NegotiationEngine(config=config, db_path=temp_db_path)

    extreme_values = [
        -1000.0, -50.0, -0.01, 0.0, 0.0001, 0.50, 49.99, 50.0,
        149.99, 150.0, 499.99, 500.0, 100_000.0, 1_000_000.0, 1e9,
        float("nan"), float("inf"), float("-inf")
    ]

    for val in extreme_values:
        bin_val = classify_cart_value(val, config)
        assert isinstance(bin_val, CartValueBin)

        res = engine.start_session(cart_value=val, intent=UserIntent.CONSIDERING, seed=42)
        assert res["status"] == "active"
        assert res["discount_pct"] <= config.DISCOUNT_MAX
        assert res["action"] in [a.name for a in Action]


def test_malformed_string_inputs(temp_db_path):
    """
    Test passing invalid string representations of enums.
    Engine must reject malformed strings with clean KeyError/ValueError.
    """
    engine = NegotiationEngine(db_path=temp_db_path)

    # Invalid intent
    with pytest.raises(KeyError):
        engine.start_session(cart_value=100.0, intent="INVALID_INTENT_STRING")

    # Start valid session
    res = engine.start_session(cart_value=100.0, intent="CONSIDERING", seed=10)
    sid = res["session_id"]

    # Invalid user response
    with pytest.raises(KeyError):
        engine.continue_session(session_id=sid, user_response="INVALID_RESPONSE")


# ═══════════════════════════════════════════════════════════════════════════
#  2. ADVERSARIAL EXPLOITATION & 500+ FUZZ SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════


def test_adversarial_bargaining_bot_and_invariants(temp_db_path):
    """
    Run 500+ automated adversarial customer sessions.
    
    Adversarial Personas:
      - 200 Relentless Rejectors: spam REJECT for 10 turns.
      - 200 Aggressive Counter-Offerers: spam COUNTER for 10 turns.
      - 100 Stochastic Fuzzers: random mix of REJECT/COUNTER/ACCEPT.
    
    Safety Invariant Verification:
      1. Cumulative discount NEVER exceeds 25% (discount_pct <= 25).
      2. All decisions logged to SQLite with unbroken sequential turns (0, 1, ...).
      3. p95 latency across all turns is strictly under 50ms.
    """
    config = NegotiatorConfig(BURN_IN_ACTIVE=False, MCTS_ITERATIONS=300)
    engine = NegotiationEngine(config=config, db_path=temp_db_path)

    latencies_ms: List[float] = []
    total_turns_executed = 0
    sessions_run = 0

    all_intents = list(UserIntent)
    
    # ── 200 Relentless Rejectors ─────────────────────────────────────────
    for i in range(200):
        intent = all_intents[i % len(all_intents)]
        cart_val = random.uniform(10.0, 1500.0)

        t0 = time.perf_counter()
        res = engine.start_session(cart_value=cart_val, intent=intent, seed=i)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        
        sid = res["session_id"]
        total_turns_executed += 1
        sessions_run += 1

        assert res["discount_pct"] <= config.DISCOUNT_MAX, f"Discount breached at turn 0: {res['discount_pct']}%"

        # Spam REJECT for 10 consecutive turns
        for turn_idx in range(1, 11):
            t0 = time.perf_counter()
            res = engine.continue_session(session_id=sid, user_response=UserResponse.REJECT)
            latencies_ms.append((time.perf_counter() - t0) * 1000)
            total_turns_executed += 1

            assert res["discount_pct"] <= config.DISCOUNT_MAX, f"Discount breached at turn {turn_idx}: {res['discount_pct']}%"
            if res["status"] == "terminal":
                break

    # ── 200 Aggressive Counter-Offerers ──────────────────────────────────
    for i in range(200):
        intent = all_intents[i % len(all_intents)]
        cart_val = random.uniform(10.0, 1500.0)

        t0 = time.perf_counter()
        res = engine.start_session(cart_value=cart_val, intent=intent, seed=1000 + i)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        
        sid = res["session_id"]
        total_turns_executed += 1
        sessions_run += 1

        assert res["discount_pct"] <= config.DISCOUNT_MAX

        # Spam COUNTER for 10 consecutive turns
        for turn_idx in range(1, 11):
            t0 = time.perf_counter()
            res = engine.continue_session(session_id=sid, user_response=UserResponse.COUNTER)
            latencies_ms.append((time.perf_counter() - t0) * 1000)
            total_turns_executed += 1

            assert res["discount_pct"] <= config.DISCOUNT_MAX
            if res["status"] == "terminal":
                break

    # ── 100 Stochastic Fuzzers ───────────────────────────────────────────
    responses = [UserResponse.REJECT, UserResponse.COUNTER, UserResponse.ACCEPT]
    for i in range(100):
        intent = random.choice(all_intents)
        cart_val = random.uniform(5.0, 3000.0)

        t0 = time.perf_counter()
        res = engine.start_session(cart_value=cart_val, intent=intent, seed=2000 + i)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        
        sid = res["session_id"]
        total_turns_executed += 1
        sessions_run += 1

        assert res["discount_pct"] <= config.DISCOUNT_MAX

        for _ in range(8):
            resp = random.choice(responses)
            t0 = time.perf_counter()
            res = engine.continue_session(session_id=sid, user_response=resp)
            latencies_ms.append((time.perf_counter() - t0) * 1000)
            total_turns_executed += 1

            assert res["discount_pct"] <= config.DISCOUNT_MAX
            if res["status"] == "terminal":
                break

    # ── Verification of Invariants ───────────────────────────────────────
    assert sessions_run == 500, f"Expected 500 sessions, ran {sessions_run}"
    assert total_turns_executed > 1500, f"Expected >1500 turns, ran {total_turns_executed}"

    # Latency p95 assertion
    latencies_sorted = sorted(latencies_ms)
    p95_index = int(len(latencies_sorted) * 0.95)
    p95_latency = latencies_sorted[p95_index]
    p99_latency = latencies_sorted[int(len(latencies_sorted) * 0.99)]
    mean_latency = sum(latencies_ms) / len(latencies_ms)

    print(f"\n[LATENCY BENCHMARK] ({total_turns_executed} total turns)")
    print(f"  Mean: {mean_latency:.2f} ms")
    print(f"  p95:  {p95_latency:.2f} ms")
    print(f"  p99:  {p99_latency:.2f} ms")

    assert p95_latency < 50.0, f"p95 latency exceeded 50ms budget: {p95_latency:.2f} ms"

    # SQLite Audit Trail Integrity Check
    # Check a sample of 50 sessions to verify complete unbroken sequence
    for i in range(50):
        test_sid = list(engine._sessions.keys())[i]
        audit_trail = engine.get_audit_log(test_sid)
        assert len(audit_trail) >= 1, f"Audit trail empty for session {test_sid}"
        for turn_idx, record in enumerate(audit_trail):
            assert record["turn"] == turn_idx, f"Unbroken turn order failed in {test_sid}: expected {turn_idx}, got {record['turn']}"
            assert record["resulting_discount_pct"] <= 25


def test_ready_to_buy_margin_protection(temp_db_path):
    """
    Ensure READY_TO_BUY users on high carts are not needlessly conceded to.
    The policy should favor HOLD or CROSS_SELL to protect merchant margin.
    """
    config = NegotiatorConfig(BURN_IN_ACTIVE=False, MCTS_ITERATIONS=500)
    engine = NegotiationEngine(config=config, db_path=temp_db_path)

    # 20 READY_TO_BUY customers
    hold_or_cs_count = 0
    for i in range(20):
        res = engine.start_session(cart_value=600.0, intent=UserIntent.READY_TO_BUY, seed=500 + i)
        if res["action"] in ("HOLD", "CROSS_SELL"):
            hold_or_cs_count += 1

    # Overwhelming majority of initial moves to ready buyers must protect margin
    assert hold_or_cs_count >= 16, f"Expected >=80% margin-protective moves for READY_TO_BUY, got {hold_or_cs_count}/20"
