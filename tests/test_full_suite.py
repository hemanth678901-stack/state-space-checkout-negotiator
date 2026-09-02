"""
Comprehensive test suite for the Checkout Negotiator.

Tests:
  1. Bug fix verification  — CONCEDE selected when it has max visits
  2. Burn-in enforcement   — CONCEDE structurally masked
  3. Discount ceiling       — Δ_max invariant across full tree
  4. Deterministic replay   — same seed → same action
  5. Reward function        — boundary conditions
  6. Engine + SQLite audit  — full session lifecycle + audit trail
  7. API schema validation  — request/response serialization
"""

import json
import os
import sqlite3
import time

# ─── 1. Bug Fix Verification ────────────────────────────────────────────────


def test_bugfix_concede_selected_when_max_visits():
    """
    REGRESSION: _best_action was a @staticmethod that called
    legal_actions(root.state) without config, defaulting to
    DEFAULT_CONFIG where BURN_IN_ACTIVE=True.  This silently
    masked CONCEDE from argmax even though it had 603/1000 visits.

    After the fix, CONCEDE must be selected.
    """
    from negotiator.mcts.tree import MCTSEngine
    from negotiator.state import make_initial_state, UserIntent, CartValueBin
    from negotiator.config import NegotiatorConfig

    config = NegotiatorConfig(BURN_IN_ACTIVE=False, MCTS_ITERATIONS=1000)
    engine = MCTSEngine(config=config, seed=42)
    state = make_initial_state(UserIntent.ABANDONING, CartValueBin.PREMIUM)

    start = time.perf_counter_ns()
    stats = engine.search_with_stats(state)
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000

    print("\n=== TEST 1: BUG FIX — CONCEDE MUST WIN ARGMAX N(s,a) ===")
    print(f"  CONCEDE:    {stats['CONCEDE']['visits']} visits, Q={stats['CONCEDE']['mean_q']:.4f}")
    print(f"  HOLD:       {stats['HOLD']['visits']} visits, Q={stats['HOLD']['mean_q']:.4f}")
    print(f"  CROSS_SELL: {stats['CROSS_SELL']['visits']} visits, Q={stats['CROSS_SELL']['mean_q']:.4f}")
    print(f"  Selected:   {stats['best_action']}")
    print(f"  Latency:    {elapsed_ms:.1f} ms")

    # CONCEDE must have the most visits AND be selected.
    assert stats["CONCEDE"]["visits"] > stats["HOLD"]["visits"], \
        f"CONCEDE visits ({stats['CONCEDE']['visits']}) should exceed HOLD ({stats['HOLD']['visits']})"
    assert stats["CONCEDE"]["visits"] > stats["CROSS_SELL"]["visits"], \
        f"CONCEDE visits ({stats['CONCEDE']['visits']}) should exceed CROSS_SELL ({stats['CROSS_SELL']['visits']})"
    assert stats["best_action"] == "CONCEDE", \
        f"Expected CONCEDE but got {stats['best_action']} — _best_action bug not fixed!"
    print("  PASSED ✓\n")


# ─── 2. Burn-in Enforcement ─────────────────────────────────────────────────


def test_burnin_masks_concede():
    """During burn-in, CONCEDE must have exactly 0 visits."""
    from negotiator.mcts.tree import MCTSEngine
    from negotiator.state import make_initial_state, UserIntent, CartValueBin
    from negotiator.config import NegotiatorConfig

    config = NegotiatorConfig(BURN_IN_ACTIVE=True, MCTS_ITERATIONS=1000)
    engine = MCTSEngine(config=config, seed=42)
    state = make_initial_state(UserIntent.CONSIDERING, CartValueBin.HIGH)

    stats = engine.search_with_stats(state)

    print("=== TEST 2: BURN-IN — CONCEDE MASKED ===")
    print(f"  CONCEDE visits: {stats['CONCEDE']['visits']}")
    print(f"  HOLD visits:    {stats['HOLD']['visits']}")
    print(f"  CROSS_SELL:     {stats['CROSS_SELL']['visits']}")
    print(f"  Selected:       {stats['best_action']}")

    assert stats["CONCEDE"]["visits"] == 0, "CONCEDE must have 0 visits during burn-in"
    assert stats["best_action"] in ("HOLD", "CROSS_SELL"), \
        f"During burn-in, only HOLD or CROSS_SELL allowed, got {stats['best_action']}"
    print("  PASSED ✓\n")


# ─── 3. Discount Ceiling Invariant ──────────────────────────────────────────


def test_discount_ceiling_invariant():
    """No node in the MCTS tree may have discount_pct > DISCOUNT_MAX."""
    from negotiator.mcts.tree import MCTSEngine
    from negotiator.mcts.node import MCTSNode
    from negotiator.state import make_initial_state, UserIntent, CartValueBin
    from negotiator.config import NegotiatorConfig

    config = NegotiatorConfig(BURN_IN_ACTIVE=False, MCTS_ITERATIONS=1000)
    engine = MCTSEngine(config=config, seed=99)
    state = make_initial_state(UserIntent.ABANDONING, CartValueBin.PREMIUM)

    root = MCTSNode(state=state)
    for _ in range(config.MCTS_ITERATIONS):
        node = engine._select(root)
        if not node.state.is_terminal(config):
            node = engine._expand(node)
        reward = engine._simulate(node.state)
        engine._backpropagate(node, reward)

    violations = []

    def _check(n, depth=0):
        if n.state.discount_pct > config.DISCOUNT_MAX:
            violations.append((depth, n.state.discount_pct))
        for children_list in n.children.values():
            for child in children_list:
                _check(child, depth + 1)

    _check(root)

    print("=== TEST 3: DISCOUNT CEILING INVARIANT ===")
    print(f"  Nodes checked: {root.tree_size()}")
    print(f"  Violations:    {len(violations)}")
    assert not violations, f"CEILING VIOLATED at {violations}"
    print("  PASSED ✓\n")


# ─── 4. Deterministic Replay ────────────────────────────────────────────────


def test_deterministic_replay():
    """Same seed → same action, same visit counts."""
    from negotiator.mcts.tree import MCTSEngine
    from negotiator.state import make_initial_state, UserIntent, CartValueBin
    from negotiator.config import NegotiatorConfig

    config = NegotiatorConfig(BURN_IN_ACTIVE=False, MCTS_ITERATIONS=500)
    state = make_initial_state(UserIntent.CONSIDERING, CartValueBin.MID)

    stats1 = MCTSEngine(config=config, seed=123).search_with_stats(state)
    stats2 = MCTSEngine(config=config, seed=123).search_with_stats(state)

    print("=== TEST 4: DETERMINISTIC REPLAY ===")
    print(f"  Run 1: {stats1['best_action']} (CONCEDE={stats1['CONCEDE']['visits']}, HOLD={stats1['HOLD']['visits']}, CS={stats1['CROSS_SELL']['visits']})")
    print(f"  Run 2: {stats2['best_action']} (CONCEDE={stats2['CONCEDE']['visits']}, HOLD={stats2['HOLD']['visits']}, CS={stats2['CROSS_SELL']['visits']})")

    assert stats1["best_action"] == stats2["best_action"], "Non-deterministic action selection!"
    assert stats1["CONCEDE"]["visits"] == stats2["CONCEDE"]["visits"], "Non-deterministic visit counts!"
    print("  PASSED ✓\n")


# ─── 5. Reward Function Boundaries ──────────────────────────────────────────


def test_reward_boundaries():
    """Verify reward normalization maps to [0, 1] for extreme cases."""
    from negotiator.reward import step_reward, terminal_reward, normalize_reward
    from negotiator.state import make_initial_state, UserIntent, CartValueBin, UserResponse
    from negotiator.actions import Action
    from negotiator.config import NegotiatorConfig

    config = NegotiatorConfig()
    state = make_initial_state(UserIntent.READY_TO_BUY, CartValueBin.PREMIUM)

    # Best case: accept at 0% on PREMIUM
    r_best = step_reward(state, Action.HOLD, UserResponse.ACCEPT, config)
    # Worst case: concede cost at 25%
    from negotiator.state import State
    worst_state = State(
        intent=UserIntent.ABANDONING,
        cart_value=CartValueBin.PREMIUM,
        discount_pct=20,
        turn=4,
        history=((-1, -1), (-1, -1), (-1, -1)),
    )
    r_worst = step_reward(worst_state, Action.CONCEDE, UserResponse.REJECT, config)

    n_best = normalize_reward(r_best, config)
    n_worst = normalize_reward(r_worst, config)

    print("=== TEST 5: REWARD BOUNDARIES ===")
    print(f"  Best raw:       {r_best:.2f} → normalized: {n_best:.4f}")
    print(f"  Worst raw:      {r_worst:.2f} → normalized: {n_worst:.4f}")

    assert 0.0 <= n_best <= 1.0, f"Normalized best out of range: {n_best}"
    assert 0.0 <= n_worst <= 1.0, f"Normalized worst out of range: {n_worst}"
    print("  PASSED ✓\n")


# ─── 6. Engine + SQLite Audit ────────────────────────────────────────────────


def test_engine_session_lifecycle():
    """Full session: start → continue (REJECT) → continue (ACCEPT). Verify SQLite trail."""
    from negotiator.engine import NegotiationEngine
    from negotiator.state import UserIntent, UserResponse
    from negotiator.config import NegotiatorConfig

    db_path = "test_negotiations.db"
    # Clean up from any previous run.
    if os.path.exists(db_path):
        os.remove(db_path)

    config = NegotiatorConfig(BURN_IN_ACTIVE=True, MCTS_ITERATIONS=200)
    engine = NegotiationEngine(config=config, db_path=db_path, templates_path=None)

    try:
        # Turn 0: start
        result0 = engine.start_session(
            cart_value=299.99,
            intent=UserIntent.CONSIDERING,
            seed=42,
        )
        sid = result0["session_id"]

        print("=== TEST 6: ENGINE + SQLITE AUDIT ===")
        print(f"  Session: {sid}")
        print(f"  Turn 0:  action={result0['action']}, discount={result0['discount_pct']}%, msg='{result0['message'][:60]}...'")

        assert result0["status"] == "active"
        assert result0["action"] in ("HOLD", "CROSS_SELL"), \
            f"During burn-in, CONCEDE should not appear, got {result0['action']}"

        # Turn 1: user rejects
        result1 = engine.continue_session(sid, UserResponse.REJECT)
        print(f"  Turn 1:  action={result1['action']}, discount={result1['discount_pct']}%")
        assert result1["status"] == "active"

        # Turn 2: user accepts
        result2 = engine.continue_session(sid, UserResponse.ACCEPT)
        print(f"  Turn 2:  outcome={result2.get('outcome', 'N/A')}, final_discount={result2.get('final_discount_pct', 'N/A')}%")
        assert result2["status"] == "terminal"
        assert result2["outcome"] == "ACCEPT"

        # Verify SQLite audit trail
        audit = engine.get_audit_log(sid)
        print(f"  Audit records: {len(audit)}")

        assert len(audit) == 2, f"Expected 2 audit records (turns 0,1), got {len(audit)}"

        # Verify audit schema
        for record in audit:
            assert "session_id" in record
            assert "turn" in record
            assert "state_tuple" in record
            assert "mcts_visit_distribution" in record
            assert "selected_action" in record
            assert "resulting_discount_pct" in record
            assert "timestamp" in record

            # Verify state_tuple is valid JSON
            state_data = json.loads(record["state_tuple"])
            assert "intent" in state_data
            assert "cart_value" in state_data
            assert "discount_pct" in state_data

            # Verify visit distribution is valid JSON
            visit_data = json.loads(record["mcts_visit_distribution"])
            assert "CONCEDE" in visit_data
            assert "HOLD" in visit_data
            assert "CROSS_SELL" in visit_data

        print("  PASSED ✓\n")

    finally:
        engine.close()
        if os.path.exists(db_path):
            os.remove(db_path)


# ─── 7. API Schema Validation ───────────────────────────────────────────────


def test_api_schemas():
    """Verify Pydantic models serialize/deserialize correctly."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from api.schemas import NegotiateRequest, NegotiateResponse, HealthResponse

    # New session request
    req_new = NegotiateRequest(cart_value=199.99, intent="CONSIDERING")
    assert req_new.session_id is None
    assert req_new.cart_value == 199.99

    # Continue session request
    req_cont = NegotiateRequest(session_id="abc-123", user_response="REJECT")
    assert req_cont.session_id == "abc-123"
    assert req_cont.user_response == "REJECT"

    # Response model
    resp = NegotiateResponse(
        session_id="abc-123",
        status="active",
        turn=0,
        action="HOLD",
        discount_pct=0,
        message="Great value!",
    )
    data = resp.model_dump()
    assert data["action"] == "HOLD"

    # Health response
    health = HealthResponse()
    assert health.action_space_size == 3

    print("=== TEST 7: API SCHEMA VALIDATION ===")
    print(f"  NegotiateRequest (new):      OK")
    print(f"  NegotiateRequest (continue): OK")
    print(f"  NegotiateResponse:           OK")
    print(f"  HealthResponse:              OK")
    print("  PASSED ✓\n")


# ─── 8. Response Template Rendering ─────────────────────────────────────────


def test_response_templates():
    """Verify templates load from YAML and render with placeholders."""
    from negotiator.engine import ResponseTemplateEngine
    from negotiator.state import make_initial_state, UserIntent, CartValueBin
    from negotiator.actions import Action
    from negotiator.config import NegotiatorConfig
    from pathlib import Path

    config = NegotiatorConfig()
    state = make_initial_state(UserIntent.CONSIDERING, CartValueBin.HIGH)
    templates_path = Path(__file__).parent / "templates" / "responses.yaml"

    engine = ResponseTemplateEngine(templates_path)

    print("=== TEST 8: RESPONSE TEMPLATES ===")
    for action in Action:
        msg = engine.render(action, state, config)
        print(f"  {action.name}: \"{msg}\"")
        assert len(msg) > 10, f"Template for {action.name} too short: '{msg}'"
        assert "{" not in msg, f"Unresolved placeholder in {action.name}: '{msg}'"

    print("  PASSED ✓\n")


# ─── Runner ─────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    test_bugfix_concede_selected_when_max_visits()
    test_burnin_masks_concede()
    test_discount_ceiling_invariant()
    test_deterministic_replay()
    test_reward_boundaries()
    test_engine_session_lifecycle()
    test_api_schemas()
    test_response_templates()

    print("=" * 60)
    print("  ALL 8 TESTS PASSED")
    print("=" * 60)
