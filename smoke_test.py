"""Smoke test: full MCTS engine integration."""

import time

from negotiator.config import NegotiatorConfig
from negotiator.mcts.node import MCTSNode
from negotiator.mcts.tree import MCTSEngine
from negotiator.state import CartValueBin, UserIntent, make_initial_state


def test_burnin():
    """Burn-in mode: CONCEDE must be completely masked."""
    config = NegotiatorConfig(BURN_IN_ACTIVE=True, MCTS_ITERATIONS=1000)
    engine = MCTSEngine(config=config, seed=42)
    state = make_initial_state(UserIntent.CONSIDERING, CartValueBin.HIGH)

    start = time.perf_counter_ns()
    stats = engine.search_with_stats(state)
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000

    print("=== BURN-IN MODE (CONCEDE masked) ===")
    print(f"  Best action:     {stats['best_action']}")
    print(f"  CONCEDE visits:  {stats['CONCEDE']['visits']}")
    print(f"  HOLD visits:     {stats['HOLD']['visits']}")
    print(f"  CROSS_SELL:      {stats['CROSS_SELL']['visits']}")
    print(f"  Tree size:       {stats['tree_size']} nodes")
    print(f"  Latency:         {elapsed_ms:.1f} ms")
    assert stats["CONCEDE"]["visits"] == 0, "CONCEDE must have 0 visits during burn-in"
    print("  PASSED\n")


def test_full_mode():
    """Full mode: all three actions available."""
    config = NegotiatorConfig(BURN_IN_ACTIVE=False, MCTS_ITERATIONS=1000)
    engine = MCTSEngine(config=config, seed=42)
    state = make_initial_state(UserIntent.ABANDONING, CartValueBin.PREMIUM)

    start = time.perf_counter_ns()
    stats = engine.search_with_stats(state)
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000

    print("=== FULL MODE (ABANDONING user, PREMIUM cart) ===")
    print(f"  Best action:     {stats['best_action']}")
    print(
        f"  CONCEDE:         {stats['CONCEDE']['visits']} visits, Q={stats['CONCEDE']['mean_q']:.4f}"
    )
    print(
        f"  HOLD:            {stats['HOLD']['visits']} visits, Q={stats['HOLD']['mean_q']:.4f}"
    )
    print(
        f"  CROSS_SELL:      {stats['CROSS_SELL']['visits']} visits, Q={stats['CROSS_SELL']['mean_q']:.4f}"
    )
    print(f"  Tree size:       {stats['tree_size']} nodes")
    print(f"  Latency:         {elapsed_ms:.1f} ms")
    assert stats["total_iterations"] == 1000
    print("  PASSED\n")


def test_discount_ceiling():
    """Verify Δ_max invariant: no node in the tree exceeds 25%."""
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

    print("=== DISCOUNT CEILING INVARIANT ===")
    print(f"  Checked {root.tree_size()} nodes")
    if violations:
        print(f"  FAILED: {len(violations)} violations: {violations}")
    else:
        print("  PASSED: No node exceeds 25% discount ceiling")
    assert not violations
    print()


def test_deterministic_replay():
    """Same seed → same action."""
    config = NegotiatorConfig(BURN_IN_ACTIVE=False, MCTS_ITERATIONS=500)
    state = make_initial_state(UserIntent.CONSIDERING, CartValueBin.MID)

    action1 = MCTSEngine(config=config, seed=123).search(state)
    action2 = MCTSEngine(config=config, seed=123).search(state)

    print("=== DETERMINISTIC REPLAY ===")
    print(f"  Run 1: {action1.name}")
    print(f"  Run 2: {action2.name}")
    assert action1 == action2, f"Non-deterministic: {action1} != {action2}"
    print("  PASSED\n")


if __name__ == "__main__":
    test_burnin()
    test_full_mode()
    test_discount_ceiling()
    test_deterministic_replay()
    print("=" * 50)
    print("ALL TESTS PASSED")
