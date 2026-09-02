"""
Opponent model — P(response | state, action) for MCTS rollouts.

Seed prior: hand-crafted probability table covering all (Intent × Action)
combinations.  Designed to be replaced by a lightweight logistic regression
trained on real session logs once burn-in threshold is reached.
"""

from __future__ import annotations

import random
from typing import Dict, Tuple

from .actions import Action
from .state import State, UserIntent, UserResponse

# ─── Seed Response Probability Table ─────────────────────────────────────────
#
# Key:   (UserIntent, Action)
# Value: (P(ACCEPT), P(REJECT), P(COUNTER))  — must sum to 1.0.
#
_RESPONSE_TABLE: Dict[
    Tuple[UserIntent, Action], Tuple[float, float, float]
] = {
    # ── BROWSING ──────────────────────────────────────────────
    (UserIntent.BROWSING, Action.CONCEDE): (0.20, 0.50, 0.30),
    (UserIntent.BROWSING, Action.HOLD): (0.05, 0.65, 0.30),
    (UserIntent.BROWSING, Action.CROSS_SELL): (0.15, 0.50, 0.35),
    # ── CONSIDERING ───────────────────────────────────────────
    (UserIntent.CONSIDERING, Action.CONCEDE): (0.50, 0.25, 0.25),
    (UserIntent.CONSIDERING, Action.HOLD): (0.15, 0.50, 0.35),
    (UserIntent.CONSIDERING, Action.CROSS_SELL): (0.30, 0.35, 0.35),
    # ── READY_TO_BUY ─────────────────────────────────────────
    (UserIntent.READY_TO_BUY, Action.CONCEDE): (0.75, 0.10, 0.15),
    (UserIntent.READY_TO_BUY, Action.HOLD): (0.60, 0.20, 0.20),
    (UserIntent.READY_TO_BUY, Action.CROSS_SELL): (0.55, 0.15, 0.30),
    # ── ABANDONING ────────────────────────────────────────────
    (UserIntent.ABANDONING, Action.CONCEDE): (0.35, 0.45, 0.20),
    (UserIntent.ABANDONING, Action.HOLD): (0.05, 0.70, 0.25),
    (UserIntent.ABANDONING, Action.CROSS_SELL): (0.10, 0.60, 0.30),
    # ── RETURNING ─────────────────────────────────────────────
    (UserIntent.RETURNING, Action.CONCEDE): (0.45, 0.30, 0.25),
    (UserIntent.RETURNING, Action.HOLD): (0.25, 0.40, 0.35),
    (UserIntent.RETURNING, Action.CROSS_SELL): (0.35, 0.30, 0.35),
}

# Defensive fallback for any missing key.
_FALLBACK_PROBS: Tuple[float, float, float] = (0.15, 0.50, 0.35)


class OpponentModel:
    """
    Stochastic user-response simulator.

    Used by MCTS during node expansion (single sample) and rollout
    (repeated samples).  All randomness is channelled through the
    caller's ``random.Random`` instance for deterministic replay.
    """

    def __init__(self) -> None:
        self._table: Dict[
            Tuple[UserIntent, Action], Tuple[float, float, float]
        ] = dict(_RESPONSE_TABLE)

    # ── Public API ────────────────────────────────────────────

    def response_probs(
        self, state: State, action: Action
    ) -> Tuple[float, float, float]:
        """Return (P(ACCEPT), P(REJECT), P(COUNTER))."""
        return self._table.get(
            (state.intent, action), _FALLBACK_PROBS
        )

    def sample_response(
        self,
        state: State,
        action: Action,
        rng: random.Random,
    ) -> UserResponse:
        """Draw one response from the conditional distribution."""
        p_accept, p_reject, _p_counter = self.response_probs(state, action)
        r = rng.random()
        if r < p_accept:
            return UserResponse.ACCEPT
        if r < p_accept + p_reject:
            return UserResponse.REJECT
        return UserResponse.COUNTER

    def acceptance_probability(
        self, state: State, action: Action
    ) -> float:
        """P(ACCEPT) — used by the ε-greedy rollout heuristic."""
        return self.response_probs(state, action)[0]
