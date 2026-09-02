"""
Action space and legality masks for the Checkout Negotiator.

|A| = 3, hard-capped.  No extensions (architecture decision locked).

Safety invariants enforced here:
  1. Discount ceiling (Δ_max) — structural, via action mask.
  2. Burn-in phase — CONCEDE masked until session threshold reached.
"""

from __future__ import annotations

from enum import IntEnum
from typing import List

from .config import DEFAULT_CONFIG, NegotiatorConfig
from .state import State


class Action(IntEnum):
    """Discrete agent actions — the only moves the MCTS tree can select."""

    CONCEDE = 0  # Offer an incremental +5 % discount
    HOLD = 1  # Maintain current offer, restate value
    CROSS_SELL = 2  # Suggest a complementary product


ALL_ACTIONS: List[Action] = list(Action)
NUM_ACTIONS: int = len(ALL_ACTIONS)


def legality_mask(
    state: State,
    config: NegotiatorConfig = DEFAULT_CONFIG,
    *,
    has_cross_sell_catalog: bool = True,
) -> List[bool]:
    """
    Return a boolean mask of legal actions for *state*.

    Index 0 = CONCEDE, 1 = HOLD, 2 = CROSS_SELL.

    This is the **single point of safety enforcement**.  Every path
    through the MCTS tree calls this function — no action is ever
    evaluated without passing through the mask first.
    """
    mask = [False] * NUM_ACTIONS

    # ── CONCEDE ──────────────────────────────────────────────
    # Legal only if adding one step does not breach the ceiling.
    can_concede = (
        state.discount_pct + config.DISCOUNT_STEP_SIZE
    ) <= config.DISCOUNT_MAX

    # Burn-in override: completely suppress CONCEDE until the
    # opponent model has been warmed up with real session data.
    if config.BURN_IN_ACTIVE:
        can_concede = False

    mask[Action.CONCEDE] = can_concede

    # ── HOLD ─────────────────────────────────────────────────
    # Always legal.  This guarantees the agent always has at
    # least one available move, preventing dead-end nodes.
    mask[Action.HOLD] = True

    # ── CROSS_SELL ───────────────────────────────────────────
    # Legal only if the catalog contains complementary items
    # for the current cart.
    mask[Action.CROSS_SELL] = has_cross_sell_catalog

    return mask


def legal_actions(
    state: State,
    config: NegotiatorConfig = DEFAULT_CONFIG,
    *,
    has_cross_sell_catalog: bool = True,
) -> List[Action]:
    """Return the list of legal Action values for *state*."""
    mask = legality_mask(
        state, config, has_cross_sell_catalog=has_cross_sell_catalog
    )
    return [a for a in ALL_ACTIONS if mask[a]]
