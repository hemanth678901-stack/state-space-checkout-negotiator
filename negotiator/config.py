"""
Hyperparameters and configuration for the Checkout Negotiator.

All locked-in values from architecture review:
  - Cart bins: electronics D2C synthetic tiers
  - Cross-sell margin: flat 30%
  - Concession step: atomic 5%
  - Action space: |A| = 3 (CONCEDE, HOLD, CROSS_SELL) — no extensions
  - Burn-in: mask CONCEDE until simulated threshold
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class NegotiatorConfig:
    """Immutable configuration — one per merchant deployment."""

    # ── State Space ──────────────────────────────────────────────
    # (label, low_inclusive, high_exclusive)
    CART_BIN_EDGES: Tuple[Tuple[str, float, float], ...] = (
        ("LOW", 0.0, 50.0),
        ("MID", 50.0, 150.0),
        ("HIGH", 150.0, 500.0),
        ("PREMIUM", 500.0, float("inf")),
    )
    # Midpoints used for reward computation (PREMIUM capped at $750 synthetic)
    CART_BIN_MIDPOINTS: Tuple[float, ...] = (25.0, 100.0, 325.0, 750.0)

    DISCOUNT_STEP_SIZE: int = 5          # Atomic concession unit (%)
    DISCOUNT_MAX: int = 25               # Hard ceiling (%)
    T_MAX: int = 5                       # Max turns per session
    HISTORY_LENGTH: int = 3              # Sliding window of (action, response) pairs

    # ── MCTS ─────────────────────────────────────────────────────
    MCTS_ITERATIONS: int = 1000          # Search budget per move
    CP: float = 1.4142135623730951       # UCB1 exploration constant (√2)

    # ── Rollout ──────────────────────────────────────────────────
    ROLLOUT_EPSILON: float = 0.3         # ε-greedy parameter for default policy

    # ── Reward ───────────────────────────────────────────────────
    ALPHA: float = 1.0                   # Conversion weight
    BETA: float = 0.8                    # Margin-erosion sensitivity
    GAMMA_CS: float = 0.3                # Cross-sell uplift weight
    GAMMA_DISCOUNT: float = 0.95         # Temporal discount factor
    LAMBDA_ABANDON: float = 0.5          # Abandonment penalty multiplier
    CROSS_SELL_MARGIN: float = 0.30      # Flat assumed margin on cross-sell items

    # ── Burn-in ──────────────────────────────────────────────────
    BURN_IN_THRESHOLD: int = 10_000      # Sessions needed before enabling CONCEDE
    BURN_IN_ACTIVE: bool = True          # Flip to False after threshold reached

    # ── Reward Normalization Bounds ──────────────────────────────
    # Conservative bounds for mapping raw reward → [0, 1].
    # Worst: immediate abandon on PREMIUM cart.
    # Best:  immediate accept at 0% discount on PREMIUM cart (step + terminal).
    R_MIN: float = -500.0
    R_MAX: float = 1500.0


DEFAULT_CONFIG = NegotiatorConfig()
