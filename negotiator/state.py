"""
State representation for the Checkout Negotiator.

State tuple: S = (Intent, CartValue, DiscountPct, Turn, History)

All components are discrete and hashable, making each State usable as
a dict key / set member for transposition detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .config import DEFAULT_CONFIG, NegotiatorConfig

# ─── Enumerations ────────────────────────────────────────────────────────────


class UserIntent(IntEnum):
    """Quantized user intent inferred from behavioural signals."""

    BROWSING = 0
    CONSIDERING = 1
    READY_TO_BUY = 2
    ABANDONING = 3
    RETURNING = 4


class CartValueBin(IntEnum):
    """Discretized cart value — bin edges defined in NegotiatorConfig."""

    LOW = 0  # $0 – $50
    MID = 1  # $50 – $150
    HIGH = 2  # $150 – $500
    PREMIUM = 3  # $500+


class UserResponse(IntEnum):
    """Possible user responses to an agent action."""

    ACCEPT = 0
    REJECT = 1
    COUNTER = 2


# ─── History Encoding ────────────────────────────────────────────────────────

# Sentinel values for empty history slots.
NO_OP_ACTION: int = -1
NO_RESPONSE: int = -1

# A single history entry is (action_id, response_id).
HistoryEntry = tuple[int, int]


# ─── State Dataclass ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class State:
    """
    Immutable, hashable state node for the MCTS tree.

    Frozen so that two States with identical fields compare equal
    and share a hash — essential for transposition tables.
    """

    intent: UserIntent
    cart_value: CartValueBin
    discount_pct: int  # 0, 5, 10, 15, 20, 25
    turn: int  # 0 .. T_MAX
    history: tuple[HistoryEntry, ...]  # len == config.HISTORY_LENGTH

    # ── Queries ───────────────────────────────────────────────

    def is_terminal(self, config: NegotiatorConfig = DEFAULT_CONFIG) -> bool:
        """A state is terminal if the game is over."""
        # Turn limit exhausted.
        if self.turn >= config.T_MAX:
            return True
        # Last user response was ACCEPT → conversion happened.
        return bool(self.history and self.history[-1][1] == UserResponse.ACCEPT)

    @property
    def last_response(self) -> UserResponse | None:
        """Return the most recent UserResponse, or None if history is empty."""
        if self.history and self.history[-1][1] != NO_RESPONSE:
            return UserResponse(self.history[-1][1])
        return None

    def cart_midpoint(self, config: NegotiatorConfig = DEFAULT_CONFIG) -> float:
        """Return the synthetic dollar midpoint for this cart bin."""
        return config.CART_BIN_MIDPOINTS[self.cart_value]


# ─── Factories ───────────────────────────────────────────────────────────────


def make_initial_state(
    intent: UserIntent,
    cart_value: CartValueBin,
    config: NegotiatorConfig = DEFAULT_CONFIG,
) -> State:
    """Construct the root state at the start of a checkout session."""
    empty_history: tuple[HistoryEntry, ...] = tuple(
        (NO_OP_ACTION, NO_RESPONSE) for _ in range(config.HISTORY_LENGTH)
    )
    return State(
        intent=intent,
        cart_value=cart_value,
        discount_pct=0,
        turn=0,
        history=empty_history,
    )


def classify_cart_value(
    raw_value: float,
    config: NegotiatorConfig = DEFAULT_CONFIG,
) -> CartValueBin:
    """Bucket a raw dollar amount into a CartValueBin."""
    for label, low, high in config.CART_BIN_EDGES:
        if low <= raw_value < high:
            return CartValueBin[label]
    # Fallback — should only hit for negative values (defensive).
    return CartValueBin.LOW


# ─── Intent Transition Model ────────────────────────────────────────────────

# Maps (current_intent, user_response) → [(next_intent, probability), ...]
# ACCEPT is always terminal so it is omitted from this table.
INTENT_TRANSITIONS: dict[
    UserIntent, dict[UserResponse, list[tuple[UserIntent, float]]]
] = {
    UserIntent.BROWSING: {
        UserResponse.REJECT: [
            (UserIntent.BROWSING, 0.6),
            (UserIntent.ABANDONING, 0.4),
        ],
        UserResponse.COUNTER: [(UserIntent.CONSIDERING, 1.0)],
    },
    UserIntent.CONSIDERING: {
        UserResponse.REJECT: [
            (UserIntent.ABANDONING, 0.4),
            (UserIntent.CONSIDERING, 0.6),
        ],
        UserResponse.COUNTER: [(UserIntent.CONSIDERING, 1.0)],
    },
    UserIntent.READY_TO_BUY: {
        UserResponse.REJECT: [
            (UserIntent.CONSIDERING, 0.7),
            (UserIntent.ABANDONING, 0.3),
        ],
        UserResponse.COUNTER: [(UserIntent.READY_TO_BUY, 1.0)],
    },
    UserIntent.ABANDONING: {
        UserResponse.REJECT: [
            (UserIntent.ABANDONING, 0.8),
            (UserIntent.BROWSING, 0.2),
        ],
        UserResponse.COUNTER: [(UserIntent.CONSIDERING, 1.0)],
    },
    UserIntent.RETURNING: {
        UserResponse.REJECT: [
            (UserIntent.CONSIDERING, 0.5),
            (UserIntent.ABANDONING, 0.5),
        ],
        UserResponse.COUNTER: [
            (UserIntent.CONSIDERING, 0.8),
            (UserIntent.READY_TO_BUY, 0.2),
        ],
    },
}


def transition_intent(
    intent: UserIntent,
    response: UserResponse,
    rng_value: float,
) -> UserIntent:
    """
    Resolve the next intent given a uniform random draw in [0, 1).

    Deterministic for a fixed *rng_value*, enabling reproducible replays.
    """
    if response == UserResponse.ACCEPT:
        return intent  # Terminal — intent is irrelevant.

    transitions = INTENT_TRANSITIONS.get(intent, {}).get(response, [(intent, 1.0)])
    cumulative = 0.0
    for next_intent, prob in transitions:
        cumulative += prob
        if rng_value < cumulative:
            return next_intent
    # Floating-point safety net.
    return transitions[-1][0]


# ─── State Transition ────────────────────────────────────────────────────────


def apply_transition(
    state: State,
    action_id: int,
    response: UserResponse,
    new_intent: UserIntent,
    new_discount_pct: int,
    config: NegotiatorConfig = DEFAULT_CONFIG,
) -> State:
    """
    Produce the successor state after an (action, response) pair.

    NOTE: Cart value bin is held constant through the session.
    Cross-sell revenue is captured in the reward function, not the state,
    to avoid inflating the state space.
    """
    # Slide the history window: drop oldest entry, append newest.
    new_history = state.history[1:] + ((action_id, int(response)),)
    return State(
        intent=new_intent,
        cart_value=state.cart_value,
        discount_pct=new_discount_pct,
        turn=state.turn + 1,
        history=new_history,
    )
