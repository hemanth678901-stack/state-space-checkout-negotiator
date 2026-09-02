"""
Pydantic schemas for the Checkout Negotiator REST API.

All request/response models are strict — no extra fields allowed.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ─── Request Models ──────────────────────────────────────────────────────────


class NegotiateRequest(BaseModel):
    """
    Unified negotiate endpoint request.

    * To **start** a session: provide ``cart_value`` and ``intent``.
    * To **continue** a session: provide ``session_id`` and ``user_response``.
    """

    session_id: str | None = Field(
        default=None,
        description="Existing session ID. Omit to start a new session.",
    )
    cart_value: float | None = Field(
        default=None,
        ge=0.0,
        description="Raw cart value in USD. Required for new sessions.",
    )
    intent: (
        Literal["BROWSING", "CONSIDERING", "READY_TO_BUY", "ABANDONING", "RETURNING"]
        | None
    ) = Field(
        default=None,
        description="Inferred user intent. Required for new sessions.",
    )
    user_response: Literal["ACCEPT", "REJECT", "COUNTER"] | None = Field(
        default=None,
        description="User's response to the previous offer. Required when continuing.",
    )
    seed: int | None = Field(
        default=None,
        description="RNG seed for deterministic replay (testing only).",
    )


# ─── Response Models ─────────────────────────────────────────────────────────


class ActionStats(BaseModel):
    """Per-action MCTS statistics."""

    visits: int
    mean_q: float
    visit_share: float


class NegotiateResponse(BaseModel):
    """Response from the /negotiate endpoint."""

    session_id: str
    status: Literal["active", "terminal"]
    turn: int = 0
    action: str | None = Field(
        default=None,
        description="Selected action: CONCEDE, HOLD, or CROSS_SELL.",
    )
    discount_pct: int = 0
    message: str = ""
    outcome: str | None = Field(
        default=None,
        description="Terminal outcome: ACCEPT, REJECT, or turn limit.",
    )
    final_discount_pct: int | None = None
    stats: dict[str, ActionStats] | None = None
    tree_size: int | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "0.1.0"
    burn_in_active: bool = True
    mcts_iterations: int = 1000
    discount_max_pct: int = 25
    action_space_size: int = 3
