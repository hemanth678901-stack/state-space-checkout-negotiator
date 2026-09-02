"""
FastAPI routes for the Checkout Negotiator.

Endpoints:
  POST /negotiate  — Start or continue a negotiation session.
  GET  /negotiate/{session_id}/audit — Retrieve audit trail.
  GET  /health     — Service health check.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from negotiator.config import NegotiatorConfig
from negotiator.engine import NegotiationEngine
from negotiator.state import UserIntent, UserResponse

from .schemas import HealthResponse, NegotiateRequest, NegotiateResponse

# ─── App & Engine ────────────────────────────────────────────────────────────

app = FastAPI(
    title="Checkout Negotiator",
    description=(
        "MCTS-driven checkout negotiation engine. "
        "No LLM in the decision loop — pure bounded-state search."
    ),
    version="0.1.0",
)

# Engine is instantiated once at import time.
# In production, swap to a per-worker factory via lifespan events.
_config = NegotiatorConfig()
_engine = NegotiationEngine(config=_config, db_path="negotiations.db")


# ─── Lifespan ────────────────────────────────────────────────────────────────

@app.on_event("shutdown")
def _shutdown() -> None:
    _engine.close()


# ─── Routes ──────────────────────────────────────────────────────────────────


@app.post("/negotiate", response_model=NegotiateResponse)
def negotiate(request: NegotiateRequest) -> NegotiateResponse:
    """
    Unified negotiation endpoint.

    **Start a session** — provide ``cart_value`` and ``intent``:
    ```json
    {"cart_value": 299.99, "intent": "CONSIDERING"}
    ```

    **Continue a session** — provide ``session_id`` and ``user_response``:
    ```json
    {"session_id": "abc-123", "user_response": "REJECT"}
    ```
    """
    try:
        if request.session_id is None:
            # ── New session ──────────────────────────────────
            if request.cart_value is None or request.intent is None:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "New sessions require 'cart_value' and 'intent'. "
                        "To continue an existing session, provide 'session_id'."
                    ),
                )
            intent = UserIntent[request.intent]
            result = _engine.start_session(
                cart_value=request.cart_value,
                intent=intent,
                seed=request.seed,
            )
        else:
            # ── Continue session ─────────────────────────────
            if request.user_response is None:
                raise HTTPException(
                    status_code=422,
                    detail="Continuing a session requires 'user_response'.",
                )
            response = UserResponse[request.user_response]
            result = _engine.continue_session(
                session_id=request.session_id,
                user_response=response,
            )

        return NegotiateResponse(**result)

    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid enum value: {exc}",
        ) from exc


@app.get("/negotiate/{session_id}/audit")
def get_audit(session_id: str) -> list:
    """Retrieve the full audit trail for a session."""
    log = _engine.get_audit_log(session_id)
    if not log:
        raise HTTPException(
            status_code=404,
            detail=f"No records found for session '{session_id}'.",
        )
    return log


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Service health and configuration summary."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        burn_in_active=_config.BURN_IN_ACTIVE,
        mcts_iterations=_config.MCTS_ITERATIONS,
        discount_max_pct=_config.DISCOUNT_MAX,
        action_space_size=3,
    )
