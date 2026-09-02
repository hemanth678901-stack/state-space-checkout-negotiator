"""
Orchestrator: raw session input → state assembly → MCTS search →
audit logging → action execution → templated response.

This is the single entry point that ties together every component.
No LLM calls.  No external network I/O in the decision loop.
"""

from __future__ import annotations

import json
import random as _stdlib_random
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .actions import Action
from .config import DEFAULT_CONFIG, NegotiatorConfig
from .mcts.tree import MCTSEngine
from .opponent_model import OpponentModel
from .state import (
    State,
    UserIntent,
    UserResponse,
    apply_transition,
    classify_cart_value,
    make_initial_state,
    transition_intent,
)

# ═══════════════════════════════════════════════════════════════
#  SQLITE AUDIT TRAIL
# ═══════════════════════════════════════════════════════════════


class AuditLogger:
    """
    Lightweight SQLite logger for every negotiation decision.

    Schema records the full decision context:
      session_id, turn, state_tuple, mcts_visit_distribution,
      selected_action, resulting_discount_pct, timestamp.
    """

    def __init__(self, db_path: str = "negotiations.db") -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS negotiations (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id              TEXT    NOT NULL,
                turn                    INTEGER NOT NULL,
                state_tuple             TEXT    NOT NULL,
                mcts_visit_distribution TEXT    NOT NULL,
                selected_action         TEXT    NOT NULL,
                resulting_discount_pct  INTEGER NOT NULL,
                timestamp               TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_negotiations_session
                ON negotiations(session_id);

            CREATE INDEX IF NOT EXISTS idx_negotiations_ts
                ON negotiations(timestamp);
            """
        )
        self._conn.commit()

    def log(
        self,
        session_id: str,
        turn: int,
        state: State,
        visit_dist: dict[str, Any],
        action: Action,
        discount_pct: int,
    ) -> None:
        """Insert one negotiation decision record."""
        state_dict = {
            "intent": state.intent.name,
            "cart_value": state.cart_value.name,
            "discount_pct": state.discount_pct,
            "turn": state.turn,
            "history": [list(h) for h in state.history],
        }
        self._conn.execute(
            """INSERT INTO negotiations
               (session_id, turn, state_tuple, mcts_visit_distribution,
                selected_action, resulting_discount_pct, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                turn,
                json.dumps(state_dict),
                json.dumps(visit_dist),
                action.name,
                discount_pct,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    def get_session_log(self, session_id: str) -> list[dict[str, Any]]:
        """Return all decision records for a session, ordered by turn."""
        cursor = self._conn.execute(
            """SELECT session_id, turn, state_tuple,
                      mcts_visit_distribution, selected_action,
                      resulting_discount_pct, timestamp
               FROM negotiations
               WHERE session_id = ?
               ORDER BY turn""",
            (session_id,),
        )
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def close(self) -> None:
        self._conn.close()


# ═══════════════════════════════════════════════════════════════
#  RESPONSE TEMPLATE ENGINE
# ═══════════════════════════════════════════════════════════════

_DEFAULT_TEMPLATES_PATH = Path(__file__).parent.parent / "templates" / "responses.yaml"


class ResponseTemplateEngine:
    """
    Maps discrete actions to human-readable messages via YAML templates.

    Placeholder interpolation only — no LLM, no network.
    Placeholders: {discount_pct}, {cart_value}, {cart_bin}, {intent}
    """

    def __init__(self, templates_path: Path | None = None) -> None:
        path = templates_path or _DEFAULT_TEMPLATES_PATH
        if path.exists():
            with open(path, encoding="utf-8") as f:
                self._templates: dict[str, Any] = yaml.safe_load(f)
        else:
            self._templates = self._fallback_templates()

    @staticmethod
    def _fallback_templates() -> dict[str, Any]:
        return {
            "CONCEDE": {
                "messages": ["We'd like to offer you {discount_pct}% off your order!"]
            },
            "HOLD": {"messages": ["This is already excellent value at {cart_value}."]},
            "CROSS_SELL": {
                "messages": ["Complete your purchase with a complementary item."]
            },
        }

    def render(
        self,
        action: Action,
        state: State,
        config: NegotiatorConfig,
    ) -> str:
        """Pick a random template and interpolate placeholders."""
        action_templates = self._templates.get(action.name, {})
        messages: list[str] = action_templates.get("messages", [""])
        template = _stdlib_random.choice(messages)

        discount_pct = state.discount_pct
        if action == Action.CONCEDE:
            discount_pct += config.DISCOUNT_STEP_SIZE

        return template.format(
            discount_pct=discount_pct,
            cart_value=f"${state.cart_midpoint(config):.0f}",
            cart_bin=state.cart_value.name,
            intent=state.intent.name,
        )


# ═══════════════════════════════════════════════════════════════
#  SESSION CONTEXT
# ═══════════════════════════════════════════════════════════════


@dataclass
class SessionContext:
    """In-memory state for one active negotiation session."""

    session_id: str
    state: State  # State from which the last MCTS search was run
    last_action: Action | None = None
    committed_discount_pct: int = 0  # Discount AFTER our last action
    mcts_seed: int | None = None
    is_terminal: bool = False
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ═══════════════════════════════════════════════════════════════
#  NEGOTIATION ENGINE
# ═══════════════════════════════════════════════════════════════


class NegotiationEngine:
    """
    Top-level orchestrator for checkout negotiations.

    Lifecycle per session:
      1. ``start_session()``    → creates initial state, runs MCTS, returns first offer.
      2. ``continue_session()`` → incorporates user response, runs MCTS, returns next offer.
      3. Repeat until terminal (ACCEPT / turn limit / HARD_ABANDON).

    Thread Safety
    -------------
    Not thread-safe.  In a multi-worker deployment, each worker should
    instantiate its own ``NegotiationEngine`` with a shared DB path.
    """

    def __init__(
        self,
        config: NegotiatorConfig = DEFAULT_CONFIG,
        db_path: str = "negotiations.db",
        templates_path: Path | None = None,
    ) -> None:
        self.config = config
        self.opponent = OpponentModel()
        self.audit = AuditLogger(db_path)
        self.templates = ResponseTemplateEngine(templates_path)
        self._sessions: dict[str, SessionContext] = {}

    # ── Public API ────────────────────────────────────────────

    def start_session(
        self,
        cart_value: float,
        intent: UserIntent | str,
        session_id: str | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Initialize a new negotiation and return the first offer."""
        sid = session_id or str(uuid.uuid4())

        # Coerce intent if string passed
        if isinstance(intent, str):
            intent_enum = UserIntent[intent]
        else:
            intent_enum = intent

        cart_bin = classify_cart_value(float(cart_value), self.config)
        state = make_initial_state(intent_enum, cart_bin, self.config)

        ctx = SessionContext(
            session_id=sid,
            state=state,
            mcts_seed=seed,
            is_terminal=False,
        )
        self._sessions[sid] = ctx

        return self._run_turn(ctx)

    def continue_session(
        self,
        session_id: str,
        user_response: UserResponse | str,
    ) -> dict[str, Any]:
        """Advance an existing session with the user's response."""
        ctx = self._sessions.get(session_id)
        if ctx is None:
            raise ValueError(f"Unknown session: {session_id}")

        if ctx.last_action is None:
            raise ValueError(f"Session {session_id} has not been started yet.")

        # Coerce user response if string passed
        if isinstance(user_response, str):
            resp_enum = UserResponse[user_response]
        else:
            resp_enum = user_response

        # ── Guard against continuing already terminal sessions ─
        if ctx.is_terminal or ctx.state.is_terminal(self.config):
            return {
                "session_id": session_id,
                "status": "terminal",
                "turn": ctx.state.turn,
                "outcome": "ALREADY_TERMINATED",
                "message": "Negotiation ended — session already terminated.",
                "discount_pct": ctx.committed_discount_pct,
                "final_discount_pct": ctx.committed_discount_pct,
            }

        # ── Check for terminal ───────────────────────────────
        if resp_enum == UserResponse.ACCEPT:
            return self._finalize(ctx, resp_enum)

        # ── Transition intent stochastically ─────────────────
        new_intent = transition_intent(
            ctx.state.intent,
            resp_enum,
            _stdlib_random.random(),
        )

        # ── Build successor state ────────────────────────────
        next_state = apply_transition(
            state=ctx.state,
            action_id=int(ctx.last_action),
            response=resp_enum,
            new_intent=new_intent,
            new_discount_pct=ctx.committed_discount_pct,
            config=self.config,
        )

        # ── Check for turn-limit terminal ────────────────────
        if next_state.is_terminal(self.config):
            ctx.is_terminal = True
            ctx.state = next_state
            return {
                "session_id": session_id,
                "status": "terminal",
                "turn": next_state.turn,
                "outcome": "TURN_LIMIT",
                "message": "Negotiation ended — turn limit reached.",
                "discount_pct": next_state.discount_pct,
                "final_discount_pct": next_state.discount_pct,
            }

        # ── Run next MCTS turn ───────────────────────────────
        ctx = SessionContext(
            session_id=session_id,
            state=next_state,
            mcts_seed=ctx.mcts_seed,
            created_at=ctx.created_at,
            is_terminal=False,
        )
        self._sessions[session_id] = ctx

        return self._run_turn(ctx)

    def get_session(self, session_id: str) -> SessionContext | None:
        """Return the in-memory session context, if it exists."""
        return self._sessions.get(session_id)

    def get_audit_log(self, session_id: str) -> list[dict[str, Any]]:
        """Return the SQLite audit trail for a session."""
        return self.audit.get_session_log(session_id)

    def close(self) -> None:
        """Release resources."""
        self.audit.close()

    # ── Internal ──────────────────────────────────────────────

    def _run_turn(self, ctx: SessionContext) -> dict[str, Any]:
        """Execute one MCTS search and return the structured result."""
        # Derive per-turn seed for deterministic but varying searches.
        seed = None
        if ctx.mcts_seed is not None:
            seed = ctx.mcts_seed + ctx.state.turn

        engine = MCTSEngine(
            config=self.config,
            opponent=self.opponent,
            seed=seed,
        )
        stats = engine.search_with_stats(ctx.state)

        action = Action[stats["best_action"]]

        # Compute committed discount after this action.
        new_discount = ctx.state.discount_pct
        if action == Action.CONCEDE:
            new_discount += self.config.DISCOUNT_STEP_SIZE

        # Build visit distribution for logging.
        visit_dist = {name: stats[name] for name in ("CONCEDE", "HOLD", "CROSS_SELL")}

        # ── Audit log ────────────────────────────────────────
        self.audit.log(
            session_id=ctx.session_id,
            turn=ctx.state.turn,
            state=ctx.state,
            visit_dist=visit_dist,
            action=action,
            discount_pct=new_discount,
        )

        # ── Render NL response ───────────────────────────────
        message = self.templates.render(action, ctx.state, self.config)

        # ── Update session context ───────────────────────────
        ctx.last_action = action
        ctx.committed_discount_pct = new_discount
        self._sessions[ctx.session_id] = ctx

        return {
            "session_id": ctx.session_id,
            "status": "active",
            "turn": ctx.state.turn,
            "action": action.name,
            "discount_pct": new_discount,
            "message": message,
            "stats": visit_dist,
            "tree_size": stats["tree_size"],
        }

    def _finalize(self, ctx: SessionContext, response: UserResponse) -> dict[str, Any]:
        """Handle terminal ACCEPT response."""
        ctx.is_terminal = True
        return {
            "session_id": ctx.session_id,
            "status": "terminal",
            "turn": ctx.state.turn,
            "outcome": response.name,
            "message": "Deal accepted! Thank you for your purchase.",
            "discount_pct": ctx.committed_discount_pct,
            "final_discount_pct": ctx.committed_discount_pct,
        }
