"""
State-Space Bounded Checkout Negotiator — Interactive Web Demo Dashboard.

Features:
  1. Customer Checkout View: Live shopping cart, dynamic pricing, and negotiation chat.
  2. Real-Time MCTS Inspector: Live state tuple S=(I, V, Δ, t, h), UCT visit distributions, Q-values, and latency metrics.
  3. Live SQLite Audit Trail: Real-time ledger queries directly from SQLite.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import pandas as pd
import streamlit as st

from negotiator.actions import Action
from negotiator.config import NegotiatorConfig
from negotiator.engine import NegotiationEngine
from negotiator.state import CartValueBin, UserIntent, UserResponse

# ─── Page Configuration ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="State-Space Bounded Checkout Negotiator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS Styling ─────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #9CA3AF;
        margin-bottom: 1.5rem;
    }
    .card-box {
        background-color: #1F2937;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 15px;
    }
    .metric-card {
        background-color: #111827;
        border: 1px solid #374151;
        border-radius: 6px;
        padding: 12px;
        text-align: center;
    }
    .badge-active {
        background-color: #065F46;
        color: #34D399;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-concede {
        background-color: #7C2D12;
        color: #F87171;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-hold {
        background-color: #1E3A8A;
        color: #60A5FA;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-cross {
        background-color: #4C1D95;
        color: #C084FC;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Initialize Session State ────────────────────────────────────────────────

if "engine" not in st.session_state:
    config = NegotiatorConfig(BURN_IN_ACTIVE=False, MCTS_ITERATIONS=600)
    st.session_state.engine = NegotiationEngine(config=config, db_path="negotiations.db")

if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_discount_pct" not in st.session_state:
    st.session_state.current_discount_pct = 0

if "cart_items" not in st.session_state:
    st.session_state.cart_items = [
        {"name": "Studio ANC Pro Headphones", "price": 299.99, "qty": 1, "selected": True},
        {"name": "Magnetic 65W Fast Charger", "price": 49.99, "qty": 1, "selected": True},
        {"name": "Braided Kevlar USB-C Cable", "price": 19.99, "qty": 1, "selected": False},
    ]

if "session_status" not in st.session_state:
    st.session_state.session_status = "idle"  # idle, active, accepted, terminated

if "last_mcts_stats" not in st.session_state:
    st.session_state.last_mcts_stats = None

if "last_latency_ms" not in st.session_state:
    st.session_state.last_latency_ms = 0.0

if "user_intent" not in st.session_state:
    st.session_state.user_intent = "CONSIDERING"


# ─── Top Bar Header ──────────────────────────────────────────────────────────

st.markdown('<div class="main-header">⚡ State-Space Bounded Checkout Negotiator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">MCTS-driven e-commerce pricing engine with hard safety bounds and verified audit trails.</div>', unsafe_allow_html=True)

# ─── Layout: Left (Cart & Negotiation), Right (MCTS Inspector & Audit) ───────

col_left, col_right = st.columns([1.1, 1.4], gap="large")

# ═══════════════════════════════════════════════════════════════════════════
#  LEFT COLUMN: CUSTOMER CHECKOUT & LIVE CHAT
# ═══════════════════════════════════════════════════════════════════════════

with col_left:
    st.subheader("🛒 Customer Cart & Checkout")

    # Cart Items Table
    raw_subtotal = 0.0
    with st.container(border=True):
        st.caption("Your Shopping Bag")
        for i, item in enumerate(st.session_state.cart_items):
            c_check, c_name, c_qty, c_price = st.columns([0.15, 0.45, 0.2, 0.2])
            selected = c_check.checkbox("", value=item["selected"], key=f"item_sel_{i}")
            item["selected"] = selected
            c_name.markdown(f"**{item['name']}**")
            qty = c_qty.number_input("Qty", min_value=1, max_value=10, value=item["qty"], key=f"qty_{i}", label_visibility="collapsed")
            item["qty"] = qty
            line_total = item["price"] * qty
            c_price.markdown(f"${line_total:.2f}")
            if selected:
                raw_subtotal += line_total

    # Price Calculation
    discount_multiplier = st.session_state.current_discount_pct / 100.0
    discount_amount = raw_subtotal * discount_multiplier
    final_total = max(0.0, raw_subtotal - discount_amount)

    # Order Summary Box
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Subtotal", f"${raw_subtotal:.2f}")
        c2.metric("Discount Applied", f"{st.session_state.current_discount_pct}%", f"-${discount_amount:.2f}")
        c3.metric("Payable Total", f"${final_total:.2f}")

    # Session Initialization / Control
    st.subheader("💬 AI Negotiator")

    intent_options = ["CONSIDERING", "BROWSING", "READY_TO_BUY", "ABANDONING", "RETURNING"]
    selected_intent = st.selectbox(
        "Simulated Customer Intent Signal:",
        intent_options,
        index=intent_options.index(st.session_state.user_intent),
        help="In production, this is inferred in real-time from dwell time, scroll speed, and cursor intent."
    )
    st.session_state.user_intent = selected_intent

    # Start Session Button if Idle
    if st.session_state.session_status == "idle":
        if st.button("🚀 Start Negotiation", use_container_width=True, type="primary"):
            t0 = time.perf_counter()
            res = st.session_state.engine.start_session(
                cart_value=raw_subtotal,
                intent=UserIntent[selected_intent],
            )
            st.session_state.last_latency_ms = (time.perf_counter() - t0) * 1000
            st.session_state.active_session_id = res["session_id"]
            st.session_state.session_status = "active"
            st.session_state.current_discount_pct = res["discount_pct"]
            st.session_state.last_mcts_stats = res
            st.session_state.chat_history.append({
                "role": "assistant",
                "action": res["action"],
                "text": res["message"],
                "turn": res["turn"],
                "discount": res["discount_pct"],
            })
            st.rerun()

    # Active Chat Container
    with st.container(border=True):
        if not st.session_state.chat_history:
            st.info("Click 'Start Negotiation' above to initiate pricing optimization.")
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    with st.chat_message("user"):
                        st.write(msg["text"])
                else:
                    with st.chat_message("assistant", avatar="⚡"):
                        badge_class = "badge-concede" if msg.get("action") == "CONCEDE" else ("badge-hold" if msg.get("action") == "HOLD" else "badge-cross")
                        st.markdown(f"<span class='{badge_class}'>{msg.get('action', 'SYSTEM')}</span> — {msg['text']}", unsafe_allow_html=True)

    # Customer Action Controls
    if st.session_state.session_status == "active":
        st.caption("Respond to AI's offer:")
        b1, b2, b3 = st.columns(3)

        if b1.button("🤝 Accept Deal", use_container_width=True, type="primary"):
            res = st.session_state.engine.continue_session(
                st.session_state.active_session_id, UserResponse.ACCEPT
            )
            st.session_state.session_status = "accepted"
            st.session_state.chat_history.append({"role": "user", "text": "I accept this offer! Complete purchase."})
            st.session_state.chat_history.append({"role": "assistant", "action": "ACCEPT", "text": res["message"]})
            st.rerun()

        if b2.button("📉 Ask for More Discount", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "text": "Can you offer a better discount?"})
            t0 = time.perf_counter()
            res = st.session_state.engine.continue_session(
                st.session_state.active_session_id, UserResponse.COUNTER
            )
            st.session_state.last_latency_ms = (time.perf_counter() - t0) * 1000
            st.session_state.last_mcts_stats = res
            if res["status"] == "terminal":
                st.session_state.session_status = "terminated"
            else:
                st.session_state.current_discount_pct = res["discount_pct"]
            st.session_state.chat_history.append({
                "role": "assistant",
                "action": res.get("action", "TERMINAL"),
                "text": res["message"],
                "turn": res.get("turn", 0),
                "discount": res.get("discount_pct", st.session_state.current_discount_pct),
            })
            st.rerun()

        if b3.button("❌ Reject Offer", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "text": "No, that's not good enough."})
            t0 = time.perf_counter()
            res = st.session_state.engine.continue_session(
                st.session_state.active_session_id, UserResponse.REJECT
            )
            st.session_state.last_latency_ms = (time.perf_counter() - t0) * 1000
            st.session_state.last_mcts_stats = res
            if res["status"] == "terminal":
                st.session_state.session_status = "terminated"
            else:
                st.session_state.current_discount_pct = res["discount_pct"]
            st.session_state.chat_history.append({
                "role": "assistant",
                "action": res.get("action", "TERMINAL"),
                "text": res["message"],
                "turn": res.get("turn", 0),
                "discount": res.get("discount_pct", st.session_state.current_discount_pct),
            })
            st.rerun()

    elif st.session_state.session_status in ("accepted", "terminated"):
        st.success(f"Session finished with status: **{st.session_state.session_status.upper()}** (Final Discount: {st.session_state.current_discount_pct}%)")
        if st.button("🔄 Start New Customer Session", use_container_width=True):
            st.session_state.session_status = "idle"
            st.session_state.active_session_id = None
            st.session_state.chat_history = []
            st.session_state.current_discount_pct = 0
            st.session_state.last_mcts_stats = None
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
#  RIGHT COLUMN: REAL-TIME MCTS INSPECTOR & LIVE AUDIT TRAIL
# ═══════════════════════════════════════════════════════════════════════════

with col_right:
    tab_mcts, tab_audit = st.tabs(["🧠 Real-Time MCTS Inspector", "📜 Live SQLite Audit Ledger"])

    # ── TAB 1: MCTS INSPECTOR ──────────────────────────────────────────────
    with tab_mcts:
        st.subheader("Monte Carlo Tree Search Decision State")

        if st.session_state.last_mcts_stats and "stats" in st.session_state.last_mcts_stats:
            stats = st.session_state.last_mcts_stats
            dist = stats.get("stats", {})

            # Key KPI Cards
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Selected Move", stats.get("action", "N/A"))
            kpi2.metric("Search Latency", f"{st.session_state.last_latency_ms:.1f} ms")
            kpi3.metric("Tree Nodes", f"{stats.get('tree_size', 0):,}")
            kpi4.metric("Ceiling Invariant", "Δ ≤ 25% (HARD)")

            # Current State Tuple S = (I, V, Δ, t, h)
            st.markdown("#### State Tuple: $S = (I, V, \\Delta, t, h)$")
            st_col1, st_col2, st_col3, st_col4 = st.columns(4)
            st_col1.info(f"**Intent ($I$):**\n{st.session_state.user_intent}")
            st_col2.info(f"**Cart Value ($V$):**\n${raw_subtotal:.2f}")
            st_col3.info(f"**Discount Margin ($\\Delta$):**\n{st.session_state.current_discount_pct}%")
            st_col4.info(f"**Turn ($t$):**\nTurn {stats.get('turn', 0)} / 5")

            # UCT Visit Distribution & Q-Values Table / Visual
            st.markdown("#### Action Space Distribution ($|A| = 3$)")

            dist_data = []
            for act_name in ["CONCEDE", "HOLD", "CROSS_SELL"]:
                a_info = dist.get(act_name, {"visits": 0, "mean_q": 0.0, "visit_share": 0.0})
                dist_data.append({
                    "Action": act_name,
                    "UCB1 Visits": a_info.get("visits", 0),
                    "Visit Share (%)": f"{a_info.get('visit_share', 0.0)*100:.1f}%",
                    "Expected Q-Value": round(a_info.get("mean_q", 0.0), 4),
                    "Status": "✅ WINNER (argmax N)" if act_name == stats.get("action") else "Evaluated",
                })

            df_dist = pd.DataFrame(dist_data)
            st.dataframe(df_dist, use_container_width=True, hide_index=True)

            # Visit share bar chart
            chart_df = pd.DataFrame({
                "Action": ["CONCEDE", "HOLD", "CROSS_SELL"],
                "Visits": [dist.get(a, {}).get("visits", 0) for a in ["CONCEDE", "HOLD", "CROSS_SELL"]]
            }).set_index("Action")
            st.bar_chart(chart_df, color="#4F46E5")

        else:
            st.info("Start a negotiation session on the left to inspect the live MCTS tree search, state tuple, and visit distributions.")

    # ── TAB 2: LIVE AUDIT TRAIL ────────────────────────────────────────────
    with tab_audit:
        st.subheader("SQLite Decision Ledger (negotiations.db)")

        c_refresh, c_filter = st.columns([0.3, 0.7])
        c_refresh.button("🔄 Refresh Ledger")

        # Query SQLite
        if os.path.exists("negotiations.db"):
            conn = sqlite3.connect("negotiations.db")
            query = """
                SELECT id, session_id, turn, selected_action, resulting_discount_pct, 
                       state_tuple, mcts_visit_distribution, timestamp
                FROM negotiations
                ORDER BY id DESC
                LIMIT 50
            """
            try:
                df_audit = pd.read_sql_query(query, conn)
                conn.close()

                if not df_audit.empty:
                    st.caption(f"Showing last {len(df_audit)} immutable decision records:")
                    st.dataframe(
                        df_audit[["id", "session_id", "turn", "selected_action", "resulting_discount_pct", "timestamp"]],
                        use_container_width=True,
                        hide_index=True,
                    )

                    # Expand selected record
                    st.markdown("#### Inspect Trace Record")
                    selected_id = st.selectbox("Select Record ID for Full State & Distribution Dump:", df_audit["id"].tolist())
                    record = df_audit[df_audit["id"] == selected_id].iloc[0]

                    st.json({
                        "id": int(record["id"]),
                        "session_id": str(record["session_id"]),
                        "turn": int(record["turn"]),
                        "selected_action": str(record["selected_action"]),
                        "resulting_discount_pct": int(record["resulting_discount_pct"]),
                        "state_tuple": json.loads(record["state_tuple"]),
                        "mcts_visit_distribution": json.loads(record["mcts_visit_distribution"]),
                        "timestamp": str(record["timestamp"]),
                    })
                else:
                    st.write("No audit records in database yet.")
            except Exception as e:
                st.error(f"Error querying audit database: {e}")
        else:
            st.info("negotiations.db has not been created yet.")
