"""
MCTS search engine for the Checkout Negotiator.

Implements all four phases of Monte Carlo Tree Search:
  1. SELECT      — UCB1 tree traversal with action-mask enforcement.
  2. EXPAND      — One-child expansion with stochastic opponent sampling.
  3. SIMULATE    — ε-greedy rollout to terminal state.
  4. BACKPROPAGATE — Reward propagation up the tree ancestry.

No neural networks.  No LLM calls.  Pure table-lookup search within
a bounded discrete state space.  Deterministic under a fixed RNG seed.
"""

from __future__ import annotations

import math
import random

from ..actions import Action, legal_actions
from ..config import DEFAULT_CONFIG, NegotiatorConfig
from ..opponent_model import OpponentModel
from ..reward import normalize_reward, step_reward, terminal_reward
from ..state import (
    State,
    apply_transition,
    transition_intent,
)
from .node import MCTSNode


class MCTSEngine:
    """
    Monte Carlo Tree Search policy for checkout negotiation.

    Usage
    -----
    >>> engine = MCTSEngine(seed=42)
    >>> best = engine.search(root_state)
    >>> print(best)           # Action.HOLD
    >>> stats = engine.search_with_stats(root_state)
    >>> print(stats)          # detailed per-action breakdown

    Thread Safety
    -------------
    Not thread-safe.  Each concurrent session must create its own
    ``MCTSEngine`` instance (they are lightweight — no GPU, no model).
    """

    __slots__ = ("config", "opponent", "rng")

    def __init__(
        self,
        config: NegotiatorConfig = DEFAULT_CONFIG,
        opponent: OpponentModel | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.opponent = opponent or OpponentModel()
        self.rng = random.Random(seed)

    # ═══════════════════════════════════════════════════════════
    #  PUBLIC API
    # ═══════════════════════════════════════════════════════════

    def search(self, root_state: State) -> Action:
        """
        Run MCTS for ``config.MCTS_ITERATIONS`` and return the
        best action at the root.

        Final selection uses **most-visited** (robust child), not
        highest-mean-Q, following standard MCTS best practice.
        """
        root = MCTSNode(state=root_state)

        for _ in range(self.config.MCTS_ITERATIONS):
            node = self._select(root)
            if not node.state.is_terminal(self.config):
                node = self._expand(node)
            reward = self._simulate(node.state)
            self._backpropagate(node, reward)

        return self._best_action(root)

    def search_with_stats(self, root_state: State) -> dict:
        """
        Run MCTS and return a diagnostic dict with per-action stats.

        Useful for debugging, logging, and post-hoc audit of the
        negotiation decision.
        """
        root = MCTSNode(state=root_state)

        for _ in range(self.config.MCTS_ITERATIONS):
            node = self._select(root)
            if not node.state.is_terminal(self.config):
                node = self._expand(node)
            reward = self._simulate(node.state)
            self._backpropagate(node, reward)

        stats: dict = {}
        for action in Action:
            n = root.action_visits[action]
            q = root.action_rewards[action]
            stats[action.name] = {
                "visits": n,
                "mean_q": round(q / n, 6) if n > 0 else 0.0,
                "visit_share": (
                    round(n / root.visit_count, 4) if root.visit_count > 0 else 0.0
                ),
            }
        stats["best_action"] = self._best_action(root).name
        stats["total_iterations"] = root.visit_count
        stats["tree_size"] = root.tree_size()
        return stats

    # ═══════════════════════════════════════════════════════════
    #  PHASE 1 — SELECTION (UCB1)
    # ═══════════════════════════════════════════════════════════

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Descend the tree from *node* until we reach either:
          (a) a terminal state, or
          (b) a node with at least one unexplored legal action.

        At decision nodes → pick action via masked UCB1.
        At chance branches → sample uniformly among existing children
        for the selected action (stochastic opponent simulation).
        """
        while not node.state.is_terminal(self.config):
            legal = legal_actions(node.state, self.config)
            unexplored = [a for a in legal if a not in node.expanded_actions()]

            if unexplored:
                # Node has untried actions → stop here for expansion.
                return node

            if not node.children:
                # No children at all (should not happen — HOLD is
                # always legal), but guard defensively.
                return node

            # All legal actions have been tried at least once.
            # Select the best via UCB1.
            best_action = self._ucb1_select(node, legal)
            children_for_action = node.children.get(best_action, [])
            if not children_for_action:
                return node

            # Chance node: pick a child uniformly.
            # Over many iterations this converges to the true
            # response distribution because expansion already
            # samples from P(r|S,a).
            node = self.rng.choice(children_for_action)

        return node

    def _ucb1_select(self, node: MCTSNode, legal: list[Action]) -> Action:
        """
        Upper Confidence Bound for Trees (UCT / UCB1).

        a* = argmax_a [ Q(s,a)/N(s,a)  +  Cp · √(ln N(s) / N(s,a)) ]

        Actions not in *legal* are structurally excluded — they can
        never be selected, preserving the discount ceiling invariant.
        """
        ln_parent = math.log(node.visit_count) if node.visit_count > 0 else 0.0
        best_score = -math.inf
        best_action = legal[0]  # Safe default — list is never empty.

        for action in legal:
            n_a = node.action_visits[action]
            if n_a == 0:
                # Unvisited action gets infinite priority — ensures
                # every legal action is tried at least once.
                return action

            exploitation = node.action_rewards[action] / n_a
            exploration = self.config.CP * math.sqrt(ln_parent / n_a)
            score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    # ═══════════════════════════════════════════════════════════
    #  PHASE 2 — EXPANSION
    # ═══════════════════════════════════════════════════════════

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Expand one unexplored legal action from *node*.

        1. Pick a random untried legal action.
        2. Sample an opponent response from P(r | S, a).
        3. Compute the deterministic successor state.
        4. Create and attach the child node.

        Returns the newly created child.
        """
        legal = legal_actions(node.state, self.config)
        unexplored = [a for a in legal if a not in node.expanded_actions()]

        if not unexplored:
            return node  # Fully expanded — nothing to add.

        action = self.rng.choice(unexplored)

        # ── Sample opponent response ─────────────────────────
        response = self.opponent.sample_response(node.state, action, self.rng)

        # ── Compute successor state ─────────────────────────
        new_discount = node.state.discount_pct
        if action == Action.CONCEDE:
            new_discount += self.config.DISCOUNT_STEP_SIZE

        new_intent = transition_intent(node.state.intent, response, self.rng.random())

        child_state = apply_transition(
            state=node.state,
            action_id=int(action),
            response=response,
            new_intent=new_intent,
            new_discount_pct=new_discount,
            config=self.config,
        )

        child = MCTSNode(
            state=child_state,
            parent=node,
            incoming_action=action,
            incoming_response=int(response),
        )
        node.add_child(action, child)
        return child

    # ═══════════════════════════════════════════════════════════
    #  PHASE 3 — SIMULATION (ε-GREEDY ROLLOUT)
    # ═══════════════════════════════════════════════════════════

    def _simulate(self, state: State) -> float:
        """
        ε-greedy rollout from *state* to a terminal state.

        With probability (1 − ε): pick the action with highest
        P(ACCEPT) from the opponent model (greedy heuristic).
        With probability ε: pick a random legal action.

        Returns the normalized cumulative discounted reward.
        """
        cumulative = 0.0
        gamma_pow = 1.0
        current = state

        while not current.is_terminal(self.config):
            legal = legal_actions(current, self.config)
            if not legal:
                break  # Defensive — should never happen.

            # ── ε-greedy action selection ────────────────────
            if self.rng.random() > self.config.ROLLOUT_EPSILON:
                # Greedy: pick the action most likely to convert.
                action = max(
                    legal,
                    key=lambda a: self.opponent.acceptance_probability(current, a),
                )
            else:
                action = self.rng.choice(legal)

            # ── Sample opponent response ─────────────────────
            response = self.opponent.sample_response(current, action, self.rng)

            # ── Accumulate discounted step reward ────────────
            r = step_reward(current, action, response, self.config)
            cumulative += gamma_pow * r
            gamma_pow *= self.config.GAMMA_DISCOUNT

            # ── Transition to successor state ────────────────
            new_discount = current.discount_pct
            if action == Action.CONCEDE:
                new_discount += self.config.DISCOUNT_STEP_SIZE

            new_intent = transition_intent(current.intent, response, self.rng.random())

            current = apply_transition(
                state=current,
                action_id=int(action),
                response=response,
                new_intent=new_intent,
                new_discount_pct=new_discount,
                config=self.config,
            )

        # ── Terminal reward ──────────────────────────────────
        cumulative += gamma_pow * terminal_reward(current, self.config)

        return normalize_reward(cumulative, self.config)

    # ═══════════════════════════════════════════════════════════
    #  PHASE 4 — BACKPROPAGATION
    # ═══════════════════════════════════════════════════════════

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """
        Propagate *reward* from *node* up to the root.

        Updates both the node-level aggregates (visit_count,
        total_reward) and the parent's per-action aggregates
        (action_visits, action_rewards) for the incoming action.
        """
        current: MCTSNode | None = node
        while current is not None:
            current.visit_count += 1
            current.total_reward += reward

            if current.parent is not None and current.incoming_action is not None:
                action = current.incoming_action
                current.parent.action_visits[action] += 1
                current.parent.action_rewards[action] += reward

            current = current.parent

    # ═══════════════════════════════════════════════════════════
    #  ACTION SELECTION
    # ═══════════════════════════════════════════════════════════

    def _best_action(self, root: MCTSNode) -> Action:
        """
        Select the final action at the root by **most-visited child**
        (robust child selection), not highest mean Q.

        Most-visited is standard MCTS best practice — it is more
        robust to outlier rollouts than argmax(Q/N).

        BUG FIX: Previously a @staticmethod that called
        legal_actions(root.state) without config, falling back to
        DEFAULT_CONFIG where BURN_IN_ACTIVE=True.  This silently
        masked CONCEDE from the argmax even when the search tree
        had explored it.  Now uses self.config.
        """
        legal = legal_actions(root.state, self.config)
        return max(legal, key=lambda a: root.action_visits[a])
