"""
MCTS node data structure.

Each node represents one state in the search tree, with edges to
children keyed by (action, response) pairs.  Because the opponent
is stochastic, a single action from a parent may produce *multiple*
children — one per distinct sampled response.  This is the standard
"chance node" pattern for MCTS in stochastic environments.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..actions import Action
from ..state import State


@dataclass
class MCTSNode:
    """
    A single node in the MCTS search tree.

    Invariants
    ----------
    * ``visit_count == sum(action_visits.values())``  after backprop.
    * ``total_reward == sum(action_rewards.values())`` after backprop.
    * Every child's ``incoming_action`` and ``incoming_response`` are set.
    """

    state: State
    parent: MCTSNode | None = field(default=None, repr=False)
    incoming_action: Action | None = None
    incoming_response: int | None = None  # UserResponse int value

    # ── Children ──────────────────────────────────────────────
    # Keyed by Action.  Each action maps to a *list* of child nodes
    # because the opponent's stochastic response creates multiple
    # possible successor states from the same parent action.
    children: dict[Action, list[MCTSNode]] = field(default_factory=dict, repr=False)

    # ── Visit Statistics ──────────────────────────────────────
    visit_count: int = 0
    total_reward: float = 0.0

    # Per-action aggregates at *this* node.
    action_visits: dict[Action, int] = field(
        default_factory=lambda: {a: 0 for a in Action}
    )
    action_rewards: dict[Action, float] = field(
        default_factory=lambda: {a: 0.0 for a in Action}
    )

    # ── Queries ───────────────────────────────────────────────

    @property
    def is_leaf(self) -> bool:
        """True if no children have been expanded yet."""
        return len(self.children) == 0

    @property
    def mean_value(self) -> float:
        """Average backpropagated reward."""
        if self.visit_count == 0:
            return 0.0
        return self.total_reward / self.visit_count

    def action_mean(self, action: Action) -> float:
        """Mean Q-value for a specific action at this node."""
        n = self.action_visits[action]
        if n == 0:
            return 0.0
        return self.action_rewards[action] / n

    # ── Mutation ──────────────────────────────────────────────

    def add_child(self, action: Action, child: MCTSNode) -> None:
        """Attach *child* under *action*, creating the list if needed."""
        if action not in self.children:
            self.children[action] = []
        self.children[action].append(child)

    # ── Introspection ─────────────────────────────────────────

    def expanded_actions(self) -> set[Action]:
        """Set of actions for which at least one child exists."""
        return set(self.children.keys())

    def child_count(self) -> int:
        """Total number of child nodes across all actions."""
        return sum(len(ch) for ch in self.children.values())

    def tree_size(self) -> int:
        """Recursive count of all descendant nodes (including self)."""
        count = 1
        for children_list in self.children.values():
            for child in children_list:
                count += child.tree_size()
        return count
