"""
Tabular Q-Learning Agent
========================
Epsilon-greedy Q-learning with a discrete state/action table.
Designed for the bilateral teleoperation voltage-control task.
"""

from __future__ import annotations

import numpy as np

import config as cfg


class QLearningAgent:
    """Tabular Q-learning with epsilon-greedy exploration."""

    def __init__(
        self,
        state_dims: tuple[int, ...],
        n_actions: int,
        seed: int | None = None,
    ):
        self.state_dims = state_dims
        self.n_actions  = n_actions
        self.rng        = np.random.default_rng(seed)

        # Q-table:  shape = (*state_dims, n_actions)
        self.q_table = np.zeros((*state_dims, n_actions), dtype=np.float64)

        # Visit counter (optional, for diagnostics)
        self.visit_count = np.zeros_like(self.q_table, dtype=np.int64)

        # Hyper-parameters
        self.lr            = cfg.LEARNING_RATE
        self.gamma         = cfg.DISCOUNT_FACTOR
        self.epsilon       = cfg.EPSILON_START
        self.epsilon_min   = cfg.EPSILON_END
        self.epsilon_decay = cfg.EPSILON_DECAY

    # ------------------------------------------------------------------ #
    #  Action selection                                                   #
    # ------------------------------------------------------------------ #
    def select_action(self, state: tuple[int, ...]) -> int:
        """Epsilon-greedy: explore with probability ε, exploit otherwise."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))

        q_values = self.q_table[state]
        max_q    = q_values.max()
        best     = np.flatnonzero(q_values == max_q)
        return int(self.rng.choice(best))          # random tie-break

    # ------------------------------------------------------------------ #
    #  Q-learning update                                                  #
    # ------------------------------------------------------------------ #
    def update(
        self,
        state: tuple[int, ...],
        action: int,
        reward: float,
        next_state: tuple[int, ...],
        done: bool,
    ) -> None:
        """Standard one-step Q-learning (off-policy TD(0))."""
        idx = state + (action,)
        current_q = self.q_table[idx]

        target = reward if done else reward + self.gamma * self.q_table[next_state].max()

        self.q_table[idx] += self.lr * (target - current_q)
        self.visit_count[idx] += 1

    # ------------------------------------------------------------------ #
    #  Exploration decay                                                  #
    # ------------------------------------------------------------------ #
    def decay_epsilon(self) -> None:
        """Multiplicative ε decay — call once per episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------ #
    #  Persistence                                                        #
    # ------------------------------------------------------------------ #
    def save(self, path: str = "q_table.npy") -> None:
        np.save(path, self.q_table)

    def load(self, path: str = "q_table.npy") -> None:
        self.q_table = np.load(path)

    # ------------------------------------------------------------------ #
    #  Diagnostics                                                        #
    # ------------------------------------------------------------------ #
    def coverage(self) -> float:
        """Fraction of (state, action) pairs visited at least once."""
        return float((self.visit_count > 0).sum() / self.visit_count.size)

    def __repr__(self) -> str:
        total = int(np.prod(self.state_dims)) * self.n_actions
        return (
            f"QLearningAgent(states={self.state_dims}, actions={self.n_actions}, "
            f"Q-entries={total:,}, ε={self.epsilon:.4f}, "
            f"coverage={self.coverage():.1%})"
        )
