"""
Sparse Tabular Q-Learning Agent
===============================
Epsilon-greedy Q-learning with sparse state storage:
only visited states are allocated.
"""

from __future__ import annotations

import numpy as np

import config as cfg


class QLearningAgent:
    """Tabular Q-learning with epsilon-greedy exploration (sparse state map)."""

    def __init__(
        self,
        state_dims: tuple[int, ...],
        n_actions: int,
        seed: int | None = None,
    ):
        self.state_dims = tuple(int(d) for d in state_dims)
        self.n_actions = int(n_actions)
        self.rng = np.random.default_rng(seed)

        # Sparse storage: state-index tuple -> action-value vector
        self.q_table: dict[tuple[int, ...], np.ndarray] = {}
        self.visit_count: dict[tuple[int, ...], np.ndarray] = {}
        self._zero_q = np.zeros(self.n_actions, dtype=np.float64)

        # Fast diagnostics counters
        self._visited_states = 0
        self._visited_state_actions = 0

        # Hyper-parameters
        self.lr = cfg.LEARNING_RATE
        self.gamma = cfg.DISCOUNT_FACTOR
        self.epsilon = cfg.EPSILON_START
        self.epsilon_min = cfg.EPSILON_END
        self.epsilon_decay = cfg.EPSILON_DECAY

    @staticmethod
    def _as_key(state: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(int(s) for s in state)

    def q_values(self, state: tuple[int, ...]) -> np.ndarray:
        """Return Q(s, :) for a state (zero-vector if unseen)."""
        return self.q_table.get(self._as_key(state), self._zero_q)

    def _ensure_state_rows(self, key: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        q_row = self.q_table.get(key)
        if q_row is None:
            q_row = np.zeros(self.n_actions, dtype=np.float64)
            self.q_table[key] = q_row

        v_row = self.visit_count.get(key)
        if v_row is None:
            v_row = np.zeros(self.n_actions, dtype=np.int64)
            self.visit_count[key] = v_row
            self._visited_states += 1

        return q_row, v_row

    # ------------------------------------------------------------------ #
    #  Action selection                                                   #
    # ------------------------------------------------------------------ #
    def select_action(self, state: tuple[int, ...]) -> int:
        """Epsilon-greedy: explore with probability epsilon, exploit otherwise."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))

        q_values = self.q_values(state)
        max_q = q_values.max()
        best = np.flatnonzero(q_values == max_q)
        return int(self.rng.choice(best))

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
        key = self._as_key(state)
        next_key = self._as_key(next_state)

        q_row, visits = self._ensure_state_rows(key)
        current_q = q_row[action]

        next_max = 0.0 if done else float(self.q_table.get(next_key, self._zero_q).max())
        target = reward if done else reward + self.gamma * next_max

        q_row[action] += self.lr * (target - current_q)
        if visits[action] == 0:
            self._visited_state_actions += 1
        visits[action] += 1

    # ------------------------------------------------------------------ #
    #  Exploration decay                                                  #
    # ------------------------------------------------------------------ #
    def decay_epsilon(self) -> None:
        """Multiplicative epsilon decay, called once per episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------ #
    #  Persistence                                                        #
    # ------------------------------------------------------------------ #
    def save(self, path: str = "q_table.npy") -> None:
        payload = {
            "format": "sparse_q_v1",
            "state_dims": self.state_dims,
            "n_actions": self.n_actions,
            "q_table": self.q_table,
            "visit_count": self.visit_count,
            "epsilon": float(self.epsilon),
            "visited_states": int(self._visited_states),
            "visited_state_actions": int(self._visited_state_actions),
        }
        np.save(path, payload, allow_pickle=True)

    def _load_sparse_payload(self, payload: dict) -> None:
        saved_actions = int(payload.get("n_actions", self.n_actions))
        if saved_actions != self.n_actions:
            raise ValueError(
                f"n_actions mismatch: file has {saved_actions}, agent expects {self.n_actions}"
            )

        loaded_dims = tuple(int(x) for x in payload.get("state_dims", self.state_dims))
        if loaded_dims != self.state_dims:
            raise ValueError(
                f"state_dims mismatch: file has {loaded_dims}, agent expects {self.state_dims}. "
                "Retrain or load a compatible model."
            )

        raw_q = payload.get("q_table", {})
        raw_visits = payload.get("visit_count", {})

        self.q_table = {}
        self.visit_count = {}
        for key, vals in raw_q.items():
            k = self._as_key(key)
            row = np.asarray(vals, dtype=np.float64).reshape(-1)
            if row.size != self.n_actions:
                continue
            self.q_table[k] = row

        for key, vals in raw_visits.items():
            k = self._as_key(key)
            row = np.asarray(vals, dtype=np.int64).reshape(-1)
            if row.size != self.n_actions:
                continue
            self.visit_count[k] = row

        # Ensure every Q row has a visit row.
        for key in self.q_table:
            if key not in self.visit_count:
                self.visit_count[key] = np.zeros(self.n_actions, dtype=np.int64)

        self._visited_states = len(self.visit_count)
        self._visited_state_actions = int(
            sum(int((row > 0).sum()) for row in self.visit_count.values())
        )
        self.epsilon = float(payload.get("epsilon", self.epsilon))

    def _load_dense_array(self, dense: np.ndarray) -> None:
        if dense.ndim < 2:
            raise ValueError("Invalid dense Q-table format.")
        if dense.shape[-1] != self.n_actions:
            raise ValueError(
                f"n_actions mismatch: file has {dense.shape[-1]}, agent expects {self.n_actions}"
            )

        loaded_dims = tuple(int(x) for x in dense.shape[:-1])
        if loaded_dims != self.state_dims:
            raise ValueError(
                f"state_dims mismatch: file has {loaded_dims}, agent expects {self.state_dims}. "
                "Retrain or load a compatible model."
            )

        self.q_table = {}
        self.visit_count = {}

        nonzero_states = np.argwhere(np.any(dense != 0.0, axis=-1))
        for idx in nonzero_states:
            key = tuple(int(i) for i in idx.tolist())
            self.q_table[key] = np.array(dense[key], dtype=np.float64)
            self.visit_count[key] = np.zeros(self.n_actions, dtype=np.int64)

        self._visited_states = len(self.visit_count)
        self._visited_state_actions = 0

    def load(self, path: str = "q_table.npy") -> None:
        raw = np.load(path, allow_pickle=True)

        if isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == ():
            payload = raw.item()
            if isinstance(payload, dict) and payload.get("format") == "sparse_q_v1":
                self._load_sparse_payload(payload)
                return

        if isinstance(raw, np.ndarray):
            self._load_dense_array(raw)
            return

        raise ValueError(f"Unsupported Q-table file format: {path}")

    # ------------------------------------------------------------------ #
    #  Diagnostics                                                        #
    # ------------------------------------------------------------------ #
    def discovered_states(self) -> int:
        return int(self._visited_states)

    def coverage(self) -> float:
        """
        Fraction of visited (state,action) pairs over reached-state action slots:
        visited_pairs / (reached_states * n_actions).
        """
        if self._visited_states == 0:
            return 0.0
        return float(self._visited_state_actions / (self._visited_states * self.n_actions))

    def __repr__(self) -> str:
        full_entries = int(np.prod(self.state_dims)) * self.n_actions
        return (
            f"QLearningAgent(states={self.state_dims}, actions={self.n_actions}, "
            f"full-Q-entries={full_entries:,}, discovered-states={self.discovered_states():,}, "
            f"action-coverage={self.coverage():.1%}, epsilon={self.epsilon:.4f})"
        )
