"""Adaptive tabular Q-learning agent with explicit environment context."""

from __future__ import annotations

from q_learning_agent import QLearningAgent


class AdaptiveQLearningAgent(QLearningAgent):
    """Same core update as QLearningAgent, with helper for contextual states."""

    @staticmethod
    def build_state(base_state: tuple[int, ...], env_id: int) -> tuple[int, ...]:
        return tuple(base_state) + (int(env_id),)
