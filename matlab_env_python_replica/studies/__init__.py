"""Replica-only RL study utilities for the SimuOriginal environment."""

from .dqn import train_dqn_variant, evaluate_dqn
from .qlearning import train_qlearning_variant, evaluate_qlearning
from .rewarding import (
    RewardVariant,
    ReplicaRewardEnv,
    baseline_reward_variant,
    build_core_reward_variants,
    build_full_reward_variants,
    equal_gradient_reward_variant,
    normalized_force_shape_reward_variant,
    normalized_legacy_transparency_reward_variant,
    reward_variant_from_name,
    tracking_effort_no_force_transparency_reward_variant,
    tracking_jerk_no_force_transparency_reward_variant,
)

__all__ = [
    "RewardVariant",
    "ReplicaRewardEnv",
    "baseline_reward_variant",
    "build_core_reward_variants",
    "build_full_reward_variants",
    "equal_gradient_reward_variant",
    "normalized_force_shape_reward_variant",
    "normalized_legacy_transparency_reward_variant",
    "reward_variant_from_name",
    "tracking_effort_no_force_transparency_reward_variant",
    "tracking_jerk_no_force_transparency_reward_variant",
    "train_qlearning_variant",
    "evaluate_qlearning",
    "train_dqn_variant",
    "evaluate_dqn",
]
