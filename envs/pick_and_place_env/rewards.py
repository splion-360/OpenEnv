# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

try:
    from . import config as cfg
    from .models import RewardBreakdown
except ImportError:
    try:
        import config as cfg  # type: ignore
        from models import RewardBreakdown  # type: ignore
    except ImportError:
        from pick_and_place_env import config as cfg  # type: ignore
        from pick_and_place_env.models import RewardBreakdown  # type: ignore


def _place_reward_active(
    gripper_contact: bool, cube_height: float, success: bool, goal_reached_once: bool
) -> bool:
    return (
        gripper_contact
        or cube_height >= (cfg.PHASE.lift_threshold * 0.5)
        or success
        or goal_reached_once
    )


def _phase_index(phase: cfg.Phase) -> int:
    ordered = {
        cfg.Phase.REACHING: 0,
        cfg.Phase.GRASPING: 1,
        cfg.Phase.LIFTING: 2,
        cfg.Phase.PLACING: 3,
    }
    return ordered[phase]


def _velocity_penalty(
    ee_speed_mps: float,
    safe_speed_mps: float,
    is_active: bool,
) -> float:
    if not is_active:
        return 0.0
    return -max(0.0, ee_speed_mps - safe_speed_mps)


def compute_reward_breakdown(
    previous_phase: cfg.Phase,
    current_phase: cfg.Phase,
    gripper_to_cube: float,
    cube_to_goal: float,
    cube_height: float,
    gripper_contact: bool,
    goal_reached_once: bool,
    ee_linear_velocity: list[float],
    step_dt: float,
    success: bool,
) -> RewardBreakdown:
    reach_reward = -gripper_to_cube
    grasp_reward = 1.0 if gripper_contact else 0.0
    lift_reward = max(0.0, cube_height)
    place_reward = (
        -cube_to_goal
        if _place_reward_active(
            gripper_contact, cube_height, success, goal_reached_once
        )
        else 0.0
    )
    phase_transition_reward = (
        1.0 if _phase_index(current_phase) > _phase_index(previous_phase) else 0.0
    )
    ee_speed_mps = math.dist(ee_linear_velocity, [0.0, 0.0, 0.0]) / max(step_dt, 1e-9)
    approach_velocity_reward = _velocity_penalty(
        ee_speed_mps=ee_speed_mps,
        safe_speed_mps=cfg.REWARD.max_safe_approach_speed_mps,
        is_active=(gripper_to_cube <= cfg.REWARD.approach_distance_threshold),
    )
    place_velocity_reward = _velocity_penalty(
        ee_speed_mps=ee_speed_mps,
        safe_speed_mps=cfg.REWARD.max_safe_place_speed_mps,
        is_active=(
            goal_reached_once or cube_to_goal <= cfg.REWARD.place_distance_threshold
        ),
    )
    success_reward = 1.0 if success else 0.0

    total_reward = (
        +(cfg.REWARD.reach_weight * reach_reward)
        + (cfg.REWARD.grasp_weight * grasp_reward)
        + (cfg.REWARD.lift_weight * lift_reward)
        + (cfg.REWARD.place_weight * place_reward)
        + (cfg.REWARD.phase_transition_weight * phase_transition_reward)
        + (cfg.REWARD.approach_velocity_weight * approach_velocity_reward)
        + (cfg.REWARD.place_velocity_weight * place_velocity_reward)
        + (cfg.REWARD.success_weight * success_reward)
    )

    return RewardBreakdown(
        reach=reach_reward,
        grasp=grasp_reward,
        lift=lift_reward,
        place=place_reward,
        phase_transition=phase_transition_reward,
        approach_velocity=approach_velocity_reward,
        place_velocity=place_velocity_reward,
        success=success_reward,
        total=total_reward,
    )
