# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


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
    gripper_contact: bool, cube_height: float, success: bool
) -> bool:
    return gripper_contact or cube_height >= (cfg.PHASE.lift_threshold * 0.5) or success


def compute_reward_breakdown(
    gripper_to_cube: float,
    cube_to_goal: float,
    cube_height: float,
    gripper_contact: bool,
    success: bool,
    cbf_reward: float = 0.0,
) -> RewardBreakdown:
    reach_reward = -gripper_to_cube
    grasp_reward = 1.0 if gripper_contact else 0.0
    lift_reward = max(0.0, cube_height)
    place_reward = (
        -cube_to_goal
        if _place_reward_active(gripper_contact, cube_height, success)
        else 0.0
    )
    success_reward = 1.0 if success else 0.0

    total_reward = (
        (cfg.REWARD.format_weight * 0.0)
        + (cfg.REWARD.bounds_weight * 0.0)
        + (cfg.REWARD.reach_weight * reach_reward)
        + (cfg.REWARD.grasp_weight * grasp_reward)
        + (cfg.REWARD.lift_weight * lift_reward)
        + (cfg.REWARD.place_weight * place_reward)
        + (cfg.REWARD.success_weight * success_reward)
        + (cfg.REWARD.cbf_weight * cbf_reward)
    )

    return RewardBreakdown(
        format=0.0,
        bounds=0.0,
        reach=reach_reward,
        grasp=grasp_reward,
        lift=lift_reward,
        place=place_reward,
        success=success_reward,
        cbf=cbf_reward,
        total=total_reward,
    )
