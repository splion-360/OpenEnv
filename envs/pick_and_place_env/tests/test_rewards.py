# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pick_and_place_env import config as cfg
from pick_and_place_env.rewards import compute_reward_breakdown


def test_phase_transition_bonus_only_on_forward_progress() -> None:
    breakdown = compute_reward_breakdown(
        previous_phase=cfg.Phase.REACHING,
        current_phase=cfg.Phase.GRASPING,
        gripper_to_cube=0.20,
        cube_to_goal=0.30,
        cube_height=0.0,
        gripper_contact=False,
        goal_reached_once=False,
        ee_linear_velocity=[0.0, 0.0, 0.0],
        step_dt=0.04,
        success=False,
    )
    assert breakdown.phase_transition == 1.0

    no_bonus = compute_reward_breakdown(
        previous_phase=cfg.Phase.LIFTING,
        current_phase=cfg.Phase.GRASPING,
        gripper_to_cube=0.20,
        cube_to_goal=0.30,
        cube_height=0.0,
        gripper_contact=False,
        goal_reached_once=False,
        ee_linear_velocity=[0.0, 0.0, 0.0],
        step_dt=0.04,
        success=False,
    )
    assert no_bonus.phase_transition == 0.0


def test_approach_velocity_penalty_triggers_near_cube() -> None:
    breakdown = compute_reward_breakdown(
        previous_phase=cfg.Phase.REACHING,
        current_phase=cfg.Phase.REACHING,
        gripper_to_cube=0.05,
        cube_to_goal=0.30,
        cube_height=0.0,
        gripper_contact=False,
        goal_reached_once=False,
        ee_linear_velocity=[0.02, 0.0, 0.0],
        step_dt=0.04,
        success=False,
    )
    assert breakdown.approach_velocity < 0.0
    assert breakdown.place_velocity == 0.0


def test_place_velocity_penalty_triggers_near_goal() -> None:
    breakdown = compute_reward_breakdown(
        previous_phase=cfg.Phase.LIFTING,
        current_phase=cfg.Phase.PLACING,
        gripper_to_cube=0.10,
        cube_to_goal=0.04,
        cube_height=0.06,
        gripper_contact=True,
        goal_reached_once=False,
        ee_linear_velocity=[0.02, 0.0, 0.0],
        step_dt=0.04,
        success=False,
    )
    assert breakdown.place_velocity < 0.0
