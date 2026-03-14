# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from pick_and_place_env.models import PickAndPlaceAction
from pick_and_place_env.server.pick_and_place_env_environment import (
    PickAndPlaceEnvironment,
)
from pick_and_place_env.simulator import FETCH, FetchPickAndPlaceSimulator


def test_simulator_peek_step_restores_state() -> None:
    simulator = FetchPickAndPlaceSimulator()
    try:
        simulator.reset(seed=0)
        before_qpos = np.array(simulator._env.unwrapped.data.qpos, copy=True)
        before_qvel = np.array(simulator._env.unwrapped.data.qvel, copy=True)
        before_goal = np.array(simulator._env.unwrapped.goal, copy=True)
        before_elapsed_steps = simulator._env._elapsed_steps

        candidate_action = PickAndPlaceAction(
            dx=1.0,
            dy=0.0,
            dz=0.0,
            gripper=1.0,
        )
        simulator.peek_step(candidate_action)

        after_qpos = np.array(simulator._env.unwrapped.data.qpos, copy=True)
        after_qvel = np.array(simulator._env.unwrapped.data.qvel, copy=True)
        after_goal = np.array(simulator._env.unwrapped.goal, copy=True)
        after_elapsed_steps = simulator._env._elapsed_steps

        assert np.allclose(before_qpos, after_qpos)
        assert np.allclose(before_qvel, after_qvel)
        assert np.allclose(before_goal, after_goal)
        assert before_elapsed_steps == after_elapsed_steps
    finally:
        simulator.close()


def test_environment_step_filters_unsafe_nominal_action() -> None:
    environment = PickAndPlaceEnvironment(enable_cbf_filter=True)
    try:
        environment.reset(seed=0)

        nominal_action = PickAndPlaceAction(
            dx=1.0,
            dy=0.0,
            dz=0.0,
            gripper=1.0,
        )
        observation = environment.step(nominal_action)

        assert environment.state.proposed_action is not None
        assert environment.state.last_action is not None
        assert environment.state.proposed_action.dx == 1.0
        assert environment.state.cbf_intervened is True
        assert 0.0 <= environment.state.cbf_scale < 1.0
        assert environment.state.last_action.dx < environment.state.proposed_action.dx
        assert environment.state.cbf_residual >= 0.0
        assert observation.metadata["cbf_intervened"] is True
        assert observation.metadata["cbf_scale"] == environment.state.cbf_scale
    finally:
        environment._simulator.close()


def test_compute_joint_limit_margin_matches_limited_fetch_joints() -> None:
    simulator = FetchPickAndPlaceSimulator()
    try:
        snapshot = simulator.reset(seed=0)
        env = simulator._env.unwrapped

        limited_joint_margins = []
        for joint_name, joint_angle in zip(
            FETCH.arm_joint_names,
            snapshot.proprioception.joint_angles,
        ):
            joint_id = env._model_names.joint_name2id[joint_name]
            if not bool(env.model.jnt_limited[joint_id]):
                continue

            lower_limit, upper_limit = env.model.jnt_range[joint_id]
            limited_joint_margins.append(
                min(joint_angle - lower_limit, upper_limit - joint_angle)
            )

        computed_margin = simulator.compute_joint_limit_margin(
            snapshot.proprioception.joint_angles
        )

        assert len(limited_joint_margins) == 4
        assert np.isclose(computed_margin, min(limited_joint_margins))
    finally:
        simulator.close()


def test_task_success_requires_goal_then_return_home() -> None:
    environment = PickAndPlaceEnvironment(enable_cbf_filter=False)
    try:
        environment.reset(seed=0)
        home = list(environment.state.home_ee_pos)
        away_from_home = [home[0] + 0.20, home[1], home[2]]

        success, _ = environment._update_task_success(
            cube_to_goal=0.20,
            ee_pos=away_from_home,
        )
        assert success is False
        assert environment.state.goal_reached_once is False

        success, _ = environment._update_task_success(
            cube_to_goal=0.0,
            ee_pos=away_from_home,
        )
        assert success is False
        assert environment.state.goal_reached_once is True

        success, home_distance = environment._update_task_success(
            cube_to_goal=0.20,
            ee_pos=home,
        )
        assert home_distance <= environment._goal_radius
        assert success is True
    finally:
        environment._simulator.close()
