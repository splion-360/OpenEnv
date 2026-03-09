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
from pick_and_place_env.simulator import FetchPickAndPlaceSimulator


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
            gripper="open",
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
    environment = PickAndPlaceEnvironment()
    try:
        environment.reset(seed=0, obs_mode="state")

        nominal_action = PickAndPlaceAction(
            dx=1.0,
            dy=0.0,
            dz=0.0,
            gripper="open",
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
