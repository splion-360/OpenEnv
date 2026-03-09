# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pick And Place environment implementation.

This module provides a stateful pick-and-place environment skeleton with the
OpenEnv interface. It tracks robot, object, and goal state with simple geometry
until the MuJoCo integration is added.
"""

import math
import random
from typing import Any, Optional
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment

    from .. import config as cfg
    from ..cbf import compute_safety_margins
    from ..models import (
        PickAndPlaceAction,
        PickAndPlaceObservation,
        PickAndPlaceState,
        Proprioception,
    )
    from ..observations import build_observation, detect_phase
    from ..rewards import compute_reward_breakdown
except ImportError:
    from openenv.core.env_server.interfaces import Environment

    try:
        import config as cfg  # type: ignore
        from cbf import compute_safety_margins  # type: ignore
        from models import (  # type: ignore
            PickAndPlaceAction,
            PickAndPlaceObservation,
            PickAndPlaceState,
            Proprioception,
        )
        from observations import build_observation, detect_phase  # type: ignore
        from rewards import compute_reward_breakdown  # type: ignore
    except ImportError:
        from pick_and_place_env import config as cfg  # type: ignore
        from pick_and_place_env.cbf import compute_safety_margins  # type: ignore
        from pick_and_place_env.models import (  # type: ignore
            PickAndPlaceAction,
            PickAndPlaceObservation,
            PickAndPlaceState,
            Proprioception,
        )
        from pick_and_place_env.observations import (  # type: ignore
            build_observation,
            detect_phase,
        )
        from pick_and_place_env.rewards import compute_reward_breakdown  # type: ignore


class PickAndPlaceEnvironment(Environment):
    """
    Stateful skeleton for a pick-and-place environment.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, max_steps: int = cfg.TASK.max_steps):
        """
        Initialize the Pick And Place environment.

        Args:
            max_steps: Maximum number of steps per episode
        """
        self._rng = random.Random()
        self._max_steps = max_steps
        self._table_height = cfg.GEOMETRY.table_height
        self._grasp_distance = cfg.GEOMETRY.grasp_distance
        self._goal_radius = cfg.GEOMETRY.goal_radius
        self._workspace_low = list(cfg.GEOMETRY.workspace_low)
        self._workspace_high = list(cfg.GEOMETRY.workspace_high)
        self._state = self._make_state()

    def _make_state(
        self,
        episode_id: Optional[str] = None,
        obs_mode: cfg.ObsMode = cfg.ObsMode.MULTIMODAL,
        ee_pos: Optional[list[float]] = None,
        cube_pos: Optional[list[float]] = None,
        goal_pos: Optional[list[float]] = None,
    ) -> PickAndPlaceState:
        ee_pos = ee_pos or list(cfg.GEOMETRY.ee_start_pos)
        cube_pos = cube_pos or list(cfg.GEOMETRY.cube_start_pos)
        goal_pos = goal_pos or list(cfg.GEOMETRY.goal_pos)

        proprioception = Proprioception(
            ee_pos=list(ee_pos),
            ee_quat=list(cfg.GEOMETRY.ee_quat),
            gripper_width=cfg.TASK.gripper_open_width,
            joint_angles=[],
        )
        safety = compute_safety_margins([0.0, 0.0, 0.0], ee_pos)

        return PickAndPlaceState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            obs_mode=obs_mode,
            phase=cfg.Phase.REACHING,
            max_steps=self._max_steps,
            ee_pos=list(ee_pos),
            ee_quat=list(cfg.GEOMETRY.ee_quat),
            cube_pos=list(cube_pos),
            goal_pos=list(goal_pos),
            cube_height=0.0,
            gripper_contact=False,
            proprioception=proprioception,
            reward_breakdown=compute_reward_breakdown(success=False),
            safety_margins=safety,
            last_action=None,
            success=False,
        )

    def _distance(self, left: list[float], right: list[float]) -> float:
        return math.dist(left, right)

    def _clip_position(self, position: list[float]) -> list[float]:
        clipped = []
        for index, value in enumerate(position):
            clipped.append(
                min(max(value, self._workspace_low[index]), self._workspace_high[index])
            )
        return clipped

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        obs_mode: cfg.ObsMode = cfg.ObsMode.MULTIMODAL,
        **kwargs: Any,
    ) -> PickAndPlaceObservation:
        """
        Reset the environment and return the first observation.

        Args:
            seed: Optional seed for deterministic placement sampling
            episode_id: Optional episode identifier
            obs_mode: Observation mode for the episode
            **kwargs: Additional reset arguments

        Returns:
            PickAndPlaceObservation with the initial scene state
        """
        if seed is not None:
            self._rng.seed(seed)

        cube_pos = [
            cfg.GEOMETRY.cube_start_pos[0]
            + self._rng.uniform(-cfg.RESET.cube_x_jitter, cfg.RESET.cube_x_jitter),
            self._rng.uniform(-cfg.RESET.cube_y_jitter, cfg.RESET.cube_y_jitter),
            self._table_height,
        ]
        goal_pos = [
            cfg.GEOMETRY.goal_pos[0]
            + self._rng.uniform(-cfg.RESET.goal_x_jitter, cfg.RESET.goal_x_jitter),
            self._rng.uniform(-cfg.RESET.goal_y_jitter, cfg.RESET.goal_y_jitter),
            self._table_height,
        ]

        self._state = self._make_state(
            episode_id=episode_id,
            obs_mode=obs_mode,
            cube_pos=cube_pos,
            goal_pos=goal_pos,
        )

        return build_observation(self._state, reward=0.0, done=False)

    def step(
        self,
        action: PickAndPlaceAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> PickAndPlaceObservation:
        """
        Execute one step in the environment.

        Args:
            action: PickAndPlaceAction containing end-effector deltas
            timeout_s: Optional timeout for compatibility with the base interface
            **kwargs: Additional step arguments

        Returns:
            PickAndPlaceObservation with updated state
        """
        del timeout_s, kwargs

        previous_ee = list(self._state.ee_pos)
        next_ee = self._clip_position(
            [
                previous_ee[0] + action.dx,
                previous_ee[1] + action.dy,
                previous_ee[2] + action.dz,
            ]
        )
        delta = [
            next_ee[0] - previous_ee[0],
            next_ee[1] - previous_ee[1],
            next_ee[2] - previous_ee[2],
        ]

        cube_pos = list(self._state.cube_pos)
        carried = self._state.gripper_contact
        close_enough_to_grasp = (
            self._distance(next_ee, cube_pos) <= self._grasp_distance
        )

        if action.gripper == cfg.GripperCommand.CLOSE and close_enough_to_grasp:
            carried = True

        if carried and action.gripper == cfg.GripperCommand.CLOSE:
            cube_pos = [
                next_ee[0],
                next_ee[1],
                max(self._table_height, next_ee[2] - 0.01),
            ]
        elif carried and action.gripper == cfg.GripperCommand.OPEN:
            carried = False
            cube_pos[2] = self._table_height

        cube_height = max(0.0, cube_pos[2] - self._table_height)
        cube_to_goal = self._distance(cube_pos, self._state.goal_pos)
        success = (
            action.gripper == cfg.GripperCommand.OPEN
            and cube_to_goal <= self._goal_radius
        )

        self._state.step_count += 1
        self._state.ee_pos = next_ee
        self._state.cube_pos = cube_pos
        self._state.cube_height = cube_height
        self._state.gripper_contact = carried
        self._state.success = success
        self._state.last_action = action
        self._state.proprioception = Proprioception(
            ee_pos=list(next_ee),
            ee_quat=list(self._state.ee_quat),
            gripper_width=(
                cfg.TASK.gripper_open_width
                if action.gripper == cfg.GripperCommand.OPEN
                else cfg.TASK.gripper_closed_width
            ),
            joint_angles=[],
        )
        self._state.safety_margins = compute_safety_margins(delta, next_ee)
        self._state.reward_breakdown = compute_reward_breakdown(success=success)
        reward = self._state.reward_breakdown.total
        self._state.phase = detect_phase(self._state, cube_to_goal)

        done = self._state.success or self._state.step_count >= self._state.max_steps
        echoed_message = action.message or ""

        return build_observation(
            self._state,
            reward=reward,
            done=done,
            echoed_message=echoed_message,
        )

    @property
    def state(self) -> PickAndPlaceState:
        """Get the current environment state."""
        return self._state
