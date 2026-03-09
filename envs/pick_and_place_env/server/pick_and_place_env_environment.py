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
    from ..models import (
        PickAndPlaceAction,
        PickAndPlaceObservation,
        PickAndPlaceState,
        Proprioception,
        RewardBreakdown,
        SafetyMargins,
    )
except ImportError:
    from openenv.core.env_server.interfaces import Environment

    try:
        import config as cfg  # type: ignore
        from models import (  # type: ignore
            PickAndPlaceAction,
            PickAndPlaceObservation,
            PickAndPlaceState,
            Proprioception,
            RewardBreakdown,
            SafetyMargins,
        )
    except ImportError:
        from pick_and_place_env import config as cfg  # type: ignore
        from pick_and_place_env.models import (  # type: ignore
            PickAndPlaceAction,
            PickAndPlaceObservation,
            PickAndPlaceState,
            Proprioception,
            RewardBreakdown,
            SafetyMargins,
        )


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
        self._lift_threshold = cfg.GEOMETRY.lift_threshold
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
        safety = self._compute_safety_margins([0.0, 0.0, 0.0], ee_pos)

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
            reward_breakdown=RewardBreakdown(),
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

    def _compute_safety_margins(
        self,
        delta: list[float],
        position: list[float],
    ) -> SafetyMargins:
        workspace_margin = min(
            position[0] - self._workspace_low[0],
            self._workspace_high[0] - position[0],
            position[1] - self._workspace_low[1],
            self._workspace_high[1] - position[1],
            position[2] - self._workspace_low[2],
            self._workspace_high[2] - position[2],
        )
        velocity_margin = cfg.TASK.max_action_delta_meters - self._distance(
            delta, [0.0, 0.0, 0.0]
        )
        joint_limit_margin = 1.0

        return SafetyMargins(
            workspace=workspace_margin,
            velocity=velocity_margin,
            joint_limit=joint_limit_margin,
            minimum=min(workspace_margin, velocity_margin, joint_limit_margin),
        )

    def _detect_phase(self, cube_to_goal: float) -> cfg.Phase:
        if self._state.success:
            return cfg.Phase.PLACING
        if not self._state.gripper_contact:
            return cfg.Phase.REACHING
        if self._state.cube_height < self._lift_threshold:
            return cfg.Phase.GRASPING
        if cube_to_goal < self._goal_radius * 1.5:
            return cfg.Phase.PLACING
        return cfg.Phase.LIFTING

    def _build_scene_text(self) -> str:
        return (
            f"Phase: {self._state.phase}. "
            f"Gripper at ({self._state.ee_pos[0]:.2f}, {self._state.ee_pos[1]:.2f}, {self._state.ee_pos[2]:.2f}). "
            f"Cube at ({self._state.cube_pos[0]:.2f}, {self._state.cube_pos[1]:.2f}, {self._state.cube_pos[2]:.2f}). "
            f"Cube height: {self._state.cube_height:.2f}m. "
            f"Goal: tray at ({self._state.goal_pos[0]:.2f}, {self._state.goal_pos[1]:.2f}, {self._state.goal_pos[2]:.2f}). "
            f"Step {self._state.step_count}/{self._state.max_steps}."
        )

    def _build_observation(
        self,
        reward: float,
        done: bool,
        echoed_message: str = "",
    ) -> PickAndPlaceObservation:
        return PickAndPlaceObservation(
            rgb_overhead=None,
            rgb_wrist=None,
            scene_text=self._build_scene_text(),
            instruction=cfg.TASK.instruction,
            phase=self._state.phase,
            obs_mode=self._state.obs_mode,
            proprioception=self._state.proprioception,
            reward_breakdown=self._state.reward_breakdown,
            safety_margins=self._state.safety_margins,
            echoed_message=echoed_message,
            message_length=len(echoed_message),
            done=done,
            reward=reward,
            metadata={
                "step": self._state.step_count,
                "success": self._state.success,
            },
        )

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

        return self._build_observation(reward=0.0, done=False)

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
        self._state.safety_margins = self._compute_safety_margins(delta, next_ee)
        reward = 1.0 if success else 0.0
        self._state.reward_breakdown = RewardBreakdown(success=reward, total=reward)
        self._state.phase = self._detect_phase(cube_to_goal)

        done = self._state.success or self._state.step_count >= self._state.max_steps
        echoed_message = action.message or ""

        return self._build_observation(
            reward=reward,
            done=done,
            echoed_message=echoed_message,
        )

    @property
    def state(self) -> PickAndPlaceState:
        """Get the current environment state."""
        return self._state
