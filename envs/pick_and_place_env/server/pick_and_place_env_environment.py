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
from typing import Any, Optional
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment

    from .. import config as cfg
    from ..cbf import compute_cbf_residual, compute_safety_margins, scale_action
    from ..models import (
        PickAndPlaceAction,
        PickAndPlaceObservation,
        PickAndPlaceState,
    )
    from ..observations import build_observation, detect_phase
    from ..rewards import compute_reward_breakdown
    from ..simulator import FetchPickAndPlaceSimulator, SimulatorSnapshot
except ImportError:
    from openenv.core.env_server.interfaces import Environment

    try:
        import config as cfg  # type: ignore
        from cbf import (  # type: ignore
            compute_cbf_residual,
            compute_safety_margins,
            scale_action,
        )
        from models import (  # type: ignore
            PickAndPlaceAction,
            PickAndPlaceObservation,
            PickAndPlaceState,
        )
        from observations import build_observation, detect_phase  # type: ignore
        from rewards import compute_reward_breakdown  # type: ignore
        from simulator import (  # type: ignore
            FetchPickAndPlaceSimulator,
            SimulatorSnapshot,
        )
    except ImportError:
        from pick_and_place_env import config as cfg  # type: ignore
        from pick_and_place_env.cbf import (  # type: ignore
            compute_cbf_residual,
            compute_safety_margins,
            scale_action,
        )
        from pick_and_place_env.models import (  # type: ignore
            PickAndPlaceAction,
            PickAndPlaceObservation,
            PickAndPlaceState,
        )
        from pick_and_place_env.observations import (  # type: ignore
            build_observation,
            detect_phase,
        )
        from pick_and_place_env.rewards import compute_reward_breakdown  # type: ignore
        from pick_and_place_env.simulator import (  # type: ignore
            FetchPickAndPlaceSimulator,
            SimulatorSnapshot,
        )


class PickAndPlaceEnvironment(Environment):
    """
    Stateful skeleton for a pick-and-place environment.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        max_steps: int = cfg.TASK.max_steps,
        enable_cbf_filter: bool = cfg.CBF.enable_filter,
    ):
        """
        Initialize the Pick And Place environment.

        Args:
            max_steps: Maximum number of steps per episode
            enable_cbf_filter: Whether to apply CBF action filtering at runtime
        """
        self._max_steps = max_steps
        self._enable_cbf_filter = enable_cbf_filter
        self._simulator = FetchPickAndPlaceSimulator(max_episode_steps=max_steps)
        self._goal_radius = self._simulator.distance_threshold
        self._max_velocity_mps = (
            math.sqrt(3.0)
            * self._simulator.position_action_scale_meters
            / self._simulator.step_dt
        )
        self._state = self._make_state(snapshot=self._simulator.reset())

    def _make_state(
        self,
        snapshot: SimulatorSnapshot,
        episode_id: Optional[str] = None,
    ) -> PickAndPlaceState:
        ee_pos = list(snapshot.ee_pos)
        ee_quat = list(snapshot.ee_quat)
        cube_pos = list(snapshot.cube_pos)
        goal_pos = list(snapshot.goal_pos)
        cube_height = snapshot.cube_height
        gripper_contact = snapshot.gripper_contact
        proprioception = snapshot.proprioception
        joint_angles = list(proprioception.joint_angles)
        safety = compute_safety_margins(
            ee_pos,
            ee_linear_velocity=list(snapshot.ee_linear_velocity),
            step_dt=self._simulator.step_dt,
            max_velocity_mps=self._max_velocity_mps,
            joint_limit_margin=self._simulator.compute_joint_limit_margin(joint_angles),
        )

        cube_to_goal = self._distance(cube_pos, goal_pos)
        state = PickAndPlaceState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            phase=cfg.Phase.REACHING,
            max_steps=self._max_steps,
            ee_pos=list(ee_pos),
            ee_quat=list(ee_quat),
            home_ee_pos=list(ee_pos),
            cube_pos=list(cube_pos),
            goal_pos=list(goal_pos),
            cube_height=cube_height,
            gripper_contact=gripper_contact,
            proprioception=proprioception,
            safety_margins=safety,
            proposed_action=None,
            last_action=None,
            cbf_intervened=False,
            cbf_scale=1.0,
            cbf_residual=0.0,
            success=False,
            goal_reached_once=False,
        )
        state.phase = detect_phase(state, cube_to_goal, self._goal_radius)
        return state

    def _distance(self, left: list[float], right: list[float]) -> float:
        x_delta = left[0] - right[0]
        y_delta = left[1] - right[1]
        z_delta = left[2] - right[2]
        return (x_delta * x_delta + y_delta * y_delta + z_delta * z_delta) ** 0.5

    def _candidate_safety(
        self,
        action: PickAndPlaceAction,
    ) -> tuple[SimulatorSnapshot, Any, float]:
        candidate_snapshot = self._simulator.peek_step(action)
        candidate_safety = compute_safety_margins(
            list(candidate_snapshot.ee_pos),
            ee_linear_velocity=list(candidate_snapshot.ee_linear_velocity),
            step_dt=self._simulator.step_dt,
            max_velocity_mps=self._max_velocity_mps,
            joint_limit_margin=self._simulator.compute_joint_limit_margin(
                candidate_snapshot.proprioception.joint_angles
            ),
        )
        candidate_residual = compute_cbf_residual(
            self._state.safety_margins,
            candidate_safety,
        )
        return candidate_snapshot, candidate_safety, candidate_residual

    def _filter_action_with_cbf(
        self,
        action: PickAndPlaceAction,
    ) -> tuple[PickAndPlaceAction, Any, float, float]:
        candidate_snapshot, candidate_safety, candidate_residual = (
            self._candidate_safety(action)
        )
        if candidate_residual >= 0.0:
            return action, candidate_safety, candidate_residual, 1.0

        zero_action = scale_action(action, 0.0)
        zero_snapshot, zero_safety, zero_residual = self._candidate_safety(zero_action)
        if zero_residual < 0.0:
            return zero_action, zero_safety, zero_residual, 0.0

        best_scale = 0.0
        best_action = zero_action
        best_safety = zero_safety
        best_residual = zero_residual
        low = 0.0
        high = 1.0

        for _ in range(cfg.CBF.binary_search_iterations):
            scale = (low + high) / 2.0
            scaled_action = scale_action(action, scale)
            _, scaled_safety, scaled_residual = self._candidate_safety(scaled_action)
            if scaled_residual >= 0.0:
                low = scale
                best_scale = scale
                best_action = scaled_action
                best_safety = scaled_safety
                best_residual = scaled_residual
            else:
                high = scale

        return best_action, best_safety, best_residual, best_scale

    def _update_task_success(
        self,
        cube_to_goal: float,
        ee_pos: list[float],
    ) -> tuple[bool, float]:
        if cube_to_goal <= self._goal_radius:
            self._state.goal_reached_once = True

        home_distance = self._distance(ee_pos, self._state.home_ee_pos)
        home_reached = home_distance <= self._goal_radius
        success = self._state.goal_reached_once and home_reached
        return success, home_distance

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> PickAndPlaceObservation:
        """
        Reset the environment and return the first observation.

        Args:
            seed: Optional seed for deterministic placement sampling
            episode_id: Optional episode identifier
            **kwargs: Additional reset arguments

        Returns:
            PickAndPlaceObservation with the initial scene state
        """
        snapshot = self._simulator.reset(seed=seed)
        self._state = self._make_state(
            snapshot=snapshot,
            episode_id=episode_id,
        )
        rgb_overhead, rgb_wrist = self._simulator.render_observation_images()

        return build_observation(
            self._state,
            reward=0.0,
            done=False,
            rgb_overhead=rgb_overhead,
            rgb_wrist=rgb_wrist,
        )

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

        proposed_action = action
        if self._enable_cbf_filter:
            (
                filtered_action,
                candidate_safety,
                candidate_residual,
                candidate_scale,
            ) = self._filter_action_with_cbf(proposed_action)
            snapshot = self._simulator.step(filtered_action)
        else:
            filtered_action = proposed_action
            candidate_scale = 1.0
            snapshot = self._simulator.step(filtered_action)
            candidate_safety = compute_safety_margins(
                list(snapshot.ee_pos),
                ee_linear_velocity=list(snapshot.ee_linear_velocity),
                step_dt=self._simulator.step_dt,
                max_velocity_mps=self._max_velocity_mps,
                joint_limit_margin=self._simulator.compute_joint_limit_margin(
                    snapshot.proprioception.joint_angles
                ),
            )
            candidate_residual = compute_cbf_residual(
                self._state.safety_margins,
                candidate_safety,
            )
        cube_to_goal = self._distance(snapshot.cube_pos, snapshot.goal_pos)
        success, home_distance = self._update_task_success(
            cube_to_goal=cube_to_goal,
            ee_pos=list(snapshot.ee_pos),
        )
        self._state.step_count += 1
        self._state.ee_pos = list(snapshot.ee_pos)
        self._state.ee_quat = list(snapshot.ee_quat)
        self._state.cube_pos = list(snapshot.cube_pos)
        self._state.goal_pos = list(snapshot.goal_pos)
        self._state.cube_height = snapshot.cube_height
        self._state.gripper_contact = snapshot.gripper_contact
        self._state.success = success
        self._state.proposed_action = proposed_action
        self._state.last_action = filtered_action
        self._state.cbf_intervened = self._enable_cbf_filter and (
            filtered_action != proposed_action
        )
        self._state.cbf_scale = candidate_scale
        self._state.cbf_residual = candidate_residual
        self._state.proprioception = snapshot.proprioception
        self._state.safety_margins = candidate_safety
        self._state.reward_breakdown = compute_reward_breakdown(
            gripper_to_cube=self._distance(snapshot.ee_pos, snapshot.cube_pos),
            cube_to_goal=cube_to_goal,
            cube_height=snapshot.cube_height,
            gripper_contact=snapshot.gripper_contact,
            success=success,
        )
        reward = self._state.reward_breakdown.total
        self._state.phase = detect_phase(
            self._state,
            cube_to_goal,
            self._goal_radius,
        )

        done = self._state.success or snapshot.terminated or snapshot.truncated
        rgb_overhead, rgb_wrist = self._simulator.render_observation_images()

        observation = build_observation(
            self._state,
            reward=reward,
            done=done,
            rgb_overhead=rgb_overhead,
            rgb_wrist=rgb_wrist,
        )
        observation.metadata["home_distance"] = home_distance
        observation.metadata["home_reached"] = home_distance <= self._goal_radius
        return observation

    def __del__(self) -> None:
        try:
            self._simulator.close()
        except Exception:
            pass

    @property
    def state(self) -> PickAndPlaceState:
        """Get the current environment state."""
        return self._state
