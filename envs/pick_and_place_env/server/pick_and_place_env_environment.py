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
    from ..cbf import compute_safety_margins
    from ..models import (
        PickAndPlaceAction,
        PickAndPlaceObservation,
        PickAndPlaceState,
        Proprioception,
    )
    from ..observations import build_observation, detect_phase
    from ..rewards import compute_reward_breakdown
    from ..simulator import FetchPickAndPlaceSimulator, SimulatorSnapshot
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
        from simulator import (  # type: ignore
            FetchPickAndPlaceSimulator,
            SimulatorSnapshot,
        )
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
        from pick_and_place_env.simulator import (  # type: ignore
            FetchPickAndPlaceSimulator,
            SimulatorSnapshot,
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
        self._max_steps = max_steps
        self._simulator = FetchPickAndPlaceSimulator(max_episode_steps=max_steps)
        self._state = self._make_state(snapshot=None)

    def _make_state(
        self,
        snapshot: Optional[SimulatorSnapshot],
        episode_id: Optional[str] = None,
        obs_mode: cfg.ObsMode = cfg.ObsMode.MULTIMODAL,
    ) -> PickAndPlaceState:
        if snapshot is None:
            ee_pos = list(cfg.GEOMETRY.ee_start_pos)
            ee_quat = list(cfg.GEOMETRY.ee_quat)
            cube_pos = list(cfg.GEOMETRY.cube_start_pos)
            goal_pos = list(cfg.GEOMETRY.goal_pos)
            cube_height = 0.0
            gripper_contact = False
            success = False
            proprioception = Proprioception(
                ee_pos=ee_pos,
                ee_quat=ee_quat,
                gripper_width=cfg.TASK.gripper_open_width,
                joint_angles=[],
            )
            reward_breakdown = compute_reward_breakdown(success=False)
            safety = compute_safety_margins([0.0, 0.0, 0.0], ee_pos)
        else:
            ee_pos = list(snapshot.ee_pos)
            ee_quat = list(snapshot.ee_quat)
            cube_pos = list(snapshot.cube_pos)
            goal_pos = list(snapshot.goal_pos)
            cube_height = snapshot.cube_height
            gripper_contact = snapshot.gripper_contact
            success = bool(snapshot.info.get("is_success", False))
            proprioception = snapshot.proprioception
            reward_breakdown = compute_reward_breakdown(success=success)
            safety = compute_safety_margins([0.0, 0.0, 0.0], ee_pos)

        cube_to_goal = self._distance(cube_pos, goal_pos)
        state = PickAndPlaceState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            obs_mode=obs_mode,
            phase=cfg.Phase.REACHING,
            max_steps=self._max_steps,
            ee_pos=list(ee_pos),
            ee_quat=list(ee_quat),
            cube_pos=list(cube_pos),
            goal_pos=list(goal_pos),
            cube_height=cube_height,
            gripper_contact=gripper_contact,
            proprioception=proprioception,
            reward_breakdown=reward_breakdown,
            safety_margins=safety,
            last_action=None,
            success=success,
        )
        state.phase = detect_phase(state, cube_to_goal)
        return state

    def _distance(self, left: list[float], right: list[float]) -> float:
        return math.dist(left, right)

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
        snapshot = self._simulator.reset(seed=seed)
        self._state = self._make_state(
            snapshot=snapshot,
            episode_id=episode_id,
            obs_mode=obs_mode,
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
        snapshot = self._simulator.step(action)
        cube_to_goal = self._distance(snapshot.cube_pos, snapshot.goal_pos)
        success = bool(snapshot.info.get("is_success", False))
        self._state.step_count += 1
        self._state.ee_pos = list(snapshot.ee_pos)
        self._state.ee_quat = list(snapshot.ee_quat)
        self._state.cube_pos = list(snapshot.cube_pos)
        self._state.goal_pos = list(snapshot.goal_pos)
        self._state.cube_height = snapshot.cube_height
        self._state.gripper_contact = snapshot.gripper_contact
        self._state.success = success
        self._state.last_action = action
        self._state.proprioception = snapshot.proprioception
        self._state.safety_margins = compute_safety_margins(
            [
                snapshot.ee_pos[0] - previous_ee[0],
                snapshot.ee_pos[1] - previous_ee[1],
                snapshot.ee_pos[2] - previous_ee[2],
            ],
            list(snapshot.ee_pos),
        )
        self._state.reward_breakdown = compute_reward_breakdown(success=success)
        reward = self._state.reward_breakdown.total
        self._state.phase = detect_phase(self._state, cube_to_goal)

        done = self._state.success or snapshot.terminated or snapshot.truncated
        echoed_message = action.message or ""

        return build_observation(
            self._state,
            reward=reward,
            done=done,
            echoed_message=echoed_message,
        )

    def __del__(self) -> None:
        try:
            self._simulator.close()
        except Exception:
            pass

    @property
    def state(self) -> PickAndPlaceState:
        """Get the current environment state."""
        return self._state
