# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pick And Place environment client.

This module provides the client for connecting to the Pick And Place
environment server via WebSocket for persistent sessions.
"""

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import (
    PickAndPlaceAction,
    PickAndPlaceObservation,
    PickAndPlaceState,
    Proprioception,
    RewardBreakdown,
    SafetyMargins,
)


class PickAndPlaceEnv(
    EnvClient[PickAndPlaceAction, PickAndPlaceObservation, PickAndPlaceState]
):
    """
    Client for the Pick And Place environment.

    This client maintains a persistent WebSocket connection to the environment
    server, enabling efficient multi-step interactions with lower latency.
    """

    @staticmethod
    def _parse_proprioception(payload: Dict[str, Any]) -> Proprioception:
        return Proprioception(
            ee_pos=payload.get("ee_pos", [0.0, 0.0, 0.0]),
            ee_quat=payload.get("ee_quat", [1.0, 0.0, 0.0, 0.0]),
            gripper_width=payload.get("gripper_width", 0.0),
            joint_angles=payload.get("joint_angles", []),
        )

    @staticmethod
    def _parse_reward_breakdown(payload: Dict[str, Any]) -> RewardBreakdown:
        return RewardBreakdown(
            format=payload.get("format", 0.0),
            bounds=payload.get("bounds", 0.0),
            reach=payload.get("reach", 0.0),
            grasp=payload.get("grasp", 0.0),
            lift=payload.get("lift", 0.0),
            place=payload.get("place", 0.0),
            success=payload.get("success", 0.0),
            cbf=payload.get("cbf", 0.0),
            total=payload.get("total", 0.0),
        )

    @staticmethod
    def _parse_safety_margins(payload: Dict[str, Any]) -> SafetyMargins:
        return SafetyMargins(
            workspace=payload.get("workspace", 0.0),
            velocity=payload.get("velocity", 0.0),
            joint_limit=payload.get("joint_limit", 0.0),
            minimum=payload.get("minimum", 0.0),
        )

    def _step_payload(self, action: PickAndPlaceAction) -> Dict[str, Any]:
        """
        Convert PickAndPlaceAction to JSON payload for step requests.

        Args:
            action: PickAndPlaceAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        payload: Dict[str, Any] = {
            "dx": action.dx,
            "dy": action.dy,
            "dz": action.dz,
            "gripper": action.gripper,
        }

        if action.metadata:
            payload["metadata"] = action.metadata

        # Keep the scaffold server usable until the real server step lands.
        payload["message"] = action.message or ""

        return payload

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[PickAndPlaceObservation]:
        """
        Parse server response into StepResult[PickAndPlaceObservation].

        Args:
            payload: JSON response from server

        Returns:
            StepResult with PickAndPlaceObservation
        """
        obs_data = payload.get("observation", {})

        observation = PickAndPlaceObservation(
            rgb_overhead=obs_data.get("rgb_overhead"),
            rgb_wrist=obs_data.get("rgb_wrist"),
            scene_text=obs_data.get("scene_text", ""),
            instruction=obs_data.get(
                "instruction", "Pick up the cube and place it in the tray."
            ),
            phase=obs_data.get("phase", "reaching"),
            obs_mode=obs_data.get("obs_mode", "multimodal"),
            proprioception=self._parse_proprioception(
                obs_data.get("proprioception", {})
            ),
            reward_breakdown=self._parse_reward_breakdown(
                obs_data.get("reward_breakdown", {})
            ),
            safety_margins=self._parse_safety_margins(
                obs_data.get("safety_margins", {})
            ),
            echoed_message=obs_data.get("echoed_message", ""),
            message_length=obs_data.get("message_length", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> PickAndPlaceState:
        """
        Parse server response into PickAndPlaceState.

        Args:
            payload: JSON response from state request

        Returns:
            PickAndPlaceState object
        """
        return PickAndPlaceState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            obs_mode=payload.get("obs_mode", "multimodal"),
            phase=payload.get("phase", "reaching"),
            max_steps=payload.get("max_steps", 100),
            ee_pos=payload.get("ee_pos", [0.0, 0.0, 0.0]),
            ee_quat=payload.get("ee_quat", [1.0, 0.0, 0.0, 0.0]),
            cube_pos=payload.get("cube_pos", [0.0, 0.0, 0.0]),
            goal_pos=payload.get("goal_pos", [0.0, 0.0, 0.0]),
            cube_height=payload.get("cube_height", 0.0),
            gripper_contact=payload.get("gripper_contact", False),
            proprioception=self._parse_proprioception(
                payload.get("proprioception", {})
            ),
            reward_breakdown=self._parse_reward_breakdown(
                payload.get("reward_breakdown", {})
            ),
            safety_margins=self._parse_safety_margins(
                payload.get("safety_margins", {})
            ),
            last_action=(
                PickAndPlaceAction(**payload["last_action"])
                if payload.get("last_action")
                else None
            ),
            success=payload.get("success", False),
        )
