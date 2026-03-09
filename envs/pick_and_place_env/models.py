# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Pick And Place environment.

These models define the wire contract for a MuJoCo-based pick-and-place task.
Legacy scaffold fields are kept temporarily so the existing placeholder server
continues to boot until the server/client migration is completed.
"""

from __future__ import annotations

from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field

from openenv.core.env_server.types import Action, Observation, State

try:
    from . import config as cfg
except ImportError:
    import config as cfg  # type: ignore

Vector3 = Annotated[list[float], Field(min_length=3, max_length=3)]
Quaternion = Annotated[list[float], Field(min_length=4, max_length=4)]


class _StrictModel(BaseModel):
    """Shared strict configuration for nested wire models."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )


class Proprioception(_StrictModel):
    """Robot proprioceptive state exposed to the policy."""

    ee_pos: Vector3 = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="End-effector position in world coordinates [x, y, z]",
    )
    ee_quat: Quaternion = Field(
        default_factory=lambda: [1.0, 0.0, 0.0, 0.0],
        description="End-effector orientation as quaternion [w, x, y, z]",
    )
    gripper_width: float = Field(
        default=0.0,
        ge=0.0,
        description="Current gripper opening width in meters",
    )
    joint_angles: list[float] = Field(
        default_factory=list,
        description="Robot arm joint angles in radians",
    )


class RewardBreakdown(_StrictModel):
    """Reward components logged for debugging and training analysis."""

    format: float = Field(default=0.0, description="Reward for valid action format")
    bounds: float = Field(default=0.0, description="Reward for staying within limits")
    reach: float = Field(default=0.0, description="Dense reaching reward")
    grasp: float = Field(default=0.0, description="Binary grasp reward")
    lift: float = Field(default=0.0, description="Dense lifting reward")
    place: float = Field(default=0.0, description="Dense placing reward")
    success: float = Field(default=0.0, description="Sparse task completion reward")
    cbf: float = Field(default=0.0, description="Control barrier function reward")
    total: float = Field(default=0.0, description="Total reward for the last step")


class SafetyMargins(_StrictModel):
    """Latest safety margins used by the control barrier function."""

    workspace: float = Field(
        default=0.0,
        description="Margin to workspace boundary; negative means violation",
    )
    velocity: float = Field(
        default=0.0,
        description="Velocity safety margin; negative means violation",
    )
    joint_limit: float = Field(
        default=0.0,
        description="Joint-limit safety margin; negative means violation",
    )
    minimum: float = Field(
        default=0.0,
        description="Minimum active safety margin across all constraints",
    )


class PickAndPlaceAction(Action):
    """Structured robot-arm control action."""

    dx: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Normalized X control in [-1, 1] for the Fetch action space",
    )
    dy: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Normalized Y control in [-1, 1] for the Fetch action space",
    )
    dz: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Normalized Z control in [-1, 1] for the Fetch action space",
    )
    gripper: cfg.GripperCommand = Field(
        default=cfg.GripperCommand.OPEN,
        description="Desired gripper command for this step",
    )
    message: Optional[str] = Field(
        default=None,
        description="Legacy scaffold field kept temporarily for compatibility",
    )


class PickAndPlaceObservation(Observation):
    """Multimodal observation returned by the environment."""

    rgb_overhead: Optional[str] = Field(
        default=None,
        description="Base64-encoded overhead RGB image",
    )
    rgb_wrist: Optional[str] = Field(
        default=None,
        description="Base64-encoded wrist-camera RGB image",
    )
    scene_text: str = Field(
        default="",
        description="Natural language description of the current scene",
    )
    instruction: str = Field(
        default=cfg.TASK.instruction,
        description="Task instruction shown to the policy",
    )
    phase: cfg.Phase = Field(
        default=cfg.Phase.REACHING,
        description="Current deterministic task phase",
    )
    obs_mode: cfg.ObsMode = Field(
        default=cfg.ObsMode.MULTIMODAL,
        description="Observation mode active for the current episode",
    )
    proprioception: Proprioception = Field(
        default_factory=Proprioception,
        description="Robot proprioceptive features",
    )
    reward_breakdown: RewardBreakdown = Field(
        default_factory=RewardBreakdown,
        description="Reward components for the latest step",
    )
    safety_margins: SafetyMargins = Field(
        default_factory=SafetyMargins,
        description="Latest CBF safety margins",
    )
    echoed_message: str = Field(
        default="",
        description="Legacy scaffold field kept temporarily for compatibility",
    )
    message_length: int = Field(
        default=0,
        ge=0,
        description="Legacy scaffold field kept temporarily for compatibility",
    )


class PickAndPlaceState(State):
    """Environment state tracked across an episode."""

    obs_mode: cfg.ObsMode = Field(
        default=cfg.ObsMode.MULTIMODAL,
        description="Observation mode configured for the episode",
    )
    phase: cfg.Phase = Field(
        default=cfg.Phase.REACHING,
        description="Current task phase",
    )
    max_steps: int = Field(
        default=cfg.TASK.max_steps,
        ge=1,
        description="Maximum number of steps allowed in the episode",
    )
    ee_pos: Vector3 = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="End-effector position [x, y, z]",
    )
    ee_quat: Quaternion = Field(
        default_factory=lambda: [1.0, 0.0, 0.0, 0.0],
        description="End-effector orientation quaternion [w, x, y, z]",
    )
    cube_pos: Vector3 = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="Cube position [x, y, z]",
    )
    goal_pos: Vector3 = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="Goal tray position [x, y, z]",
    )
    cube_height: float = Field(
        default=0.0,
        description="Cube height above the table in meters",
    )
    gripper_contact: bool = Field(
        default=False,
        description="Whether the gripper is currently contacting the cube",
    )
    proprioception: Proprioception = Field(
        default_factory=Proprioception,
        description="Latest robot proprioceptive state",
    )
    reward_breakdown: RewardBreakdown = Field(
        default_factory=RewardBreakdown,
        description="Latest reward component values",
    )
    safety_margins: SafetyMargins = Field(
        default_factory=SafetyMargins,
        description="Latest safety margins from the CBF module",
    )
    last_action: Optional[PickAndPlaceAction] = Field(
        default=None,
        description="Most recent action applied to the environment",
    )
    success: bool = Field(
        default=False,
        description="Whether the task has been completed successfully",
    )


Proprioception.model_rebuild()
RewardBreakdown.model_rebuild()
SafetyMargins.model_rebuild()
PickAndPlaceAction.model_rebuild()
PickAndPlaceObservation.model_rebuild()
PickAndPlaceState.model_rebuild()
