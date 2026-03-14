# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the Pick And Place environment."""

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
    ee_linear_velocity: Vector3 = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="End-effector linear velocity in x/y/z as provided by Fetch observations",
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

    reach: float = Field(default=0.0, description="Dense reaching reward")
    grasp: float = Field(default=0.0, description="Binary grasp reward")
    lift: float = Field(default=0.0, description="Dense lifting reward")
    place: float = Field(default=0.0, description="Dense placing reward")
    phase_transition: float = Field(
        default=0.0, description="Discrete bonus when advancing phase"
    )
    approach_velocity: float = Field(
        default=0.0, description="Penalty for moving too fast near cube"
    )
    place_velocity: float = Field(
        default=0.0, description="Penalty for moving too fast near goal"
    )
    success: float = Field(default=0.0, description="Sparse task completion reward")
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
    gripper: float = Field(
        default=1.0,
        ge=-1.0,
        le=1.0,
        description="Normalized Fetch gripper control in [-1, 1]",
    )


class PickAndPlaceObservation(Observation):
    """Policy-facing multimodal observation."""

    rgb_overhead: Optional[str] = Field(
        default=None,
        description="Base64-encoded overhead RGB image",
    )
    rgb_wrist: Optional[str] = Field(
        default=None,
        description="Base64-encoded wrist-camera RGB image",
    )
    instruction: str = Field(
        default=cfg.TASK.instruction_goal,
        description="Task instruction shown to the policy",
    )
    steps_remaining: int = Field(
        default=cfg.TASK.max_steps,
        ge=0,
        description="Number of environment steps remaining before truncation",
    )
    home_ee_pos: Vector3 = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="Home end-effector position to return to after placement",
    )
    is_grasped: bool = Field(
        default=False,
        description="Whether the cube is currently grasped by the gripper",
    )


class PickAndPlaceState(State):
    """Environment state tracked across an episode."""

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
    home_ee_pos: Vector3 = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="Episode home end-effector position [x, y, z]",
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
    proposed_action: Optional[PickAndPlaceAction] = Field(
        default=None,
        description="Most recent nominal action proposed before CBF filtering",
    )
    last_action: Optional[PickAndPlaceAction] = Field(
        default=None,
        description="Most recent action executed in the environment",
    )
    cbf_intervened: bool = Field(
        default=False,
        description="Whether the CBF filter modified the proposed action",
    )
    cbf_scale: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Scale factor applied to the proposed Cartesian action",
    )
    cbf_residual: float = Field(
        default=0.0,
        description="Discrete-time CBF residual for the executed action",
    )
    success: bool = Field(
        default=False,
        description="Whether the task has been completed successfully",
    )
    goal_reached_once: bool = Field(
        default=False,
        description="Whether cube has reached goal at least once this episode",
    )


Proprioception.model_rebuild()
RewardBreakdown.model_rebuild()
SafetyMargins.model_rebuild()
PickAndPlaceAction.model_rebuild()
PickAndPlaceObservation.model_rebuild()
PickAndPlaceState.model_rebuild()
