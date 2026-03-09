# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from enum import StrEnum
from dataclasses import dataclass


class ObsMode(StrEnum):
    VISION = "vision"
    STATE = "state"
    MULTIMODAL = "multimodal"


class GripperCommand(StrEnum):
    OPEN = "open"
    CLOSE = "close"


class Phase(StrEnum):
    REACHING = "reaching"
    GRASPING = "grasping"
    LIFTING = "lifting"
    PLACING = "placing"


@dataclass(frozen=True)
class TaskConfig:
    instruction: str = "Pick up the cube and place it in the tray."
    max_steps: int = 100
    gripper_open_width: float = 0.08
    gripper_closed_width: float = 0.0


@dataclass(frozen=True)
class SimulatorConfig:
    env_id: str = "FetchPickAndPlace-v4"
    control_min: float = -1.0
    control_max: float = 1.0
    position_action_scale_meters: float = 0.05
    grip_site_name: str = "robot0:grip"
    arm_joint_names: tuple[str, ...] = (
        "robot0:shoulder_pan_joint",
        "robot0:shoulder_lift_joint",
        "robot0:upperarm_roll_joint",
        "robot0:elbow_flex_joint",
        "robot0:forearm_roll_joint",
        "robot0:wrist_flex_joint",
        "robot0:wrist_roll_joint",
    )


@dataclass(frozen=True)
class GeometryConfig:
    table_height: float = 0.42
    grasp_distance: float = 0.03
    lift_threshold: float = 0.05
    goal_radius: float = 0.05
    workspace_low: tuple[float, float, float] = (0.35, -0.15, 0.42)
    workspace_high: tuple[float, float, float] = (0.75, 0.15, 0.70)
    ee_start_pos: tuple[float, float, float] = (0.42, 0.0, 0.62)
    ee_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    cube_start_pos: tuple[float, float, float] = (0.50, 0.0, 0.42)
    goal_pos: tuple[float, float, float] = (0.70, 0.0, 0.42)


@dataclass(frozen=True)
class ResetConfig:
    cube_x_jitter: float = 0.03
    cube_y_jitter: float = 0.04
    goal_x_jitter: float = 0.02
    goal_y_jitter: float = 0.05


TASK = TaskConfig()
SIMULATOR = SimulatorConfig()
GEOMETRY = GeometryConfig()
RESET = ResetConfig()
