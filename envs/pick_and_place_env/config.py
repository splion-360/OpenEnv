# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from enum import StrEnum


class Phase(StrEnum):
    REACHING = "reaching"
    GRASPING = "grasping"
    LIFTING = "lifting"
    PLACING = "placing"


@dataclass(frozen=True)
class TaskConfig:
    instruction_goal: str = "Move the cube to the target position, then return the gripper to the home position."
    max_steps: int = 100


@dataclass(frozen=True)
class PhaseConfig:
    lift_threshold: float = 0.05


@dataclass(frozen=True)
class SafetyConfig:
    workspace_low: tuple[float, float, float] = (1.10, 0.55, 0.38)
    workspace_high: tuple[float, float, float] = (1.60, 1.00, 0.90)


@dataclass(frozen=True)
class RewardConfig:
    reach_weight: float = 1.0
    grasp_weight: float = 2.0
    lift_weight: float = 1.5
    place_weight: float = 2.0
    phase_transition_weight: float = 1.0
    approach_velocity_weight: float = 0.4
    place_velocity_weight: float = 0.4
    success_weight: float = 10.0
    approach_distance_threshold: float = 0.08
    place_distance_threshold: float = 0.08
    max_safe_approach_speed_mps: float = 0.25
    max_safe_place_speed_mps: float = 0.20


@dataclass(frozen=True)
class CBFConfig:
    enable_filter: bool = False
    gamma: float = 0.2
    binary_search_iterations: int = 8
    joint_velocity_limit_scale: float = 1.0


TASK = TaskConfig()
PHASE = PhaseConfig()
SAFETY = SafetyConfig()
REWARD = RewardConfig()
CBF = CBFConfig()
