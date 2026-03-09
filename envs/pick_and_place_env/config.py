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


@dataclass(frozen=True)
class PhaseConfig:
    lift_threshold: float = 0.05


@dataclass(frozen=True)
class SafetyConfig:
    workspace_low: tuple[float, float, float] = (1.10, 0.55, 0.38)
    workspace_high: tuple[float, float, float] = (1.60, 1.00, 0.90)


@dataclass(frozen=True)
class RewardConfig:
    format_weight: float = 1.0
    bounds_weight: float = 0.5
    reach_weight: float = 1.0
    grasp_weight: float = 2.0
    lift_weight: float = 1.5
    place_weight: float = 2.0
    success_weight: float = 10.0
    cbf_weight: float = 0.8


@dataclass(frozen=True)
class CBFConfig:
    gamma: float = 0.2
    binary_search_iterations: int = 8


TASK = TaskConfig()
PHASE = PhaseConfig()
SAFETY = SafetyConfig()
REWARD = RewardConfig()
CBF = CBFConfig()
