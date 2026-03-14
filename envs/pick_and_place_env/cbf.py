# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

try:
    from . import config as cfg
    from .models import PickAndPlaceAction, SafetyMargins
except ImportError:
    try:
        import config as cfg  # type: ignore
        from models import PickAndPlaceAction, SafetyMargins  # type: ignore
    except ImportError:
        from pick_and_place_env import config as cfg  # type: ignore
        from pick_and_place_env.models import (  # type: ignore
            PickAndPlaceAction,
            SafetyMargins,
        )


def compute_safety_margins(
    position: list[float],
    ee_linear_velocity: list[float],
    step_dt: float,
    max_velocity_mps: float,
    joint_limit_margin: float,
    joint_velocity_margin: float,
) -> SafetyMargins:
    workspace_margin = min(
        position[0] - cfg.SAFETY.workspace_low[0],
        cfg.SAFETY.workspace_high[0] - position[0],
        position[1] - cfg.SAFETY.workspace_low[1],
        cfg.SAFETY.workspace_high[1] - position[1],
        position[2] - cfg.SAFETY.workspace_low[2],
        cfg.SAFETY.workspace_high[2] - position[2],
    )
    ee_speed_mps = math.dist(ee_linear_velocity, [0.0, 0.0, 0.0]) / max(step_dt, 1e-9)
    velocity_margin = max_velocity_mps - ee_speed_mps

    return SafetyMargins(
        workspace=workspace_margin,
        velocity=velocity_margin,
        joint_limit=joint_limit_margin,
        joint_velocity=joint_velocity_margin,
        minimum=min(
            workspace_margin,
            velocity_margin,
            joint_limit_margin,
            joint_velocity_margin,
        ),
    )


def scale_action(action: PickAndPlaceAction, scale: float) -> PickAndPlaceAction:
    return PickAndPlaceAction(
        dx=action.dx * scale,
        dy=action.dy * scale,
        dz=action.dz * scale,
        gripper=action.gripper,
    )


def compute_cbf_residual(
    current_margins: SafetyMargins,
    next_margins: SafetyMargins,
) -> float:
    gamma = cfg.CBF.gamma
    return min(
        next_margins.velocity - ((1.0 - gamma) * current_margins.velocity),
        next_margins.joint_limit - ((1.0 - gamma) * current_margins.joint_limit),
        next_margins.joint_velocity - ((1.0 - gamma) * current_margins.joint_velocity),
    )
