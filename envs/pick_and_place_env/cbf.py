# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

try:
    from . import config as cfg
    from .models import SafetyMargins
except ImportError:
    try:
        import config as cfg  # type: ignore
        from models import SafetyMargins  # type: ignore
    except ImportError:
        from pick_and_place_env import config as cfg  # type: ignore
        from pick_and_place_env.models import SafetyMargins  # type: ignore


def compute_safety_margins(
    delta: list[float],
    position: list[float],
) -> SafetyMargins:
    workspace_margin = min(
        position[0] - cfg.GEOMETRY.workspace_low[0],
        cfg.GEOMETRY.workspace_high[0] - position[0],
        position[1] - cfg.GEOMETRY.workspace_low[1],
        cfg.GEOMETRY.workspace_high[1] - position[1],
        position[2] - cfg.GEOMETRY.workspace_low[2],
        cfg.GEOMETRY.workspace_high[2] - position[2],
    )
    velocity_margin = cfg.TASK.max_action_delta_meters - math.dist(
        delta, [0.0, 0.0, 0.0]
    )
    joint_limit_margin = 1.0

    return SafetyMargins(
        workspace=workspace_margin,
        velocity=velocity_margin,
        joint_limit=joint_limit_margin,
        minimum=min(workspace_margin, velocity_margin, joint_limit_margin),
    )
