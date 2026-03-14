# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

try:
    from . import config as cfg
    from .models import PickAndPlaceObservation, PickAndPlaceState
except ImportError:
    try:
        import config as cfg  # type: ignore
        from models import PickAndPlaceObservation, PickAndPlaceState  # type: ignore
    except ImportError:
        from pick_and_place_env import config as cfg  # type: ignore
        from pick_and_place_env.models import (  # type: ignore
            PickAndPlaceObservation,
            PickAndPlaceState,
        )


def detect_phase(
    state: PickAndPlaceState,
    cube_to_goal: float,
    place_radius: float,
) -> cfg.Phase:
    if state.success:
        return cfg.Phase.PLACING
    if not state.gripper_contact:
        return cfg.Phase.REACHING
    if state.cube_height < cfg.PHASE.lift_threshold:
        return cfg.Phase.GRASPING
    if cube_to_goal < place_radius * 1.5:
        return cfg.Phase.PLACING
    return cfg.Phase.LIFTING


def build_observation(
    state: PickAndPlaceState,
    reward: float,
    done: bool,
    rgb_overhead: str | None = None,
    rgb_wrist: str | None = None,
) -> PickAndPlaceObservation:
    return PickAndPlaceObservation(
        rgb_overhead=rgb_overhead,
        rgb_wrist=rgb_wrist,
        instruction=cfg.TASK.instruction_goal,
        steps_remaining=max(0, state.max_steps - state.step_count),
        home_ee_pos=list(state.home_ee_pos),
        is_grasped=state.gripper_contact,
        done=done,
        reward=reward,
        metadata={
            "step": state.step_count,
            "steps_remaining": max(0, state.max_steps - state.step_count),
            "success": state.success,
            "phase": state.phase,
            "proprioception": state.proprioception.model_dump(),
            "reward_breakdown": state.reward_breakdown.model_dump(),
            "safety_margins": state.safety_margins.model_dump(),
            "goal_reached_once": state.goal_reached_once,
            "cbf_intervened": state.cbf_intervened,
            "cbf_scale": state.cbf_scale,
            "cbf_residual": state.cbf_residual,
            "proposed_action": (
                state.proposed_action.model_dump() if state.proposed_action else None
            ),
            "executed_action": (
                state.last_action.model_dump() if state.last_action else None
            ),
        },
    )
