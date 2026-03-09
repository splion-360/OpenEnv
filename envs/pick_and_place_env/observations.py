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


def build_scene_text(state: PickAndPlaceState) -> str:
    return (
        f"Phase: {state.phase}. "
        f"Gripper at ({state.ee_pos[0]:.2f}, {state.ee_pos[1]:.2f}, {state.ee_pos[2]:.2f}). "
        f"Cube at ({state.cube_pos[0]:.2f}, {state.cube_pos[1]:.2f}, {state.cube_pos[2]:.2f}). "
        f"Cube height: {state.cube_height:.2f}m. "
        f"Goal: tray at ({state.goal_pos[0]:.2f}, {state.goal_pos[1]:.2f}, {state.goal_pos[2]:.2f})."
    )


def build_observation(
    state: PickAndPlaceState,
    reward: float,
    done: bool,
    echoed_message: str = "",
    rgb_overhead: str | None = None,
    rgb_wrist: str | None = None,
) -> PickAndPlaceObservation:
    include_images = state.obs_mode != cfg.ObsMode.STATE
    include_text = state.obs_mode != cfg.ObsMode.VISION

    return PickAndPlaceObservation(
        rgb_overhead=rgb_overhead if include_images else None,
        rgb_wrist=rgb_wrist if include_images else None,
        scene_text=build_scene_text(state) if include_text else "",
        instruction=cfg.TASK.instruction,
        phase=state.phase,
        obs_mode=state.obs_mode,
        steps_remaining=max(0, state.max_steps - state.step_count),
        proprioception=state.proprioception,
        reward_breakdown=state.reward_breakdown,
        safety_margins=state.safety_margins,
        echoed_message=echoed_message,
        message_length=len(echoed_message),
        done=done,
        reward=reward,
        metadata={
            "step": state.step_count,
            "steps_remaining": max(0, state.max_steps - state.step_count),
            "success": state.success,
        },
    )
