# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Custom Gradio visualization tab for the pick and place environment."""

from __future__ import annotations

import base64
import io
import json
from typing import Any, Dict, List, Optional

import gradio as gr
from PIL import Image

from openenv.core.env_server.serialization import serialize_observation
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from .. import config as cfg
except ImportError:
    import config as cfg  # type: ignore


def _decode_image(value: Any) -> Optional[Image.Image]:
    """Decode a base64 image field from the observation payload."""
    if not isinstance(value, str) or not value:
        return None

    image_data = value
    if value.startswith("data:"):
        _, _, image_data = value.partition(",")

    try:
        return Image.open(io.BytesIO(base64.b64decode(image_data))).convert("RGB")
    except Exception:
        return None


def _extract_images(
    response: Dict[str, Any],
) -> tuple[Optional[Image.Image], Optional[Image.Image]]:
    """Extract the decoded overhead and wrist images from a response payload."""
    observation = response.get("observation", {})
    if not isinstance(observation, dict):
        return (None, None)

    return (
        _decode_image(observation.get("rgb_overhead")),
        _decode_image(observation.get("rgb_wrist")),
    )


def _format_action(action: Any) -> str:
    """Format a nominal or executed action for compact display."""
    if action is None:
        return "`None`"

    return (
        f"`dx={action.dx:.3f}, dy={action.dy:.3f}, dz={action.dz:.3f}, "
        f"gripper={action.gripper}`"
    )


def _format_summary(response: Dict[str, Any], state: Any | None = None) -> str:
    """Build a compact markdown summary for the latest observation."""
    observation = response.get("observation", {})
    if not isinstance(observation, dict):
        return "*No observation data*"

    lines: List[str] = ["# Visualization"]

    phase = observation.get("phase")
    obs_mode = observation.get("obs_mode")
    if phase:
        lines.append(f"**Phase:** `{phase}`")
    if obs_mode:
        lines.append(f"**Observation mode:** `{obs_mode}`")
    if "steps_remaining" in observation:
        lines.append(f"**Steps remaining:** `{observation['steps_remaining']}`")

    scene_text = observation.get("scene_text")
    if scene_text:
        lines.append("")
        lines.append(f"**Scene:** {scene_text}")

    reward = response.get("reward")
    done = response.get("done")
    if reward is not None or done is not None:
        lines.append("")
    if reward is not None:
        lines.append(f"**Reward:** `{reward}`")
    if done is not None:
        lines.append(f"**Done:** `{done}`")

    if state is not None:
        velocity = state.proprioception.ee_linear_velocity
        lines.append("")
        lines.append(
            "**EE linear velocity (x,y,z):** "
            f"`[{velocity[0]:.4f}, {velocity[1]:.4f}, {velocity[2]:.4f}]`"
        )
        lines.append(f"**CBF intervened:** `{state.cbf_intervened}`")
        lines.append(f"**CBF scale:** `{state.cbf_scale:.3f}`")
        lines.append(f"**CBF residual:** `{state.cbf_residual:.6f}`")
        lines.append(f"**Proposed action:** {_format_action(state.proposed_action)}")
        lines.append(f"**Executed action:** {_format_action(state.last_action)}")

    return "\n".join(lines)


def build_pick_and_place_gradio_app(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],
    metadata: Optional[EnvironmentMetadata],
    is_chat_env: bool,
    title: str,
    quick_start_md: str,
) -> gr.Blocks:
    """Build the visualization tab for pick_and_place_env."""

    async def reset_env(obs_mode: str):
        try:
            observation = await web_manager._run_sync_in_thread_pool(
                web_manager.env.reset,
                obs_mode=cfg.ObsMode(obs_mode),
            )
            state = web_manager.env.state
            response = serialize_observation(observation)
            web_manager.episode_state.episode_id = state.episode_id
            web_manager.episode_state.step_count = 0
            web_manager.episode_state.current_observation = response["observation"]
            web_manager.episode_state.action_logs = []
            web_manager.episode_state.is_reset = True
            await web_manager._send_state_update()
            overhead_image, wrist_image = _extract_images(response)
            return (
                _format_summary(response, state),
                overhead_image,
                wrist_image,
                json.dumps(response, indent=2),
                f"Environment reset successfully in {obs_mode} mode.",
            )
        except Exception as error:
            return ("", None, None, "", f"Error: {error}")

    async def step_form(*values):
        action_data = {}
        for index, field in enumerate(action_fields):
            if index >= len(values):
                break

            value = values[index]
            if field.get("type") == "checkbox":
                action_data[field["name"]] = bool(value)
            elif value is not None and value != "":
                action_data[field["name"]] = value

        try:
            response = await web_manager.step_environment(action_data)
            state = web_manager.env.state
            overhead_image, wrist_image = _extract_images(response)
            return (
                _format_summary(response, state),
                overhead_image,
                wrist_image,
                json.dumps(response, indent=2),
                "Step complete.",
            )
        except Exception as error:
            return ("", None, None, "", f"Error: {error}")

    def get_state_sync():
        try:
            return json.dumps(web_manager.get_state(), indent=2), "State refreshed."
        except Exception as error:
            return f"Error: {error}", f"Error: {error}"

    with gr.Blocks(title=f"{title} — Visualization") as blocks:
        summary = gr.Markdown(
            "# Visualization\n\nClick **Reset** to load the current overhead and wrist cameras."
        )
        with gr.Row():
            overhead_image = gr.Image(label="Overhead Camera", interactive=False)
            wrist_image = gr.Image(label="Wrist Camera", interactive=False)

        with gr.Group():
            obs_mode_input = gr.Dropdown(
                choices=[mode.value for mode in cfg.ObsMode],
                value=cfg.ObsMode.MULTIMODAL.value,
                label="Observation Mode",
                allow_custom_value=False,
            )
            step_inputs = []
            for field in action_fields:
                name = field["name"]
                label = name.replace("_", " ").title()
                placeholder = field.get("placeholder", "")
                field_type = field.get("type", "text")

                if field_type == "checkbox":
                    component = gr.Checkbox(label=label)
                elif field_type == "number":
                    step_value = field.get("multiple_of")
                    if step_value is None:
                        step_value = (
                            1 if field.get("schema_type") == "integer" else 1e-6
                        )
                    component = gr.Number(
                        label=label,
                        value=field.get("default_value"),
                        minimum=field.get("min_value"),
                        maximum=field.get("max_value"),
                        step=step_value,
                    )
                elif field_type == "select":
                    component = gr.Dropdown(
                        choices=field.get("choices") or [],
                        label=label,
                        allow_custom_value=False,
                    )
                else:
                    component = gr.Textbox(label=label, placeholder=placeholder)

                step_inputs.append(component)

            with gr.Row():
                step_button = gr.Button("Step", variant="primary")
                reset_button = gr.Button("Reset", variant="secondary")
                state_button = gr.Button("Get state", variant="secondary")

            status = gr.Textbox(label="Status", interactive=False)
            raw_json = gr.Code(
                label="Raw JSON response",
                language="json",
                interactive=False,
            )

        reset_button.click(
            fn=reset_env,
            inputs=[obs_mode_input],
            outputs=[summary, overhead_image, wrist_image, raw_json, status],
        )
        step_button.click(
            fn=step_form,
            inputs=step_inputs,
            outputs=[summary, overhead_image, wrist_image, raw_json, status],
        )
        state_button.click(fn=get_state_sync, outputs=[raw_json, status])

    return blocks
