# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run end-to-end PickAndPlace episodes with optional live rendering."""

from __future__ import annotations

import argparse
import base64
import io
import math
import random
import time
from dataclasses import dataclass
from typing import Optional

from PIL import Image

from pick_and_place_env import config as cfg
from pick_and_place_env.models import PickAndPlaceAction
from pick_and_place_env.server.pick_and_place_env_environment import (
    PickAndPlaceEnvironment,
)

from logger import EpisodeConsoleLogger

__test__ = False


def _decode_base64_image(value: str | None) -> Optional[Image.Image]:
    if not value:
        return None
    image_data = value
    if value.startswith("data:"):
        _, _, image_data = value.partition(",")
    return Image.open(io.BytesIO(base64.b64decode(image_data))).convert("RGB")


class LiveRenderer:
    """Simple Tk-based dual camera live renderer."""

    def __init__(self, fps: float) -> None:
        from PIL import ImageTk
        import tkinter as tk

        self._ImageTk = ImageTk
        self._fps = max(1.0, fps)

        self._root = tk.Tk()
        self._root.title("PickAndPlace Live Render")
        self._left_label = tk.Label(self._root, text="Overhead")
        self._left_label.grid(row=0, column=0, padx=8, pady=8)
        self._right_label = tk.Label(self._root, text="Wrist")
        self._right_label.grid(row=0, column=1, padx=8, pady=8)
        self._status = tk.Label(self._root, text="Starting...")
        self._status.grid(row=1, column=0, columnspan=2, sticky="w", padx=8, pady=8)

        self._left_photo = None
        self._right_photo = None

    def update(
        self,
        *,
        overhead: Image.Image | None,
        wrist: Image.Image | None,
        episode: int,
        step: int,
        reward: float | None,
        done: bool,
    ) -> None:
        if overhead is not None:
            self._left_photo = self._ImageTk.PhotoImage(overhead)
            self._left_label.configure(image=self._left_photo)
        if wrist is not None:
            self._right_photo = self._ImageTk.PhotoImage(wrist)
            self._right_label.configure(image=self._right_photo)

        reward_str = "None" if reward is None else f"{reward:.4f}"
        self._status.configure(
            text=f"Episode {episode} | Step {step} | Reward {reward_str} | Done {done}"
        )
        self._root.update_idletasks()
        self._root.update()
        time.sleep(1.0 / self._fps)

    def close(self) -> None:
        try:
            self._root.destroy()
        except Exception:
            pass


@dataclass
class RunSummary:
    episodes_completed: int
    total_steps: int
    max_steps_seen: int
    success_episodes: int


def _expected_total(state) -> float:
    rb = state.reward_breakdown
    return (
        cfg.REWARD.reach_weight * rb.reach
        + cfg.REWARD.grasp_weight * rb.grasp
        + cfg.REWARD.lift_weight * rb.lift
        + cfg.REWARD.place_weight * rb.place
        + cfg.REWARD.phase_transition_weight * rb.phase_transition
        + cfg.REWARD.approach_velocity_weight * rb.approach_velocity
        + cfg.REWARD.place_velocity_weight * rb.place_velocity
        + cfg.REWARD.success_weight * rb.success
    )


def _robot_state_snapshot(state) -> dict[str, object]:
    return {
        "phase": state.phase,
        "ee_pos": state.ee_pos,
        "cube_pos": state.cube_pos,
        "goal_pos": state.goal_pos,
        "cube_height": f"{state.cube_height:.6f}",
        "gripper_contact": state.gripper_contact,
        "goal_reached_once": state.goal_reached_once,
    }


def run_episodes(
    episodes: int,
    seed: int,
    render: bool,
    fps: float,
    logger: EpisodeConsoleLogger,
) -> RunSummary:
    rng = random.Random(seed)
    env = PickAndPlaceEnvironment(enable_cbf_filter=False)
    renderer = LiveRenderer(fps=fps) if render else None

    completed = 0
    total_steps = 0
    max_steps_seen = 0
    successes = 0

    try:
        for episode in range(episodes):
            observation = env.reset(seed=seed + episode)
            done = bool(observation.done)
            step = 0

            if renderer is not None:
                renderer.update(
                    overhead=_decode_base64_image(observation.rgb_overhead),
                    wrist=_decode_base64_image(observation.rgb_wrist),
                    episode=episode,
                    step=step,
                    reward=observation.reward,
                    done=done,
                )

            while not done and step < cfg.TASK.max_steps:
                action = PickAndPlaceAction(
                    dx=rng.uniform(-1.0, 1.0),
                    dy=rng.uniform(-1.0, 1.0),
                    dz=rng.uniform(-1.0, 1.0),
                    gripper=rng.uniform(-1.0, 1.0),
                )
                observation = env.step(action)
                state = env.state
                breakdown = state.reward_breakdown

                values = (
                    breakdown.reach,
                    breakdown.grasp,
                    breakdown.lift,
                    breakdown.place,
                    breakdown.phase_transition,
                    breakdown.approach_velocity,
                    breakdown.place_velocity,
                    breakdown.success,
                    breakdown.total,
                )
                if not all(math.isfinite(v) for v in values):
                    raise RuntimeError(
                        f"Non-finite reward component at episode={episode} step={step}: {breakdown}"
                    )

                expected = _expected_total(state)
                if not math.isclose(
                    breakdown.total, expected, rel_tol=1e-9, abs_tol=1e-9
                ):
                    raise RuntimeError(
                        f"Reward mismatch episode={episode} step={step}: "
                        f"total={breakdown.total} expected={expected}"
                    )

                done = bool(observation.done)
                step += 1
                total_steps += 1

                if renderer is not None:
                    renderer.update(
                        overhead=_decode_base64_image(observation.rgb_overhead),
                        wrist=_decode_base64_image(observation.rgb_wrist),
                        episode=episode,
                        step=step,
                        reward=observation.reward,
                        done=done,
                    )

            completed += 1
            max_steps_seen = max(max_steps_seen, step)
            if env.state.success:
                successes += 1

            logger.log_episode_summary(
                episode_index=episode,
                step_count=step,
                total_reward=env.state.reward_breakdown.total,
                success=env.state.success,
                reward_breakdown=env.state.reward_breakdown.model_dump(),
                robot_state=_robot_state_snapshot(env.state),
            )

    finally:
        if renderer is not None:
            renderer.close()
        env._simulator.close()

    return RunSummary(
        episodes_completed=completed,
        total_steps=total_steps,
        max_steps_seen=max_steps_seen,
        success_episodes=successes,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--fps", type=float, default=15.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = EpisodeConsoleLogger()
    summary = run_episodes(
        episodes=args.episodes,
        seed=args.seed,
        render=args.render,
        fps=args.fps,
        logger=logger,
    )
    logger.log_metrics(
        title="End-to-End Validation Result",
        metrics={
            "status": "VALIDATION_OK",
            "episodes_completed": summary.episodes_completed,
            "total_steps": summary.total_steps,
            "max_steps_seen": summary.max_steps_seen,
            "success_episodes": summary.success_episodes,
        },
    )


if __name__ == "__main__":
    main()
