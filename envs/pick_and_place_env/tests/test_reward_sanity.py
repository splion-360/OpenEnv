# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reward sanity runner for pick_and_place_env."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

from pick_and_place_env import config as cfg
from pick_and_place_env.models import PickAndPlaceAction
from pick_and_place_env.server.pick_and_place_env_environment import (
    PickAndPlaceEnvironment,
)

from logger import EpisodeConsoleLogger

__test__ = False


@dataclass
class RewardSanitySummary:
    episodes_completed: int
    total_steps: int
    toward_vs_away_passes: int
    toward_vs_away_failures: int
    phase_transition_events: int
    approach_velocity_events: int
    place_velocity_events: int
    success_events: int
    success_episodes: int


def _l2(left: list[float], right: list[float]) -> float:
    dx = left[0] - right[0]
    dy = left[1] - right[1]
    dz = left[2] - right[2]
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _to_normalized_delta(
    from_pos: list[float],
    to_pos: list[float],
    scale_meters: float,
) -> tuple[float, float, float]:
    dx = (to_pos[0] - from_pos[0]) / scale_meters
    dy = (to_pos[1] - from_pos[1]) / scale_meters
    dz = (to_pos[2] - from_pos[2]) / scale_meters
    return (
        max(-1.0, min(1.0, dx)),
        max(-1.0, min(1.0, dy)),
        max(-1.0, min(1.0, dz)),
    )


def _heuristic_action(
    env: PickAndPlaceEnvironment, rng: random.Random
) -> PickAndPlaceAction:
    state = env.state
    scale = env._simulator.position_action_scale_meters

    ee = list(state.ee_pos)
    cube = list(state.cube_pos)
    goal = list(state.goal_pos)
    home = list(state.home_ee_pos)

    if state.goal_reached_once:
        dx, dy, dz = _to_normalized_delta(ee, home, scale)
        return PickAndPlaceAction(dx=dx, dy=dy, dz=dz, gripper=1.0)

    if not state.gripper_contact:
        close_xy = _l2([ee[0], ee[1], 0.0], [cube[0], cube[1], 0.0]) < 0.02
        target = [cube[0], cube[1], cube[2] + (0.0 if close_xy else 0.03)]
        dx, dy, dz = _to_normalized_delta(ee, target, scale)

        # Deliberately inject aggressive motion near cube to verify approach velocity shaping.
        if _l2(ee, cube) <= cfg.REWARD.approach_distance_threshold:
            dx = max(-1.0, min(1.0, dx + rng.choice([-1.0, 1.0]) * 0.4))
            dy = max(-1.0, min(1.0, dy + rng.choice([-1.0, 1.0]) * 0.4))
            dz = max(-1.0, min(1.0, dz + rng.choice([-1.0, 1.0]) * 0.4))
        return PickAndPlaceAction(dx=dx, dy=dy, dz=dz, gripper=1.0)

    if state.cube_height < (cfg.PHASE.lift_threshold + 0.01):
        return PickAndPlaceAction(dx=0.0, dy=0.0, dz=1.0, gripper=-1.0)

    target = [goal[0], goal[1], goal[2] + 0.03]
    if _l2(cube, goal) < 0.03:
        target = [goal[0], goal[1], goal[2]]
    dx, dy, dz = _to_normalized_delta(ee, target, scale)

    # Deliberately inject aggressive motion near goal to verify place velocity shaping.
    if _l2(cube, goal) <= cfg.REWARD.place_distance_threshold:
        dx = max(-1.0, min(1.0, dx + rng.choice([-1.0, 1.0]) * 0.5))
        dy = max(-1.0, min(1.0, dy + rng.choice([-1.0, 1.0]) * 0.5))
        dz = max(-1.0, min(1.0, dz + rng.choice([-1.0, 1.0]) * 0.5))

    return PickAndPlaceAction(dx=dx, dy=dy, dz=dz, gripper=-1.0)


def run_reward_sanity(
    *,
    episodes: int,
    seed: int,
    strict: bool,
    logger: EpisodeConsoleLogger,
) -> RewardSanitySummary:
    env = PickAndPlaceEnvironment(enable_cbf_filter=False)
    rng = random.Random(seed)

    episodes_completed = 0
    total_steps = 0
    toward_vs_away_passes = 0
    toward_vs_away_failures = 0
    phase_transition_events = 0
    approach_velocity_events = 0
    place_velocity_events = 0
    success_events = 0
    success_episodes = 0

    try:
        for episode in range(episodes):
            env.reset(seed=seed + episode)
            state = env.state
            ee = list(state.ee_pos)
            cube = list(state.cube_pos)
            scale = env._simulator.position_action_scale_meters

            toward_dx, toward_dy, toward_dz = _to_normalized_delta(ee, cube, scale)
            away_dx, away_dy, away_dz = -toward_dx, -toward_dy, -toward_dz
            toward_snapshot = env._simulator.peek_step(
                PickAndPlaceAction(
                    dx=toward_dx,
                    dy=toward_dy,
                    dz=toward_dz,
                    gripper=1.0,
                )
            )
            away_snapshot = env._simulator.peek_step(
                PickAndPlaceAction(
                    dx=away_dx,
                    dy=away_dy,
                    dz=away_dz,
                    gripper=1.0,
                )
            )
            toward_dist = _l2(toward_snapshot.ee_pos, toward_snapshot.cube_pos)
            away_dist = _l2(away_snapshot.ee_pos, away_snapshot.cube_pos)
            if toward_dist < away_dist:
                toward_vs_away_passes += 1
            else:
                toward_vs_away_failures += 1

            done = False
            episode_steps = 0
            episode_reward_total = 0.0
            while not done and episode_steps < cfg.TASK.max_steps:
                action = _heuristic_action(env, rng)
                observation = env.step(action)
                done = bool(observation.done)

                rb = env.state.reward_breakdown
                episode_steps += 1
                total_steps += 1
                episode_reward_total += rb.total

                if rb.phase_transition > 0.0:
                    phase_transition_events += 1
                if rb.approach_velocity < 0.0:
                    approach_velocity_events += 1
                if rb.place_velocity < 0.0:
                    place_velocity_events += 1
                if rb.success > 0.0:
                    success_events += 1

            episodes_completed += 1
            if env.state.success:
                success_episodes += 1

            logger.log_episode_summary(
                episode_index=episode,
                step_count=episode_steps,
                total_reward=episode_reward_total,
                success=env.state.success,
                reward_breakdown=env.state.reward_breakdown.model_dump(),
                robot_state={
                    "phase": env.state.phase,
                    "ee_pos": env.state.ee_pos,
                    "cube_pos": env.state.cube_pos,
                    "goal_pos": env.state.goal_pos,
                    "goal_reached_once": env.state.goal_reached_once,
                    "steps_remaining": max(
                        0, cfg.TASK.max_steps - env.state.step_count
                    ),
                },
            )

    finally:
        env._simulator.close()

    summary = RewardSanitySummary(
        episodes_completed=episodes_completed,
        total_steps=total_steps,
        toward_vs_away_passes=toward_vs_away_passes,
        toward_vs_away_failures=toward_vs_away_failures,
        phase_transition_events=phase_transition_events,
        approach_velocity_events=approach_velocity_events,
        place_velocity_events=place_velocity_events,
        success_events=success_events,
        success_episodes=success_episodes,
    )

    if strict:
        if summary.toward_vs_away_failures > 0:
            raise RuntimeError(
                f"Directional reach sanity failed for {summary.toward_vs_away_failures} episodes"
            )
        if summary.phase_transition_events == 0:
            raise RuntimeError("No phase transition rewards observed")
        if summary.approach_velocity_events == 0:
            raise RuntimeError("No approach velocity penalties observed")
        if summary.place_velocity_events == 0:
            raise RuntimeError("No place velocity penalties observed")
        if summary.success_events == 0:
            raise RuntimeError("No success rewards observed")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = EpisodeConsoleLogger(name="pick_and_place.reward_sanity")
    summary = run_reward_sanity(
        episodes=args.episodes,
        seed=args.seed,
        strict=args.strict,
        logger=logger,
    )
    logger.log_metrics(
        title="Reward Sanity Result",
        metrics=summary.__dict__,
    )


if __name__ == "__main__":
    main()
