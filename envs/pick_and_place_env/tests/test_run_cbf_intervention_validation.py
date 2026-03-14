# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Validate CBF intervention behavior under aggressive random actions."""

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
class CBFValidationSummary:
    episodes_completed: int
    total_steps: int
    interventions: int
    intervention_rate: float
    min_scale: float
    avg_scale: float
    min_residual: float
    regulation_violations: int
    success_episodes: int


def _sample_high_action(rng: random.Random, min_abs_delta: float) -> PickAndPlaceAction:
    def axis_value() -> float:
        magnitude = rng.uniform(min_abs_delta, 1.0)
        return magnitude if rng.random() < 0.5 else -magnitude

    return PickAndPlaceAction(
        dx=axis_value(),
        dy=axis_value(),
        dz=axis_value(),
        gripper=rng.uniform(-1.0, 1.0),
    )


def _is_scaled_down(proposed: PickAndPlaceAction, executed: PickAndPlaceAction) -> bool:
    tolerance = 1e-9
    return (
        abs(executed.dx) <= abs(proposed.dx) + tolerance
        and abs(executed.dy) <= abs(proposed.dy) + tolerance
        and abs(executed.dz) <= abs(proposed.dz) + tolerance
    )


def validate_cbf(
    *,
    episodes: int,
    seed: int,
    min_abs_delta: float,
    required_intervention_rate: float,
    logger: EpisodeConsoleLogger,
) -> CBFValidationSummary:
    rng = random.Random(seed)
    env = PickAndPlaceEnvironment(
        max_steps=cfg.TASK.max_steps,
        enable_cbf_filter=True,
    )

    steps = 0
    interventions = 0
    scale_sum = 0.0
    min_scale = 1.0
    min_residual = float("inf")
    regulation_violations = 0
    successes = 0

    try:
        for episode in range(episodes):
            observation = env.reset(seed=seed + episode)
            done = bool(observation.done)
            episode_steps = 0
            episode_interventions = 0

            while not done and env.state.step_count < cfg.TASK.max_steps:
                action = _sample_high_action(rng, min_abs_delta=min_abs_delta)
                observation = env.step(action)
                done = bool(observation.done)

                proposed = env.state.proposed_action
                executed = env.state.last_action
                if proposed is None or executed is None:
                    raise RuntimeError(
                        "Expected proposed/executed actions to be present"
                    )

                steps += 1
                episode_steps += 1
                scale_sum += env.state.cbf_scale
                min_scale = min(min_scale, env.state.cbf_scale)
                min_residual = min(min_residual, env.state.cbf_residual)

                if env.state.cbf_intervened:
                    interventions += 1
                    episode_interventions += 1
                    if env.state.cbf_scale >= 1.0:
                        regulation_violations += 1

                if not _is_scaled_down(proposed, executed):
                    regulation_violations += 1

            if env.state.success:
                successes += 1

            logger.log_episode_summary(
                episode_index=episode,
                step_count=episode_steps,
                total_reward=env.state.reward_breakdown.total,
                success=env.state.success,
                reward_breakdown=env.state.reward_breakdown.model_dump(),
                robot_state={
                    "phase": env.state.phase,
                    "ee_pos": env.state.ee_pos,
                    "cube_pos": env.state.cube_pos,
                    "goal_pos": env.state.goal_pos,
                    "goal_reached_once": env.state.goal_reached_once,
                    "cbf_intervened_last_step": env.state.cbf_intervened,
                    "cbf_scale_last_step": f"{env.state.cbf_scale:.6f}",
                    "cbf_residual_last_step": f"{env.state.cbf_residual:.6f}",
                },
            )

    finally:
        env._simulator.close()

    intervention_rate = interventions / max(steps, 1)
    if intervention_rate < required_intervention_rate:
        raise RuntimeError(
            "CBF intervention rate below expected threshold: "
            f"{intervention_rate:.4f} < {required_intervention_rate:.4f}"
        )

    if regulation_violations > 0:
        raise RuntimeError(
            f"Detected {regulation_violations} regulation invariant violations"
        )

    return CBFValidationSummary(
        episodes_completed=episodes,
        total_steps=steps,
        interventions=interventions,
        intervention_rate=intervention_rate,
        min_scale=min_scale,
        avg_scale=(scale_sum / max(steps, 1)),
        min_residual=min_residual,
        regulation_violations=regulation_violations,
        success_episodes=successes,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-abs-delta",
        type=float,
        default=0.8,
        help="Minimum absolute value for sampled dx/dy/dz components",
    )
    parser.add_argument(
        "--required-intervention-rate",
        type=float,
        default=0.05,
        help="Fail if intervention rate falls below this threshold",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = EpisodeConsoleLogger()
    summary = validate_cbf(
        episodes=args.episodes,
        seed=args.seed,
        min_abs_delta=args.min_abs_delta,
        required_intervention_rate=args.required_intervention_rate,
        logger=logger,
    )
    logger.log_metrics(
        title="CBF Validation Result",
        metrics={
            "status": "CBF_VALIDATION_OK",
            "episodes_completed": summary.episodes_completed,
            "total_steps": summary.total_steps,
            "interventions": summary.interventions,
            "intervention_rate": f"{summary.intervention_rate:.6f}",
            "min_scale": f"{summary.min_scale:.6f}",
            "avg_scale": f"{summary.avg_scale:.6f}",
            "min_residual": f"{summary.min_residual:.6f}",
            "regulation_violations": summary.regulation_violations,
            "success_episodes": summary.success_episodes,
        },
    )


if __name__ == "__main__":
    main()
