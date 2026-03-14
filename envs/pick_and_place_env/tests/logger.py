# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Console logger utilities for pick_and_place validation scripts."""

from __future__ import annotations

import logging
from pprint import pformat
from typing import Any, Mapping


class EpisodeConsoleLogger:
    """Structured console logger using stdlib logging + text tables."""

    def __init__(self, *, name: str = "pick_and_place.validation") -> None:
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            )
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)
            self._logger.propagate = False

    def _render_table(self, title: str, rows: list[tuple[str, str]]) -> str:
        key_width = max((len(key) for key, _ in rows), default=3)
        val_width = max((len(value) for _, value in rows), default=5)
        border = "+" + "-" * (key_width + 2) + "+" + "-" * (val_width + 2) + "+"
        lines = [title, border]
        for key, value in rows:
            lines.append(f"| {key.ljust(key_width)} | {value.ljust(val_width)} |")
        lines.append(border)
        return "\n".join(lines)

    def log_episode_summary(
        self,
        *,
        episode_index: int,
        step_count: int,
        total_reward: float,
        success: bool,
        reward_breakdown: Mapping[str, float],
        robot_state: Mapping[str, Any],
    ) -> None:
        summary_rows = [
            ("episode", str(episode_index)),
            ("steps", str(step_count)),
            ("total_reward", f"{total_reward:.6f}"),
            ("success", str(success)),
        ]
        reward_rows = [
            (str(key), f"{float(value):.6f}") for key, value in reward_breakdown.items()
        ]
        state_rows = [
            (str(key), pformat(value, compact=True))
            for key, value in robot_state.items()
        ]

        self._logger.info("\n%s", self._render_table("Episode Summary", summary_rows))
        self._logger.info("\n%s", self._render_table("Reward Breakdown", reward_rows))
        self._logger.info("\n%s", self._render_table("Robot State", state_rows))

    def log_run_summary(
        self,
        *,
        episodes_completed: int,
        total_steps: int,
        success_episodes: int,
    ) -> None:
        rows = [
            ("episodes_completed", str(episodes_completed)),
            ("total_steps", str(total_steps)),
            ("success_episodes", str(success_episodes)),
        ]
        self._logger.info("\n%s", self._render_table("Run Summary", rows))

    def log_metrics(self, *, title: str, metrics: Mapping[str, Any]) -> None:
        rows = [
            (str(key), pformat(value, compact=True)) for key, value in metrics.items()
        ]
        self._logger.info("\n%s", self._render_table(title, rows))
