# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    import mujoco

    from . import config as cfg
    from .models import PickAndPlaceAction, Proprioception
except ImportError:
    try:
        import config as cfg  # type: ignore
        import gymnasium as gym  # type: ignore
        import gymnasium_robotics  # type: ignore # noqa: F401
        import mujoco  # type: ignore
        from models import PickAndPlaceAction, Proprioception  # type: ignore
    except ImportError:
        import gymnasium as gym  # type: ignore
        import gymnasium_robotics  # type: ignore # noqa: F401
        import mujoco  # type: ignore

        from pick_and_place_env import config as cfg  # type: ignore
        from pick_and_place_env.models import (  # type: ignore
            PickAndPlaceAction,
            Proprioception,
        )


@dataclass(frozen=True)
class SimulatorSnapshot:
    ee_pos: list[float]
    ee_quat: list[float]
    cube_pos: list[float]
    goal_pos: list[float]
    cube_height: float
    gripper_contact: bool
    proprioception: Proprioception
    raw_reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class FetchPickAndPlaceSimulator:
    """
    Wrapper for the headless pick and place environment in MuJoCo
    """

    def __init__(self, max_episode_steps: int = cfg.TASK.max_steps):
        self._env = gym.make(
            cfg.SIMULATOR.env_id,
            max_episode_steps=max_episode_steps,
        )

    def reset(self, seed: int | None = None) -> SimulatorSnapshot:
        observation, info = self._env.reset(seed=seed)
        return self._snapshot(
            observation=observation,
            reward=0.0,
            terminated=False,
            truncated=False,
            info=info,
        )

    def step(self, action: PickAndPlaceAction) -> SimulatorSnapshot:
        observation, reward, terminated, truncated, info = self._env.step(
            self._to_env_action(action)
        )
        return self._snapshot(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def close(self) -> None:
        self._env.close()

    def _to_env_action(self, action: PickAndPlaceAction) -> np.ndarray:
        gripper_value = 1.0 if action.gripper == cfg.GripperCommand.OPEN else -1.0
        return np.array(
            [
                action.dx,
                action.dy,
                action.dz,
                gripper_value,
            ],
            dtype=np.float32,
        )

    def _snapshot(
        self,
        observation: dict[str, np.ndarray],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> SimulatorSnapshot:

        obs = observation["observation"]
        ee_pos = obs[0:3].tolist()
        cube_pos = obs[3:6].tolist()
        goal_pos = observation["desired_goal"].tolist()
        gripper_width = float(obs[9] + obs[10])

        ee_quat = self._get_gripper_quaternion()
        joint_angles = self._get_arm_joint_angles()
        cube_height = max(0.0, cube_pos[2] - cfg.GEOMETRY.table_height)

        gripper_contact = self._estimate_gripper_contact(
            object_rel_pos=obs[6:9],
            gripper_width=gripper_width,
            cube_height=cube_height,
        )

        proprioception = Proprioception(
            ee_pos=ee_pos,
            ee_quat=ee_quat,
            gripper_width=gripper_width,
            joint_angles=joint_angles,
        )

        return SimulatorSnapshot(
            ee_pos=ee_pos,
            ee_quat=ee_quat,
            cube_pos=cube_pos,
            goal_pos=goal_pos,
            cube_height=cube_height,
            gripper_contact=gripper_contact,
            proprioception=proprioception,
            raw_reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info=dict(info),
        )

    def _get_gripper_quaternion(self) -> list[float]:
        env = self._env.unwrapped
        site_id = env._model_names.site_name2id[cfg.SIMULATOR.grip_site_name]
        matrix = np.asarray(env.data.site_xmat[site_id], dtype=np.float64).reshape(9)
        quaternion = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, matrix)
        return quaternion.tolist()

    def _get_arm_joint_angles(self) -> list[float]:
        env = self._env.unwrapped
        return [
            float(
                env.data.qpos[
                    env.model.jnt_qposadr[env._model_names.joint_name2id[name]]
                ]
            )
            for name in cfg.SIMULATOR.arm_joint_names
        ]

    def _estimate_gripper_contact(
        self,
        object_rel_pos: np.ndarray,
        gripper_width: float,
        cube_height: float,
    ) -> bool:
        return bool(
            np.linalg.norm(object_rel_pos) <= cfg.GEOMETRY.grasp_distance
            and (
                gripper_width <= cfg.TASK.gripper_open_width * 0.5
                or cube_height >= cfg.GEOMETRY.lift_threshold
            )
        )
