# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import base64
import io
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

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
class FetchConfig:
    env_id: str = "FetchPickAndPlace-v4"
    position_action_scale_meters: float = 0.05
    grip_site_name: str = "robot0:grip"
    overhead_camera_name: str = "external_camera_0"
    wrist_camera_name: str = "gripper_camera_rgb"
    render_width: int = 224
    render_height: int = 224
    object_geom_name: str = "object0"
    finger_geom_names: tuple[str, str] = (
        "robot0:r_gripper_finger_link",
        "robot0:l_gripper_finger_link",
    )
    arm_joint_names: tuple[str, ...] = (
        "robot0:shoulder_pan_joint",
        "robot0:shoulder_lift_joint",
        "robot0:upperarm_roll_joint",
        "robot0:elbow_flex_joint",
        "robot0:forearm_roll_joint",
        "robot0:wrist_flex_joint",
        "robot0:wrist_roll_joint",
    )


FETCH = FetchConfig()


@dataclass(frozen=True)
class SimulatorSnapshot:
    ee_pos: list[float]
    ee_quat: list[float]
    ee_linear_velocity: list[float]
    cube_pos: list[float]
    goal_pos: list[float]
    cube_height: float
    gripper_contact: bool
    proprioception: Proprioception
    raw_reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@dataclass(frozen=True)
class SimulatorState:
    mj_state: np.ndarray
    goal: np.ndarray
    elapsed_steps: int | None


class FetchPickAndPlaceSimulator:
    """
    Wrapper for the headless pick and place environment in MuJoCo
    """

    def __init__(self, max_episode_steps: int = cfg.TASK.max_steps):
        self._env = gym.make(
            FETCH.env_id,
            max_episode_steps=max_episode_steps,
        )
        self._renderers: dict[str, MujocoRenderer] = {}
        self._state_spec = (
            mujoco.mjtState.mjSTATE_FULLPHYSICS
            | mujoco.mjtState.mjSTATE_MOCAP_POS
            | mujoco.mjtState.mjSTATE_MOCAP_QUAT
            | mujoco.mjtState.mjSTATE_CTRL
        )
        self._validate_render_backend()

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

    def peek_step(self, action: PickAndPlaceAction) -> SimulatorSnapshot:
        saved_state = self._capture_state()
        try:
            return self.step(action)
        finally:
            self._restore_state(saved_state)

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
        self._env.close()

    @property
    def distance_threshold(self) -> float:
        return float(self._env.unwrapped.distance_threshold)

    @property
    def table_height(self) -> float:
        return float(self._env.unwrapped.height_offset)

    @property
    def position_action_scale_meters(self) -> float:
        return FETCH.position_action_scale_meters

    @property
    def step_dt(self) -> float:
        env = self._env.unwrapped
        return float(env.n_substeps) * float(env.model.opt.timestep)

    def compute_joint_limit_margin(self, joint_angles: list[float]) -> float:
        env = self._env.unwrapped
        min_margin = float("inf")

        for joint_name, joint_angle in zip(FETCH.arm_joint_names, joint_angles):
            joint_id = env._model_names.joint_name2id[joint_name]
            if not bool(env.model.jnt_limited[joint_id]):
                continue

            lower_limit, upper_limit = env.model.jnt_range[joint_id]
            joint_margin = min(joint_angle - lower_limit, upper_limit - joint_angle)
            min_margin = min(min_margin, float(joint_margin))

        return min_margin

    def render_observation_images(
        self,
    ) -> tuple[str | None, str | None]:
        return (
            self._render_camera(FETCH.overhead_camera_name),
            self._render_camera(FETCH.wrist_camera_name),
        )

    def _to_env_action(self, action: PickAndPlaceAction) -> np.ndarray:
        return np.array(
            [
                action.dx,
                action.dy,
                action.dz,
                action.gripper,
            ],
            dtype=np.float32,
        )

    def _capture_state(self) -> SimulatorState:
        env = self._env.unwrapped
        mj_state = np.empty(
            mujoco.mj_stateSize(env.model, self._state_spec),
            dtype=np.float64,
        )
        mujoco.mj_getState(env.model, env.data, mj_state, self._state_spec)
        return SimulatorState(
            mj_state=mj_state,
            goal=np.array(env.goal, copy=True),
            elapsed_steps=getattr(self._env, "_elapsed_steps", None),
        )

    def _restore_state(self, saved_state: SimulatorState) -> None:
        env = self._env.unwrapped
        mujoco.mj_setState(env.model, env.data, saved_state.mj_state, self._state_spec)
        mujoco.mj_forward(env.model, env.data)
        env.goal = np.array(saved_state.goal, copy=True)
        if hasattr(self._env, "_elapsed_steps"):
            self._env._elapsed_steps = saved_state.elapsed_steps

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
        ee_linear_velocity = obs[20:23].tolist()
        cube_pos = obs[3:6].tolist()
        goal_pos = observation["desired_goal"].tolist()
        gripper_width = float(obs[9] + obs[10])

        ee_quat = self._get_gripper_quaternion()
        joint_angles = self._get_arm_joint_angles()
        cube_height = max(0.0, cube_pos[2] - self.table_height)

        gripper_contact = self._has_gripper_contact()

        proprioception = Proprioception(
            ee_pos=ee_pos,
            ee_quat=ee_quat,
            ee_linear_velocity=ee_linear_velocity,
            gripper_width=gripper_width,
            joint_angles=joint_angles,
        )

        return SimulatorSnapshot(
            ee_pos=ee_pos,
            ee_quat=ee_quat,
            ee_linear_velocity=ee_linear_velocity,
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

    def _render_camera(self, camera_name: str) -> str:
        renderer = self._renderers.get(camera_name)
        if renderer is None:
            env = self._env.unwrapped
            renderer = MujocoRenderer(
                env.model,
                env.data,
                width=FETCH.render_width,
                height=FETCH.render_height,
                camera_name=camera_name,
            )
            self._renderers[camera_name] = renderer

        frame = renderer.render("rgb_array")
        image = Image.fromarray(frame)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _validate_render_backend(self) -> None:
        try:
            self._render_camera(FETCH.overhead_camera_name)
        except Exception as error:
            backend = os.getenv("MUJOCO_GL", "<unset>")
            raise RuntimeError(
                "MuJoCo offscreen rendering backend is not available. "
                f"Current MUJOCO_GL={backend}. "
                "Set MUJOCO_GL=egl (preferred) or MUJOCO_GL=osmesa before starting "
                "the server."
            ) from error

    def _get_gripper_quaternion(self) -> list[float]:
        env = self._env.unwrapped
        site_id = env._model_names.site_name2id[FETCH.grip_site_name]
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
            for name in FETCH.arm_joint_names
        ]

    def _has_gripper_contact(self) -> bool:
        env = self._env.unwrapped
        geom_names = env._model_names.geom_id2name

        for index in range(env.data.ncon):
            contact = env.data.contact[index]
            geom_1 = geom_names.get(contact.geom1)
            geom_2 = geom_names.get(contact.geom2)
            contact_names = {geom_1, geom_2}

            if FETCH.object_geom_name not in contact_names:
                continue
            if any(name in contact_names for name in FETCH.finger_geom_names):
                return True

        return False
