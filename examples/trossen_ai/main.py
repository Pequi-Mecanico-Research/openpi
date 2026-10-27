#!/usr/bin/env python3
"""
Trossen Arm <-> OpenPI Policy Server Bridge (Bimanual Version)

Bridge between a bimanual widowx and the OpenPI policy server.
Handles:
1. Collecting observations from the arm (joint positions, images)
2. Sending observations to the policy server via WebSocket
3. Receiving action predictions
4. Executing actions on the arm

Usage:
    python main.py --mode autonomous --task_prompt "grab and handover red cube"

    Arm + mobile base (publishes cmd_vel/ang_vel on ROS 2, see Dockerfile):
    python main.py --mode autonomous_base --base_topic /cmd_vel --task_prompt "move the cube from the table"

    Test mode (no movement):
    python main.py --mode test --task_prompt "grab and handover red cube"
"""

import argparse
from collections import defaultdict
import logging
import signal
import sys
import threading
import time

import cv2
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots import make_robot_from_config
from lerobot_robot_trossen.config_widowxai_follower import WidowXAIFollowerConfig
from lerobot_robot_trossen.config_bi_widowxai_follower import BiWidowXAIFollowerRobotConfig
import numpy as np
from openpi_client import websocket_client_policy
from scipy.interpolate import PchipInterpolator

from ros2_base_bridge import BaseVelocityBridge

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Dimensões da base móvel no fim do vetor de ação: [cmd_vel (linear), ang_vel (angular)].
BASE_ACTION_DIM = 2


class TrossenOpenPIBridge:
    """Bridge between a Trossen AI Stationary Kit and OpenPI policy server."""

    def __init__(
        self,
        policy_server_host: str = "localhost",
        policy_server_port: int = 8000,
        control_frequency: int = 30,
        test_mode: str = "autonomous",  # "autonomous" (arm), "autonomous_base" (arm + base) or "test"
        max_steps: int = 1000,
        base_topic: str = "/cmd_vel",
    ):
        self.control_frequency = control_frequency
        self.max_steps = max_steps
        self.dt = 1.0 / control_frequency
        self.test_mode = test_mode

        logger.info(f"Connecting to policy server at {policy_server_host}:{policy_server_port}")
        self.policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=policy_server_host, port=policy_server_port
        )

        robot_config = WidowXAIFollowerConfig(
            id="Clear_Table_Task",
            ip_address="192.168.1.2",
            min_time_to_move_multiplier=4.0,
            loop_rate=30,
            cameras={
                "cam_high": RealSenseCameraConfig(
                    serial_number_or_name="944122073060", width=640, height=480, fps=30, use_depth=False
                ),
                "cam_wrist": RealSenseCameraConfig(
                    serial_number_or_name="243322072096", width=640, height=480, fps=30, use_depth=False
                ),
                "cam_low": RealSenseCameraConfig(
                    serial_number_or_name="102122061119", width=640, height=480, fps=30, use_depth=False
                ),
            },
        )
        
        """robot_config = BiWidowXAIFollowerRobotConfig(
            left_arm_ip_address="192.168.1.23",
            right_arm_ip_address="192.168.1.24",
            min_time_to_move_multiplier=4.0,
            id="bimanual_follower",
            cameras={
                "cam_high": RealSenseCameraConfig(
                    serial_number_or_name="235422300510", width=640, height=480, fps=30, use_depth=False
                ),
                "cam_right_wrist": RealSenseCameraConfig(
                    serial_number_or_name="243322072096", width=640, height=480, fps=30, use_depth=False
                ),
            },
        )"""
        
        # A base só é conectada no modo autonomous_base; no modo teste os comandos são apenas impressos
        # e no modo autonomous as dims da base são ignoradas.
        # Iniciada antes do braço para falhar cedo caso o ROS 2 não esteja disponível.
        self.base = BaseVelocityBridge(topic=base_topic) if test_mode == "autonomous_base" else None

        self.robot = make_robot_from_config(robot_config)
        self.robot.connect()

        self.current_action_chunk = None
        self.action_chunk_idx = 0
        self.episode_step = 0
        self.is_running = False
        self.rate_of_inference = 50 # Number of control steps per policy inference (matches README and Pi-0 paper)

        # Web control panel state: mutable prompt, pause/abort signals, live status, and episode history.
        self.task_prompt = ""
        self.pause_requested = threading.Event()
        self.abort_requested = threading.Event()
        self.status = {
            "is_running": False,
            "is_paused": False,
            "phase": "idle",
            "episode_step": 0,
            "max_steps": self.max_steps,
            "prompt": "",
            "hz": 0.0,
            "loop_ms": 0.0,
            "mode": self.test_mode,
        }
        self.history = []
        self._episode_counter = 0

        # Safe pose the arm returns to when an episode completes or is aborted.
        # Matches WidowXAIFollowerConfig's own default `staged_positions`:
        # [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, left_carriage_joint (gripper, meters)].
        self.ready_position: np.ndarray | None = np.array(
            [0, np.pi / 3, np.pi / 6, np.pi / 5, 0, 0, 0]
        )

        self.temporal_ensemble_coefficient = None  #Testar 0.01 # Temporal ensembling weight (can be set to None for no ensembling)

        # FIFO Buffer for actions
        self.action_buffer = defaultdict(list)
        self.action_buffer_size = (
            self.max_steps + 64
        )  # Buffer size to hold actions for the entire episode (margin >= largest chunk returned by the policy)

        self.arm_action_dim = len(self.robot.action_features)  # 7 dims (6 joints + gripper)
        self.action_dim = self.arm_action_dim + BASE_ACTION_DIM  # + cmd_vel e ang_vel da base

    def execute_action(self, action: np.ndarray):
        """Execute action on the arm and, if the action includes it, on the mobile base."""
        full_action = action.copy()
        arm_action = full_action[: self.arm_action_dim]
        base_action = full_action[self.arm_action_dim : self.arm_action_dim + BASE_ACTION_DIM] if self.base is not None else np.zeros(BASE_ACTION_DIM)

        if self.test_mode == "test":
            logger.info(f"TEST MODE: Would execute arm action: {arm_action}")
            if base_action.size == BASE_ACTION_DIM:
                logger.info(f"TEST MODE: Would publish base cmd_vel={base_action[0]:.4f} ang_vel={base_action[1]:.4f}")
            return
        if self.test_mode in ("autonomous", "autonomous_base"):
            joint_features = list(self.robot.action_features.keys())
            action_dict = {k: arm_action[i] for i, k in enumerate(joint_features)}
            self.robot.send_action(action_dict)

            if self.base is not None and base_action.size == BASE_ACTION_DIM:
                self.base.send(base_action[0], base_action[1])
            # Executa de maneira sequêncial as ações até que elas finalizem e faça outra pergunta ao modelo.
        else:
            logger.error(f"Unknown mode: {self.test_mode}. No action executed.")

    def stop_base(self):
        """Send zero velocity to the mobile base (no-op in test mode)."""
        if self.base is not None:
            self.base.stop()

    def move_to_start_position(self, goal_position: np.ndarray, duration: float = 5.0):
        """The first position queried from the policy depends on the training data.
        Assuming the first position is a "stage" position will result in a large jump if the arm is not already there.
        To avoid this, we smoothly move the arm to a first action/position before sending the rest of the actions.
        We use PCHIP interpolation for smooth trajectory generation and give it enough time to reach the position to prevent
        jumps and triggering safety stops (velocity limits)."""

        joint_pos_keys = [k for k in self.robot.get_observation().keys() if k.endswith(".pos")]
        current_pose = np.array([self.robot.get_observation()[k] for k in joint_pos_keys])
        # Example stage_pose for bimanual WidowX arms.
        # Each value corresponds to a joint position (in radians) for the 14 joints:
        # [left_joint_0, left_joint_1, left_joint_2, left_joint_3, left_joint_4, left_joint_5, left_left_carriage_joint,
        #  right_joint_0, right_joint_1, right_joint_2, right_joint_3, right_joint_4, right_joint_5, right_left_carriage_joint]
        # The values below represent a "stage" pose, e.g. arms up and open, ready for task start.
        # stage_pose = np.array([0, np.pi/3, np.pi/6, np.pi/5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # Movimento só do braço: a base fica parada e as dims de velocidade da ação são ignoradas.
        self.stop_base()
        waypoints = np.array([current_pose, goal_position[: self.arm_action_dim]])
        timepoints = np.array([0, duration])  # Use the provided duration
        interpolator_position = PchipInterpolator(timepoints, waypoints, axis=0)

        start_time = time.time()
        end_time = start_time + timepoints[-1]

        while time.time() < end_time:
            loop_start_time = time.time()
            current_time = loop_start_time - start_time
            positions = interpolator_position(current_time)
            self.execute_action(positions)

    def blend_pi05_inference(
        self, previous_keyframe: np.ndarray, next_keyframe: np.ndarray, blend_duration: float = 0.5
    ):
        """Smoothly move the arm between the last keyframe actually executed from the outgoing
        action chunk and the first keyframe of the newly inferred chunk. Avoids the jerk caused by
        snapping directly between two independently inferred chunks. Uses the same PCHIP
        interpolation pattern as move_to_start_position, but over a much shorter duration.
        blend_duration should stay comfortably above the arm driver's own per-call ramp time
        (min_time_to_move_multiplier / loop_rate, ~0.13s with this bridge's config) since each
        execute_action call here also gets its own firmware-level ramp to the target."""
        waypoints = np.array([previous_keyframe, next_keyframe])
        timepoints = np.array([0, blend_duration])
        interpolator_position = PchipInterpolator(timepoints, waypoints, axis=0)

        start_time = time.time()
        end_time = start_time + timepoints[-1]

        while time.time() < end_time:
            current_time = time.time() - start_time
            positions = interpolator_position(current_time)
            self.execute_action(positions)

    def move_to_ready_position(self, duration: float = 3.0):
        """Smoothly return the arm to a safe 'ready' pose after an episode ends (completed or aborted).
        No-op with a warning until self.ready_position is configured for this robot."""
        if self.ready_position is None:
            logger.warning("ready_position is not configured; skipping move to ready pose.")
            return
        logger.info("Moving to ready position...")
        self.move_to_start_position(self.ready_position, duration=duration)

    def run_episode(self, task_prompt: str = "look down", max_steps: int | None = None):
        """Run a single episode of policy execution."""
        if max_steps is not None:
            self.max_steps = max_steps
            self.action_buffer_size = self.max_steps + 64
        logger.info(f"Starting episode with prompt: '{task_prompt}' (max_steps={self.max_steps})")
        self.episode_step = 0
        self.action_chunk_idx = 0
        self.current_action_chunk = None
        self.is_running = True
        self.pause_requested.clear()
        self.abort_requested.clear()
        self.task_prompt = task_prompt
        is_first_step = True

        self._episode_counter += 1
        history_entry = {
            "id": self._episode_counter,
            "prompt": task_prompt,
            "start_time": time.time(),
            "end_time": None,
            "steps_completed": 0,
            "stop_reason": None,
            "outcome": None,
        }
        self.history.append(history_entry)
        stop_reason = "completed"

        try:
            while self.is_running and self.episode_step < self.max_steps:
                if self.abort_requested.is_set():
                    stop_reason = "aborted"
                    break

                if self.pause_requested.is_set():
                    self.stop_base()
                    self.status["is_paused"] = True
                    self.status["phase"] = "paused"
                    time.sleep(self.dt)
                    continue
                self.status["is_paused"] = False
                self.status["phase"] = "running"

                start_loop_time = time.perf_counter()

                # Request new action chunk after consuming the previous one
                if self.current_action_chunk is None or self.action_chunk_idx >= self.rate_of_inference:
                    observation_dict = self.robot.get_observation()

                    # Extract joint positions from observation
                    joint_pos_keys = [k for k in observation_dict.keys() if k.endswith(".pos")]
                    joint_positions = np.array([observation_dict[k] for k in joint_pos_keys])

                    # Transform and resize images from all cameras
                    cameras = list(self.robot._cameras_ft.keys())
                    for cam in cameras:
                        image_hwc = observation_dict[cam]
                        # convert BGR to RGB
                        image_resized = cv2.resize(image_hwc, (224, 224))
                        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
                        image_chw = np.transpose(image_rgb, (2, 0, 1))
                        observation_dict[cam] = image_chw

                    # Create observation for policy to follow the ALOHA format
                    """observation = {
                        "state": joint_positions,
                        "images": {cam: observation_dict[cam] for cam in cameras},
                        "prompt": self.task_prompt,
                    }"""

                    observation = {
                        "observation/state": joint_positions,
                        "observation/image": observation_dict["cam_high"],
                        "observation/wrist_image": observation_dict["cam_wrist"],
                        "observation/low_image": observation_dict["cam_low"],
                        "prompt": self.task_prompt,
                    }

                    logger.info(f"Step {self.episode_step}: Requesting new action chunk")
                    # Momento da inferência feita ao modelo do pi05
                    response = self.policy_client.infer(observation)
                    if self.current_action_chunk is not None:
                        previous_keyframe = self.current_action_chunk[self.action_chunk_idx - 1]
                        next_keyframe = response["actions"][0]
                        self.blend_pi05_inference(previous_keyframe, next_keyframe)
                    self.current_action_chunk = response["actions"]
                    chunk_len = self.current_action_chunk.shape[0]  # actual chunk length returned by the policy

                    for k in range(chunk_len):
                        future_t = self.episode_step + k
                        if future_t < self.action_buffer_size:
                            self.action_buffer[future_t].append(self.current_action_chunk[k])

                    self.action_chunk_idx = 0
                    logger.info(f"Received action chunk: {self.current_action_chunk.shape}")

                # Select action using temporal ensembling if enabled
                ## TODO: Estudar mais sobre temporal_ensemble_coefficient
                if self.temporal_ensemble_coefficient is not None:
                    if len([self.episode_step]) == 0:
                        a_t = np.zeros(self.action_dim)
                    else:
                        candidates = np.array(self.action_buffer[self.episode_step])  # shape: (N, 14)
                        weights = self._get_weights(len(candidates))  # shape: (N,)
                        a_t = np.average(candidates, axis=0, weights=weights)  # shape: (14,)
                else:
                    a_t = self.current_action_chunk[self.action_chunk_idx]
                # Execute the current action
                if is_first_step:
                    logger.info("Moving to start position to avoid large jumps...")
                    self.move_to_start_position(a_t, duration=5.0)
                    is_first_step = False
                else:
                    self.execute_action(a_t)

                self.action_chunk_idx += 1
                self.episode_step += 1

                dt_s = time.perf_counter() - start_loop_time
                busy_wait_time = self.dt - dt_s

                # Busy wait to maintain control frequency
                if busy_wait_time > 0:
                    time.sleep(busy_wait_time)
                loop_s = time.perf_counter() - start_loop_time
                logger.info(f"time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

                self.status = {
                    "is_running": True,
                    "is_paused": False,
                    "phase": "running",
                    "episode_step": self.episode_step,
                    "max_steps": self.max_steps,
                    "prompt": self.task_prompt,
                    "hz": (1 / loop_s) if loop_s > 0 else 0.0,
                    "loop_ms": loop_s * 1e3,
                    "mode": self.test_mode,
                }

            if stop_reason in ("completed", "aborted"):
                self.status["phase"] = "returning_to_ready"
                self.move_to_ready_position()
        except Exception:
            logger.exception("Episode aborted due to an error")
            stop_reason = "error"
        finally:
            self.stop_base()
            self.is_running = False
            history_entry["end_time"] = time.time()
            history_entry["steps_completed"] = self.episode_step
            history_entry["stop_reason"] = stop_reason
            self.status["is_running"] = False
            self.status["is_paused"] = False
            self.status["phase"] = "idle"
            logger.info(f"Episode completed after {self.episode_step} steps (reason: {stop_reason})")

    def _get_weights(self, num_preds: int) -> np.ndarray:
        weights = np.exp(-self.temporal_ensemble_coefficient * np.arange(num_preds))
        return weights / weights.sum()

    def autonomous_mode(self, task_prompt: str = "look down"):
        """Run in autonomous mode where the arm executes policy predictions."""
        logger.info("Starting autonomous mode")
        self.run_episode(task_prompt=task_prompt)

    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")
        if self.base is not None:
            self.base.close()
        self.robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trossen AI Stationary Kit <-> OpenPI Policy Server Bridge")
    parser.add_argument("--policy_host", default="localhost", help="Policy server host")
    parser.add_argument("--policy_port", type=int, default=8000, help="Policy server port")
    parser.add_argument("--control_freq", type=int, default=30, help="Control frequency in Hz")
    parser.add_argument(
        "--mode",
        choices=["autonomous", "autonomous_base", "test"],
        default="autonomous",
        help="Operation mode: autonomous (arm only), autonomous_base (arm + mobile base) or test (no movement)",
    )
    parser.add_argument("--task_prompt", default="move the arm to the left", help="Task description for the policy")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--base_topic", default="/cmd_vel", help="ROS 2 geometry_msgs/Twist topic of the mobile base")
    args = parser.parse_args()

    bridge = TrossenOpenPIBridge(
        policy_server_host=args.policy_host,
        policy_server_port=args.policy_port,
        control_frequency=args.control_freq,
        test_mode=args.mode,
        max_steps=args.max_steps,
        base_topic=args.base_topic,
    )
    
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0)) if args.mode == "autonomous_base" else None

    try:
        bridge.autonomous_mode(task_prompt=args.task_prompt)
    except KeyboardInterrupt:
        logger.info("Interrupted by user, returning arm to start position...")
    finally:
        bridge.cleanup()
