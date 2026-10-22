"""Isaac Lab 3 camera/state probe. Holds the robot; never executes VLA actions.

Run from the IsaacLab project with its Python environment. The optional --query
sends the saved observation to an existing pi05_libero OpenPI server, but saves
the returned actions instead of applying them. Scene transfer is not validated.
"""

import argparse
import json
from pathlib import Path


def main():
    import gymnasium as gym
    import numpy as np
    import torch
    from PIL import Image

    import isaaclab.sim as sim_utils
    import isaaclab_tasks  # noqa: F401 -- register official tasks
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils.math import axis_angle_from_quat, subtract_frame_transforms
    from isaaclab_rl.entrypoints.simple_agents import create_zero_action_policy
    from isaaclab_tasks.contrib.lift.config.franka.ik_rel_env_cfg import (
        FrankaCubeLiftEnvCfg,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("logs/openpi_camera_check"))
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--query", action="store_true")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--prompt", default="pick up the cube")
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.warmup_steps < 1:
        parser.error("--warmup-steps must be positive")
    # Some Lab 3 revisions still expose this launcher flag; newer revisions
    # discover cameras from the scene configuration.
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = True

    cfg = FrankaCubeLiftEnvCfg()
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 60.0
    # Disambiguate this diagnostic from previous hand-edited lift rewards.
    # No learning or success evaluation is performed here.
    cfg.scene.external_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ExternalCamera",
        width=256,
        height=256,
        data_types=["rgb"],
        update_period=0.0,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=36.0,
            clipping_range=(0.01, 20.0),
        ),
    )
    hand_path = cfg.scene.ee_frame.target_frames[0].prim_path
    cfg.scene.wrist_camera = CameraCfg(
        prim_path=hand_path + "/OpenPIWristCamera",
        width=256,
        height=256,
        data_types=["rgb"],
        update_period=0.0,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=16.0,
            horizontal_aperture=36.0,
            clipping_range=(0.01, 10.0),
        ),
        # Identity rotation in the installed CameraCfg's quaternion convention.
        # ROS optical +Z looks along the hand's +Z; verify the image before use.
        offset=CameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.03), convention="ros"),
    )

    def tensor(value):
        return value if isinstance(value, torch.Tensor) else value.torch

    with launch_simulation(cfg, args):
        env = gym.make("IsaacContrib-Lift-Cube-Franka-IK-Rel", cfg=cfg)
        try:
            base = env.unwrapped
            env.reset()
            hold = create_zero_action_policy(env)
            origin = tensor(base.scene.env_origins)[0:1]
            camera = base.scene["external_camera"]
            camera.set_world_poses_from_view(
                eyes=origin + torch.tensor([[1.2, -1.2, 1.1]], device=origin.device),
                targets=origin + torch.tensor([[0.5, 0.0, 0.15]], device=origin.device),
            )
            with torch.inference_mode():
                for _ in range(args.warmup_steps):
                    env.step(hold())

                output = args.output.expanduser().resolve()
                output.mkdir(parents=True, exist_ok=True)
                images = {}
                image_report = {}
                for name in ("external_camera", "wrist_camera"):
                    rgb = tensor(base.scene[name].data.output["rgb"])[0, ..., :3]
                    rgb = rgb.detach().cpu().numpy()
                    if rgb.dtype != np.uint8:
                        raise RuntimeError(f"{name}: expected uint8 RGB, got {rgb.dtype}")
                    if rgb.ndim != 3 or rgb.shape[-1] != 3:
                        raise RuntimeError(f"{name}: invalid RGB shape {rgb.shape}")
                    images[name] = rgb
                    Image.fromarray(rgb).save(output / f"{name}.png")
                    image_report[name] = {
                        "shape": list(rgb.shape),
                        "dtype": str(rgb.dtype),
                        "min": int(rgb.min()),
                        "max": int(rgb.max()),
                        "std": float(rgb.std()),
                    }

                robot = base.scene["robot"]
                ee = base.scene["ee_frame"]
                ee_pos = tensor(ee.data.target_pos_w)[:, 0, :]
                ee_quat = tensor(ee.data.target_quat_w)[:, 0, :]
                position_b, rotation_b = subtract_frame_transforms(
                    tensor(robot.data.root_pos_w),
                    tensor(robot.data.root_quat_w),
                    ee_pos,
                    ee_quat,
                )
                finger_ids, finger_names = robot.find_joints("panda_finger.*")
                if len(finger_ids) != 2:
                    raise RuntimeError(f"Expected two fingers; found {finger_names}")
                fingers = tensor(robot.data.joint_pos)[:, finger_ids]
                state = torch.cat(
                    (position_b, axis_angle_from_quat(rotation_b), fingers), dim=-1
                )[0].detach().cpu().numpy().astype(np.float32)
                if state.shape != (8,) or not np.isfinite(state).all():
                    raise RuntimeError(f"Invalid state: {state}")

            report = {
                "task": "IsaacContrib-Lift-Cube-Franka-IK-Rel",
                "state": state.tolist(),
                "state_layout": "EE xyz in robot root frame; rotation vector; two finger joint positions",
                "finger_names": list(finger_names),
                "images": image_report,
                "validated": "capture only; LIBERO reference frames, cameras and action mapping are not calibrated",
            }
            (output / "observation.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            print(json.dumps(report, indent=2), flush=True)
            print(f"Saved camera check to: {output}", flush=True)

            if args.query:
                from openpi_client import image_tools
                from openpi_client.websocket_client_policy import WebsocketClientPolicy

                client = WebsocketClientPolicy(host=args.server_host, port=args.server_port)
                obs = {
                    "observation/image": image_tools.resize_with_pad(images["external_camera"], 224, 224),
                    "observation/wrist_image": image_tools.resize_with_pad(images["wrist_camera"], 224, 224),
                    "observation/state": state,
                    "prompt": args.prompt,
                }
                actions = np.asarray(client.infer(obs)["actions"])
                if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
                    raise RuntimeError(f"Unexpected OpenPI action chunk: {actions.shape}")
                np.save(output / "predicted_actions.npy", actions)
                print(f"OpenPI returned {actions.shape}; saved, NOT executed.", flush=True)
                print(f"First predicted action: {actions[0]}", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
