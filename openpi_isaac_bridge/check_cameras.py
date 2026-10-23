"""Isaac Lab 3 observation and deterministic motion probe; never executes VLA actions.

Run from the IsaacLab project with its Python environment. The optional --query
sends the saved observation to an existing pi05_libero OpenPI server, but saves
the returned actions instead of applying them. Scene transfer is not validated.
"""

import argparse
import json
import math
import time
from pathlib import Path


def translation_probe_action(axis, step_m, ik_scale):
    """Create one bounded relative-IK test action, with the gripper OPEN.

    This is NOT a converter for normalized LIBERO model output.
    """
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be x, y or z")
    if not math.isfinite(step_m) or not 0 < step_m <= 0.002:
        raise ValueError("test step must be finite and in (0, 0.002] metres")
    if not math.isfinite(ik_scale) or ik_scale <= 0:
        raise ValueError("IK scale must be finite and positive")
    raw = step_m / ik_scale
    if abs(raw) > 1.0:
        raise ValueError("test command would exceed normalized magnitude 1")
    action = [0.0] * 7
    action[("x", "y", "z").index(axis)] = raw
    action[6] = 1.0  # Isaac BinaryJointPositionAction: positive = OPEN
    return action


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
    parser.add_argument("--motion-test", choices=("none", "x", "y", "z", "gripper"), default="none")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--prompt", default="pick up the cube")
    parser.add_argument("--light-intensity", type=float, default=1000.0)
    parser.add_argument("--external-eye", type=float, nargs=3, default=(0.95, -0.75, 0.65))
    parser.add_argument("--external-target", type=float, nargs=3, default=(0.5, 0.0, 0.10))
    parser.add_argument("--wrist-offset", type=float, nargs=3, default=(0.10, 0.0, 0.02))
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.query and args.motion_test != "none":
        parser.error("run --query and --motion-test separately; VLA actions are never executed")
    if args.warmup_steps < 1:
        parser.error("--warmup-steps must be positive")
    if not math.isfinite(args.light_intensity) or args.light_intensity <= 0:
        parser.error("--light-intensity must be finite and positive")
    for name in ("external_eye", "external_target", "wrist_offset"):
        if not all(math.isfinite(value) for value in getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must contain finite coordinates")
    if tuple(args.external_eye) == tuple(args.external_target):
        parser.error("--external-eye and --external-target must differ")
    # Some Lab 3 revisions still expose this launcher flag; newer revisions
    # discover cameras from the scene configuration.
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = True

    cfg = FrankaCubeLiftEnvCfg()
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 60.0
    # The command target axes/spheres are real rendered geometry, so disable
    # their creation before constructing the scene, not just in the web viewer.
    cfg.commands.object_pose.debug_vis = False
    cfg.scene.ee_frame.debug_vis = False
    cfg.scene.light.spawn.intensity = args.light_intensity
    # Disambiguate this diagnostic from previous hand-edited lift rewards.
    # No learning or success evaluation is performed here.
    cfg.scene.external_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ExternalCamera",
        width=256,
        height=256,
        data_types=["rgb"],
        update_period=0.0,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=32.0,
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
        # Keep the lens farther outside the palm. Set its initial orientation
        # explicitly below instead of assuming the hand's optical axis/roll.
        update_latest_camera_pose=True,
        offset=CameraCfg.OffsetCfg(pos=tuple(args.wrist_offset), convention="ros"),
    )

    def tensor(value):
        return value if isinstance(value, torch.Tensor) else value.torch

    with launch_simulation(cfg, args):
        env = gym.make("IsaacContrib-Lift-Cube-Franka-IK-Rel", cfg=cfg)
        try:
            base = env.unwrapped
            env.reset()
            framework_hold = create_zero_action_policy(env)
            action_manager = base.action_manager
            if list(action_manager.active_terms) != ["arm_action", "gripper_action"]:
                raise RuntimeError("Unexpected action order; cannot assume a 6-D IK term plus a gripper")
            arm_term = action_manager.get_term("arm_action")
            grip_term = action_manager.get_term("gripper_action")
            if arm_term.action_dim != 6 or grip_term.action_dim != 1:
                raise RuntimeError("Expected a 6-D arm action and a 1-D gripper action")
            if not arm_term.cfg.controller.use_relative_mode:
                raise RuntimeError("This diagnostic requires relative IK")

            def hold():
                action = framework_hold().clone()
                if tuple(action.shape) != (1, 7):
                    raise RuntimeError(f"Unexpected hold action shape: {tuple(action.shape)}")
                # Literal zero CLOSES this binary gripper. Keep it explicitly
                # open during image capture and translation calibration.
                action[:, 6] = 1.0
                return action
            origin = tensor(base.scene.env_origins)[0:1]
            camera = base.scene["external_camera"]
            camera.set_world_poses_from_view(
                eyes=origin + torch.tensor([args.external_eye], device=origin.device),
                targets=origin + torch.tensor([args.external_target], device=origin.device),
            )
            with torch.inference_mode():
                for _ in range(args.warmup_steps):
                    env.step(hold())

                # Calibration-only: aim at the settled cube ONCE using its
                # ground-truth position. This is not a target-tracking camera
                # and must not be copied as an online VLA observation policy.
                # Parent-relative mounting remains on the hand after setting.
                wrist = base.scene["wrist_camera"]
                wrist_eye = tensor(wrist.data.pos_w).clone()
                wrist_target = tensor(base.scene["object"].data.root_pos_w).clone()
                wrist_target[:, 2] += 0.02
                wrist.set_world_poses_from_view(eyes=wrist_eye, targets=wrist_target)
                # Force fresh renders after changing the camera pose; reuse the
                # exact sensor/step path already verified by the first capture.
                for _ in range(5):
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
                "gripper_during_capture": "open (Isaac action +1)",
                "images": image_report,
                "camera_setup": {
                    "debug_markers": False,
                    "light_intensity": args.light_intensity,
                    "external_eye_env": args.external_eye,
                    "external_target_env": args.external_target,
                    "wrist_offset_hand": args.wrist_offset,
                    "wrist_initial_aim": "one-time calibration from ground-truth cube position; not for online tracking",
                    "wrist_eye_world_at_calibration": wrist_eye[0].cpu().tolist(),
                    "wrist_target_world_at_calibration": wrist_target[0].cpu().tolist(),
                },
                "validated": "capture only; LIBERO reference frames, cameras and action mapping are not calibrated",
            }
            (output / "observation.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            print(json.dumps(report, indent=2), flush=True)
            print(f"Saved camera check to: {output}", flush=True)

            if args.motion_test != "none":
                def read_pose():
                    pose_pos, _ = subtract_frame_transforms(
                        tensor(robot.data.root_pos_w),
                        tensor(robot.data.root_quat_w),
                        tensor(ee.data.target_pos_w)[:, 0, :],
                        tensor(ee.data.target_quat_w)[:, 0, :],
                    )
                    return pose_pos[0].detach().clone()

                def checked_step(action, initial_pos):
                    result = env.step(action)
                    if bool(tensor(result[2]).any()) or bool(tensor(result[3]).any()):
                        raise RuntimeError("Environment reset during calibration; discard this measurement")
                    delta = read_pose() - initial_pos
                    if not bool(torch.isfinite(delta).all()) or float(delta.norm()) > 0.05:
                        raise RuntimeError("Calibration stopped: invalid state or EE displacement exceeded 5 cm")
                    time.sleep(float(base.step_dt))

                with torch.inference_mode():
                    before = read_pose()
                    measured = {"mode": args.motion_test, "ee_before_root": before.cpu().tolist()}
                    if args.motion_test == "gripper":
                        phases = []
                        for label, sign in (("open", 1.0), ("close", -1.0), ("reopen", 1.0)):
                            for _ in range(30):
                                action = hold()
                                action[:, 6] = sign
                                checked_step(action, before)
                            phases.append({
                                "phase": label,
                                "isaac_action": sign,
                                "finger_positions_m": tensor(robot.data.joint_pos)[0, finger_ids].cpu().tolist(),
                            })
                        measured["phases"] = phases
                    else:
                        # Scalar scale verified from the actual loaded term.
                        # Each target is offset 1 mm from the current IK pose;
                        # ten steps do NOT guarantee an achieved 10 mm motion.
                        action_values = translation_probe_action(
                            args.motion_test, 0.001, float(arm_term.cfg.scale)
                        )
                        action = torch.tensor([action_values], dtype=torch.float32, device=base.device)
                        for _ in range(10):
                            checked_step(action, before)
                        for _ in range(5):
                            checked_step(hold(), before)
                        measured["requested_step_m"] = 0.001
                        measured["command_steps"] = 10
                        measured["ik_scale"] = float(arm_term.cfg.scale)
                        measured["raw_isaac_action"] = action_values
                    after = read_pose()
                    measured["ee_after_root"] = after.cpu().tolist()
                    measured["delta_root_m"] = (after - before).cpu().tolist()
                    measured["note"] = "deterministic diagnostic only; no OpenPI actions executed; inspect direction and achieved displacement"
                report_path = output / f"motion_{args.motion_test}.json"
                report_path.write_text(json.dumps(measured, indent=2), encoding="utf-8")
                print("[Motion test] " + json.dumps(measured, indent=2), flush=True)
                print(f"Saved calibration to: {report_path}", flush=True)

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
