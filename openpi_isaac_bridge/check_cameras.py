"""Isaac Lab 3 observation and motion diagnostics.

Run from the IsaacLab project with its Python environment. The optional --query
sends the saved observation to an existing pi05_libero OpenPI server, but saves
the returned actions instead of applying them. Only the explicit openpi-step
motion mode executes norm-limited model actions with the gripper forced open.
Scene transfer is not validated.
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


def rotation_probe_action(axis, step_deg, ik_scale):
    """Bounded root-frame rotation vector in radians, with an open gripper."""
    if axis not in ("rx", "ry", "rz"):
        raise ValueError("rotation axis must be rx, ry or rz")
    if not math.isfinite(step_deg) or not 0 < step_deg <= 0.5:
        raise ValueError("rotation step must be finite and in (0, 0.5] degrees")
    if not math.isfinite(ik_scale) or ik_scale <= 0:
        raise ValueError("IK scale must be finite and positive")
    raw = math.radians(step_deg) / ik_scale
    if abs(raw) > 1.0:
        raise ValueError("test command would exceed normalized magnitude 1")
    action = [0.0] * 7
    action[3 + ("rx", "ry", "rz").index(axis)] = raw
    action[6] = 1.0
    return action


def tracking_action(position_error, rotation_error, ik_scale, rotation_step_deg=0.5):
    """Convert root-frame pose errors to separately norm-limited IK commands."""
    if not math.isfinite(ik_scale) or ik_scale <= 0:
        raise ValueError("IK scale must be finite and positive")
    if not math.isfinite(rotation_step_deg) or not 0.1<=rotation_step_deg<=0.5:
        raise ValueError("rotation step must be finite and in [0.1, 0.5] degrees")
    action = []
    for error, limit in ((position_error, 0.001), (rotation_error, math.radians(rotation_step_deg))):
        if len(error) != 3 or not all(math.isfinite(v) for v in error):
            raise ValueError("pose errors must be finite three-vectors")
        length = math.sqrt(sum(v * v for v in error))
        factor = min(1.0, limit / length) if length else 1.0
        action.extend(v * factor / ik_scale for v in error)
    if max(abs(v) for v in action) > 1.0:
        raise ValueError("tracking command would exceed normalized magnitude 1")
    return action + [1.0]


def main():
    import gymnasium as gym
    import numpy as np
    import torch
    from PIL import Image

    import isaaclab.sim as sim_utils
    import isaaclab_tasks  # noqa: F401 -- register official tasks
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils.math import apply_delta_pose, axis_angle_from_quat, compute_pose_error, quat_apply_inverse, subtract_frame_transforms
    from libero_mapping import decode_libero_action, mapping_test_plan, stream_test_plan
    from step_response import summarize_response
    from gain_probe import GAIN_PROFILES, configure_gain_profile
    from physics_trace import PhysicsJointRecorder
    from joint_identification import summarize_joint_step
    from stream_scenarios import validate_stream_scenarios
    from grasp_probe import run_grasp_probe
    from grasp_scenarios import validate_grasp_scenarios
    from openpi_step_probe import run_openpi_steps, libero_state_vector, read_server_metadata
    from approach_monitor import ApproachMonitor
    from replay_probe import REPLAY_MODES, load_replay, run_replay
    from kinematic_attribution import JacobianTap, enrich_kinematic_row, summarize_kinematics, resolved_body_identity, resolve_sensor_body_path
    from isaaclab_rl.entrypoints.simple_agents import create_zero_action_policy
    from isaaclab_tasks.contrib.lift.config.franka.ik_rel_env_cfg import (
        FrankaCubeLiftEnvCfg,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("logs/openpi_camera_check"))
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--query", action="store_true")
    parser.add_argument("--gain-profile", choices=tuple(GAIN_PROFILES), default="baseline")
    parser.add_argument("--record-jacobians", action="store_true", help="capture actual IK Jacobians during frozen replay only")
    parser.add_argument("--record-physics", action="store_true", help="record actual joint targets/state at every existing physics substep")
    parser.add_argument("--joint-gains-json", type=Path, help="optional explicit per-arm-joint gain overrides; no automatic tuning")
    parser.add_argument("--joint-step-rad", type=float, default=0.003, help="isolated joint-step magnitude, at most 0.005 rad")
    parser.add_argument("--stream-start-offset", type=float, nargs=3, help="preposition EE in root axes; hold orientation; norm <= 0.025 m")
    parser.add_argument("--stream-amplitude-scale", type=float, default=1.0)
    parser.add_argument("--grasp-object-offset", type=float, nargs=2, default=(0.0,0.0))
    parser.add_argument("--grasp-object-yaw", type=float, default=0.0)
    parser.add_argument("--grasp-orient-step-deg", type=float, default=0.5)
    parser.add_argument("--grasp-seed", type=int, default=0)
    parser.add_argument("--replay-actions-json", type=Path, default=Path(__file__).with_name("openpi_replay_actions.json"))
    parser.add_argument("--openpi-approach-long", action="store_true", help="explicit 400-step approach budget and larger total envelope")
    parser.add_argument("--openpi-approach", action="store_true", help="segmented diagnostic-only approach, up to 60 controls")
    parser.add_argument("--openpi-replan-steps", type=int, choices=(1,5), default=1)
    parser.add_argument("--openpi-steps", type=int, default=20)
    parser.add_argument("--openpi-state-frame", choices=("world", "root"), default="world")
    parser.add_argument("--motion-test", choices=(
        "none", "x", "y", "z", "gripper", "rx", "ry", "rz",
        "track-x", "track-y", "track-z", "track-rx", "track-ry", "track-rz", "libero-map", "libero-stream", "step-response", "joint-step", "grasp", "openpi-step", *REPLAY_MODES,
    ), default="none")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--prompt", default="pick up the cube")
    parser.add_argument("--light-intensity", type=float, default=1000.0)
    parser.add_argument("--external-eye", type=float, nargs=3, default=(0.95, -0.75, 0.65))
    parser.add_argument("--external-target", type=float, nargs=3, default=(0.5, 0.0, 0.10))
    parser.add_argument("--wrist-offset", type=float, nargs=3, default=(0.10, 0.0, 0.02))
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.openpi_approach_long and not args.openpi_approach:
        parser.error("--openpi-approach-long requires --openpi-approach")
    if args.openpi_approach and (args.motion_test!="openpi-step" or args.openpi_replan_steps!=5):
        parser.error("--openpi-approach requires openpi-step and --openpi-replan-steps 5")
    if args.openpi_replan_steps!=1 and args.motion_test!="openpi-step":
        parser.error("--openpi-replan-steps requires openpi-step")
    if args.record_jacobians:
        if args.motion_test not in REPLAY_MODES:
            parser.error("--record-jacobians requires a frozen replay mode")
        args.record_physics = True
    if not 1<=args.openpi_steps<=(400 if args.openpi_approach_long else 60 if args.openpi_approach else 20):
        parser.error("--openpi-steps must be 1..20, 1..60 with approach, or 1..400 with explicit long approach")
    grasp_condition = {"name": "direct", "object_offset_xy_m": list(args.grasp_object_offset),
        "object_yaw_deg": args.grasp_object_yaw, "orient_step_deg": args.grasp_orient_step_deg, "seed": args.grasp_seed}
    try:
        validate_grasp_scenarios([grasp_condition])
    except ValueError as error:
        parser.error(str(error))
    if args.motion_test!="grasp" and (any(args.grasp_object_offset) or args.grasp_object_yaw!=0 or args.grasp_orient_step_deg!=0.5 or args.grasp_seed!=0):
        parser.error("grasp condition options require --motion-test grasp")
    if args.stream_start_offset is not None or args.stream_amplitude_scale != 1.0:
        if args.motion_test != "libero-stream":
            parser.error("stream position/amplitude options require --motion-test libero-stream")
        try:
            validate_stream_scenarios([{"name": "direct", "start_offset_m": args.stream_start_offset or [0, 0, 0],
                                        "amplitude_scale": args.stream_amplitude_scale}])
        except ValueError as error:
            parser.error(str(error))
    if args.query and args.motion_test != "none":
        parser.error("--query is preview only; choose a separate motion mode, including openpi-step for bounded model motion")
    if args.motion_test in ("joint-step", "grasp", "openpi-step", *REPLAY_MODES):
        args.record_physics = True
    if not math.isfinite(args.joint_step_rad) or not 0 < args.joint_step_rad <= 0.005:
        parser.error("--joint-step-rad must be finite and in (0, 0.005]")
    if args.gain_profile != "baseline" and args.motion_test not in ("step-response", "libero-stream", "joint-step", "grasp", "openpi-step", *REPLAY_MODES):
        parser.error("experimental gain profiles require step-response, libero-stream, joint-step, or grasp")
    if args.record_physics and args.motion_test not in ("step-response", "libero-stream", "joint-step", "grasp", "openpi-step", *REPLAY_MODES):
        parser.error("--record-physics requires step-response, libero-stream, joint-step, or grasp")
    if args.joint_gains_json and args.motion_test not in ("step-response", "libero-stream", "joint-step", "grasp", "openpi-step", *REPLAY_MODES):
        parser.error("per-joint gains require step-response, libero-stream, joint-step, or grasp")
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
    joint_overrides = json.loads(args.joint_gains_json.read_text(encoding="utf-8")) if args.joint_gains_json else None
    gain_config = configure_gain_profile(cfg.scene.robot.actuators, args.gain_profile, joint_overrides)
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 60.0
    if args.motion_test in ("libero-map", "libero-stream", "step-response", "joint-step", "grasp", "openpi-step", *REPLAY_MODES):
        # LIBERO's default 20 Hz is a simulation period, not wall-clock speed.
        decimation = round(0.05 / float(cfg.sim.dt))
        if decimation < 1 or not math.isclose(decimation * float(cfg.sim.dt), 0.05, abs_tol=1e-8):
            raise RuntimeError("Physics dt cannot represent a 50 ms mapping test period")
        cfg.decimation = decimation
        cfg.episode_length_s = 300.0
    if args.motion_test == "grasp":
        cfg.seed = args.grasp_seed
        cfg.events.reset_object_position.params["pose_range"] = {
            "x": (args.grasp_object_offset[0],)*2, "y": (args.grasp_object_offset[1],)*2,
            "z": (0.0, 0.0), "yaw": (math.radians(args.grasp_object_yaw),)*2}
    elif args.motion_test in ("openpi-step", *REPLAY_MODES):
        cfg.seed = 0
        cfg.events.reset_object_position.params["pose_range"] = {"x": (0.0,0.0), "y": (0.0,0.0), "z": (0.0,0.0)}
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
        physics_recorder = None
        jacobian_tap = None
        direct_joint_target = None
        original_arm_apply = None
        had_arm_apply = False
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
                if args.motion_test in ("openpi-step", *REPLAY_MODES):
                    # Fixed scene reference for this diagnostic, independent of
                    # actual object position. Never re-aim during the loop.
                    wrist_target = origin + torch.tensor([[0.5,0.0,0.04]],device=origin.device)
                else:
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
                arm_ids, arm_names = robot.find_joints("panda_joint[1-7]")
                if len(arm_ids) != 7:
                    raise RuntimeError("Expected seven Panda arm joints for gain audit")

                def joint_field(name):
                    try:
                        values = tensor(getattr(robot.data, name))[0, arm_ids].detach().clone()
                        return values, None
                    except (AttributeError, NotImplementedError) as error:
                        return None, str(error)

                def json_vector(values):
                    return [float(v) if math.isfinite(float(v)) else str(float(v)) for v in values.cpu().tolist()]

                runtime_audit = {}
                for name in ("joint_stiffness", "joint_damping", "joint_vel_limits", "joint_effort_limits"):
                    values, unavailable = joint_field(name)
                    runtime_audit[name] = {"values": json_vector(values)} if values is not None else {"unavailable": unavailable}
                actuator_audit = {"configuration": gain_config, "joint_names": arm_names, "runtime": runtime_audit,
                    "ik_configuration": {name: getattr(arm_term.cfg.controller, name, None) for name in (
                        "command_type", "use_relative_mode", "ik_method", "ik_params",
                    )},
                    "effort_note": "implicit-drive applied_torque is approximate telemetry; control-period samples do not capture physics-step peaks"}
                (output / "actuator_audit.json").write_text(json.dumps(actuator_audit, indent=2, default=str), encoding="utf-8")
                print(f"[Actuator audit] profile={args.gain_profile}; saved {output / 'actuator_audit.json'}", flush=True)
                telemetry_samples = []

                def sample_joint_telemetry():
                    row = {}
                    for field in ("joint_vel", "applied_torque"):
                        values, unavailable = joint_field(field)
                        if values is not None:
                            if not bool(torch.isfinite(values).all()):
                                raise RuntimeError(f"Non-finite {field} during actuator experiment")
                            row[field] = values.cpu().tolist()
                        else:
                            row[field + "_unavailable"] = unavailable
                    telemetry_samples.append(row)
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
                    "wrist_initial_aim": ("fixed scene reference [0.5,0,0.04]; no object tracking" if args.motion_test in ("openpi-step", *REPLAY_MODES) else
                                          "one-time calibration from ground-truth cube position; not for online tracking"),
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

            physics_context = {}
            if args.record_physics:
                if getattr(base, "_physics_handles_decimation", False):
                    raise RuntimeError("Physics-owned decimation hides substeps; this recorder cannot label them as individual physics samples")
                command_buffers = getattr(robot.actuators, "target_command", None)
                if command_buffers is not None:
                    position_target_buffer = command_buffers.position
                    velocity_target_buffer = command_buffers.velocity
                    command_source = "robot.actuators.target_command"
                else:
                    position_target_buffer = robot.data.joint_pos_target
                    velocity_target_buffer = getattr(robot.data, "joint_vel_target", None)
                    command_source = "robot.data.joint_pos_target (legacy)"

                def required_joint_values(field):
                    values, unavailable = joint_field(field)
                    if values is None:
                        raise RuntimeError(f"Required physics state {field} unavailable: {unavailable}")
                    if not bool(torch.isfinite(values).all()):
                        raise RuntimeError(f"Non-finite physics state {field}")
                    return values.cpu().tolist()

                def command_values(buffer):
                    values = tensor(buffer)[0, arm_ids].detach().clone()
                    if not bool(torch.isfinite(values).all()):
                        raise RuntimeError("Non-finite joint target")
                    return values.cpu().tolist()

                def kinematic_pose():
                    sensor_pos,sensor_quat=read_pose()
                    ik_pos,ik_quat=arm_term._compute_frame_pose()
                    return {"sensor_pos_root_m":sensor_pos.cpu().tolist(),
                            "sensor_quat_root_xyzw":sensor_quat.cpu().tolist(),
                            "ik_pos_root_m":tensor(ik_pos)[0].detach().cpu().tolist(),
                            "ik_quat_root_xyzw":tensor(ik_quat)[0].detach().cpu().tolist()}

                if args.record_jacobians:
                    import inspect
                    for method in ("_compute_frame_pose", "_compute_frame_jacobian"):
                        if not callable(getattr(arm_term,method,None)):
                            raise RuntimeError(f"Jacobian recording requires IK {method}")
                    if list(arm_term._joint_names) != list(arm_names) or not robot.is_fixed_base:
                        raise RuntimeError("Jacobian attribution requires fixed base and identical seven-arm-joint ordering")
                    from pxr import UsdPhysics
                    body_resolution={"sensor_path":cfg.scene.ee_frame.target_frames[0].prim_path,
                        "robot_path_pattern":robot.cfg.prim_path,
                        "env_regex_ns":base.scene.env_regex_ns,
                        "ik_body_pattern":arm_term.cfg.body_name,
                        "private_body_name_repr":repr(getattr(arm_term,"_body_name",None)),
                        "robot_body_names":list(robot.body_names),"used_ik_body_index":int(arm_term._body_idx)}
                    try:
                        stage=sim_utils.get_current_stage()
                        if stage is None:
                            raise RuntimeError("No active USD stage for full-path body resolution")
                        rigid_paths=[str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
                        body_resolution["usd_rigid_body_paths"]=rigid_paths
                        matched=resolve_sensor_body_path(body_resolution["sensor_path"],robot.cfg.prim_path,
                            base.scene.env_regex_ns,rigid_paths,list(robot.body_names))
                        body_resolution.update(matched)
                        sensor_body_ids=matched["sensor_body_ids"]
                        sensor_body_names=matched["sensor_body_names"]
                        sensor_parent=sensor_body_names[0]
                        ik_body_ids,ik_body_names=robot.find_bodies(arm_term.cfg.body_name)
                        body_resolution.update(resolved_body_identity(sensor_body_ids,sensor_body_names,
                            ik_body_ids,ik_body_names,arm_term._body_idx))
                    except Exception as error:
                        body_resolution["resolution_error"]=str(error)
                        raise
                    finally:
                        (output/"jacobian_body_resolution.json").write_text(json.dumps(body_resolution,indent=2),encoding="utf-8")
                        print("[Jacobian bodies] "+json.dumps(body_resolution),flush=True)
                    implementation=[]
                    for method in ("_compute_frame_pose", "_compute_frame_jacobian"):
                        try:
                            implementation.append(inspect.getsource(getattr(arm_term,method)))
                        except (OSError,TypeError):
                            implementation.append(f"Source unavailable for {method}")
                    for property_name in ("jacobian_w", "jacobian_b"):
                        descriptor=inspect.getattr_static(type(arm_term),property_name,None)
                        try:
                            implementation.append(inspect.getsource(descriptor.fget if isinstance(descriptor,property) else descriptor))
                        except (OSError,TypeError):
                            implementation.append(f"Source unavailable for {property_name}")
                    (output/"jacobian_implementation.txt").write_text("\n\n".join(implementation),encoding="utf-8")
                    def capture_jacobian(result):
                        matrix=tensor(result)[0].detach().clone()
                        if matrix.shape != (6,7) or not bool(torch.isfinite(matrix).all()):
                            raise RuntimeError("Expected finite actual IK Jacobian with shape 6x7")
                        controller=arm_term._ik_controller
                        return {**kinematic_pose(), "actual_ik_jacobian_root":matrix.cpu().tolist(),
                            "joint_pos_rad":required_joint_values("joint_pos"),
                            "ik_target_pos_root_m":tensor(controller.ee_pos_des)[0].detach().cpu().tolist(),
                            "ik_target_quat_root_xyzw":tensor(controller.ee_quat_des)[0].detach().cpu().tolist()}
                    jacobian_tap=JacobianTap(arm_term,capture_jacobian)

                def read_physics_command():
                    effort, unavailable = joint_field("applied_torque")
                    actual_target = command_values(position_target_buffer)
                    if direct_joint_target is not None:
                        expected_target = direct_joint_target[0].detach().cpu().tolist()
                        if max(abs(a-b) for a, b in zip(actual_target, expected_target)) > 1e-6:
                            raise RuntimeError("Fixed joint target was overwritten; discard isolated-step run")
                    captured = {"kinematics_before":jacobian_tap.take()} if jacobian_tap is not None else {}
                    return {
                        **captured, "position_target_rad": actual_target,
                        "velocity_target_rad_s": command_values(velocity_target_buffer) if velocity_target_buffer is not None else None,
                        "position_before_rad": required_joint_values("joint_pos"),
                        "estimated_effort_nm": effort.cpu().tolist() if effort is not None else None,
                        "estimated_effort_unavailable": unavailable,
                    }

                def read_physics_state():
                    return {"position_rad": required_joint_values("joint_pos"),
                            "velocity_rad_s": required_joint_values("joint_vel"),
                            **({"kinematics_after":kinematic_pose()} if jacobian_tap is not None else {})}

                physics_recorder = PhysicsJointRecorder(
                    base.scene, float(base.physics_dt), float(base.step_dt),
                    read_physics_command, read_physics_state, output / "physics_joint_trace.jsonl",
                    {"joint_names": arm_names, "gain_profile": args.gain_profile,
                     "command_source": command_source,
                     "time_origin": "first recorded diagnostic control period, excluding warmup",
                     "effort_note": "estimate captured after command write before integration; not a measured solver torque",
                     "target_note": ("fixed seven-joint targets; IK arm application bypassed" if args.motion_test == "joint-step" else
                                     "post-write position target used for the following physics integration; IK may update it every physics step"),
                     "record_jacobians":args.record_jacobians,
                     "kinematic_capture_setup":({
                         "action_class":type(arm_term).__module__+"."+type(arm_term).__qualname__,
                         "ik_body":ik_body_names[0],"sensor_parent":sensor_parent,"body_resolution":body_resolution,
                         "ik_offset_pos_m":list(arm_term.cfg.body_offset.pos) if arm_term.cfg.body_offset else None,
                         "ik_offset_quat_xyzw":list(arm_term.cfg.body_offset.rot) if arm_term.cfg.body_offset else None,
                         "sensor_offset_pos_m":list(cfg.scene.ee_frame.target_frames[0].offset.pos),
                         "sensor_offset_quat_xyzw":list(cfg.scene.ee_frame.target_frames[0].offset.rot),
                         "capture_timing":"actual IK Jacobian before compute/target-write; command read after target-write; state read after one physics update",
                         "source_file":"jacobian_implementation.txt",
                         "sensor_transform":"Jv_sensor=Jv_ik+Jw cross (p_sensor-p_ik); all columns in joint_names order"}
                         if args.record_jacobians else None),
                     "jacobian_note":"actual IK method output copied before each integration; sensor point shifted using omega cross offset; root axes; local first-order attribution"},
                    enrich_row=enrich_kinematic_row if args.record_jacobians else None,
                )
                print(f"[Physics trace] physics_dt={base.physics_dt}; control_dt={base.step_dt}; "
                      f"substeps={physics_recorder.expected_substeps}; joints={arm_names}", flush=True)

            if args.motion_test != "none":
                def read_pose():
                    pose_pos, pose_quat = subtract_frame_transforms(
                        tensor(robot.data.root_pos_w),
                        tensor(robot.data.root_quat_w),
                        tensor(ee.data.target_pos_w)[:, 0, :],
                        tensor(ee.data.target_quat_w)[:, 0, :],
                    )
                    return pose_pos[0].detach().clone(), pose_quat[0].detach().clone()

                def rotation_delta(initial_pos, initial_quat, current_pos, current_quat):
                    # q_after * inverse(q_before): a rotation vector expressed
                    # in robot root axes. Subtracting absolute rotation vectors
                    # would be incorrect around the initial near-pi orientation.
                    _, delta = compute_pose_error(
                        initial_pos[None], initial_quat[None],
                        current_pos[None], current_quat[None], rot_error_type="axis_angle",
                    )
                    return delta[0]

                def checked_step(action, initial_pos):
                    if args.motion_test in ("grasp", "openpi-step", *REPLAY_MODES):
                        previous_pos, previous_quat = read_pose()
                    if physics_recorder is not None:
                        physics_recorder.begin_control({**physics_context, "raw_isaac_action": action[0].detach().cpu().tolist()})
                    if jacobian_tap is not None:
                        jacobian_tap.active=True
                    try:
                        result = env.step(action)
                    finally:
                        if jacobian_tap is not None:
                            jacobian_tap.active=False
                    if physics_recorder is not None:
                        physics_recorder.end_control()
                    if bool(tensor(result[2]).any()) or bool(tensor(result[3]).any()):
                        raise RuntimeError("Environment reset during calibration; discard this measurement")
                    current_pos, current_quat = read_pose()
                    delta = current_pos - initial_pos
                    max_displacement = 0.6 if args.motion_test == "grasp" else ((0.35 if args.openpi_approach_long else 0.07 if args.openpi_approach else 0.03) if args.motion_test in ("openpi-step", *REPLAY_MODES) else 0.05)
                    if not bool(torch.isfinite(delta).all()) or float(delta.norm()) > max_displacement:
                        raise RuntimeError(f"Calibration stopped: invalid state or EE displacement exceeded {max_displacement} m")
                    angle = rotation_delta(before, before_quat, current_pos, current_quat)
                    max_angle = math.pi + 1e-5 if args.motion_test == "grasp" else math.radians((90 if args.openpi_approach_long else 18 if args.openpi_approach else 6) if args.motion_test in ("openpi-step", *REPLAY_MODES) else 15)
                    if not bool(torch.isfinite(angle).all()) or float(angle.norm()) > max_angle:
                        raise RuntimeError("Calibration stopped: invalid orientation or rotation exceeded diagnostic envelope")
                    if args.motion_test in ("grasp", "openpi-step", *REPLAY_MODES):
                        if float((current_pos-previous_pos).norm()) > 0.01 or float(rotation_delta(previous_pos, previous_quat, current_pos, current_quat).norm()) > math.radians(3):
                            raise RuntimeError("Diagnostic stopped: excessive EE change in one control period")
                        q = tensor(robot.data.joint_pos)[0, arm_ids]
                        limits = tensor(robot.data.soft_joint_pos_limits)[0, arm_ids]
                        if not bool(((q >= limits[:, 0]-0.001) & (q <= limits[:, 1]+0.001)).all()):
                            raise RuntimeError("Diagnostic stopped: arm outside soft joint limits")
                    time.sleep(float(base.step_dt))
                    sample_joint_telemetry()

                with torch.inference_mode():
                    before, before_quat = read_pose()
                    measured = {"mode": args.motion_test, "gain_profile": args.gain_profile,
                        "actuator_audit": actuator_audit, "ee_before_root": before.cpu().tolist()}
                    if args.motion_test in ("openpi-step", *REPLAY_MODES):
                        if args.motion_test == "openpi-step":
                            from openpi_client import image_tools
                            from openpi_client.websocket_client_policy import WebsocketClientPolicy
                        if list(finger_names)!=["panda_finger_joint1","panda_finger_joint2"]:
                            raise RuntimeError("Unexpected finger ordering for LIBERO state adapter")
                        client = None
                        object_origin = tensor(base.scene["object"].data.root_pos_w)[0].detach().clone()

                        def policy_states():
                            p,q = read_pose()
                            wp = tensor(ee.data.target_pos_w)[0,0].detach().clone()
                            wq = tensor(ee.data.target_quat_w)[0,0].detach().clone()
                            fingers = tensor(robot.data.joint_pos)[0,finger_ids].detach().clone()
                            root_state = np.asarray(libero_state_vector(p.cpu().tolist(),q.cpu().tolist(),fingers.cpu().tolist()),dtype=np.float32)
                            world_state = np.asarray(libero_state_vector(wp.cpu().tolist(),wq.cpu().tolist(),fingers.cpu().tolist()),dtype=np.float32)
                            if root_state.shape!=(8,) or not np.isfinite(root_state).all() or not np.isfinite(world_state).all():
                                raise RuntimeError("Invalid live policy state")
                            return p,q,root_state,world_state

                        def policy_observe(index):
                            p,q,root_state,world_state = policy_states()
                            selected = world_state if args.openpi_state_frame=="world" else root_state
                            observation={"observation/state":selected,"prompt":args.prompt}
                            files=[];image_stats={}
                            for name,key,label in (("external_camera","observation/image","external"),("wrist_camera","observation/wrist_image","wrist")):
                                rgb=tensor(base.scene[name].data.output["rgb"])[0,...,:3].detach().cpu().numpy().copy()
                                processed=image_tools.resize_with_pad(rgb,224,224)
                                if processed.shape!=(224,224,3) or processed.dtype!=np.uint8:
                                    raise RuntimeError("Policy image must be uint8 RGB 224x224x3")
                                filename=f"openpi_input_{index:03d}_{label}.png"
                                Image.fromarray(processed).save(output/filename)
                                files.append(filename);observation[key]=processed
                                image_stats[label]={"shape":list(processed.shape),"std":float(processed.std()),"min":int(processed.min()),"max":int(processed.max())}
                            return observation,{"image_files":files,"image_stats":image_stats,
                                "state_frame":args.openpi_state_frame,"state_sent":selected.tolist(),
                                "state_world":world_state.tolist(),"state_root":root_state.tolist(),
                                "isaac_finger_positions_m":tensor(robot.data.joint_pos)[0,finger_ids].cpu().tolist(),
                                "ee_before_root_m":p.cpu().tolist(),"ee_before_quat_root":q.cpu().tolist(),
                                "ee_before_quat_world":tensor(ee.data.target_quat_w)[0,0].cpu().tolist()}

                        def policy_infer(observation):
                            nonlocal client
                            if client is None:
                                client=WebsocketClientPolicy(host=args.server_host,port=args.server_port)
                                measured["openpi_setup"]["server_metadata"]=read_server_metadata(client)
                            return client.infer(observation)

                        def policy_execute(mapped,index):
                            p,q,_,_=policy_states()
                            root_quat=tensor(robot.data.root_quat_w).detach().clone()
                            delta_root=torch.cat((
                                quat_apply_inverse(root_quat,torch.tensor([mapped["bounded_delta_world_m"]],device=base.device)),
                                quat_apply_inverse(root_quat,torch.tensor([mapped["bounded_delta_world_rad"]],device=base.device))),dim=-1)
                            target_pos,target_quat=apply_delta_pose(p[None],q[None],delta_root)
                            pe,re=compute_pose_error(p[None],q[None],target_pos,target_quat,rot_error_type="axis_angle")
                            values=tracking_action(pe[0].cpu().tolist(),re[0].cpu().tolist(),float(arm_term.cfg.scale),rotation_step_deg=.25)
                            values[6]=1.0
                            physics_context.update(case="openpi",phase=("model-step" if args.motion_test=="openpi-step" else ("replay" if index<20 else "zero-input-tail")),case_control_step=index+1)
                            physics_context.update(sensor_target_pos_root_m=target_pos[0].cpu().tolist(),
                                sensor_target_quat_root_xyzw=target_quat[0].cpu().tolist())
                            checked_step(torch.tensor([values],dtype=torch.float32,device=base.device),before)
                            after,after_q,root_state,world_state=policy_states()
                            object_after=tensor(base.scene["object"].data.root_pos_w)[0].detach().clone()
                            if not bool(torch.isfinite(object_after).all()) or float((object_after-object_origin).norm())>0.02:
                                raise RuntimeError("Object moved more than 2 cm or became invalid during first model-motion probe")
                            target_pe,target_re=compute_pose_error(after[None],after_q[None],target_pos,target_quat,rot_error_type="axis_angle")
                            return {"ee_before_root_m":p.cpu().tolist(),"ee_before_quat_root":q.cpu().tolist(),
                                "period_position_error_m":float(target_pe.norm()),
                                "period_rotation_error_deg":math.degrees(float(target_re.norm())),
                                "target_pos_root_m":target_pos[0].cpu().tolist(),"target_quat_root":target_quat[0].cpu().tolist(),
                                "bounded_delta_root_m_rad":delta_root[0].cpu().tolist(),"raw_isaac_action":values,
                                "ee_after_root_m":after.cpu().tolist(),"ee_after_quat_root":after_q.cpu().tolist(),
                                "actual_delta_root_m":(after-p).cpu().tolist(),
                                "actual_delta_rotation_root_rad":rotation_delta(p,q,after,after_q).cpu().tolist(),
                                "state_after_root":root_state.tolist(),"state_after_world":world_state.tolist(),
                                "object_pos_world_m_for_diagnostic_only":object_after.cpu().tolist()}

                        measured["openpi_setup"]={"host":args.server_host,"port":args.server_port,"prompt":args.prompt,
                            "server_metadata":{"status":"not_connected"},
                            "seed":0,"reset_pose_range":cfg.events.reset_object_position.params["pose_range"],
                            "robot_gravity_disabled":cfg.scene.robot.spawn.rigid_props.disable_gravity,
                            "ee_sensor_offset_m":list(cfg.scene.ee_frame.target_frames[0].offset.pos),
                            "ik_offset_m":list(arm_term.cfg.body_offset.pos),
                            "robot_root_pos_world_m":tensor(robot.data.root_pos_w)[0].cpu().tolist(),
                            "robot_root_quat_world":tensor(robot.data.root_quat_w)[0].cpu().tolist(),
                            "state_frame":args.openpi_state_frame,"state_layout":"EE xyz + rotation vector + two finger joint positions",
                            "rotation_vector_convention":"official LIBERO _quat2axisangle; xyzw; no w>=0 sign canonicalization",
                            "finger_state_convention":"Isaac [q1,q2] -> robosuite-1.4 Panda [q1,-q2]; metre units",
                            "image_preprocessing":"RGB resize/pad 224; no 180-degree flip",
                            "wrist_initial_aim":"fixed scene reference; no live object re-aiming",
                            "frame_assumption":"LIBERO and Isaac world axes assumed aligned; scene registration remains unverified",
                            "object_truth_usage":"diagnostic stop/progress checks only, not policy input or action target",
                            "segmented_approach":args.openpi_approach,"extended_approach":args.openpi_approach_long,
                            "motion_envelope":{"displacement_m":.35 if args.openpi_approach_long else .07 if args.openpi_approach else .03,
                                               "rotation_deg":90 if args.openpi_approach_long else 18 if args.openpi_approach else 6},
                            "timing":"synchronous inference pauses simulation; one 50ms simulated period per request; not real-time 20Hz",
                            "replan_steps":args.openpi_replan_steps,
                            "observation_semantics":"fresh observation each control step; sent only when inference_performed=true; cached-step images are diagnostics",
                            "gripper":"forced open; model request recorded but not executed"}
                        if args.motion_test in REPLAY_MODES:
                            replay_data = load_replay(args.replay_actions_json)
                            if not math.isclose(float(base.step_dt), .05, abs_tol=1e-8):
                                raise RuntimeError("Replay requires 50 ms control period")
                            def replay_state():
                                p,q=read_pose()
                                return {"position_m":p.cpu().tolist(),"quat_xyzw":q.cpu().tolist()}
                            measured["replay_setup"] = dict(measured["openpi_setup"],
                                timing="fixed action replay; no inference; 50ms simulated period",
                                source_commit=replay_data["source_commit"], fixture_sha256=replay_data["sha256"])
                            measured["replay_result"] = run_replay(replay_data,args.motion_test,replay_state,policy_execute,
                                announce=lambda message:print(message,flush=True))
                            del measured["openpi_setup"]
                        else:
                            monitor=None
                            if args.openpi_approach:
                                _,_,_,initial_world_state=policy_states()
                                monitor=ApproachMonitor(initial_world_state[:3].tolist(),object_origin.cpu().tolist())
                            measured["openpi_result"]=run_openpi_steps(policy_observe,policy_infer,policy_execute,
                                steps=args.openpi_steps,announce=lambda message:print(message,flush=True),replan_steps=args.openpi_replan_steps,approach_monitor=monitor,extended_approach=args.openpi_approach_long)
                        measured["control_dt_s"]=float(base.step_dt)
                    elif args.motion_test == "grasp":
                        def grasp_state():
                            p, q = read_pose()
                            obj = base.scene["object"]
                            op, oq = subtract_frame_transforms(
                                tensor(robot.data.root_pos_w), tensor(robot.data.root_quat_w),
                                tensor(obj.data.root_pos_w), tensor(obj.data.root_quat_w))
                            rel_p, rel_q = subtract_frame_transforms(p[None], q[None], op, oq)
                            return {"ee_pos_root_m": p.cpu().tolist(), "ee_quat_root": q.cpu().tolist(),
                                    "object_pos_root_m": op[0].cpu().tolist(), "object_quat_root": oq[0].cpu().tolist(),
                                    "object_in_ee_pos_m": rel_p[0].cpu().tolist(), "object_in_ee_quat": rel_q[0].cpu().tolist(),
                                    "finger_positions_m": tensor(robot.data.joint_pos)[0, finger_ids].cpu().tolist()}

                        grasp_step = 0
                        def grasp_pose_step(target, quat, gripper, phase):
                            nonlocal grasp_step
                            grasp_step += 1
                            physics_context.update(case="grasp", phase=phase, case_control_step=grasp_step)
                            p, q = read_pose()
                            pe, re = compute_pose_error(p[None], q[None],
                                torch.tensor([target], device=base.device), torch.tensor([quat], device=base.device),
                                rot_error_type="axis_angle")
                            values = tracking_action(pe[0].cpu().tolist(), re[0].cpu().tolist(), float(arm_term.cfg.scale),
                                rotation_step_deg=args.grasp_orient_step_deg if phase=="orient" else 0.5)
                            values[6] = gripper
                            checked_step(torch.tensor([values], dtype=torch.float32, device=base.device), before)
                        measured["grasp_setup"] = {
                            "seed": args.grasp_seed, "condition": grasp_condition,
                            "object_spawn_pose": str(cfg.scene.object.init_state),
                            "object_usd": str(cfg.scene.object.spawn.usd_path), "object_scale": cfg.scene.object.spawn.scale,
                            "robot_gravity_disabled": cfg.scene.robot.spawn.rigid_props.disable_gravity,
                            "reset_pose_range": cfg.events.reset_object_position.params["pose_range"],
                            "ee_sensor_offset_m": list(cfg.scene.ee_frame.target_frames[0].offset.pos),
                            "ik_offset_m": list(arm_term.cfg.body_offset.pos),
                            "feedback": "ee_frame sensor in robot root frame; quaternion xyzw",
                            "velocity_limit": "IK translation increment <=1 mm; orient rotation uses condition limit; other rotation <=0.5 deg per 50 ms",
                            "yaw_policy": "align top-down grasp with measured settled object yaw"}
                        measured["grasp_result"] = run_grasp_probe(grasp_state, grasp_pose_step,
                            announce=lambda message: print(message, flush=True), orient_step_deg=args.grasp_orient_step_deg)
                        measured["control_dt_s"] = float(base.step_dt)
                    elif args.motion_test == "joint-step":
                        if physics_recorder is None:
                            raise RuntimeError("Isolated joint steps require physics tracing")
                        arm_positions = tensor(robot.data.joint_pos)[:, arm_ids].detach().clone()
                        anchor = arm_positions.clone()
                        direct_joint_target = anchor.clone()
                        commands = getattr(robot.actuators, "target_command", None)
                        def apply_fixed_arm_targets():
                            if commands is not None and callable(getattr(commands, "set_position_index", None)):
                                commands.set_position_index(value=direct_joint_target, joint_ids=arm_ids)
                                commands.set_velocity_index(value=torch.zeros_like(direct_joint_target), joint_ids=arm_ids)
                            else:
                                robot.set_joint_position_target_index(target=direct_joint_target, joint_ids=arm_ids)
                                robot.set_joint_velocity_target_index(target=torch.zeros_like(direct_joint_target), joint_ids=arm_ids)
                        # Hand action remains normal/open. Only the arm's final
                        # application is replaced; no IK position target reaches
                        # the actuator. The trace checks every written target.
                        had_arm_apply = "apply_actions" in arm_term.__dict__
                        original_arm_apply = arm_term.apply_actions
                        arm_term.apply_actions = apply_fixed_arm_targets
                        limits = tensor(robot.data.soft_joint_pos_limits)[0, arm_ids].detach().clone()
                        phases = []
                        for step in range(20):
                            physics_context.update(case="joint-settle", phase="settle", case_control_step=step+1)
                            checked_step(hold(), before)
                        for joint_name in ("panda_joint2", "panda_joint4", "panda_joint6"):
                            index = arm_names.index(joint_name)
                            for sign in (1.0, -1.0):
                                label = f"{joint_name}/{'positive' if sign > 0 else 'negative'}"
                                initial_positions = required_joint_values("joint_pos")
                                step_rad = sign * args.joint_step_rad
                                direct_joint_target = anchor.clone()
                                direct_joint_target[0, index] += step_rad
                                if not bool(((direct_joint_target[0] >= limits[:, 0]) &
                                             (direct_joint_target[0] <= limits[:, 1])).all()):
                                    raise RuntimeError("Joint step target is outside soft joint limits")
                                first_row = len(physics_recorder.rows)
                                for step in range(60):
                                    physics_context.update(case=label, phase="step", case_control_step=step+1)
                                    checked_step(hold(), before)
                                rows = physics_recorder.rows[first_row:]
                                phase = summarize_joint_step(rows, index, initial_positions, step_rad, arm_names)
                                phase["fixed_position_targets_rad"] = direct_joint_target[0].cpu().tolist()
                                phases.append(phase)
                                print(f"[Joint step] {label}: gain50={phase['gain_at_50ms']}, "
                                      f"overshoot={phase['overshoot_pct']:.4f}%, "
                                      f"settle5pct={phase['settling_5pct_s']}", flush=True)
                                direct_joint_target = anchor.clone()
                                for step in range(60):
                                    physics_context.update(case=label, phase="return", case_control_step=step+1)
                                    checked_step(hold(), before)
                                returned = required_joint_values("joint_pos")
                                phase["return_max_joint_error_rad"] = max(abs(a-b) for a,b in zip(returned, anchor[0].cpu().tolist()))
                                if phase["return_max_joint_error_rad"] > 0.0003:
                                    raise RuntimeError("Joint did not return within 0.0003 rad; stop before the next case")
                        measured.update({"joint_steps": phases, "control_dt_s": float(base.step_dt),
                            "joint_step_rad": args.joint_step_rad,
                            "anchor_position_targets_rad": anchor[0].cpu().tolist(),
                            "scope": "fixed joint-space position steps for joints 2/4/6; all other arm position targets held, hand open, IK arm application bypassed",
                            "target_validation": "every physics write checked against the seven fixed position targets within 1e-6 rad"})
                    elif args.motion_test == "step-response":
                        if not math.isclose(float(base.step_dt), 0.05, abs_tol=1e-8):
                            raise RuntimeError("Step response requires a 50 ms control period")
                        if not callable(getattr(arm_term, "_compute_frame_pose", None)):
                            raise RuntimeError("This diagnostic needs Differential IK _compute_frame_pose; report installed version")
                        controller = getattr(arm_term, "_ik_controller", None)
                        if controller is None or not all(hasattr(controller, name) for name in ("ee_pos_des", "ee_quat_des")):
                            raise RuntimeError("This diagnostic needs IK controller desired pose access; report installed version")

                        def read_ik_pose():
                            pos, quat = arm_term._compute_frame_pose()
                            return tensor(pos)[0].detach().clone(), tensor(quat)[0].detach().clone()

                        sensor_pos, sensor_quat = read_pose()
                        ik_pos, ik_quat = read_ik_pose()
                        sensor_cfg = cfg.scene.ee_frame.target_frames[0]
                        body_offset = arm_term.cfg.body_offset
                        alignment = {
                            "ik_body_name": arm_term.cfg.body_name,
                            "ik_offset_pos": list(body_offset.pos) if body_offset is not None else None,
                            "ik_offset_rot": list(body_offset.rot) if body_offset is not None else None,
                            "sensor_target_prim": sensor_cfg.prim_path,
                            "sensor_offset_pos": list(sensor_cfg.offset.pos),
                            "sensor_offset_rot": list(sensor_cfg.offset.rot),
                            "sensor_minus_ik_pos_root_m": (sensor_pos - ik_pos).cpu().tolist(),
                            "sensor_to_ik_distance_m": float((sensor_pos - ik_pos).norm()),
                            "sensor_to_ik_rotation_deg": math.degrees(float(rotation_delta(ik_pos, ik_quat, sensor_pos, sensor_quat).norm())),
                        }
                        phases = []
                        for feedback_name, feedback_reader in (("sensor", read_pose), ("ik", read_ik_pose)):
                            for axis_index, axis in ((0, "x"), (3, "rx")):
                                start_pos, start_quat = feedback_reader()
                                step_delta = torch.zeros((1, 6), device=base.device)
                                step_delta[0, axis_index] = 0.001 if axis_index == 0 else math.radians(0.5)
                                target_pos, target_quat = apply_delta_pose(start_pos[None], start_quat[None], step_delta)
                                trace = []
                                for step in range(60):
                                    physics_context.update(case=f"{feedback_name}/{axis}", phase="step", case_control_step=step + 1)
                                    current_pos, current_quat = feedback_reader()
                                    pos_error, rot_error = compute_pose_error(
                                        current_pos[None], current_quat[None], target_pos, target_quat, rot_error_type="axis_angle",
                                    )
                                    command = tracking_action(pos_error[0].cpu().tolist(), rot_error[0].cpu().tolist(), float(arm_term.cfg.scale))
                                    checked_step(torch.tensor([command], dtype=torch.float32, device=base.device), before)
                                    actual_pos, actual_quat = feedback_reader()
                                    remaining_pos, remaining_rot = compute_pose_error(
                                        actual_pos[None], actual_quat[None], target_pos, target_quat, rot_error_type="axis_angle",
                                    )
                                    sensor_now, sensor_rot_now = read_pose()
                                    ik_now, ik_rot_now = read_ik_pose()
                                    actual_rotation = rotation_delta(start_pos, start_quat, actual_pos, actual_quat)
                                    gain = float((actual_pos - start_pos)[0] / 0.001) if axis_index == 0 else float(actual_rotation[0] / math.radians(0.5))
                                    desired_pos = tensor(controller.ee_pos_des)[0].detach().clone()
                                    desired_quat = tensor(controller.ee_quat_des)[0].detach().clone()
                                    trace.append({
                                        "step": step + 1, "time_s": (step + 1) * 0.05, "gain": gain,
                                        "position_error_m": float(remaining_pos.norm()),
                                        "rotation_error_deg": math.degrees(float(remaining_rot.norm())),
                                        "raw_isaac_action": command,
                                        "feedback_pos_root_m": actual_pos.cpu().tolist(),
                                        "feedback_quat_root": actual_quat.cpu().tolist(),
                                        "sensor_pos_root_m": sensor_now.cpu().tolist(),
                                        "sensor_quat_root": sensor_rot_now.cpu().tolist(),
                                        "ik_pos_root_m": ik_now.cpu().tolist(), "ik_quat_root": ik_rot_now.cpu().tolist(),
                                        "controller_target_pos_root_m": desired_pos.cpu().tolist(),
                                        "controller_target_quat_root": desired_quat.cpu().tolist(),
                                        "controller_target_minus_fixed_pos_m": (desired_pos - target_pos[0]).cpu().tolist(),
                                        "controller_target_vs_fixed_position_m": float((desired_pos - target_pos[0]).norm()),
                                        "controller_target_vs_fixed_rotation_deg": math.degrees(float(rotation_delta(
                                            target_pos[0], target_quat[0], desired_pos, desired_quat,
                                        ).norm())),
                                    })
                                response = summarize_response(trace)
                                phase = {"feedback": feedback_name, "axis": axis,
                                    "target_delta_root_m_rad": step_delta[0].cpu().tolist(),
                                    "target_pos_root_m": target_pos[0].cpu().tolist(),
                                    "target_quat_root": target_quat[0].cpu().tolist(),
                                    **response, "trace": trace}
                                phases.append(phase)
                                checkpoint_gains = {round(r["time_s"], 2): round(r["gain"], 4) for r in response["checkpoints"]}
                                print(f"[Step response] {feedback_name}/{axis}: gains={checkpoint_gains}, "
                                      f"t90={response['t90_sustained_s']}, settle5pct={response['settling_5pct_s']}", flush=True)
                                # Return to the same feedback point's starting
                                # pose before the next case; do not score this
                                # repositioning as part of the step response.
                                for return_step in range(60):
                                    physics_context.update(case=f"{feedback_name}/{axis}", phase="return", case_control_step=return_step + 1)
                                    current_pos, current_quat = feedback_reader()
                                    pos_error, rot_error = compute_pose_error(
                                        current_pos[None], current_quat[None], start_pos[None], start_quat[None], rot_error_type="axis_angle",
                                    )
                                    command = tracking_action(pos_error[0].cpu().tolist(), rot_error[0].cpu().tolist(), float(arm_term.cfg.scale))
                                    checked_step(torch.tensor([command], dtype=torch.float32, device=base.device), before)
                                return_pos, return_quat = feedback_reader()
                                phase["return_position_error_m"] = float((return_pos - start_pos).norm())
                                phase["return_rotation_error_deg"] = math.degrees(float(rotation_delta(start_pos, start_quat, return_pos, return_quat).norm()))
                        measured.update({"control_dt_s": 0.05, "frame_alignment": alignment, "phases": phases,
                            "scope": "fixed-target closed-loop step response at x=1 mm and rx=0.5 deg; comparison of sensor and IK feedback points",
                            "diagnostic_api": "private IK frame reader and desired-pose fields; read-only diagnostic access",
                            "timing_note": "times are simulated, not wall-clock or inference latency; gain timing measures this IK/servo loop, not OSC equivalence"})
                    elif args.motion_test == "libero-stream":
                        if not math.isclose(float(base.step_dt), 0.05, abs_tol=1e-8):
                            raise RuntimeError("Streaming test requires 20 Hz simulation control")
                        if args.stream_start_offset is not None:
                            target_pos = before[None] + torch.tensor([args.stream_start_offset], device=base.device)
                            target_quat = before_quat[None].clone()
                            start_trace = []
                            for step in range(80):
                                physics_context.update(case="stream-preposition", phase="settle", case_control_step=step+1)
                                p, q = read_pose()
                                pe, re = compute_pose_error(p[None], q[None], target_pos, target_quat, rot_error_type="axis_angle")
                                values = tracking_action(pe[0].cpu().tolist(), re[0].cpu().tolist(), float(arm_term.cfg.scale))
                                checked_step(torch.tensor([values], dtype=torch.float32, device=base.device), before)
                                p, q = read_pose()
                                pe, re = compute_pose_error(p[None], q[None], target_pos, target_quat, rot_error_type="axis_angle")
                                start_trace.append({"step": step+1, "position_error_m": float(pe.norm()),
                                                    "rotation_error_deg": math.degrees(float(re.norm()))})
                            settled = all(t["position_error_m"] <= 0.00025 and t["rotation_error_deg"] <= 0.1 for t in start_trace[-5:])
                            measured["stream_start"] = {"offset_root_m": args.stream_start_offset,
                                "target_pos_root_m": target_pos[0].cpu().tolist(), "actual_pos_root_m": p.cpu().tolist(),
                                "actual_quat_root": q.cpu().tolist(), "arm_positions_rad": joint_field("joint_pos")[0].cpu().tolist(),
                                "orientation_policy": "hold initial EE orientation", "settled_last_5_steps": settled,
                                "trace": start_trace}
                            print(f"[Stream start] offset={args.stream_start_offset}; settled={settled}", flush=True)
                            if not settled:
                                raise RuntimeError("Stream start did not settle; discard run before testing axes")
                        measured["stream_amplitude_scale"] = args.stream_amplitude_scale
                        axes = []
                        for axis in ("x", "y", "z", "rx", "ry", "rz"):
                            axis_start_pos, axis_start_quat = read_pose()
                            trace = []
                            boundaries = {}
                            stop_starts = {}
                            last_phase = None
                            for step, (phase, libero_action) in enumerate(stream_test_plan(axis, args.stream_amplitude_scale)):
                                physics_context.update(case=axis, phase=phase, case_control_step=step + 1)
                                wall_start = time.perf_counter()
                                decoded = decode_libero_action(libero_action)
                                current_pos, current_quat = read_pose()
                                if phase != last_phase and phase.startswith("stop"):
                                    stop_starts[phase] = (current_pos.clone(), current_quat.clone())
                                root_quat = tensor(robot.data.root_quat_w).clone()
                                delta_root = torch.cat((
                                    quat_apply_inverse(root_quat, torch.tensor([decoded["delta_world_m"]], device=base.device)),
                                    quat_apply_inverse(root_quat, torch.tensor([decoded["delta_world_rad"]], device=base.device)),
                                ), dim=-1)
                                # One target from measured CURRENT pose per
                                # input period. Do not accumulate desired poses
                                # or secretly wait for each target to converge.
                                target_pos, target_quat = apply_delta_pose(current_pos[None], current_quat[None], delta_root)
                                pos_error, rot_error = compute_pose_error(
                                    current_pos[None], current_quat[None], target_pos, target_quat,
                                    rot_error_type="axis_angle",
                                )
                                action_values = tracking_action(
                                    pos_error[0].cpu().tolist(), rot_error[0].cpu().tolist(), float(arm_term.cfg.scale),
                                )
                                checked_step(torch.tensor([action_values], dtype=torch.float32, device=base.device), before)
                                actual_pos, actual_quat = read_pose()
                                remaining_pos, remaining_rot = compute_pose_error(
                                    actual_pos[None], actual_quat[None], target_pos, target_quat,
                                    rot_error_type="axis_angle",
                                )
                                trace.append({
                                    "step": step + 1, "phase": phase, "time_s": (step + 1) * 0.05,
                                    "libero_action": libero_action, "raw_isaac_action": action_values,
                                    "target_pos_root_m": target_pos[0].cpu().tolist(),
                                    "target_quat_root": target_quat[0].cpu().tolist(),
                                    "ee_pos_root_m": actual_pos.cpu().tolist(),
                                    "ee_quat_root": actual_quat.cpu().tolist(),
                                    "position_error_m": float(remaining_pos.norm()),
                                    "rotation_error_deg": math.degrees(float(remaining_rot.norm())),
                                    "wall_period_s": time.perf_counter() - wall_start,
                                })
                                boundaries[phase] = (actual_pos.clone(), actual_quat.clone())
                                if phase == "out":
                                    nominal_out = delta_root[0] * 20
                                last_phase = phase
                            out_pos, out_quat = boundaries["out"]
                            final_pos, final_quat = boundaries["stop_back"]
                            out_delta_pos = out_pos - axis_start_pos
                            out_delta_rot = rotation_delta(axis_start_pos, axis_start_quat, out_pos, out_quat)
                            expected = nominal_out[:3] if axis in ("x", "y", "z") else nominal_out[3:]
                            actual = out_delta_pos if axis in ("x", "y", "z") else out_delta_rot
                            gain = float(torch.dot(actual, expected) / torch.dot(expected, expected))
                            stop_drifts = []
                            for phase in ("stop_out", "stop_back"):
                                start_pos, start_quat = stop_starts[phase]
                                end_pos, end_quat = boundaries[phase]
                                stop_drifts.append({
                                    "phase": phase,
                                    "position_m": float((end_pos - start_pos).norm()),
                                    "rotation_deg": math.degrees(float(rotation_delta(start_pos, start_quat, end_pos, end_quat).norm())),
                                })
                            final_pos_error = float((final_pos - axis_start_pos).norm())
                            final_rot_error = math.degrees(float(rotation_delta(axis_start_pos, axis_start_quat, final_pos, final_quat).norm()))
                            max_pos_error = max(row["position_error_m"] for row in trace)
                            max_rot_error = max(row["rotation_error_deg"] for row in trace)
                            fingers_m = tensor(robot.data.joint_pos)[0, finger_ids].cpu().tolist()
                            passed = (
                                0.7 <= gain <= 1.3 and final_pos_error <= 0.001 and final_rot_error <= 0.3
                                and max_pos_error <= 0.001 and max_rot_error <= 0.5
                                and all(d["position_m"] <= 0.0005 and d["rotation_deg"] <= 0.2 for d in stop_drifts)
                                and all(abs(q - 0.04) <= 0.002 for q in fingers_m)
                            )
                            result = {
                                "axis": axis, "passed": passed, "outbound_gain": gain,
                                "nominal_out_delta_root_m_rad": nominal_out.cpu().tolist(),
                                "actual_out_delta_root_m": out_delta_pos.cpu().tolist(),
                                "actual_out_delta_rotation_root_deg": torch.rad2deg(out_delta_rot).cpu().tolist(),
                                "return_position_error_m": final_pos_error, "return_rotation_error_deg": final_rot_error,
                                "max_period_position_error_m": max_pos_error, "max_period_rotation_error_deg": max_rot_error,
                                "rms_period_position_error_m": math.sqrt(sum(r["position_error_m"]**2 for r in trace) / len(trace)),
                                "rms_period_rotation_error_deg": math.sqrt(sum(r["rotation_error_deg"]**2 for r in trace) / len(trace)),
                                "mean_wall_period_s": sum(r["wall_period_s"] for r in trace) / len(trace),
                                "stop_drifts": stop_drifts, "finger_positions_m": fingers_m, "trace": trace,
                            }
                            axes.append(result)
                            print(f"[LIBERO stream] {axis}: passed={passed}, gain={gain:.4f}, "
                                  f"return={final_pos_error:.6g} m / {final_rot_error:.6g} deg", flush=True)
                        measured.update({
                            "control_dt_s": 0.05, "axes": axes,
                            "all_stream_tests_passed": all(a["passed"] for a in axes),
                            "target_semantics": "current-pose-relative target per 50 ms; no desired-pose accumulation or extra settling steps",
                            "scope": "small deterministic input stream only; no OpenPI inference or OSC dynamics equivalence",
                            "wall_timing_note": "simulation runs synchronously; wall periods include compute/render and pacing, not a real-time guarantee",
                            "thresholds": {"gain_range": [0.7, 1.3], "return_position_m": 0.001,
                                "return_rotation_deg": 0.3, "max_period_position_m": 0.001,
                                "max_period_rotation_deg": 0.5, "stop_drift_m": 0.0005, "stop_drift_deg": 0.2},
                        })
                    elif args.motion_test == "libero-map":
                        if not math.isclose(float(base.step_dt), 0.05, abs_tol=1e-8):
                            raise RuntimeError("Mapping test requires 20 Hz simulation control")
                        phases = []
                        previous_gripper = 1.0
                        for label, libero_action in mapping_test_plan():
                            decoded = decode_libero_action(libero_action, previous_gripper)
                            previous_gripper = decoded["isaac_gripper"]
                            start_pos, start_quat = read_pose()
                            # OSC deltas use world axes. Do not assume the
                            # robot base quaternion is identity in Isaac.
                            root_quat = tensor(robot.data.root_quat_w).clone()
                            world_pos_delta = torch.tensor([decoded["delta_world_m"]], device=base.device)
                            world_rot_delta = torch.tensor([decoded["delta_world_rad"]], device=base.device)
                            delta_root = torch.cat((
                                quat_apply_inverse(root_quat, world_pos_delta),
                                quat_apply_inverse(root_quat, world_rot_delta),
                            ), dim=-1)
                            target_pos, target_quat = apply_delta_pose(
                                start_pos[None], start_quat[None], delta_root,
                            )
                            trace = []
                            for step in range(60):
                                current_pos, current_quat = read_pose()
                                pos_error, rot_error = compute_pose_error(
                                    current_pos[None], current_quat[None], target_pos, target_quat,
                                    rot_error_type="axis_angle",
                                )
                                action_values = tracking_action(
                                    pos_error[0].cpu().tolist(), rot_error[0].cpu().tolist(),
                                    float(arm_term.cfg.scale),
                                )
                                action_values[6] = decoded["isaac_gripper"]
                                checked_step(torch.tensor([action_values], dtype=torch.float32, device=base.device), before)
                                actual_pos, actual_quat = read_pose()
                                remaining_pos, remaining_rot = compute_pose_error(
                                    actual_pos[None], actual_quat[None], target_pos, target_quat,
                                    rot_error_type="axis_angle",
                                )
                                trace.append({
                                    "step": step + 1,
                                    "time_s": (step + 1) * float(base.step_dt),
                                    "raw_isaac_action": action_values,
                                    "ee_pos_root_m": actual_pos.cpu().tolist(),
                                    "ee_quat_root": actual_quat.cpu().tolist(),
                                    "delta_root_m": (actual_pos - start_pos).cpu().tolist(),
                                    "delta_rotation_root_deg": torch.rad2deg(rotation_delta(
                                        start_pos, start_quat, actual_pos, actual_quat,
                                    )).cpu().tolist(),
                                    "position_error_m": float(remaining_pos.norm()),
                                    "rotation_error_deg": math.degrees(float(remaining_rot.norm())),
                                })
                            fingers_m = tensor(robot.data.joint_pos)[0, finger_ids].cpu().tolist()
                            gripper_ok = (
                                all(abs(q - 0.04) <= 0.002 for q in fingers_m)
                                if decoded["isaac_gripper"] > 0 else
                                all(abs(q) <= 0.002 for q in fingers_m)
                            )
                            phase = {
                                "phase": label,
                                "mapping": decoded,
                                "target_delta_root_m_rad": delta_root[0].cpu().tolist(),
                                "target_pos_root_m": target_pos[0].cpu().tolist(),
                                "target_quat_root": target_quat[0].cpu().tolist(),
                                "first_period_response": trace[0],
                                "final_position_error_m": trace[-1]["position_error_m"],
                                "final_rotation_error_deg": trace[-1]["rotation_error_deg"],
                                "finger_positions_m": fingers_m,
                                "gripper_position_passed": gripper_ok,
                                "target_tracking_passed": all(
                                    row["position_error_m"] <= 0.00025 and row["rotation_error_deg"] <= 0.1
                                    for row in trace[-5:]
                                ),
                                "trace": trace,
                            }
                            phases.append(phase)
                            print(f"[LIBERO mapping] {label}: target={phase['target_tracking_passed']}, "
                                  f"gripper={gripper_ok}, pos_error={phase['final_position_error_m']:.6g} m, "
                                  f"rot_error={phase['final_rotation_error_deg']:.6g} deg", flush=True)
                        measured.update({
                            "control_dt_s": float(base.step_dt),
                            "ik_scale": float(arm_term.cfg.scale),
                            "steps_per_target": 60,
                            "phases": phases,
                            "all_mapping_targets_passed": all(
                                p["target_tracking_passed"] and p["gripper_position_passed"] for p in phases
                            ),
                            "scope": "default OSC mapping and settled IK targets only; not OSC dynamics equivalence",
                            "frame_assumption": "LIBERO world axes aligned with Isaac world axes; scene registration unverified",
                            "timing_note": "each mapped target held for 3 simulated seconds; first_period_response is after 50 ms; not model execution timing",
                        })
                    elif args.motion_test.startswith("track-"):
                        axis = args.motion_test.removeprefix("track-")
                        target_delta = torch.zeros((1, 6), device=base.device)
                        axis_index = ("x", "y", "z", "rx", "ry", "rz").index(axis)
                        target_delta[0, axis_index] = 0.005 if axis_index < 3 else math.radians(2)
                        target_pos, target_quat = apply_delta_pose(
                            before[None], before_quat[None], target_delta,
                        )
                        trace = []
                        for step in range(120):
                            current_pos, current_quat = read_pose()
                            pos_error, rot_error = compute_pose_error(
                                current_pos[None], current_quat[None], target_pos, target_quat,
                                rot_error_type="axis_angle",
                            )
                            action_values = tracking_action(
                                pos_error[0].cpu().tolist(), rot_error[0].cpu().tolist(),
                                float(arm_term.cfg.scale),
                            )
                            checked_step(torch.tensor([action_values], dtype=torch.float32, device=base.device), before)
                            actual_pos, actual_quat = read_pose()
                            remaining_pos, remaining_rot = compute_pose_error(
                                actual_pos[None], actual_quat[None], target_pos, target_quat,
                                rot_error_type="axis_angle",
                            )
                            trace.append({
                                "step": step + 1,
                                "time_s": (step + 1) * float(base.step_dt),
                                "raw_isaac_action": action_values,
                                "ee_pos_root_m": actual_pos.cpu().tolist(),
                                "ee_quat_root": actual_quat.cpu().tolist(),
                                "position_error_m": float(remaining_pos.norm()),
                                "rotation_error_deg": math.degrees(float(remaining_rot.norm())),
                            })
                        measured.update({
                            "target_pos_root_m": target_pos[0].cpu().tolist(),
                            "target_quat_root": target_quat[0].cpu().tolist(),
                            "target_delta_root_m_rad": target_delta[0].cpu().tolist(),
                            "ik_scale": float(arm_term.cfg.scale),
                            "control_dt_s": float(base.step_dt),
                            "command_steps": len(trace),
                            "position_tolerance_m": 0.00025,
                            "rotation_tolerance_deg": 0.1,
                            "reached_and_held_last_5_steps": all(
                                row["position_error_m"] <= 0.00025 and row["rotation_error_deg"] <= 0.1
                                for row in trace[-5:]
                            ),
                            "final_position_error_m": trace[-1]["position_error_m"],
                            "final_rotation_error_deg": trace[-1]["rotation_error_deg"],
                            "trace": trace,
                        })
                    elif args.motion_test == "gripper":
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
                        is_rotation = args.motion_test in ("rx", "ry", "rz")
                        action_values = (
                            rotation_probe_action(args.motion_test, 0.5, float(arm_term.cfg.scale))
                            if is_rotation else
                            translation_probe_action(args.motion_test, 0.001, float(arm_term.cfg.scale))
                        )
                        action = torch.tensor([action_values], dtype=torch.float32, device=base.device)
                        for _ in range(10):
                            checked_step(action.clone(), before)
                        for _ in range(5):
                            checked_step(hold(), before)
                        if is_rotation:
                            measured["requested_step_deg"] = 0.5
                            measured["rotation_frame"] = "robot root; positive right-hand rotation"
                        else:
                            measured["requested_step_m"] = 0.001
                        measured["command_steps"] = 10
                        measured["ik_scale"] = float(arm_term.cfg.scale)
                        measured["raw_isaac_action"] = action_values
                    after, after_quat = read_pose()
                    measured["ee_after_root"] = after.cpu().tolist()
                    measured["delta_root_m"] = (after - before).cpu().tolist()
                    delta_rot = rotation_delta(before, before_quat, after, after_quat)
                    measured["ee_before_quat_root"] = before_quat.cpu().tolist()
                    measured["ee_after_quat_root"] = after_quat.cpu().tolist()
                    measured["delta_rotation_root_rad"] = delta_rot.cpu().tolist()
                    measured["delta_rotation_root_deg"] = torch.rad2deg(delta_rot).cpu().tolist()
                    measured["translation_drift_norm_m"] = float((after - before).norm())
                    measured["note"] = ("bounded model motion; gripper forced open; inspect direction and achieved displacement, not grasp success" if args.motion_test in ("openpi-step", *REPLAY_MODES) else
                                        "deterministic diagnostic only; no OpenPI actions executed; inspect direction and achieved displacement")
                report_path = output / f"motion_{args.motion_test}.json"
                telemetry_summary = {"sample_count": len(telemetry_samples), "joint_names": arm_names,
                    "note": "sampled once per control step, includes repositioning; implicit effort is estimated, not a measured torque"}
                for field, limit_field in (("joint_vel", "joint_vel_limits"), ("applied_torque", "joint_effort_limits")):
                    samples = [row[field] for row in telemetry_samples if field in row]
                    if samples:
                        peaks = [max(abs(row[i]) for row in samples) for i in range(7)]
                        telemetry_summary[field + "_abs_peak"] = peaks
                        limits = runtime_audit[limit_field].get("values")
                        if limits is not None:
                            telemetry_summary[field + "_peak_limit_ratio"] = [
                                peak / limit if isinstance(limit, (int, float)) and limit > 0 else None
                                for peak, limit in zip(peaks, limits)
                            ]
                    else:
                        telemetry_summary[field + "_unavailable"] = True
                measured["joint_telemetry_summary"] = telemetry_summary
                if physics_recorder is not None:
                    highres_summary = physics_recorder.summary()
                    if args.record_jacobians:
                        highres_summary["kinematic_attribution_summary"]=summarize_kinematics(physics_recorder.rows)
                    (output / "physics_joint_summary.json").write_text(json.dumps(highres_summary, indent=2), encoding="utf-8")
                    measured["physics_joint_summary"] = highres_summary
                    print(f"[Physics trace] saved {highres_summary['recorded_substeps']} physics samples "
                          f"in {highres_summary['recorded_controls']} controls", flush=True)
                (output / "joint_telemetry.json").write_text(json.dumps(telemetry_samples, indent=2), encoding="utf-8")
                report_path.write_text(json.dumps(measured, indent=2), encoding="utf-8")
                summary = {k: v for k, v in measured.items() if k != "trace"}
                if physics_recorder is not None:
                    summary["physics_joint_summary"] = {k: v for k, v in measured["physics_joint_summary"].items() if k != "cases"}
                if args.motion_test in ("libero-map", "step-response"):
                    summary["phases"] = [{
                        k: v for k, v in phase.items() if k not in ("trace", "first_period_response")
                    } for phase in measured["phases"]]
                if args.motion_test == "libero-stream":
                    summary["axes"] = [{k: v for k, v in axis.items() if k != "trace"} for axis in measured["axes"]]
                if args.motion_test == "grasp":
                    summary["grasp_result"] = {k: v for k, v in measured["grasp_result"].items() if k != "trace"}
                if args.motion_test in ("openpi-step", *REPLAY_MODES):
                    result_key = "replay_result" if args.motion_test in REPLAY_MODES else "openpi_result"
                    summary[result_key] = {k: v for k, v in measured[result_key].items() if k != "trace"}
                print("[Motion test] " + json.dumps(summary, indent=2), flush=True)
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
                # --query remains decode-only; execution is a separate mode.
                preview = []
                previous_gripper = 1.0
                for predicted_action in actions:
                    decoded = decode_libero_action(predicted_action.tolist(), previous_gripper)
                    previous_gripper = decoded["isaac_gripper"]
                    preview.append(decoded)
                (output / "predicted_action_mapping.json").write_text(json.dumps({
                    "executed": False,
                    "frame_assumption": "LIBERO world axes; registration to Isaac scene unverified",
                    "actions": preview,
                }, indent=2), encoding="utf-8")
                print(f"OpenPI returned {actions.shape}; saved, NOT executed.", flush=True)
                print(f"First predicted action: {actions[0]}", flush=True)
        finally:
            if physics_recorder is not None:
                physics_recorder.close()
            if jacobian_tap is not None:
                jacobian_tap.close()
            if original_arm_apply is not None:
                if had_arm_apply:
                    arm_term.apply_actions = original_arm_apply
                else:
                    del arm_term.apply_actions
            env.close()


if __name__ == "__main__":
    main()
