"""Read-only substep tracing around scene write/update, without extra sim steps.

This diagnostic supports environment-owned decimation only. Targets are read
after write_data_to_sim and paired with state after the following scene.update.
"""

import json
import math


class PhysicsJointRecorder:
    def __init__(self, scene, physics_dt, control_dt, read_command, read_state, path, metadata):
        if not math.isfinite(physics_dt) or not math.isfinite(control_dt) or physics_dt <= 0 or control_dt <= 0:
            raise ValueError("physics and control periods must be finite and positive")
        ratio = control_dt / physics_dt
        if not math.isclose(ratio, round(ratio), abs_tol=1e-8):
            raise ValueError("control period must be an integer multiple of physics dt")
        if round(ratio) <= 1:
            raise ValueError("physics sampling must be finer than the control period")
        self.scene, self.physics_dt, self.control_dt = scene, physics_dt, control_dt
        self.expected_substeps = round(ratio)
        self.read_command, self.read_state = read_command, read_state
        self.path, self.metadata = path, metadata
        self.context = None
        self.pending = None
        self.rows = []
        self.control_count = 0
        self.substep_count = 0
        self.original_write = scene.write_data_to_sim
        self.original_update = scene.update
        self.had_write = "write_data_to_sim" in scene.__dict__
        self.had_update = "update" in scene.__dict__
        self.file = path.open("w", encoding="utf-8", buffering=1)
        scene.write_data_to_sim = self._write
        scene.update = self._update

    def begin_control(self, context):
        if self.context is not None:
            raise RuntimeError("previous control period is unfinished")
        self.context = dict(context)
        self.control_count += 1
        self.period_substeps = 0

    def _write(self, *args, **kwargs):
        result = self.original_write(*args, **kwargs)
        if self.context is not None:
            if self.pending is not None:
                raise RuntimeError("multiple physics writes without an intervening update")
            self.pending = self.read_command()
        return result

    def _update(self, *args, **kwargs):
        result = self.original_update(*args, **kwargs)
        if self.context is not None:
            dt = kwargs.get("dt", args[0] if args else None)
            if dt is None or not math.isclose(float(dt), self.physics_dt, abs_tol=1e-9):
                raise RuntimeError("scene update does not correspond to exactly one physics step")
            if self.pending is None:
                raise RuntimeError("physics update without a captured command")
            self.period_substeps += 1
            self.substep_count += 1
            row = {
                **self.context, "control_index": self.control_count,
                "physics_substep": self.period_substeps,
                "command_time_s": (self.substep_count - 1) * self.physics_dt,
                "time_s": self.substep_count * self.physics_dt,
                **self.pending, **self.read_state(),
            }
            self.rows.append(row)
            self.file.write(json.dumps(row, allow_nan=False) + "\n")
            self.pending = None
        return result

    def end_control(self):
        if self.context is None or self.pending is not None or self.period_substeps != self.expected_substeps:
            raise RuntimeError("incomplete substep trace; discard this run")
        self.context = None

    def summary(self):
        groups = {}
        for row in self.rows:
            key = (row.get("case", "unlabelled"), row.get("phase", "unlabelled"))
            groups.setdefault(key, []).append(row)
        cases = []
        for (case, phase), rows in groups.items():
            joints = []
            for i, name in enumerate(self.metadata["joint_names"]):
                targets = [r["position_target_rad"][i] for r in rows]
                actual = [r["position_rad"][i] for r in rows]
                errors = [target - value for target, value in zip(targets, actual)]
                span = max(targets) - min(targets)
                command_excursion = max(abs(target - rows[0]["position_before_rad"][i]) for target in targets)
                rms = math.sqrt(sum(error**2 for error in errors) / len(errors))
                joint = {
                    "joint": name, "sample_count": len(rows),
                    "initial_position_rad": rows[0]["position_before_rad"][i],
                    "target_range_rad": span, "command_excursion_rad": command_excursion,
                    "actual_range_rad": max(actual) - min(actual),
                    "rms_tracking_error_rad": rms, "max_abs_tracking_error_rad": max(abs(e) for e in errors),
                    "normalized_rms_error": rms / command_excursion if command_excursion >= 1e-5 else None,
                    "excited_above_1e_minus5_rad": command_excursion >= 1e-5,
                    "peak_abs_velocity_rad_s": max(abs(r["velocity_rad_s"][i]) for r in rows),
                    "max_target_change_per_physics_step_rad": max(
                        (abs(b-a) for a, b in zip(targets, targets[1:])), default=0),
                }
                torques = [r["estimated_effort_nm"][i] for r in rows if r.get("estimated_effort_nm") is not None]
                joint["peak_abs_estimated_effort_nm"] = max((abs(v) for v in torques), default=None)
                joints.append(joint)
            cases.append({"case": case, "phase": phase, "joints": joints})
        return {
            **self.metadata, "physics_dt_s": self.physics_dt, "control_dt_s": self.control_dt,
            "expected_substeps_per_control": self.expected_substeps,
            "recorded_controls": self.control_count, "recorded_substeps": self.substep_count,
            "trace_file": str(self.path), "cases": cases,
            "scope": "IK joint targets may move within a control period; these are tracking statistics, not isolated-joint step identification",
        }

    def close(self):
        if self.had_write:
            self.scene.write_data_to_sim = self.original_write
        else:
            del self.scene.write_data_to_sim
        if self.had_update:
            self.scene.update = self.original_update
        else:
            del self.scene.update
        self.file.close()
