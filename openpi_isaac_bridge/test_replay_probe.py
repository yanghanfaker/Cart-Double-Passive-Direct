import copy
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from openpi_step_probe import bounded_openpi_action
from replay_probe import REPLAY_MODES, load_replay, replay_mapping, run_replay, quaternion_distance_deg
from gain_cases import validate_gain_cases
import run_gain_sweep


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.data = load_replay(Path(__file__).with_name("openpi_replay_actions.json"))

    def test_initial_angle_removes_float32_norm_bias_and_sign(self):
        q = self.data["initial_pose_root"]["quat_xyzw"]
        self.assertAlmostEqual(quaternion_distance_deg(q,q), 0, places=12)
        self.assertAlmostEqual(quaternion_distance_deg(q,[-x for x in q]), 0, places=12)
        self.assertAlmostEqual(quaternion_distance_deg([0,0,0,1],
            [0,0,math.sin(math.radians(.2)/2),math.cos(math.radians(.2)/2)]), .2, places=10)
        with self.assertRaises(ValueError):
            quaternion_distance_deg([0,0,0,0],q)

    def test_orientation_mismatch_stops_before_command(self):
        state = copy.deepcopy(self.data["initial_pose_root"])
        state["quat_xyzw"] = [0,0,0,1]
        result = run_replay(self.data, REPLAY_MODES[0], lambda:state,
                            lambda *a:self.fail("must not execute"), announce=lambda x:None)
        self.assertFalse(result["probe_completed"])
        self.assertEqual(result["completed_steps"],0)

    def test_joint_cases_change_exactly_one_gain_of_one_joint(self):
        cases = validate_gain_cases(json.loads(Path(__file__).with_name("replay_joint_gain_cases.json").read_text()))
        self.assertEqual(len(cases),9)
        base = cases[0]
        expected = {(j,k) for j in (1,3,6,7) for k in ("stiffness","damping")}
        observed = set()
        for case in cases[1:]:
            self.assertEqual(case["profile"],base["profile"])
            for joint in (2,4):
                self.assertEqual(case["joint_overrides"][f"panda_joint{joint}"],base["joint_overrides"][f"panda_joint{joint}"])
            extra = set(case["joint_overrides"])-set(base["joint_overrides"])
            self.assertEqual(len(extra),1)
            name = extra.pop()
            gains = case["joint_overrides"][name]
            changes = [key for key,value in gains.items() if value != {"stiffness":900,"damping":20}[key]]
            self.assertEqual(len(changes),1)
            key = changes[0]
            self.assertEqual(gains[key], {"stiffness":1125,"damping":25}[key])
            observed.add((int(name.removeprefix("panda_joint")),key))
        self.assertEqual(observed,expected)

    def test_component_removal_preserves_original_bounded_components(self):
        for action in self.data["actions"]:
            original = bounded_openpi_action(action)
            for mode in REPLAY_MODES:
                mapped = replay_mapping(action, mode)
                for field, removed in (("bounded_delta_world_m", mode.endswith("rotation")),
                                       ("bounded_delta_world_rad", mode.endswith("translation"))):
                    self.assertEqual(mapped[field], [0.0]*3 if removed else original[field])
                self.assertEqual(mapped["applied_isaac_gripper"], 1.0)

    def test_twenty_actions_then_twenty_zero_steps_without_inference(self):
        for mode in REPLAY_MODES:
            executed = []
            def execute(mapping, index):
                executed.append(mapping)
                return {"index": index}
            result = run_replay(self.data, mode, lambda:self.data["initial_pose_root"], execute, announce=lambda x:None)
            self.assertTrue(result["probe_completed"])
            self.assertEqual(len(executed), 40)
            for row in executed[20:]:
                self.assertEqual(row["bounded_delta_world_m"], [0]*3)
                self.assertEqual(row["bounded_delta_world_rad"], [0]*3)

    def test_reset_mismatch_stops_before_any_command(self):
        state = copy.deepcopy(self.data["initial_pose_root"])
        state["position_m"][0] += .001
        result = run_replay(self.data, REPLAY_MODES[0], lambda:state,
                            lambda *a:self.fail("must not execute"), announce=lambda x:None)
        self.assertFalse(result["probe_completed"])
        self.assertEqual(result["completed_steps"], 0)

    def test_execution_failure_preserves_partial_trace(self):
        def execute(mapping, index):
            if index == 2:
                raise RuntimeError("stop envelope")
            return {}
        result = run_replay(self.data, REPLAY_MODES[0], lambda:self.data["initial_pose_root"], execute, announce=lambda x:None)
        self.assertEqual(result["completed_steps"], 2)
        self.assertFalse(result["trace"][-1]["executed"])
        self.assertEqual(result["failure_reason"], "stop envelope")

    def test_fixture_rejects_nonfinite_future_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"actions.json"
            self.data["actions"][-1][0] = float("nan")
            path.write_text(json.dumps(self.data))
            with self.assertRaises(ValueError):
                load_replay(path)

    def test_runner_all_three_modes_without_openpi_extra(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            commands = []
            class Process:
                def __init__(self, command, **kwargs):
                    commands.append(command)
                    mode = command[command.index("--motion-test")+1]
                    output = Path(command[command.index("--output")+1])
                    report = {"joint_telemetry_summary":{}, "actuator_audit":{}, "physics_joint_summary":{},
                              "replay_result":{"probe_completed":True}, "replay_setup":{}}
                    (output/f"motion_{mode}.json").write_text(json.dumps(report))
                    self.stdout = io.StringIO("")
                def __enter__(self): return self
                def __exit__(self,*args): pass
                def wait(self): return 0
            args = ["runner", "--isaaclab-root", str(root), "--profiles", "kp2p25-kd0p25", "--modes", *REPLAY_MODES]
            with patch.object(sys,"argv",args), patch.object(run_gain_sweep.subprocess,"Popen",Process):
                self.assertEqual(run_gain_sweep.main(), 0)
            self.assertEqual(len(commands), 3)
            for command in commands:
                self.assertNotIn("openpi", command)
                self.assertIn("--replay-actions-json", command)
                self.assertIn("--record-physics", command)


if __name__ == "__main__":
    unittest.main()
