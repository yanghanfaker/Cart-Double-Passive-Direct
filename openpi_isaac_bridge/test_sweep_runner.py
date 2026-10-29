import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_gain_sweep


class SweepRoutingTests(unittest.TestCase):
    def test_cartesian_run_retains_trace_and_checks_all_six_axes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            axes = [{"axis": a, "passed": True, "outbound_gain": 0.95,
                     "trace": [{"position_error_m": 0.00001, "ee_pos_root_m": [0.4, 0, 0.3]}]}
                    for a in ("x", "y", "z", "rx", "ry", "rz")]
            class FakeProcess:
                def __init__(self, command, **kwargs):
                    output = Path(command[command.index("--output")+1])
                    report = {"joint_telemetry_summary": {}, "actuator_audit": {},
                              "all_stream_tests_passed": True, "axes": axes,
                              "thresholds": {"gain_range": [0.7, 1.3]},
                              "target_semantics": "current measured pose relative"}
                    (output / "motion_libero-stream.json").write_text(json.dumps(report))
                    self.stdout = io.StringIO("")
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def wait(self): return 0
            argv = ["run_gain_sweep.py", "--isaaclab-root", str(root), "--profiles", "baseline",
                    "--modes", "libero-stream"]
            with patch.object(sys, "argv", argv), patch.object(run_gain_sweep.subprocess, "Popen", FakeProcess):
                self.assertEqual(run_gain_sweep.main(), 0)
            manifest = json.loads((root / "logs/openpi_gain_sweep/latest_gain_sweep.json").read_text())
            run = manifest["runs"][0]
            self.assertEqual(run["axes"], axes)
            self.assertEqual(run["stream_thresholds"]["gain_range"], [0.7, 1.3])
            self.assertEqual(run["cartesian_stream_screen"]["outbound_gain_range"], [0.9, 1.1])
            self.assertTrue(run["cartesian_stream_screen"]["all_passed"])

    def test_named_cases_route_distinct_overrides_and_export_without_collisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "cases.json"
            cases = [{"name": f"case-{kd}", "profile": "kp2p25-kd0p25",
                      "joint_overrides": {"panda_joint2": {"damping": kd}}} for kd in [30, 40]]
            config.write_text(json.dumps(cases))
            launched = []
            class FakeProcess:
                def __init__(self, command, **kwargs):
                    output = Path(command[command.index("--output")+1])
                    overrides = json.loads(Path(command[command.index("--joint-gains-json")+1]).read_text())
                    launched.append((command, overrides))
                    report = {"joint_telemetry_summary": {}, "actuator_audit": overrides,
                              "physics_joint_summary": {}, "joint_steps": []}
                    (output / "motion_joint-step.json").write_text(json.dumps(report))
                    self.stdout = io.StringIO("")
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def wait(self): return 0
            argv = ["run_gain_sweep.py", "--isaaclab-root", str(root), "--gain-cases-json", str(config),
                    "--modes", "joint-step"]
            with patch.object(sys, "argv", argv), patch.object(run_gain_sweep.subprocess, "Popen", FakeProcess):
                self.assertEqual(run_gain_sweep.main(), 0)
            manifest = json.loads((root / "logs/openpi_gain_sweep/latest_gain_sweep.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["profiles"], ["case-30", "case-40"])
            self.assertEqual([o["panda_joint2"]["damping"] for _, o in launched], [30, 40])
            self.assertEqual(len({r["output"] for r in manifest["runs"]}), 2)
            for command, overrides in launched:
                self.assertEqual(command[command.index("--gain-profile")+1], "kp2p25-kd0p25")
                self.assertIn("--record-physics", command)
            self.assertEqual([r["actuator_audit"] for r in manifest["runs"]], [o for _, o in launched])


if __name__ == "__main__":
    unittest.main()
