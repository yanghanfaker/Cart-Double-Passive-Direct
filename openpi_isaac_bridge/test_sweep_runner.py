import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_gain_sweep


class SweepRoutingTests(unittest.TestCase):
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
