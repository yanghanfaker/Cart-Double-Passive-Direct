import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ExportTests(unittest.TestCase):
    def prepare(self, root):
        runs = []
        for mode in ("step-response", "libero-stream"):
            folder = root / mode
            folder.mkdir()
            (folder / "actuator_audit.json").write_text("{}")
            (folder / "physics_joint_summary.json").write_text("{}")
            (folder / "physics_joint_trace.jsonl").write_text('{"time_s":0.01}\n')
            runs.append({"profile": "test", "mode": mode, "status": "completed", "output": str(folder)})
        manifest = root / "latest.json"
        manifest.write_text(json.dumps({"status": "completed", "profiles": ["test"], "record_physics": True, "runs": runs}))
        return manifest

    def run_export(self, manifest, repo):
        return subprocess.run([sys.executable, str(Path(__file__).with_name("export_gain_results.py")),
            "--latest", str(manifest), "--repo-root", str(repo), "--include-physics-traces"], capture_output=True, text=True)

    def test_compressed_targets_and_summaries_export_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.run_export(self.prepare(root), root / "repo")
            self.assertEqual(result.returncode, 0, result.stderr)
            dest = root / "repo" / "diagnostics" / "gain_sweep"
            self.assertEqual(len(list(dest.iterdir())), 7)
            trace = gzip.decompress((dest / "test_step-response_physics_joint_trace.jsonl.gz").read_bytes())
            self.assertEqual(json.loads(trace)["time_s"], 0.01)

    def test_missing_artifact_does_not_export_partial_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.prepare(root)
            (root / "libero-stream" / "physics_joint_trace.jsonl").unlink()
            result = self.run_export(manifest, root / "repo")
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "repo" / "diagnostics").exists())

    def test_single_joint_step_mode_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.prepare(root)
            data = json.loads(manifest.read_text())
            data["modes"] = ["joint-step"]
            data["runs"] = [dict(data["runs"][0], mode="joint-step")]
            manifest.write_text(json.dumps(data))
            result = self.run_export(manifest, root / "repo")
            self.assertEqual(result.returncode, 0, result.stderr)
            dest = root / "repo" / "diagnostics" / "gain_sweep"
            self.assertEqual(len(list(dest.iterdir())), 4)
            self.assertTrue((dest / "test_joint-step_physics_joint_trace.jsonl.gz").is_file())


if __name__ == "__main__":
    unittest.main()
