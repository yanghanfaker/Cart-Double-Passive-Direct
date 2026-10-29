import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ExportTests(unittest.TestCase):
    def test_jacobian_source_exports_with_physics_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest=self.prepare(root)
            data=json.loads(manifest.read_text())
            data["record_jacobians"]=True
            for run in data["runs"]:
                (Path(run["output"])/"jacobian_implementation.txt").write_text("installed IK source")
            manifest.write_text(json.dumps(data))
            result=self.run_export(manifest,root/"repo")
            self.assertEqual(result.returncode,0,result.stderr)
            dest=root/"repo/diagnostics/gain_sweep"
            self.assertEqual((dest/"test_step-response_jacobian_implementation.txt").read_text(),"installed IK source")

    def prepare_openpi(self, root, previews):
        manifest=self.prepare(root)
        data=json.loads(manifest.read_text())
        data["modes"]=["openpi-step"]
        data["runs"]=[dict(data["runs"][0],mode="openpi-step",openpi_result={"preview_images":previews})]
        manifest.write_text(json.dumps(data))
        return manifest

    def test_policy_input_previews_are_exported_with_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest=self.prepare_openpi(root,["first.png","last.png"])
            for name in ("first.png","last.png"):
                (root/"step-response"/name).write_bytes(b"preview fixture")
            result=self.run_export(manifest,root/"repo")
            self.assertEqual(result.returncode,0,result.stderr)
            dest=root/"repo/diagnostics/gain_sweep"
            self.assertEqual(len(list(dest.iterdir())),6)
            self.assertEqual((dest/"test_openpi-step_last.png").read_bytes(),b"preview fixture")

    def test_unsafe_policy_preview_path_does_not_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest=self.prepare_openpi(root,["../outside.png"])
            result=self.run_export(manifest,root/"repo")
            self.assertNotEqual(result.returncode,0)
            self.assertFalse((root/"repo/diagnostics").exists())

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

    def test_single_joint_step_and_grasp_mode_exports(self):
        for mode in ("joint-step", "grasp"):
            with self.subTest(mode=mode):
                self.check_single_mode_exports(mode)

    def check_single_mode_exports(self, mode):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.prepare(root)
            data = json.loads(manifest.read_text())
            data["modes"] = [mode]
            data["runs"] = [dict(data["runs"][0], mode=mode)]
            manifest.write_text(json.dumps(data))
            result = self.run_export(manifest, root / "repo")
            self.assertEqual(result.returncode, 0, result.stderr)
            dest = root / "repo" / "diagnostics" / "gain_sweep"
            self.assertEqual(len(list(dest.iterdir())), 4)
            self.assertTrue((dest / f"test_{mode}_physics_joint_trace.jsonl.gz").is_file())


if __name__ == "__main__":
    unittest.main()
