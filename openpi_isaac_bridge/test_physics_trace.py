import json
from pathlib import Path
import tempfile
import unittest

from physics_trace import PhysicsJointRecorder


class Scene:
    def __init__(self):
        self.target = 0.0
        self.position = 0.0

    def write_data_to_sim(self):
        self.target += 0.001

    def update(self, dt):
        self.position += 0.0001


class PhysicsTraceTests(unittest.TestCase):
    def recorder(self, scene, path):
        return PhysicsJointRecorder(scene, 0.01, 0.05,
            lambda: {"position_target_rad": [scene.target], "position_before_rad": [scene.position]},
            lambda: {"position_rad": [scene.position], "velocity_rad_s": [0.01], "estimated_effort_nm": None},
            path, {"joint_names": ["joint1"]})

    def test_pairs_pre_step_command_and_post_step_state_and_restores_methods(self):
        scene = Scene()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            recorder = self.recorder(scene, path)
            try:
                # Warmup/reset updates outside the armed interval are ignored.
                scene.update(0.01)
                recorder.begin_control({"case": "ik/x", "phase": "step"})
                for _ in range(5):
                    scene.write_data_to_sim()
                    scene.update(0.01)
                recorder.end_control()
                summary = recorder.summary()
                self.assertEqual(summary["recorded_substeps"], 5)
                self.assertAlmostEqual(recorder.rows[0]["position_target_rad"][0], 0.001)
                self.assertAlmostEqual(recorder.rows[0]["position_before_rad"][0], 0.0001)
                self.assertAlmostEqual(recorder.rows[0]["position_rad"][0], 0.0002)
                self.assertAlmostEqual(recorder.rows[-1]["time_s"], 0.05)
                self.assertEqual(summary["cases"][0]["joints"][0]["peak_abs_estimated_effort_nm"], None)
            finally:
                recorder.close()
            self.assertNotIn("update", scene.__dict__)
            self.assertEqual(len(path.read_text().splitlines()), 5)
            json.loads(path.read_text().splitlines()[0])

    def test_rejects_missing_substeps(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene = Scene()
            recorder = self.recorder(scene, Path(tmp) / "trace.jsonl")
            try:
                recorder.begin_control({})
                scene.write_data_to_sim()
                scene.update(0.01)
                with self.assertRaises(RuntimeError):
                    recorder.end_control()
            finally:
                recorder.close()

    def test_rejects_decimated_update_instead_of_mislabelling_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene = Scene()
            recorder = self.recorder(scene, Path(tmp) / "trace.jsonl")
            try:
                recorder.begin_control({})
                scene.write_data_to_sim()
                with self.assertRaises(RuntimeError):
                    scene.update(0.05)
            finally:
                recorder.close()


if __name__ == "__main__":
    unittest.main()
