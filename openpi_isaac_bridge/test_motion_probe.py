"""CPU-only checks of the deterministic motion probe's magnitude guards."""

import math
from pathlib import Path
import runpy
import unittest

make_action = runpy.run_path(str(Path(__file__).with_name("check_cameras.py")))["translation_probe_action"]


class MotionProbeTests(unittest.TestCase):
    def test_ik_scale_is_compensated_and_gripper_stays_open(self):
        for i, axis in enumerate(("x", "y", "z")):
            for scale in (0.25, 0.5, 1.0):
                action = make_action(axis, 0.001, scale)
                self.assertAlmostEqual(action[i] * scale, 0.001)
                self.assertEqual(action[6], 1.0)
                self.assertEqual(sum(value != 0 for value in action[:6]), 1)

    def test_rejects_invalid_or_large_motion(self):
        for step in (0, -0.001, 0.01, math.nan, math.inf):
            with self.assertRaises(ValueError):
                make_action("x", step, 0.5)
        for scale in (0, -1, math.nan, math.inf, 0.00001):
            with self.assertRaises(ValueError):
                make_action("x", 0.001, scale)
        with self.assertRaises(ValueError):
            make_action("roll", 0.001, 0.5)


if __name__ == "__main__":
    unittest.main()
