"""CPU-only checks of the deterministic motion probe's magnitude guards."""

import math
from pathlib import Path
import runpy
import unittest

probe = runpy.run_path(str(Path(__file__).with_name("check_cameras.py")))
make_action = probe["translation_probe_action"]
make_rotation = probe["rotation_probe_action"]
make_tracking = probe["tracking_action"]


class MotionProbeTests(unittest.TestCase):
    def test_slow_orientation_caps_rotation_without_reducing_translation_limit(self):
        action=make_tracking([0,0,.1],[1,0,0],.5,rotation_step_deg=.25)
        self.assertAlmostEqual(math.hypot(*action[:3])*.5,.001)
        self.assertAlmostEqual(math.hypot(*action[3:6])*.5,math.radians(.25))
        self.assertEqual(action[6],1)

    def test_tracking_preserves_direction_and_limits_vector_norm(self):
        action = make_tracking([0.003, 0.004, 0], [0, -0.03, -0.04], 0.5)
        self.assertAlmostEqual(math.hypot(*action[:3]) * 0.5, 0.001)
        self.assertAlmostEqual(math.hypot(*action[3:6]) * 0.5, math.radians(0.5))
        self.assertAlmostEqual(action[0] / action[1], 3 / 4)
        self.assertLess(action[4], 0)
        self.assertEqual(make_tracking([0]*3, [0]*3, 0.5), [0]*6 + [1])

    def test_tracking_recomputes_residual_to_reach_fixed_target(self):
        # An idealized lagged servo should converge from either side without
        # accumulating a fresh offset from the original target on each step.
        for initial in (-0.002, 0.009):
            position = initial
            for _ in range(120):
                residual = 0.005 - position
                action = make_tracking([residual, 0, 0], [0]*3, 0.5)
                position += 0.3 * action[0] * 0.5
            self.assertAlmostEqual(position, 0.005, places=9)

    def test_tracking_rejects_invalid_inputs(self):
        for error in ([math.nan, 0, 0], [math.inf, 0, 0], [0, 0]):
            with self.assertRaises(ValueError):
                make_tracking(error, [0]*3, 0.5)
            with self.assertRaises(ValueError):
                make_tracking([0]*3, error, 0.5)
        for scale in (0, -1, math.nan, math.inf, 0.000001):
            with self.assertRaises(ValueError):
                make_tracking([0.005, 0, 0], [0]*3, scale)

    def test_rotation_uses_radians_correct_axis_and_open_gripper(self):
        for index, axis in enumerate(("rx", "ry", "rz"), start=3):
            for scale in (0.25, 0.5, 1.0):
                action = make_rotation(axis, 0.5, scale)
                self.assertAlmostEqual(action[index] * scale, math.pi / 360)
                self.assertEqual(action[:3], [0.0] * 3)
                self.assertEqual(action[6], 1.0)
                self.assertEqual(sum(value != 0 for value in action[:6]), 1)

    def test_rotation_rejects_invalid_or_large_commands(self):
        for step in (0, -0.5, 0.501, math.nan, math.inf):
            with self.assertRaises(ValueError):
                make_rotation("rx", step, 0.5)
        for scale in (0, -1, math.nan, math.inf, 0.00001):
            with self.assertRaises(ValueError):
                make_rotation("rx", 0.5, scale)
        with self.assertRaises(ValueError):
            make_rotation("x", 0.5, 0.5)

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
