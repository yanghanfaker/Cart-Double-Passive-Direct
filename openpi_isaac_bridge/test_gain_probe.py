import math
from types import SimpleNamespace
import unittest

from gain_probe import configure_gain_profile, scaled_gain


class Implicit:
    is_implicit_model = True


class GainProbeTests(unittest.TestCase):
    def actuator(self, pattern, kp=80, kd=4):
        return SimpleNamespace(joint_names_expr=[pattern], stiffness=kp, damping=kd, class_type=Implicit,
                               effort_limit_sim=87, velocity_limit_sim=2)

    def test_changes_only_arm_and_preserves_limits(self):
        arm = self.actuator("panda_joint[1-7]", {".*": 80})
        hand = self.actuator("panda_finger_joint.*", 2000, 100)
        report = configure_gain_profile({"arm": arm, "hand": hand}, "kp1p5-damped")
        self.assertEqual(arm.stiffness, {".*": 120})
        self.assertAlmostEqual(arm.damping, 4 * math.sqrt(1.5))
        self.assertEqual((hand.stiffness, hand.damping), (2000, 100))
        self.assertEqual((arm.effort_limit_sim, arm.velocity_limit_sim), (87, 2))
        self.assertEqual(report["groups"][0]["original_stiffness"], {".*": 80})

    def test_baseline_is_unchanged(self):
        arm = self.actuator("panda_joint.*")
        configure_gain_profile({"arm": arm}, "baseline")
        self.assertEqual((arm.stiffness, arm.damping), (80, 4))

    def test_invalid_or_mixed_configuration_is_not_partially_modified(self):
        first = self.actuator("panda_joint[1-4]")
        second = self.actuator("panda_joint[5-7]", None)
        with self.assertRaises(ValueError):
            configure_gain_profile({"a": first, "b": second}, "kp1p5")
        self.assertEqual(first.stiffness, 80)
        with self.assertRaises(ValueError):
            configure_gain_profile({"mixed": self.actuator("panda_.*")}, "kp1p5")
        with self.assertRaises(ValueError):
            configure_gain_profile({"missing": first}, "kp1p5")

    def test_invalid_gain_is_rejected(self):
        for value in (None, -1, math.nan, math.inf, True):
            with self.assertRaises(ValueError):
                scaled_gain(value, 1.5)


if __name__ == "__main__":
    unittest.main()
