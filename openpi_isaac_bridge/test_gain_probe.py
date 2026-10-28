import math
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from gain_probe import configure_gain_profile, scaled_gain


class Implicit:
    is_implicit_model = True


class GainProbeTests(unittest.TestCase):
    def test_recorded_data_candidate_changes_only_joint2_and_joint4_damping(self):
        overrides = json.loads(Path(__file__).with_name("joint_gains_j2_j4_d40.json").read_text())
        arm = self.actuator("panda_joint[1-7]", 400, 80)
        configure_gain_profile({"arm": arm}, "kp2p25-kd0p25", overrides)
        self.assertEqual(set(arm.stiffness.values()), {900})
        self.assertEqual([arm.damping[f"panda_joint{i}"] for i in range(1, 8)], [20, 40, 20, 40, 20, 20, 20])
        self.assertEqual((arm.effort_limit_sim, arm.velocity_limit_sim), (87, 2))

    def test_per_joint_override_changes_only_selected_gain_after_profile(self):
        arm = self.actuator("panda_joint[1-7]", 400, 80)
        hand = self.actuator("panda_finger_joint.*", None, None)
        report = configure_gain_profile({"arm": arm, "hand": hand}, "kp2p25-kd0p375",
                                        {"panda_joint2": {"damping": 35}})
        self.assertEqual(arm.stiffness, {f"panda_joint{i}": 900 for i in range(1, 8)})
        self.assertEqual(arm.damping["panda_joint1"], 30)
        self.assertEqual(arm.damping["panda_joint2"], 35)
        self.assertEqual((hand.stiffness, hand.damping), (None, None))
        self.assertEqual(report["groups"][0]["experiment_damping"], arm.damping)
        self.assertEqual(arm.effort_limit_sim, 87)

    def test_per_joint_override_rejects_fingers_invalid_fields_and_invalid_values(self):
        for override in ({"panda_finger_joint1": {"damping": 35}}, {"panda_joint2": {"force": 5}},
                         {"panda_joint2": {"damping": 0}}, {"panda_joint2": {"stiffness": math.nan}}):
            arm = self.actuator("panda_joint[1-7]", 400, 80)
            with self.assertRaises(ValueError):
                configure_gain_profile({"arm": arm}, "kp2p25-kd0p375", override)
            self.assertEqual((arm.stiffness, arm.damping), (400, 80))

    def test_faster_profiles_start_from_original_gains(self):
        for profile, expected in (
            ("kp1p5-kd0p375", (600, 30)), ("kp1p5-kd0p25", (600, 20)),
            ("kp2p25-kd0p25", (900, 20)), ("kp3-kd0p25", (1200, 20)),
            ("kp2p25-kd0p375", (900, 30)), ("kp2p25-kd0p5", (900, 40)),
        ):
            with self.subTest(profile=profile):
                arm = self.actuator("panda_joint[1-7]", 400, 80)
                configure_gain_profile({"arm": arm}, profile)
                self.assertEqual((arm.stiffness, arm.damping), expected)
                self.assertEqual((arm.effort_limit_sim, arm.velocity_limit_sim), (87, 2))

    def test_900_stiffness_comparison_keeps_hand_and_source_config(self):
        for profile, damping in (("kp2p25-kd0p25", 20), ("kp2p25-kd0p375", 30), ("kp2p25-kd0p5", 40)):
            with self.subTest(profile=profile):
                arm = self.actuator("panda_joint[1-7]", 400, 80)
                hand = self.actuator("panda_finger_joint.*", None, None)
                report = configure_gain_profile({"arm": arm, "hand": hand}, profile)
                self.assertEqual((arm.stiffness, arm.damping), (900, damping))
                self.assertEqual((hand.stiffness, hand.damping), (None, None))
                self.assertEqual(report["groups"][0]["original_stiffness"], 400)
                self.assertEqual(report["groups"][0]["original_damping"], 80)

    def test_damping_sweep_holds_stiffness_and_keeps_hand_unchanged(self):
        for profile, damping in (("kp1p5", 80), ("kp1p5-kd0p75", 60), ("kp1p5-kd0p5", 40)):
            with self.subTest(profile=profile):
                arm = self.actuator("panda_joint[1-7]", 400, 80)
                hand = self.actuator("panda_finger_joint.*", 2000, 100)
                configure_gain_profile({"arm": arm, "hand": hand}, profile)
                self.assertEqual((arm.stiffness, arm.damping), (600, damping))
                self.assertEqual((hand.stiffness, hand.damping), (2000, 100))
                self.assertEqual((arm.effort_limit_sim, arm.velocity_limit_sim), (87, 2))

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
