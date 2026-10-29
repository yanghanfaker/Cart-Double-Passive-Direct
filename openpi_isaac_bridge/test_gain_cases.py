import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from gain_cases import validate_gain_cases
from gain_probe import configure_gain_profile


class GainCaseTests(unittest.TestCase):
    def cases(self):
        return json.loads(Path(__file__).with_name("joint_gain_cases.json").read_text())

    def refine_cases(self):
        return json.loads(Path(__file__).with_name("joint_gain_refine_cases.json").read_text())

    def test_candidates_keep_other_joints_and_limits_unchanged(self):
        class Implicit:
            is_implicit_model = True
        self.assertEqual(len(self.cases()), 6)
        self.assertEqual(len(self.refine_cases()), 7)
        regression = json.loads(Path(__file__).with_name("cartesian_regression_cases.json").read_text())
        self.assertEqual(len(regression), 3)
        grasp = json.loads(Path(__file__).with_name("grasp_validation_cases.json").read_text())
        self.assertEqual(len(grasp), 3)
        robustness = json.loads(Path(__file__).with_name("grasp_robustness_cases.json").read_text())
        self.assertEqual(len(robustness),2)
        policy = json.loads(Path(__file__).with_name("openpi_probe_cases.json").read_text())
        self.assertEqual(len(policy),1)
        cases = validate_gain_cases(self.cases()) + validate_gain_cases(self.refine_cases()) + validate_gain_cases(regression) + validate_gain_cases(grasp) + validate_gain_cases(robustness) + validate_gain_cases(policy)
        for case in cases:
            arm = SimpleNamespace(class_type=Implicit, joint_names_expr=["panda_joint[1-7]"],
                                  stiffness=400, damping=80, joint_effort_limit=87)
            configure_gain_profile({"arm": arm}, case["profile"], case["joint_overrides"])
            for i in [1, 3, 5, 6, 7]:
                joint = f"panda_joint{i}"
                self.assertEqual(arm.stiffness[joint] if isinstance(arm.stiffness, dict) else arm.stiffness, 900)
                self.assertEqual(arm.damping[joint] if isinstance(arm.damping, dict) else arm.damping, 20)
            self.assertEqual(arm.joint_effort_limit, 87)
            for joint, gains in case["joint_overrides"].items():
                self.assertEqual(arm.stiffness[joint], gains["stiffness"])
                self.assertEqual(arm.damping[joint], gains["damping"])

    def test_invalid_and_duplicate_names_rejected(self):
        for name in ["../escape", "a/b", "..", "", "A"]:
            cases = self.cases()
            cases[0]["name"] = name
            with self.assertRaises(ValueError):
                validate_gain_cases(cases)
        cases = self.cases()
        cases.append(copy.deepcopy(cases[0]))
        with self.assertRaises(ValueError):
            validate_gain_cases(cases)

    def test_invalid_joint_gain_rejected_before_launch(self):
        cases = self.cases()
        cases[0]["joint_overrides"] = {"panda_finger_joint1": {"damping": 20}}
        with self.assertRaises(ValueError):
            validate_gain_cases(cases)


if __name__ == "__main__":
    unittest.main()
