"""CPU checks of the versioned controller mapping, independent of Isaac Sim."""

import math
import unittest

from libero_mapping import decode_libero_action, mapping_test_plan


class LiberoMappingTests(unittest.TestCase):
    def test_controller_scale_and_axis_order(self):
        result = decode_libero_action([0.1, -0.1, 0.02, 0.04, -0.04, 0.02, -1])
        for actual, expected in zip(result["delta_world_m"], [0.005, -0.005, 0.001]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(result["delta_world_rad"], [0.02, -0.02, 0.01])
        self.assertEqual(result["isaac_gripper"], 1)

    def test_gripper_sign_and_zero_hold(self):
        self.assertEqual(decode_libero_action([0]*6 + [1])["isaac_gripper"], -1)
        for previous in (-1, 1):
            self.assertEqual(decode_libero_action([0]*7, previous)["isaac_gripper"], previous)

    def test_clipping_is_reported_not_hidden(self):
        result = decode_libero_action([2, -2, 0, 0, 0, 0, 1.003])
        self.assertEqual(result["clipped_indices"], [0, 1, 6])
        self.assertEqual(result["delta_world_m"], [0.05, -0.05, 0])

    def test_invalid_input_is_rejected(self):
        for command in ([0]*6, [0]*8, [math.nan]*7, [math.inf]*7):
            with self.assertRaises(ValueError):
                decode_libero_action(command)
        with self.assertRaises(ValueError):
            decode_libero_action([0]*7, 0)

    def test_signed_plan_is_small_and_returns_each_axis(self):
        plan = mapping_test_plan()
        self.assertEqual(len(plan), 15)
        for index in range(0, 12, 2):
            positive = decode_libero_action(plan[index][1])
            negative = decode_libero_action(plan[index+1][1])
            delta = positive["delta_world_m"] + positive["delta_world_rad"]
            reverse = negative["delta_world_m"] + negative["delta_world_rad"]
            self.assertEqual(delta, [-value for value in reverse])
            expected = 0.005 if index < 6 else math.radians(2)
            self.assertAlmostEqual(math.hypot(*delta), expected)


if __name__ == "__main__":
    unittest.main()
