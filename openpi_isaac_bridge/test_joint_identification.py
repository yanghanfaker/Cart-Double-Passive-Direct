import unittest

from joint_identification import summarize_joint_step, screen_joint_steps


class JointIdentificationTests(unittest.TestCase):
    def valid_phases(self):
        return [{"joint": f"panda_joint{i}", "step_rad": sign*0.003,
                 "gain_at_50ms": 0.94, "overshoot_pct": 4.0}
                for i in (2, 4, 6) for sign in (-1, 1)]

    def test_screen_rejects_later_overshoot_despite_good_50ms_gain(self):
        phases = self.valid_phases()
        self.assertTrue(screen_joint_steps(phases)["all_passed"])
        phases[0]["overshoot_pct"] = 9.1
        self.assertFalse(screen_joint_steps(phases)["all_passed"])

    def test_screen_requires_both_signs_and_joint6_control(self):
        phases = self.valid_phases()
        self.assertFalse(screen_joint_steps(phases[:-1])["complete"])
        self.assertFalse(screen_joint_steps(phases + [phases[0]])["complete"])
        phases[-1]["gain_at_50ms"] = 0.85
        self.assertFalse(screen_joint_steps(phases)["all_passed"])

    def test_screen_rejects_missing_or_nonfinite_measurements(self):
        for gain in [None, float("nan"), float("inf")]:
            phases = self.valid_phases()
            phases[0]["gain_at_50ms"] = gain
            self.assertFalse(screen_joint_steps(phases)["all_passed"])

    def test_signed_step_and_cross_joint_motion(self):
        for step in (0.003, -0.003):
            rows = [{"command_time_s": 2+i*0.01, "time_s": 2+(i+1)*0.01,
                     "position_target_rad": [step, 0], "position_rad": [step*g, 0.00001*(i+1)]}
                    for i, g in enumerate([0.1, 0.3, 0.6, 0.8, 0.94, 1.02, 1])]
            summary = summarize_joint_step(rows, 0, [0, 0], step, ["j1", "j2"])
            self.assertAlmostEqual(summary["gain_at_50ms"], 0.94)
            self.assertAlmostEqual(summary["overshoot_pct"], 2)
            self.assertAlmostEqual(summary["t90_sustained_s"], 0.05)
            self.assertAlmostEqual(summary["settling_5pct_s"], 0.06)
            self.assertAlmostEqual(summary["coupling_peak_rad"]["j2"], 0.00007)
            self.assertAlmostEqual(summary["final_tracking_error_rad"], 0)

    def test_no_response_has_no_settling_time(self):
        rows = [{"command_time_s": 0, "time_s": 0.01, "position_target_rad": [0.003], "position_rad": [0]}]
        result = summarize_joint_step(rows, 0, [0], 0.003, ["j1"])
        self.assertIsNone(result["t90_sustained_s"])
        self.assertIsNone(result["settling_5pct_s"])
        self.assertIsNone(result["gain_at_50ms"])


if __name__ == "__main__":
    unittest.main()
