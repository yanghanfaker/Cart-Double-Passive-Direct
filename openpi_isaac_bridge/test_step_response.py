import unittest

from step_response import summarize_response


class StepResponseTests(unittest.TestCase):
    def test_50ms_band_checks_reject_overshoot(self):
        for gain, band10, band5 in ((0.53, False, False), (0.92, True, False),
                                    (0.98, True, True), (1.2, False, False)):
            with self.subTest(gain=gain):
                result = summarize_response(self.trace([gain, 1.0]))
                self.assertEqual(result["gain_at_50ms"], gain)
                self.assertEqual(result["within_10pct_at_50ms"], band10)
                self.assertEqual(result["within_5pct_at_50ms"], band5)

    def test_missing_50ms_sample_is_marked_unavailable(self):
        trace = self.trace([1.0])
        trace[0]["time_s"] = 0.1
        result = summarize_response(trace)
        self.assertIsNone(result["gain_at_50ms"])
        self.assertIsNone(result["within_5pct_at_50ms"])

    def trace(self, gains):
        return [{"step": i + 1, "time_s": (i + 1) * 0.05, "gain": gain,
                 "position_error_m": abs(1 - gain) * 0.001, "rotation_error_deg": 0}
                for i, gain in enumerate(gains)]

    def test_sustained_crossing_ignores_early_overshoot_and_dip(self):
        result = summarize_response(self.trace([0.2, 1.1, 0.8, 0.92, 0.97, 1.0]))
        self.assertAlmostEqual(result["t90_sustained_s"], 0.2)
        self.assertAlmostEqual(result["settling_5pct_s"], 0.25)
        self.assertEqual(result["peak_gain"], 1.1)
        self.assertAlmostEqual(result["overshoot_pct"], 10)
        self.assertAlmostEqual(result["tail_gain_peak_to_peak"], 0.9)

    def test_nonconvergence_is_not_reported_as_success(self):
        result = summarize_response(self.trace([0.2] * 60))
        self.assertIsNone(result["t90_sustained_s"])
        self.assertIsNone(result["settling_5pct_s"])
        self.assertEqual([r["step"] for r in result["checkpoints"]], [1, 2, 5, 10, 20, 60])

    def test_empty_trace_rejected(self):
        with self.assertRaises(ValueError):
            summarize_response([])


if __name__ == "__main__":
    unittest.main()
