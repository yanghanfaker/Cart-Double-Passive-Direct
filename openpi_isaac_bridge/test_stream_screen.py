import unittest

from stream_screen import screen_cartesian_stream


class StreamScreenTests(unittest.TestCase):
    def axes(self):
        return [{"axis": a, "outbound_gain": 0.95, "passed": True}
                for a in ("x", "y", "z", "rx", "ry", "rz")]

    def test_stricter_gain_does_not_replace_other_legacy_checks(self):
        axes = self.axes()
        self.assertTrue(screen_cartesian_stream(axes)["all_passed"])
        axes[2]["passed"] = False
        self.assertFalse(screen_cartesian_stream(axes)["all_passed"])

    def test_missing_duplicate_and_nonfinite_axis_fail(self):
        axes = self.axes()
        self.assertFalse(screen_cartesian_stream(axes[:-1])["complete"])
        self.assertFalse(screen_cartesian_stream(axes[:-1]+[axes[0]])["complete"])
        for gain in (None, float("nan"), float("inf"), True):
            axes = self.axes()
            axes[0]["outbound_gain"] = gain
            self.assertFalse(screen_cartesian_stream(axes)["all_passed"])

    def test_legacy_gain_pass_can_fail_stricter_project_screen(self):
        axes = self.axes()
        axes[0]["outbound_gain"] = 0.85
        self.assertFalse(screen_cartesian_stream(axes)["all_passed"])
        self.assertTrue(axes[0]["passed"])


if __name__ == "__main__":
    unittest.main()
