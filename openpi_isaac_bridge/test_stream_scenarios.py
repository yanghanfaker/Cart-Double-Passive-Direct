import copy
import json
from pathlib import Path
import unittest

from stream_scenarios import expand_stream_cases, validate_stream_scenarios


class StreamScenarioTests(unittest.TestCase):
    def scenarios(self):
        return json.loads(Path(__file__).with_name("cartesian_workspace_scenarios.json").read_text())

    def test_workspace_grid_preserves_gains_and_unique_output_ids(self):
        cases = json.loads(Path(__file__).with_name("cartesian_regression_cases.json").read_text())
        expanded = expand_stream_cases(cases, self.scenarios())
        self.assertEqual(len(expanded), 18)
        self.assertEqual(len({c["name"] for c in expanded}), 18)
        for c in expanded:
            original = next(o for o in cases if o["name"] == c["base_case"])
            self.assertEqual(c["joint_overrides"], original["joint_overrides"])
            self.assertEqual(c["profile"], original["profile"])

    def test_reject_invalid_offsets_scales_and_names_before_launch(self):
        for field, value in [("start_offset_m", [0,0,0.026]), ("start_offset_m", [0,0]),
                             ("start_offset_m", [0,0,float("nan")]), ("amplitude_scale", 3),
                             ("amplitude_scale", float("inf")), ("amplitude_scale", True), ("name", "../bad")]:
            scenarios = self.scenarios()
            scenarios[0][field] = value
            with self.assertRaises(ValueError): validate_stream_scenarios(scenarios)
        scenarios = self.scenarios()
        with self.assertRaises(ValueError): validate_stream_scenarios(scenarios+[copy.deepcopy(scenarios[0])])


if __name__ == "__main__":
    unittest.main()
