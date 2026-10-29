import json
from pathlib import Path
import unittest
from grasp_scenarios import expand_grasp_cases, validate_grasp_scenarios


class GraspScenarioTests(unittest.TestCase):
    def scenarios(self):
        return json.loads(Path(__file__).with_name("grasp_robustness_scenarios.json").read_text())

    def test_two_gains_eight_conditions_remain_distinct(self):
        cases=json.loads(Path(__file__).with_name("grasp_robustness_cases.json").read_text())
        expanded=expand_grasp_cases(cases,self.scenarios())
        self.assertEqual(len(expanded),16)
        self.assertEqual(len({c["name"] for c in expanded}),16)
        for c in expanded:
            original=next(o for o in cases if o["name"]==c["base_case"])
            self.assertEqual(c["joint_overrides"],original["joint_overrides"])
        self.assertEqual({s["seed"] for s in self.scenarios()},{0,1})

    def test_pose_rate_seed_and_path_bounds_checked_before_launch(self):
        for key,value in [("object_offset_xy_m",[.04,.04]),("object_yaw_deg",31),
                          ("object_yaw_deg",float("nan")),("orient_step_deg",0),
                          ("seed",True),("seed",-1),("name","../bad")]:
            scenarios=self.scenarios();scenarios[0][key]=value
            with self.assertRaises(ValueError):validate_grasp_scenarios(scenarios)
        scenarios=self.scenarios()
        with self.assertRaises(ValueError):validate_grasp_scenarios(scenarios+[scenarios[0]])

    def test_home_speed_comparison_is_matched_on_pose_and_seed(self):
        scenarios=self.scenarios()
        for repeat in (0,1):
            a=next(s for s in scenarios if s["name"]==f"home-fast-r{repeat}")
            b=next(s for s in scenarios if s["name"]==f"home-slow-r{repeat}")
            for key in ("object_offset_xy_m","object_yaw_deg","seed"):
                self.assertEqual(a[key],b[key])
            self.assertEqual(a["orient_step_deg"],.5)
            self.assertEqual(b["orient_step_deg"],.25)


if __name__=="__main__":unittest.main()
