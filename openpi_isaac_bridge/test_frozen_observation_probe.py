import copy
import math
import unittest
from frozen_observation_probe import analyze_chunk, select_snapshots


def manifest_fixture():
    runs=[]
    for i in range(3):
        def row(step):
            return {"step":step,"state_frame":"world","state_observed":[0,0,0,0,0,0,.04,-.04],
                "state_sent":None,"image_files":[f"openpi_input_{step-1:03d}_external.png",f"openpi_input_{step-1:03d}_wrist.png"],
                "object_pos_world_m_for_diagnostic_only":[0,0,1],"observation_used_for_inference":False}
        runs.append({"status":"completed","mode":"openpi-step","profile":f"repeat{i+1}","output":f"/fixture/{i}",
            "openpi_setup":{"extended_approach":True,"prompt":"pick up the cube"},
            "openpi_result":{"stopped_by_monitor":True,"trace":[row(1),row(280+20*i)]}})
    return {"status":"completed","runs":runs}


class FrozenObservationTests(unittest.TestCase):
    def test_selects_initial_and_three_last_observations_without_mutation(self):
        source=manifest_fixture(); before=copy.deepcopy(source)
        snapshots=select_snapshots(source)
        self.assertEqual([s["source_step"] for s in snapshots],[1,280,300,320])
        self.assertEqual(len(snapshots),4)
        self.assertEqual(snapshots[-1]["state"],source["runs"][-1]["openpi_result"]["trace"][-1]["state_observed"])
        self.assertEqual(source,before)

    def test_rejects_incomplete_source(self):
        source=manifest_fixture(); source["runs"].pop()
        with self.assertRaises(ValueError):select_snapshots(source)

    def test_rejects_unsafe_images_and_invalid_state(self):
        for field,value in (("image_files",["../external.png","wrist.png"]),("state_observed",[float("nan")]*8),("state_frame","root")):
            source=manifest_fixture();source["runs"][0]["openpi_result"]["trace"][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):select_snapshots(source)

    def test_separates_first_five_from_whole_chunk(self):
        chunk=[[0,0,.1,0,0,0,1] for _ in range(5)]+[[0,0,-.1,0,0,0,-1] for _ in range(5)]
        source=copy.deepcopy(chunk); result=analyze_chunk(chunk,[0]*8,[0,0,1])
        self.assertAlmostEqual(result["first5"]["decoded_local_approach_projection_sum_mm"],25)
        self.assertAlmostEqual(result["first5"]["bounded_local_approach_projection_sum_mm"],5)
        self.assertAlmostEqual(result["whole_chunk"]["bounded_local_approach_projection_sum_mm"],0)
        self.assertEqual(result["first5"]["close_requests"],5)
        self.assertTrue(all(not r["executed"] for r in result["actions"]))
        self.assertEqual(chunk,source)

    def test_records_separate_translation_and_rotation_retention(self):
        result=analyze_chunk([[1,0,0,0,0,1,1]]*5,[0]*8,[0,0,1])
        row=result["actions"][0]
        self.assertAlmostEqual(row["translation_retained_fraction"],.02)
        self.assertAlmostEqual(row["rotation_retained_fraction"],math.radians(.25)/.5)
        self.assertNotEqual(row["translation_retained_fraction"],row["rotation_retained_fraction"])

    def test_zero_motion_is_not_amplified(self):
        result=analyze_chunk([[0]*7]*5,[0]*8,[0,0,1])
        self.assertEqual(result["first5"]["bounded_translation_sum_mm"],[0,0,0])
        self.assertEqual(result["first5"]["translation_clamped"],0)

    def test_rejects_bad_future_action_and_short_chunks(self):
        for chunk in ([[0]*7]*4,[[0]*7]*5+[[0,0,0,0,0,0,float("inf")]]):
            with self.assertRaises(ValueError):analyze_chunk(chunk,[0]*8,[0,0,1])


if __name__=="__main__":unittest.main()
