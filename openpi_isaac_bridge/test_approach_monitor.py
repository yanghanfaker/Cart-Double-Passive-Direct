import math
import unittest

from approach_monitor import ApproachMonitor
from openpi_step_probe import run_openpi_steps


class ApproachTests(unittest.TestCase):
    def run_probe(self, position_at_step, steps=60, initial=.36, extended=False, replan=5):
        monitor=ApproachMonitor([0,0,initial],[0,0,0])
        report=run_openpi_steps(lambda i:({}, {"image_files":[f"image{i}.png"]}),
            lambda o:{"actions":[[0,0,-.01,0,0,0,-1]]*10},
            lambda mapping,i:{"state_after_world":[0,0,position_at_step(i+1)]+[0]*5,
                              "object_pos_world_m_for_diagnostic_only":[0,0,0]},
            steps=steps,replan_steps=replan,approach_monitor=monitor,extended_approach=extended,announce=lambda _:None)
        return report

    def test_chunk10_long_budget_uses_40_requests_and_same_twenty_segments(self):
        result=self.run_probe(lambda i:1-.0005*i,steps=400,initial=1,extended=True,replan=10)
        self.assertTrue(result["probe_completed"])
        self.assertEqual(result["inference_requests"],40)
        self.assertEqual(result["completed_steps"],400)
        self.assertEqual(len(result["approach_summary"]["segments"]),20)

    def test_chunk10_monitor_stops_inside_block_without_finishing_it(self):
        result=self.run_probe(lambda i:.086-.001*i,initial=.086,extended=True,replan=10)
        self.assertEqual(result["completed_steps"],6)
        self.assertEqual(result["trace"][-1]["chunk_action_index"],5)
        self.assertEqual(result["inference_requests"],1)
        self.assertEqual(result["termination_reason"],"near_object_boundary")


    def test_three_progress_segments_and_exact_60_step_budget(self):
        result=self.run_probe(lambda i:.36-.0005*i)
        self.assertTrue(result["probe_completed"])
        self.assertEqual(result["completed_steps"],60)
        self.assertEqual(result["inference_requests"],12)
        self.assertEqual(len(result["approach_summary"]["segments"]),3)
        for segment in result["approach_summary"]["segments"]:
            self.assertAlmostEqual(segment["distance_reduction_m"],.01)
        self.assertEqual(result["preview_images"],["image0.png","image19.png","image39.png","image59.png"])

    def test_explicit_long_mode_has_exact_400_step_and_80_request_budget(self):
        result=self.run_probe(lambda i:1-.0005*i,steps=400,initial=1,extended=True)
        self.assertTrue(result["probe_completed"])
        self.assertEqual(result["completed_steps"],400)
        self.assertEqual(result["inference_requests"],80)
        self.assertEqual(len(result["approach_summary"]["segments"]),20)

    def test_long_mode_near_boundary_stops_before_next_action(self):
        result=self.run_probe(lambda i:.3604-.0008*i,steps=400,initial=.3604,extended=True)
        self.assertEqual(result["completed_steps"],351)
        self.assertEqual(result["inference_requests"],71)
        self.assertEqual(result["termination_reason"],"near_object_boundary")
        self.assertTrue(result["approach_summary"]["near_boundary_reached"])
        self.assertEqual(result["trace"][-1]["chunk_action_index"],0)

    def test_progress_stop_still_works_after_original_60_steps(self):
        result=self.run_probe(lambda i:.36-.0005*min(i,60),steps=400,extended=True)
        self.assertEqual(result["completed_steps"],80)
        self.assertEqual(result["termination_reason"],"segment_progress_less_than_2mm")

    def test_long_budget_rejects_missing_monitor_or_more_than_500_steps(self):
        with self.assertRaises(ValueError):run_openpi_steps(None,None,None,steps=400,extended_approach=True)
        with self.assertRaises(ValueError):self.run_probe(lambda i:.36,steps=501,extended=True)

    def test_chunk10_500_step_budget_uses_50_requests_and_same_checkpoints(self):
        result=self.run_probe(lambda i:1-.0005*i,steps=500,initial=1,extended=True,replan=10)
        self.assertTrue(result["probe_completed"])
        self.assertEqual(result["completed_steps"],500)
        self.assertEqual(result["inference_requests"],50)
        self.assertEqual(len(result["approach_summary"]["segments"]),25)

    def test_500_step_budget_does_not_disable_early_stall_stop(self):
        result=self.run_probe(lambda i:.36-.0005*min(i,60),steps=500,extended=True,replan=10)
        self.assertEqual(result["completed_steps"],80)
        self.assertEqual(result["termination_reason"],"segment_progress_less_than_2mm")

    def test_stall_stops_after_executed_checkpoint(self):
        result=self.run_probe(lambda i:.36)
        self.assertFalse(result["probe_completed"])
        self.assertTrue(result["stopped_by_monitor"])
        self.assertEqual(result["completed_steps"],20)
        self.assertEqual(result["inference_requests"],4)
        self.assertTrue(result["trace"][-1]["executed"])
        self.assertEqual(result["termination_reason"],"segment_progress_less_than_2mm")

    def test_distance_increase_stops_inside_cached_block(self):
        result=self.run_probe(lambda i:.36+.003*i)
        self.assertEqual(result["completed_steps"],2)
        self.assertEqual(result["inference_requests"],1)
        self.assertEqual(result["termination_reason"],"distance_increased_more_than_5mm")

    def test_near_boundary_stops_without_closing_gripper(self):
        result=self.run_probe(lambda i:.079,initial=.081)
        self.assertEqual(result["completed_steps"],1)
        self.assertEqual(result["termination_reason"],"near_object_boundary")
        self.assertEqual(result["trace"][0]["action_mapping"]["applied_isaac_gripper"],1)

    def test_initial_near_boundary_prevents_inference_and_execution(self):
        result=self.run_probe(lambda i:self.fail("must not execute"),initial=.079)
        self.assertEqual(result["completed_steps"],0)
        self.assertEqual(result["inference_requests"],0)
        self.assertEqual(result["preview_images"],[])

    def test_invalid_state_and_sample_order_are_rejected(self):
        with self.assertRaises(ValueError):ApproachMonitor([0,0,float("nan")],[0,0,0])
        monitor=ApproachMonitor([0,0,.36],[0,0,0])
        with self.assertRaises(ValueError):monitor.update({"step":2})

    def test_60_steps_requires_explicit_monitor_and_chunk5(self):
        with self.assertRaises(ValueError):run_openpi_steps(None,None,None,steps=60)
        with self.assertRaises(ValueError):run_openpi_steps(None,None,None,steps=61,approach_monitor=ApproachMonitor([0,0,.36],[0,0,0]))
        with self.assertRaises(ValueError):run_openpi_steps(None,None,None,approach_monitor=ApproachMonitor([0,0,.36],[0,0,0]),replan_steps=1)


if __name__=="__main__":unittest.main()
