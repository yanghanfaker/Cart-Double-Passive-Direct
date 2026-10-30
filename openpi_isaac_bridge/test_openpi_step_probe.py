import math
import unittest
from openpi_step_probe import bounded_openpi_action, validate_action_chunk, preview_images, run_openpi_steps, libero_state_vector, read_server_metadata


class OpenpiStepTests(unittest.TestCase):
    def test_chunk10_executes_all_actions_before_fresh_inference(self):
        queried=[];executed=[]
        result=run_openpi_steps(lambda i:({"step":i},{"image_files":[],"state_sent":[i]}),
            lambda obs:queried.append(obs["step"]) or {"actions":[[i*.001,0,0,0,0,0,1] for i in range(10)]},
            lambda mapped,i:executed.append(mapped["input_action"][0]) or {},
            steps=20,replan_steps=10,announce=lambda _:None)
        self.assertTrue(result["probe_completed"])
        self.assertEqual(queried,[0,10])
        self.assertEqual(executed,[i*.001 for i in range(10)]*2)
        self.assertEqual([r["chunk_action_index"] for r in result["trace"]],list(range(10))*2)
        self.assertEqual(result["trace"][10]["state_sent"],[10])
        self.assertIsNone(result["trace"][9]["state_sent"])

    def test_chunk10_rejects_short_block_before_any_execution(self):
        result=run_openpi_steps(lambda i:({}, {"image_files":[]}),lambda obs:{"actions":[[0]*7]*9},
            lambda *a:self.fail("no partial block"),replan_steps=10,announce=lambda _:None)
        self.assertEqual(result["completed_steps"],0)
        self.assertIn("shorter",result["failure_reason"])

    def test_chunk5_executes_ordered_prefix_and_queries_only_after_five_steps(self):
        state=[0];queries=[];executed=[]
        def observe(index):
            return {"state":state[0]}, {"image_files":[f"step{index}.png"],"state_sent":[state[0]]}
        def infer(obs):
            queries.append(obs["state"])
            return {"actions":[[.001*(i+1),0,0,0,0,0,1] for i in range(10)]}
        def execute(action,index):
            state[0]+=1;executed.append(action["input_action"][0]);return {}
        result=run_openpi_steps(observe,infer,execute,steps=20,replan_steps=5,announce=lambda _:None)
        self.assertTrue(result["probe_completed"])
        self.assertEqual(queries,[0,5,10,15])
        self.assertEqual(executed,[.001*(i+1) for i in range(5)]*4)
        self.assertEqual(result["inference_requests"],4)
        self.assertEqual([r["state_observed"][0] for r in result["trace"]],list(range(20)))
        self.assertIsNone(result["trace"][1]["state_sent"])
        self.assertIsNone(result["trace"][1]["inference_wall_s"])
        self.assertEqual(result["trace"][6]["chunk_observation_step"],6)

    def test_short_chunk_cannot_execute_a_partial_requested_block(self):
        result=run_openpi_steps(lambda i:({}, {"image_files":[]}),
            lambda o:{"actions":[[0]*7]*4},lambda *a:self.fail("must not execute"),
            replan_steps=5,announce=lambda _:None)
        self.assertFalse(result["probe_completed"])
        self.assertEqual(result["completed_steps"],0)
        self.assertEqual(result["validated_inferences"],0)

    def test_cached_action_failure_preserves_executed_prefix(self):
        def execute(action,index):
            if index==2:raise RuntimeError("motion stop")
            return {}
        result=run_openpi_steps(lambda i:({}, {"image_files":[]}),lambda o:{"actions":[[0]*7]*10},
            execute,replan_steps=5,announce=lambda _:None)
        self.assertEqual(result["inference_requests"],1)
        self.assertEqual(result["completed_steps"],2)
        self.assertEqual(result["trace"][-1]["chunk_action_index"],2)
        self.assertFalse(result["trace"][-1]["executed"])

    def test_last_block_does_not_increase_total_requested_steps(self):
        result=run_openpi_steps(lambda i:({}, {"image_files":[]}),lambda o:{"actions":[[0]*7]*10},
            lambda *a:{},steps=7,replan_steps=5,announce=lambda _:None)
        self.assertEqual(result["completed_steps"],7)
        self.assertEqual(result["inference_requests"],2)
        self.assertEqual(result["trace"][-1]["chunk_action_index"],1)

    def test_invalid_replan_lengths_rejected(self):
        for value in (0,2,True,5.0):
            with self.assertRaises(ValueError):
                run_openpi_steps(None,None,None,replan_steps=value)

    def test_cached_server_metadata_preserves_reported_fields(self):
        class Client:
            calls=0
            def get_server_metadata(self):
                self.calls+=1
                return {"checkpoint":"reported-checkpoint","action_dim":7}
        client=Client();result=read_server_metadata(client)
        self.assertEqual(result["status"],"reported")
        self.assertEqual(result["value"]["checkpoint"],"reported-checkpoint")
        self.assertEqual(client.calls,1)

    def test_empty_unsupported_and_invalid_metadata_are_explicit(self):
        class Empty:
            def get_server_metadata(self):return {}
        class Invalid:
            def get_server_metadata(self):return {"value":float("nan")}
        self.assertEqual(read_server_metadata(Empty())["status"],"empty")
        self.assertEqual(read_server_metadata(object())["status"],"unsupported")
        self.assertEqual(read_server_metadata(Invalid())["status"],"unavailable")

    def test_state_matches_libero_signed_axis_angle_above_pi(self):
        angle=math.radians(200)
        q=[math.sin(angle/2),0,0,math.cos(angle/2)]
        state=libero_state_vector([.4,.1,.9],q,[.04,.04])
        self.assertEqual(len(state),8)
        self.assertAlmostEqual(state[3],angle)
        self.assertEqual(state[:3],[.4,.1,.9])
        opposite=libero_state_vector([0]*3,[-v for v in q],[0,0])
        self.assertAlmostEqual(opposite[3],math.radians(-160))

    def test_state_identity_and_invalid_quaternion(self):
        self.assertEqual(libero_state_vector([0]*3,[0,0,0,1],[.04,.04]),[0]*6+[.04,-.04])
        with self.assertRaises(ValueError):libero_state_vector([0]*3,[0]*4,[0,0])

    def test_finger_state_uses_robosuite_opposite_second_joint_sign(self):
        self.assertEqual(libero_state_vector([0]*3,[0,0,0,1],[.024,.023])[-2:],[.024,-.023])
        with self.assertRaises(ValueError):libero_state_vector([0]*3,[0,0,0,1],[.024,-.023])

    def test_fresh_observation_then_only_first_action_of_each_chunk_executes(self):
        state=[0];observed=[];executed=[]
        def observe(index):
            return {"state":state[0]},{"image_files":[f"obs{index}.png"]}
        def infer(obs):
            observed.append(obs["state"])
            return {"actions":[[.1,0,0,0,0,0,1],[-.1,0,0,0,0,0,-1]]}
        def execute(action,index):
            executed.append(action)
            state[0]+=1
            return {"actual_index":index}
        report=run_openpi_steps(observe,infer,execute,steps=3,announce=lambda _:None)
        self.assertTrue(report["probe_completed"])
        self.assertEqual(observed,[0,1,2])
        self.assertEqual(len(executed),3)
        self.assertTrue(all(a["bounded_delta_world_m"][0]>0 and a["applied_isaac_gripper"]==1 for a in executed))

    def test_invalid_future_action_stops_before_any_execution(self):
        executed=[]
        report=run_openpi_steps(lambda i:({}, {"image_files":["first.png"]}),
            lambda o:{"actions":[[0]*7,[float("nan")]*7]},
            lambda a,i:executed.append(a),steps=2,announce=lambda _:None)
        self.assertFalse(report["probe_completed"])
        self.assertEqual(report["completed_steps"],0)
        self.assertFalse(executed)
        self.assertEqual(report["preview_images"],["first.png"])

    def test_limits_vector_norm_preserves_direction_and_reports_gripper_override(self):
        r=bounded_openpi_action([.3,.4,0,0,-.6,-.8,1.2])
        self.assertAlmostEqual(math.hypot(*r["bounded_delta_world_m"]),.001)
        self.assertAlmostEqual(math.hypot(*r["bounded_delta_world_rad"]),math.radians(.25))
        self.assertAlmostEqual(r["bounded_delta_world_m"][0]/r["bounded_delta_world_m"][1],.75)
        self.assertEqual(r["requested_isaac_gripper"],-1)
        self.assertEqual(r["applied_isaac_gripper"],1)
        self.assertIn(6,r["clipped_indices"])

    def test_small_and_zero_actions_are_not_amplified(self):
        r=bounded_openpi_action([.001,0,0,0,0,0,-1])
        self.assertEqual(r["bounded_delta_world_m"],r["delta_world_m"])
        self.assertEqual(bounded_openpi_action([0]*7)["bounded_delta_world_rad"],[0]*3)

    def test_entire_chunk_must_be_valid_before_executing_first_action(self):
        self.assertEqual(validate_action_chunk([[0]*7]),[[0]*7])
        for chunk in ([],[[0]*6],[[0]*7,[0]*6+[float("nan")]],[[True]*7]):
            with self.assertRaises(ValueError):validate_action_chunk(chunk)

    def test_first_and_last_observation_previews_only(self):
        self.assertEqual(preview_images([]),[])
        rows=[{"image_files":[f"obs{i}_external.png",f"obs{i}_wrist.png"]} for i in range(3)]
        self.assertEqual(preview_images(rows),rows[0]["image_files"]+rows[2]["image_files"])


if __name__=="__main__":unittest.main()
