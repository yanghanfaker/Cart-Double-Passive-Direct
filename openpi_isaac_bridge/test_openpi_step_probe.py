import math
import unittest
from openpi_step_probe import bounded_openpi_action, validate_action_chunk, preview_images, run_openpi_steps, libero_state_vector


class OpenpiStepTests(unittest.TestCase):
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
