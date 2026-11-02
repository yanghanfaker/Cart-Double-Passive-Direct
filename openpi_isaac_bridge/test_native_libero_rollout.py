import json
from pathlib import Path
import tempfile
import types
import unittest
import zipfile
import numpy as np

import native_libero_rollout as native


class Images:
    @staticmethod
    def resize_with_pad(a, h, w): return a[:h, :w].copy()
    @staticmethod
    def convert_to_uint8(a): return a.astype(np.uint8)


def observation(step=0):
    image = np.zeros((256, 256, 3), dtype=np.uint8); image[-1, -1] = [7, 8, 9]
    return {'agentview_image': image, 'robot0_eye_in_hand_image': image.copy(),
        'robot0_eef_pos': np.array([.1+step*.001, .2, .3]), 'robot0_eef_quat': np.array([0., 0., 0., 1.]),
        'robot0_gripper_qpos': np.array([.04, -.04]), 'robot0_joint_pos': np.zeros(7), 'bowl_pos': np.array([0., 0., .1])}


class Env:
    def __init__(self, success=99, done=99, fail=99):
        self.actions=[]; self.success_at=success; self.done_at=done; self.fail_at=fail
        self.sim=types.SimpleNamespace(data=types.SimpleNamespace(time=0.))
    def step(self, action):
        if len(self.actions)+1 == self.fail_at: raise RuntimeError('step failed')
        self.actions.append(action); self.sim.data.time+=.05
        return observation(len(self.actions)), 0., len(self.actions)==self.done_at, {}
    def check_success(self): return len(self.actions)>=self.success_at


class Client:
    def __init__(self, bad=False, mutate=False): self.inputs=[]; self.bad=bad; self.mutate=mutate
    def infer(self, element):
        self.inputs.append(element['observation/state'].copy())
        if self.mutate: element['observation/state'][0]+=1
        a=[[.2, -.3, .4, 0., 0., 0., 1. if i==2 else -1.] for i in range(10)]
        if self.bad: a[-1][0]=float('nan')
        return {'actions': np.array(a)}


class NativeTests(unittest.TestCase):
    def test_controller_optional_attributes(self):
        r=native.controller_record(types.SimpleNamespace(name='OSC_POSE',kp=np.array([150]*6)))
        self.assertEqual(r['kp'],[150]*6);self.assertIsNone(r['control_delta'])
        self.assertEqual(r['name'],'OSC_POSE')

    def run_rollout(self, root, env, client, budget=12):
        return native.rollout(env, client, observation(), 'native task', root/'episode', Images, budget)

    def test_state_image_conventions(self):
        obs=observation(); obs['robot0_eef_quat']=np.array([0., 0., np.sqrt(.5), -np.sqrt(.5)])
        p=native.policy_observation(obs, 'task', Images)
        self.assertEqual(p['observation/state'].dtype, np.float64)
        self.assertAlmostEqual(p['observation/state'][5], 1.5*np.pi)
        np.testing.assert_array_equal(p['observation/state'][-2:], [.04, -.04])
        np.testing.assert_array_equal(p['observation/image'][0,0], [7,8,9])
        self.assertLess(obs['robot0_eef_quat'][3], 0)

    def test_invalid_observation(self):
        obs=observation(); obs['robot0_eef_pos'][0]=np.nan
        with self.assertRaises(ValueError): native.policy_observation(obs, 'task', Images)
        obs=observation(); obs['agentview_image']=np.zeros((224,224,3),dtype=np.uint8)
        with self.assertRaises(ValueError): native.policy_observation(obs, 'task', Images)

    def test_prefix_replan_and_raw_gripper_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); env=Env(); client=Client(); r=self.run_rollout(root, env, client)
            self.assertEqual((r['action_steps'],r['policy_queries']), (12,3))
            self.assertEqual([q['executed_prefix_steps'] for q in r['queries']], [5,5,2])
            self.assertEqual(env.actions[0], [.2,-.3,.4,0.,0.,0.,-1.])
            self.assertEqual(env.actions[2][-1], 1.)
            self.assertNotEqual(client.inputs[0][0], client.inputs[1][0])
            self.assertFalse(r['success']); self.assertEqual(r['stop_reason'],'budget_exhausted')
            with zipfile.ZipFile(root/'episode/queries.zip') as z:
                self.assertEqual(len(z.namelist()),18)
                self.assertEqual(json.loads(z.read('query-001/request.json'))['prompt'],'native task')
            self.assertTrue((root/'episode/preview.gif').is_file())

    def test_success_mid_chunk_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            r=self.run_rollout(Path(temp),Env(success=3),Client())
            self.assertTrue(r['success']);self.assertEqual(r['action_steps'],3);self.assertEqual(r['policy_queries'],1)

    def test_done_is_not_automatically_success(self):
        with tempfile.TemporaryDirectory() as temp:
            r=self.run_rollout(Path(temp),Env(done=2),Client())
            self.assertFalse(r['success']);self.assertEqual(r['stop_reason'],'done_without_success')

    def test_invalid_future_action_preserved_no_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);env=Env()
            with self.assertRaises(ValueError):self.run_rollout(root,env,Client(bad=True))
            r=json.loads((root/'episode/report.json').read_bytes())
            self.assertEqual(r['action_steps'],0);self.assertEqual(r['status'],'failed')
            with zipfile.ZipFile(root/'episode/queries.zip') as z:self.assertIn('nan',z.read('query-001/response.txt').decode())

    def test_failed_step_keeps_pending_action_and_partial_trace(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with self.assertRaises(RuntimeError):self.run_rollout(root,Env(fail=2),Client())
            r=json.loads((root/'episode/report.json').read_bytes())
            self.assertEqual((r['action_steps'],r['step_attempts']), (1,2))
            self.assertEqual(r['pending_action']['step'],2)

    def test_mutated_input_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            env=Env()
            with self.assertRaises(ValueError):self.run_rollout(Path(temp),env,Client(mutate=True))
            self.assertEqual(env.actions,[])


if __name__=='__main__':unittest.main()
