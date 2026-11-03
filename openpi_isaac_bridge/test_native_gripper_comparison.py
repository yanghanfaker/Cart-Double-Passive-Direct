import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
import zipfile
import numpy as np
from native_libero_rollout import rollout,gripper_command_state
from native_gripper_comparison import comparison_cases,force_open,load_responses,FrozenClient,compare_pair,episode_metrics
from test_native_libero_rollout import Env,Client,Images,observation


class GripperComparisonTests(unittest.TestCase):
    def test_matrix_and_source_dependencies(self):
        cases=comparison_cases()
        self.assertEqual(len(cases),24);self.assertEqual(len({c['name'] for c in cases}),24)
        for task in (0,2):
            for state in (0,1,2):
                group=[c for c in cases if c['task_id']==task and c['state_index']==state]
                self.assertEqual({c['mode'] for c in group},{'model-live','model-replay','open-replay','open-live'})
                self.assertEqual(group[0]['mode'],'model-live')
                self.assertTrue(all(c['seed']==7 and c['prefix_steps']==5 and c['max_controls']==220 for c in group))

    def test_gripper_transform_never_changes_pose_or_input(self):
        raw=[.1,.2,.3,.4,.5,.6,.9]
        self.assertEqual(force_open(raw),raw[:6]+[-1.])
        self.assertEqual(raw[-1],.9)
        with self.assertRaises(ValueError):force_open([0.]*6)
        with self.assertRaises(ValueError):force_open([float('nan')]+[0.]*6)

    def episodes(self,root):
        source=rollout(Env(),Client(),observation(),'task',root/'source',Images,max_steps=12)
        parsed,responses=load_responses(root/'source/report.json')
        control_client=FrozenClient(responses)
        normal=rollout(Env(),control_client,observation(),'task',root/'normal',Images,max_steps=12,query_origin='archive')
        open_client=FrozenClient(responses)
        opened=rollout(Env(),open_client,observation(),'task',root/'opened',Images,max_steps=12,query_origin='archive',action_transform=force_open)
        return source,normal,opened,control_client,open_client

    def test_archives_preserve_raw_actions_and_network_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,normal,opened,client,_=self.episodes(Path(tmp))
            self.assertEqual(source['new_policy_queries'],3)
            self.assertEqual(normal['new_policy_queries'],0);self.assertEqual(opened['new_policy_queries'],0)
            self.assertEqual(opened['policy_queries'],3)
            self.assertTrue(all(x['exact_input_match'] for x in client.checks))
            self.assertEqual([x['raw_action'] for x in source['trace']],[x['raw_action'] for x in opened['trace']])
            self.assertTrue(all(x['action'][-1]==-1 for x in opened['trace']))
            self.assertTrue(any(x['model_close_requested'] for x in opened['trace']))
            metrics=episode_metrics(opened,'bowl_pos')
            self.assertEqual(metrics['executed_close_controls'],0);self.assertEqual(metrics['first_model_close_control'],3)

    def test_pair_requires_unchanged_control_replay_before_attribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,normal,opened,_,_=self.episodes(Path(tmp))
            opened['trace'][4]['after']['robot0_eef_pos'][0]+=.001
            result=compare_pair(source,normal,opened)
            self.assertTrue(result['physical_reproduction_passed'])
            self.assertAlmostEqual(result['maximum_open_vs_normal_eef_distance_m'],.001)
            self.assertEqual(result['first_blocked_close_control'],3)
            normal['trace'][0]['after']['robot0_eef_pos'][0]+=.00001
            self.assertFalse(compare_pair(source,normal,opened)['physical_reproduction_passed'])

    def test_pose_change_or_short_replay_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,normal,opened,_,_=self.episodes(Path(tmp))
            changed=copy.deepcopy(opened);changed['trace'][0]['action'][0]+=.1
            with self.assertRaises(ValueError):compare_pair(source,normal,changed)
            opened['trace'].pop()
            with self.assertRaises(ValueError):compare_pair(source,normal,opened)

    def test_changed_counterfactual_input_logged_without_padding(self):
        client=FrozenClient([{'actions':[[0.]*7]*10,'input_sha256':'different'}])
        from native_libero_rollout import policy_observation
        client.infer(policy_observation(observation(),'task',Images))
        self.assertFalse(client.checks[0]['exact_input_match'])
        with self.assertRaises(ValueError):client.infer({})

    def test_source_trace_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rollout(Env(),Client(),observation(),'task',root/'source',Images,max_steps=12)
            path=root/'source/report.json';r=json.loads(path.read_bytes());r['trace'][0]['raw_action'][0]+=.1
            path.write_text(json.dumps(r))
            with self.assertRaises(ValueError):load_responses(path)

    def test_native_command_diagnostic_is_read_only_and_unavailability_explicit(self):
        gripper=types.SimpleNamespace(actuators=['finger'],current_action=np.array([-.7]))
        env=types.SimpleNamespace(robots=[types.SimpleNamespace(gripper=gripper)],
            sim=types.SimpleNamespace(model=types.SimpleNamespace(actuator_name2id=lambda name:0,actuator_ctrlrange=np.array([[0.,.04]])),
                                      data=types.SimpleNamespace(ctrl=np.array([.034]))))
        result=gripper_command_state(env)
        self.assertEqual(result['status'],'available');self.assertEqual(result['actuators'][0]['ctrl'],.034)
        self.assertEqual(env.sim.data.ctrl[0],.034)
        self.assertEqual(gripper_command_state(Env())['status'],'unavailable')

    def test_invalid_pose_transform_stops_before_physics(self):
        env=Env()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                rollout(env,Client(),observation(),'task',Path(tmp)/'bad',Images,max_steps=12,action_transform=lambda a:[a[0]+.1]+a[1:])
        self.assertEqual(env.actions,[])

    def test_physics_advance_during_query_preserves_response_and_stops(self):
        env=Env();original=Client()
        class Advancing:
            def infer(self,obs):
                response=original.infer(obs);env.sim.data.time+=.01;return response
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(RuntimeError):rollout(env,Advancing(),observation(),'task',root/'bad',Images)
            result=json.loads((root/'bad/report.json').read_bytes())
            self.assertEqual(result['action_steps'],0);self.assertEqual(result['new_policy_queries'],1)
            with zipfile.ZipFile(root/'bad/queries.zip') as z:self.assertIn('query-001/response.json',z.namelist())


if __name__=='__main__':unittest.main()
