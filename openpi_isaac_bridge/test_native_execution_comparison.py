import copy
from pathlib import Path
import tempfile
import types
import unittest
import numpy as np
from native_execution_comparison import execution_cases,Intervention,compare_execution,state_difference
from native_gripper_comparison import load_responses,FrozenClient
from native_libero_rollout import rollout,controller_tracking
from test_native_libero_rollout import Env,Client,Images,observation


class ExecutionComparisonTests(unittest.TestCase):
    def case(self,name):return next(c for c in execution_cases() if c['name']==f't0-s0-{name}-replay')

    def test_matrix_covers_independent_and_combined_interventions(self):
        cases=execution_cases();self.assertEqual(len(cases),60);self.assertEqual(len({c['name'] for c in cases}),60)
        for task in (0,2):
            for state in (0,1,2):
                group=[c for c in cases if (c['task_id'],c['state_index'])==(task,state)]
                self.assertEqual(len(group),10);self.assertEqual(group[0]['mode'],'identity-replay')

    def test_native_clipping_precedes_norm_cap_preserving_direction(self):
        action=[2.,1.,0.,.2,-.1,.3,.8];original=action.copy()
        out=Intervention(self.case('translation1'))(action)
        np.testing.assert_allclose(np.array(out[:3])*.05,[.001/np.sqrt(2)]*2+[0.],atol=1e-15)
        self.assertEqual(out[3:],action[3:]);self.assertEqual(action,original)

    def test_identity_and_delay_preserve_unclipped_pose(self):
        action=[2.,-.4,.8,1.5,0.,0.,1.]
        self.assertEqual(Intervention(self.case('identity'))(action),action)
        out=Intervention(self.case('delay5'))(action)
        self.assertEqual(out[:6],action[:6]);self.assertEqual(out[6],-1.)

    def test_rotation_only_and_combination(self):
        raw=[.3,.2,-.1,.4,.5,-.3,1.]
        out=Intervention(self.case('rotation0p1'))(raw)
        self.assertEqual(out[:3],raw[:3]);self.assertAlmostEqual(np.linalg.norm(np.array(out[3:6])*.5),np.deg2rad(.1))
        both=Intervention(self.case('both0p5-delay5'))(raw)
        self.assertAlmostEqual(np.linalg.norm(np.array(both[:3])*.05),.0005)
        self.assertAlmostEqual(np.linalg.norm(np.array(both[3:6])*.5),np.deg2rad(.1));self.assertEqual(both[6],-1.)

    def test_delay_is_per_consecutive_close_block_and_does_not_queue(self):
        f=Intervention(self.case('delay5'));raw=[0.]*6+[1.]
        self.assertEqual([f(raw)[6] for _ in range(7)],[-1.]*5+[1.]*2)
        self.assertEqual(f([0.]*7)[6],0.)
        self.assertEqual(f(raw)[6],-1.)
        self.assertEqual(f([0.]*6+[-.5])[6],-.5)
        self.assertEqual(f(raw)[6],-1.)

    def episodes(self,root,name='identity',success=99):
        source=rollout(Env(),Client(),observation(),'task',root/'source',Images,max_steps=12)
        _,responses=load_responses(root/'source/report.json');client=FrozenClient(responses)
        result=rollout(Env(success=success),client,observation(),'task',root/'result',Images,max_steps=12,
            query_origin='archive',action_transform=Intervention(self.case(name)),allow_pose_intervention=True)
        return source,result

    def test_identity_reproduction_and_zero_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=self.episodes(Path(tmp));r=compare_execution(a,b,self.case('identity'))
            self.assertTrue(r['identity_reproduction_passed']);self.assertEqual(b['new_policy_queries'],0)
            b['trace'][3]['after']['robot0_gripper_qpos'][1]+=.001
            self.assertFalse(compare_execution(a,b,self.case('identity'))['identity_reproduction_passed'])

    def test_explicit_pose_intervention_archives_actual_and_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=self.episodes(Path(tmp),'both0p5')
            r=compare_execution(a,b,self.case('both0p5'))
            self.assertEqual(r['pose_changed_controls'],12);self.assertEqual(r['gripper_changed_controls'],0)
            self.assertEqual(b['trace'][0]['raw_action'],a['trace'][0]['raw_action'])
            self.assertNotEqual(b['trace'][0]['action'],a['trace'][0]['action'])
            b['trace'][0]['action'][0]+=.1
            with self.assertRaises(ValueError):compare_execution(a,b,self.case('both0p5'))

    def test_short_replay_only_allowed_at_terminal_no_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=self.episodes(Path(tmp),'delay5',success=3)
            r=compare_execution(a,b,self.case('delay5'));self.assertTrue(r['early_terminal']);self.assertEqual(r['common_controls'],3)
            b['stop_reason']='budget_exhausted'
            with self.assertRaises(ValueError):compare_execution(a,b,self.case('delay5'))

    def test_pose_changes_remain_disallowed_for_live_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            env=Env()
            with self.assertRaises(ValueError):rollout(env,Client(),observation(),'task',Path(tmp)/'bad',Images,
                action_transform=Intervention(self.case('translation1')),allow_pose_intervention=True)
            self.assertEqual(env.actions,[])

    def test_read_only_controller_buffers_and_explicit_unavailability(self):
        fields=dict(goal_pos=np.array([.1,0,0]),ee_pos=np.zeros(3),ee_pos_vel=np.zeros(3),goal_ori=np.eye(3),
            ee_ori_mat=np.eye(3),ee_ori_vel=np.zeros(3),joint_pos=np.zeros(7),joint_vel=np.zeros(7),torques=np.ones(7))
        c=types.SimpleNamespace(**fields);env=types.SimpleNamespace(robots=[types.SimpleNamespace(controller=c)])
        r=controller_tracking(env);self.assertEqual(r['status'],'available');self.assertEqual(r['position_error_m'],.1)
        np.testing.assert_array_equal(c.goal_pos,[.1,0,0]);self.assertEqual(controller_tracking(Env())['status'],'unavailable')

    def test_bad_timing_source_action_and_new_query_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=self.episodes(Path(tmp))
            for mutation in ('time','raw','query'):
                changed=copy.deepcopy(b)
                if mutation=='time':changed['trace'][0]['after']['sim_time_s']+=.01
                if mutation=='raw':changed['trace'][0]['raw_action'][0]+=.1
                if mutation=='query':changed['new_policy_queries']=1
                with self.assertRaises(ValueError):compare_execution(a,changed,self.case('identity'))

    def test_snapshot_keys_cannot_be_silently_dropped(self):
        with self.assertRaises(ValueError):state_difference({'x':[1.]},{'x':[1.],'y':[2.]})


if __name__=='__main__':unittest.main()
