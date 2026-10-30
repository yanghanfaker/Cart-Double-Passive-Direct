import copy
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock,patch
import numpy as np

from openpi_grasp_logic import close_gate,bounded_goal_target,run_grasp_experiment
import run_openpi_grasp_capture as runner
from kinematic_attribution import rotate
from geometry_audit import inverse,multiply
from test_clean_scene_probe import fixtures
from run_clean_scene_capture import prepare_cases


class Sim:
    def __init__(self,mode='held',near=True):
        self.mode=mode;self.tcp=np.array([.5 if near else .4,0,.026 if near else .25]);self.obj=np.array([.5,0,.021]);self.fingers=[.04,.04]
        self.q=[1,0,0,0];self.calls=[];self.hold_count=0
    def state(self):
        return {'tcp_world_m':self.tcp.tolist(),'hand_quat_world_xyzw':self.q,'object_pos_world_m':self.obj.tolist(),
            'object_quat_world_xyzw':[0,0,0,1],'object_in_tcp_pos_m':rotate(inverse(self.q),(self.obj-self.tcp).tolist()),
            'object_in_tcp_quat_xyzw':multiply(inverse(self.q),[0,0,0,1]),'finger_positions_m':self.fingers[:]}
    def query(self,index):return {'raw_action_chunk':[[0,0,0,0,0,0,-1 if self.mode=='open' else 1] for _ in range(10)]}
    def model(self,action,grip):
        self.calls.append(('approach',grip))
        if self.mode=='thrown':self.obj[0]+=.03
        return self.state()
    def goal(self,p,q,grip,phase):
        self.calls.append((phase,grip));delta=np.asarray(p)-self.tcp;n=np.linalg.norm(delta)
        if n>.0005:delta*=.0005/n
        self.tcp+=delta
        if phase=='close':self.fingers=[0,0] if self.mode=='empty' else [.02,.02]
        if phase in ('lift','hold') and self.mode!='unheld':self.obj+=delta
        if phase=='hold' and self.mode=='slip':self.obj[2]-=.0006
        return self.state()
    def run(self,budget=1):return run_grasp_experiment(self.state,self.query,self.model,self.goal,budget,announce=lambda _:None)


class OpenPIGraspTests(unittest.TestCase):
    def test_gate_checks_alignment_height_and_orientation(self):
        s=Sim().state();self.assertTrue(close_gate(s)['allowed'])
        for mode in ('xy','low','high','up'):
            bad=copy.deepcopy(s)
            if mode=='xy':bad['tcp_world_m'][0]+=.013
            if mode=='low':bad['tcp_world_m'][2]=.01
            if mode=='high':bad['tcp_world_m'][2]=.05
            if mode=='up':bad['hand_quat_world_xyzw']=[0,0,0,1]
            with self.subTest(mode=mode):self.assertFalse(close_gate(bad)['allowed'])

    def test_verified_lift_requires_sustained_relative_pose(self):
        sim=Sim();r=sim.run()
        self.assertTrue(r['success']);self.assertEqual(r['outcome'],'verified_lift_hold')
        self.assertTrue(r['lift_attempted']);self.assertEqual(r['phase_counts']['close'],60);self.assertEqual(r['phase_counts']['hold'],40)
        self.assertGreaterEqual(r['hold_check']['min_object_lift_m'],.035)
        self.assertTrue(all(grip==-1 for phase,grip in sim.calls if phase!='approach'))

    def test_empty_close_never_lifts(self):
        r=Sim('empty').run();self.assertFalse(r['success']);self.assertFalse(r['lift_attempted'])
        self.assertEqual(r['outcome'],'close_candidate_failed');self.assertFalse(r['close_check']['obstruction_candidate'])

    def test_unheld_or_slipping_object_cannot_pass(self):
        for mode in ('unheld','slip'):
            with self.subTest(mode=mode):
                r=Sim(mode).run();self.assertFalse(r['success']);self.assertTrue(r['lift_attempted']);self.assertEqual(r['outcome'],'lift_hold_failed')

    def test_far_model_close_is_blocked_and_open_model_is_not_forced_closed(self):
        s=Sim(near=False);r=s.run();self.assertEqual(r['blocked_close_requests'],5)
        self.assertFalse(r['close_attempted']);self.assertTrue(all(grip==1 for _,grip in s.calls))
        s=Sim('open');r=s.run();self.assertFalse(r['close_attempted']);self.assertEqual(r['outcome'],'approach_query_budget_exhausted')

    def test_throw_and_stagnation_stop_before_lift(self):
        r=Sim('thrown',near=False).run();self.assertFalse(r['success']);self.assertIn('speed',r['outcome']);self.assertFalse(r['lift_attempted'])
        r=Sim('open',near=False).run(30);self.assertEqual(r['approach_controls'],100);self.assertIn('progress',r['outcome'])

    def test_goal_target_retains_caps_and_exact_tcp_point_shift(self):
        from replay_probe import quaternion_distance_deg
        hand=[.4,0,.3];q=[1,0,0,0];goal=[.5,.1,.15];qgoal=[math.cos(.2),0,0,-math.sin(.2)]
        target=bounded_goal_target(hand,q,goal,qgoal);current=np.array(hand)+rotate(q,[0,0,.097])
        self.assertAlmostEqual(np.linalg.norm(np.array(target['tcp97_target_world_m'])-current),.0005)
        self.assertLessEqual(quaternion_distance_deg(q,target['hand_target_quat_world_xyzw']),.1000001)
        np.testing.assert_allclose(np.array(target['ik107_target_world_m'])-rotate(target['hand_target_quat_world_xyzw'],[0,0,.010]),target['tcp97_target_world_m'],atol=1e-14)

    def test_invalid_future_action_and_budget_rejected(self):
        s=Sim();bad=s.query(1);bad['raw_action_chunk'][9][0]=float('nan')
        with self.assertRaises(ValueError):run_grasp_experiment(s.state,lambda _:bad,s.model,s.goal,1,announce=lambda _:None)
        for budget in (0,201,True):
            with self.assertRaises(ValueError):s.run(budget)

    def test_outer_runner_reports_unsuccessful_completed_experiment_separately(self):
        for fail in (False,True):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);source=root/'source';source.mkdir();config=prepare_cases(*fixtures())[1]
                (source/'clean_scene_request.json').write_text(json.dumps(config))
                cal=fixtures()[2];cal['candidate_geometry_within_thresholds']=True;cb=json.dumps(cal).encode();(source/'world_registration.json').write_bytes(cb)
                prior={'status':'completed','mode':'three-round-replan','completed_action_periods':15,'policy_queries':3,'final_target_reached':True,
                    'rounds':[{'input':{'prompt':'pick up the cube'}}]}
                pb=json.dumps(prior).encode();(source/'result.json').write_bytes(pb)
                m={'status':'completed','final_target_reached':True,'all_followup_inputs_changed':True,'result':'result.json',
                    'result_sha256':hashlib.sha256(pb).hexdigest(),'world_registration_sha256':hashlib.sha256(cb).hexdigest()}
                source_manifest=source/'run.json';source_manifest.write_text(json.dumps(m));commands=[]
                class Process:
                    def __init__(self,command,**kwargs):
                        commands.append(command);self.stdout=io.StringIO('')
                        if not fail:
                            out=Path(command[command.index('--output-root')+1])/'capture';out.mkdir(parents=True)
                            (out/'online_registered_result.json').write_text(json.dumps({'status':'completed','mode':'openpi-grasp','success':False,
                                'outcome':'approach_query_budget_exhausted','policy_queries':1,'native_controls_executed':5}))
                    def __enter__(self):return self
                    def __exit__(self,*args):pass
                    def wait(self):return 1 if fail else 0
                with (patch.object(sys,'argv',['grasp','--source-manifest',str(source_manifest),'--output-root',str(root/'out'),'--max-queries','1']),
                        patch.object(runner.socket,'create_connection',MagicMock()),patch.object(runner.subprocess,'Popen',Process)):
                    self.assertEqual(runner.main(),1 if fail else 0)
                self.assertEqual(len(commands),1);report=json.loads(next((root/'out').glob('*/run.json')).read_bytes())
                self.assertEqual(report['status'],'failed' if fail else 'completed');self.assertFalse(report['success'])
                sf=Path(commands[0][commands[0].index('--online-registered-json')+1]);self.assertEqual(json.loads(sf.read_bytes())['mode'],'openpi-grasp')


if __name__=='__main__':unittest.main()
