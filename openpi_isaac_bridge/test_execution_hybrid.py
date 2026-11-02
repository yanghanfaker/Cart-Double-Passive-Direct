import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile
from execution_hybrid import hybrid_cases,run_policy_approach,validate_contact_source
from execution_alignment_probe import validate_case
from live_grasp_query import feedback_check,make_live_query


class HybridTests(unittest.TestCase):
    def test_completed_reference_without_grasp_success_cannot_enable_policy(self):
        from run_execution_alignment import missing_prerequisites
        case=self.case();row={'case':hybrid_cases()[0],'status':'completed','summary':{'grasp_result':{'success':False}}}
        self.assertEqual(missing_prerequisites(case,[row]),['hybrid-reference-start grasp success'])
        row['summary']['grasp_result']['success']=True
        self.assertEqual(missing_prerequisites(case,[row]),[])

    def test_contact_source_requires_full_positive_and_negative_results_and_hashes(self):
        import hashlib
        from execution_contact import contact_cases
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);rows=[]
            for c in contact_cases():
                g={'success':not c['negative_control'],'outcome':'negative_rejected' if c['negative_control'] else 'verified_lift_hold','negative_check_passed':True}
                payload=json.dumps({'status':'completed','grasp_result':g}).encode();name=c['name']+'.json';(root/name).write_bytes(payload)
                rows.append({'case':c,'status':'completed','result':name,'summary':{'result_sha256':hashlib.sha256(payload).hexdigest()}})
            path=root/'run.json';path.write_text(json.dumps({'status':'completed','cases':rows}))
            self.assertEqual(len(validate_contact_source(path)['result_sha256']),10)
            (root/rows[0]['result']).write_bytes(b'{}')
            with self.assertRaises(ValueError):validate_contact_source(path)

    def case(self,mode='geometry',prefix=1):
        return next(c for c in hybrid_cases() if c.get('handoff_mode')==mode and c['prefix_length']==prefix and c['repeat']==1)

    def state(self,near=False):
        z=.026 if near else .12
        return {'tcp_world_m':[.66,.14,z],'hand_position_world_m':[.66,.14,z+.097],
            'hand_quat_world_xyzw':[1.,0.,0.,0.],'object_pos_world_m':[.66,.14,.021],
            'object_quat_world_xyzw':[0.,0.,0.,1.],'finger_positions_m':[.04,.04],
            'object_in_tcp_pos_m':[0.,0.,z-.021],'object_in_tcp_quat_xyzw':[1.,0.,0.,0.]}

    def test_matrix_has_order_reversal_and_success_dependency(self):
        cases=hybrid_cases();self.assertEqual(len(cases),10)
        for case in cases:validate_case(case)
        self.assertEqual(cases[0]['hybrid_role'],'reference');self.assertEqual(cases[-1]['hybrid_role'],'reference')
        first=[(c['prefix_length'],c['handoff_mode']) for c in cases if c.get('hybrid_role')=='policy' and c['repeat']==1]
        second=[(c['prefix_length'],c['handoff_mode']) for c in cases if c.get('hybrid_role')=='policy' and c['repeat']==2]
        self.assertEqual(second,list(reversed(first)))
        for c in cases[1:-1]:self.assertEqual(c['requires_success'],['hybrid-reference-start'])

    def test_geometry_handoff_can_ignore_model_open_request(self):
        state=self.state(near=True);executed=[]
        result=run_policy_approach(lambda:state,lambda i:{'raw_action_chunk':[[0.]*6+[-1.]]*10},
            lambda *args:executed.append(args),self.case())
        self.assertTrue(result['handoff']);self.assertEqual(result['model_close_requests'],0)
        self.assertEqual(result['controls'],0);self.assertEqual(executed,[])
        self.assertFalse(result['decisions'][0]['executed'])

    def test_model_close_handoff_requires_positive_request(self):
        state=self.state(near=True);calls=[]
        actions=[[0.]*6+[-1.],[0.]*6+[1.]]+[[0.]*7]*8
        result=run_policy_approach(lambda:state,lambda i:{'raw_action_chunk':actions},
            lambda *args:(calls.append(args) or state),self.case('model-close',5))
        self.assertEqual(len(calls),1);self.assertTrue(result['handoff'])
        self.assertEqual(result['handoff_prefix_index'],2);self.assertEqual(result['model_close_requests'],1)

    def test_geometry_handoff_after_motion_does_not_request_extra_inference(self):
        state=self.state();state['tcp_world_m'][2]=.041;queries=[]
        def query(index):queries.append(index);return {'raw_action_chunk':[[0.]*6+[-1.]]*10}
        def step(*args):state['tcp_world_m'][2]=.035;return copy.deepcopy(state)
        result=run_policy_approach(lambda:copy.deepcopy(state),query,step,self.case('geometry',5))
        self.assertTrue(result['handoff']);self.assertEqual(result['controls'],1);self.assertEqual(queries,[1])
        self.assertEqual(result['decisions'][0]['decision'],'executed_pose_then_program_handoff')

    def test_premature_close_is_logged_without_program_handoff(self):
        state=self.state();calls=[]
        result=run_policy_approach(lambda:state,lambda i:{'raw_action_chunk':[[0.]*6+[1.]]*10},
            lambda *args:(calls.append(args) or state),self.case('model-close',5))
        self.assertFalse(result['handoff']);self.assertEqual(len(calls),100)
        self.assertEqual(result['blocked_close_requests'],100)
        self.assertEqual(result['outcome'],'approach_progress_less_than_2mm_per100_controls')

    def test_prefix_lengths_use_equal_control_budget(self):
        for prefix,queries in ((1,600),(5,120)):
            state=self.state();count=[0]
            def step(*args):
                count[0]+=1;state['tcp_world_m'][2]-=.00003;return copy.deepcopy(state)
            result=run_policy_approach(lambda:copy.deepcopy(state),lambda i:{'raw_action_chunk':[[0.]*7]*10},step,self.case('geometry',prefix))
            self.assertEqual(result['outcome'],'approach_control_budget_exhausted')
            self.assertEqual(result['controls'],600);self.assertEqual(result['queries'],queries)

    def test_program_handoff_reuses_verified_closure_without_high_clearance_requirement(self):
        from test_execution_contact import ContactTests
        from execution_contact import run_contact_experiment
        read,step,calls=ContactTests().simulator()
        # Move callback fixture to the qualified handoff pose before invoking program refinement.
        for _ in range(110):step([.66,.14,.026],[1.,0.,0.,0.],1.,'descend')
        before=len(calls);result=run_contact_experiment(read,step,self.case(),handoff=True)
        self.assertTrue(result['success']);self.assertIn('handoff_refinement',result)
        self.assertTrue(all(c[3] not in ('orient','above','descend') for c in calls[before:]))
        self.assertEqual(sum(c[3]=='hold' for c in calls[before:]),40)

    def test_feedback_uses_actual_prefix_controls(self):
        for controls in (1,5):
            previous={'executed_prefix_steps':controls,'capture_physics_step':100,'camera_frames':{'external_camera':10,'wrist_camera':10},
                'input':{'input_sha256':'old','state_sent_float32':[0]*8,'frames':{'rgb':{'array_sha256':'old'}}}}
            current=copy.deepcopy(previous);current.update(capture_physics_step=100+5*controls,camera_frames={'external_camera':11,'wrist_camera':11})
            self.assertEqual(feedback_check(previous,current)['executed_controls_between_observations'],controls)
            current['capture_physics_step']+=5
            with self.assertRaises(RuntimeError):feedback_check(previous,current)

    def test_live_query_archives_response_and_rejects_physics_during_inference(self):
        for advance in (False,True):
            with tempfile.TemporaryDirectory() as temp:
                folder=Path(temp);(folder/'queries').mkdir();step=[10]
                runtime=SimpleNamespace(output=folder,base=SimpleNamespace(sim=SimpleNamespace(get_physics_step_count=lambda:step[0])),
                    arm=None,tensor=None,state=lambda:{'tcp_world_m':[0.,0.,0.],'hand_quat_world_xyzw':[0.,0.,0.,1.],'joint_positions_rad':[0.]*7})
                report={'queries':[],'inference_attempts':0,'policy_queries':0}
                request={'world_rotation_L_from_I':[[1,0,0],[0,1,0],[0,0,1]],'world_translation_L_from_I_m':[0,0,0],
                    'source_state_float32':[0.]*8,'registration_quat_xyzw':[0,0,0,1],
                    'source_mapped_hand_quat_xyzw':[0,0,0,1],'source_arm_positions_rad':[0.]*7}
                def infer(client,obs,req,on_response):
                    actions=[[0.]*7]*10;on_response({'actions':actions},.1)
                    if advance:step[0]+=5
                    return {**req,'raw_action_chunk':actions},.1
                geometry={'cameras':{name:{'frame':[1]} for name in ('external_camera','wrist_camera')}}
                with patch('live_grasp_query.capture_round',return_value=(geometry,10)),patch('live_grasp_query.prepare_capture_observation',return_value=({},request,{})),patch('live_grasp_query.query_once',side_effect=infer):
                    query=make_live_query(runtime,{'calibration':{},'prompt':'pick up the cube'},report,lambda:None,object())
                    if advance:
                        with self.assertRaisesRegex(RuntimeError,'physics advanced'):query(1)
                    else:self.assertEqual(len(query(1)['raw_action_chunk']),10)
                self.assertEqual(report['policy_queries'],1)
                self.assertEqual(report['queries'][0]['status'],'failed' if advance else 'validated')
                with zipfile.ZipFile(folder/'queries/query_0001.zip') as archive:
                    self.assertIn('online_model_response.json',archive.namelist())


if __name__=='__main__':unittest.main()
