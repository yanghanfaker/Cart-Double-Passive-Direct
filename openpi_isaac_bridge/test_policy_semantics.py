import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import zipfile
from execution_policy_semantics import (semantics_cases,SOURCE,grip_command,lift_window,run_policy_sequence,
    prepare_semantics_replay,load_semantics_replay,compare_identity)
from execution_alignment_probe import validate_case
import test_execution_hybrid as hybrid_fixture


class PolicySemanticsTests(unittest.TestCase):
    def case(self,role='live',grip='model'):
        return next(c for c in semantics_cases() if c.get('semantics_role')==role and c['gripper_policy']==grip)

    def state(self):
        s=hybrid_fixture.HybridTests().state();s.update(joint_positions_rad=[0.]*7,joint_velocities_rad_s=[0.]*7)
        return s

    def test_declared_matrix_fixed_gains_and_dependency_order(self):
        cases=semantics_cases();self.assertEqual(len(cases),26);self.assertEqual(len({c['name'] for c in cases}),26)
        seen=set()
        for c in cases:
            validate_case(c);self.assertTrue(set(c['requires'])<=seen);seen.add(c['name'])
            self.assertEqual(c['arm_profile'],'k1687p5-d50');self.assertEqual(c['finger_profile'],'kp350-kd50')
        live=[c for c in cases if c.get('semantics_role')=='live'];replay=[c for c in cases if c.get('semantics_role')=='replay']
        self.assertEqual((len(live),len(replay)),(16,8));self.assertEqual(live[0]['name'],SOURCE)
        key=lambda c:(c['cap_m'],c['rotation_cap_deg'],c['gripper_policy'])
        self.assertEqual([key(c) for c in live if c['repeat']==2],list(reversed([key(c) for c in live if c['repeat']==1])))
        self.assertTrue(replay[0]['identity_replay'])

    def test_gripper_sign_zero_retains_last_and_open_overrides(self):
        self.assertEqual(grip_command([0.]*6+[1.],'model'),-1.)
        self.assertEqual(grip_command([0.]*6+[-1.],'model'),1.)
        self.assertEqual(grip_command([0.]*7,'model',-1.),-1.)
        self.assertEqual(grip_command([0.]*6+[1.],'open',-1.),1.)

    def test_close_is_executed_even_inside_or_outside_old_geometry_gate(self):
        for near in (True,False):
            state=hybrid_fixture.HybridTests().state(near);calls=[]
            def step(raw,grip,q,p):calls.append(grip);return copy.deepcopy(state)
            r=run_policy_sequence(lambda:copy.deepcopy(state),lambda i:{'raw_action_chunk':[[0.]*6+[1.]]*10},step,self.case())
            self.assertEqual(r['controls'],100);self.assertEqual(calls,[-1.]*100)
            self.assertEqual(r['model_close_requests'],r['applied_close_controls'])
            self.assertNotIn('handoff',r)

    def test_open_reference_keeps_model_request_but_never_closes(self):
        state=self.state();calls=[]
        r=run_policy_sequence(lambda:state,lambda i:{'raw_action_chunk':[[0.]*6+[1.]]*10},
            lambda raw,g,q,p:(calls.append(g) or copy.deepcopy(state)),self.case(grip='open'))
        self.assertEqual(calls,[1.]*100);self.assertEqual(r['model_close_requests'],100);self.assertEqual(r['applied_close_controls'],0)

    def test_model_lift_requires_continuous_relative_pose_and_finger_evidence(self):
        state=self.state();calls=[]
        def step(raw,grip,q,p):
            calls.append((q,p));state['object_pos_world_m'][2]=.07;state['finger_positions_m']=[.02,.02]
            state['object_in_tcp_pos_m']=[0.,0.,.05];return copy.deepcopy(state)
        r=run_policy_sequence(lambda:copy.deepcopy(state),lambda i:{'raw_action_chunk':[[0.]*6+[1.]]*10},step,self.case())
        self.assertTrue(r['success']);self.assertEqual(r['controls'],40);self.assertEqual(r['queries'],8)
        self.assertEqual(calls[:6],[(1,1),(1,2),(1,3),(1,4),(1,5),(2,1)])

    def test_lift_window_rejects_empty_closed_or_open_fingers_and_slip(self):
        initial=self.state();s=copy.deepcopy(initial);s['object_pos_world_m'][2]+=.05;s['finger_positions_m']=[.02,.02]
        states=[copy.deepcopy(s) for _ in range(40)];self.assertTrue(lift_window(states,initial)['passed'])
        for fingers in ([0.,0.],[.04,.04]):
            altered=copy.deepcopy(states);altered[-1]['finger_positions_m']=fingers
            self.assertFalse(lift_window(altered,initial)['passed'])
        states[20]['object_in_tcp_pos_m'][0]+=.02;self.assertFalse(lift_window(states,initial)['passed'])
        self.assertFalse(lift_window(states[:39],initial)['passed'])

    def test_offline_uses_exact_source_horizon_and_never_queries(self):
        state=self.state();raw=[[0.]*6+[1.]]*7;calls=[]
        r=run_policy_sequence(lambda:state,lambda i:(_ for _ in ()).throw(AssertionError('network')),
            lambda a,g,q,p:(calls.append(a) or copy.deepcopy(state)),self.case('replay'),archived=raw)
        self.assertEqual(calls,raw);self.assertEqual(r['controls'],7);self.assertEqual(r['queries'],0)
        self.assertEqual(r['outcome'],'archived_prefix_complete')

    def test_model_future_invalid_action_stops_before_any_execution(self):
        raw=[[0.]*7 for _ in range(10)];raw[-1][0]=float('nan');calls=[]
        with self.assertRaises(ValueError):run_policy_sequence(self.state,lambda i:{'raw_action_chunk':raw},lambda *a:calls.append(a),self.case())
        self.assertEqual(calls,[])

    def test_floor_abort_is_reported_without_program_recovery(self):
        state=self.state()
        def step(*args):state['tcp_world_m'][2]=0.;return state
        r=run_policy_sequence(lambda:copy.deepcopy(state),lambda i:{'raw_action_chunk':[[0.]*7]*10},step,self.case())
        self.assertEqual(r['controls'],1);self.assertIn('below initial object floor',r['outcome']);self.assertFalse(r['success'])

    def test_identity_checks_fingers_object_rotation_and_velocity_not_only_tcp(self):
        state=self.state();trace=[{'before':copy.deepcopy(state),'after':copy.deepcopy(state)}]*5
        source={'trace':copy.deepcopy(trace)};self.assertTrue(compare_identity(source,trace)['passed'])
        for field in ('finger_positions_m','joint_velocities_rad_s','object_pos_world_m'):
            changed=copy.deepcopy(trace);changed[0]['after'][field][0]+=.01
            self.assertFalse(compare_identity(source,changed)['passed'])
        self.assertFalse(compare_identity(source,trace[:4])['passed'])

    def source_fixture(self,root):
        source=root/'source';source.mkdir();(source/'queries').mkdir()
        raw=[[.01,0.,0.,0.,0.,0.,1.]]*10;state=self.state();case=next(c for c in semantics_cases() if c['name']==SOURCE)
        archive=source/'queries/query_0001.zip'
        with zipfile.ZipFile(archive,'w') as z:
            z.writestr('online_model_response.json',json.dumps({'actions':raw}))
            z.writestr('online_replay_request.json',json.dumps({'raw_action_chunk':raw}))
        digest=hashlib.sha256(archive.read_bytes()).hexdigest()
        result={'status':'completed','initial':state,'policy_queries':1,'settings':{'execution_case':case},
            'queries':[{'query':1,'archive':'queries/query_0001.zip','archive_sha256':digest,'status':'validated','capture_physics_step':0,'post_query_physics_step':0}],
            'trace':[{'phase':'policy-open','native_gripper_sign':-1.,'raw_action':raw[i],
                      'target':{'source_query':1,'source_prefix_index':i+1},'before':state,'after':state} for i in range(5)]}
        path=source/'online_registered_result.json';path.write_text(json.dumps(result))
        row={'status':'completed','case':case,'summary':{'result_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}
        return path,row

    def test_replay_copies_and_verifies_actual_inference_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,row=self.source_fixture(root);req=prepare_semantics_replay(source,row,root/'replay')
            parsed,r=load_semantics_replay(req);self.assertEqual(parsed['controls'],5);self.assertEqual(parsed['actions'],[t['raw_action'] for t in r['trace']])
            (req.parent/'query_0001.zip').write_bytes(b'changed')
            with self.assertRaises(ValueError):load_semantics_replay(req)

    def test_replay_action_tamper_rejected_even_with_intact_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,row=self.source_fixture(root);path=prepare_semantics_replay(source,row,root/'replay')
            req=json.loads(path.read_bytes());req['actions'][0][0]+=.1;path.write_text(json.dumps(req))
            with self.assertRaises(ValueError):load_semantics_replay(path)

    def test_uncompleted_source_cannot_be_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,row=self.source_fixture(root);row['status']='failed'
            with self.assertRaises(ValueError):prepare_semantics_replay(source,row,root/'replay')

    def test_runtime_routes_model_to_distinct_phase_and_offline_never_connects(self):
        from execution_policy_semantics import run_semantics_runtime
        state=self.state();cal={'world_from_isaac_rotation':[[1,0,0],[0,1,0],[0,0,1]]}
        for grip in ('model','open'):
            settings={'execution_case':self.case('replay',grip),'semantics_replay_path':'unused',
                'calibration':cal,'expected_joint_gains':{},'expected_runtime_arm_gains':{}}
            req={'actions':[[.01,0.,0.,0.,0.,0.,1.]]*5};source={'settings':settings,'initial':state,
                'trace':[{'before':copy.deepcopy(state),'after':copy.deepcopy(state)} for _ in range(5)]}
            report={'trace':[]};calls=[];runtime=SimpleNamespace(state=lambda:copy.deepcopy(state),controls=0)
            def execute(target,g,phase,goal,raw):
                calls.append((g,phase));runtime.controls+=1
                report['trace'].append({'before':copy.deepcopy(state),'after':copy.deepcopy(state)})
                return copy.deepcopy(state)
            with patch('execution_policy_semantics.load_semantics_replay',return_value=(req,source)):
                result=run_semantics_runtime(runtime,settings,report,lambda:None,execute)
            self.assertEqual(calls,[(-1.,'policy-model') if grip=='model' else (1.,'policy-open')]*5)
            self.assertEqual(result['new_policy_queries'],0);self.assertEqual(report['queries'],[])
            if grip=='open':self.assertTrue(result['identity_check']['passed'])

    def test_both_program_references_execute_full_contact_sequence(self):
        from execution_contact import run_contact_experiment
        import test_execution_contact as contact_fixture
        references=[c for c in semantics_cases() if c.get('hybrid_role')=='reference']
        self.assertEqual(len(references),2)
        for case in references:
            read,step,calls=contact_fixture.ContactTests().simulator()
            result=run_contact_experiment(read,step,case)
            self.assertTrue(result['success'],case['name'])
            self.assertEqual(result['outcome'],'verified_lift_hold')
            self.assertTrue(result['close_attempted']);self.assertTrue(result['lift_attempted'])
            self.assertEqual(sum(phase=='hold' for _,_,_,phase in calls),40)

    def test_program_reference_registration_rejects_policy_and_mutated_reference(self):
        from execution_contact import run_contact_experiment
        import test_execution_contact as contact_fixture
        altered=copy.deepcopy(semantics_cases()[0]);altered['grasp_height_m']=.03
        for case in (altered,self.case()):
            read,step,calls=contact_fixture.ContactTests().simulator()
            with self.assertRaisesRegex(ValueError,'undeclared contact case'):run_contact_experiment(read,step,case)
            self.assertEqual(calls,[])


if __name__=='__main__':unittest.main()
