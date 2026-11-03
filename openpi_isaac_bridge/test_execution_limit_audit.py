import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
from execution_limit_audit import limit_cases,direction_metrics,load_replay,run_replay,run_frozen
from execution_alignment_probe import validate_case


class LimitAuditTests(unittest.TestCase):
    def test_matrix_keeps_reference_gains_and_separates_queries_from_replay(self):
        cases=limit_cases();self.assertEqual(len(cases),16)
        self.assertEqual(len({c['name'] for c in cases}),16)
        for c in cases:
            validate_case(c);self.assertEqual(c['arm_profile'],'k1687p5-d50');self.assertEqual(c['finger_profile'],'kp350-kd50')
        frozen=[c for c in cases if c['limit_audit_role']=='frozen'];replay=[c for c in cases if c['limit_audit_role']=='replay']
        self.assertEqual(len(frozen),8);self.assertEqual(sum(c['query_repeats'] for c in frozen),48)
        self.assertEqual({tuple(c['object_offset_xy_m']) for c in frozen},{(0.,0.),(-.02,0.),(.02,0.),(0.,-.02),(0.,.02)})
        self.assertEqual({(c['source_prefix'],c['cap_m']) for c in replay},{(p,c) for p in (1,5) for c in (.0005,.001,.002)})
        self.assertTrue(all('hybrid_role' not in c for c in frozen+replay))

    def test_direction_metric_rotates_from_policy_world_into_simulator_world(self):
        state={'tcp_world_m':[0.,0.,.1],'object_pos_world_m':[0.,-.1,.095]}
        R=[[0,-1,0],[1,0,0],[0,0,1]]
        result=direction_metrics([[1.,0,0,0,0,0,1]]*10,state,R)
        np.testing.assert_allclose(result['first_action']['decoded_world_delta_m'],[0.,-.05,0.])
        self.assertAlmostEqual(result['first_action']['xy_cosine_to_object'],1.)
        self.assertAlmostEqual(result['mean_first5']['cosine_to_grasp_point'],1.)

    def fixture(self,folder,prefix=1):
        name=f'hybrid-p{prefix}-geometry-r1'
        initial={'joint_positions_rad':[0.]*7,'tcp_world_m':[.5,0.,.12],
            'hand_position_world_m':[.5,0.,.217],'hand_quat_world_xyzw':[1.,0,0,0],
            'object_pos_world_m':[.51,0.,.021]}
        calibration={'world_from_isaac_rotation':np.eye(3).tolist()}
        actions=[[.1,0,-.2,0,0,0,1.]]*100
        source={'status':'completed','initial':initial,'settings':{'execution_case':{'name':name},'calibration':calibration,'expected_joint_gains':{}},
            'trace':[{'raw_action':a,'phase':'policy-approach','after':copy.deepcopy(initial)} for a in actions]}
        payload=json.dumps(source).encode();(folder/'source.json').write_bytes(payload)
        digest=hashlib.sha256(payload).hexdigest()
        manifest=json.dumps({'status':'completed','cases':[{'case':{'name':name},'summary':{'result_sha256':digest}}]}).encode()
        (folder/'source_hybrid_run.json').write_bytes(manifest)
        request={'source_prefix':prefix,'source_result_file':'source.json','source_result_sha256':digest,
            'source_manifest_sha256':hashlib.sha256(manifest).hexdigest(),'actions':actions}
        path=folder/'p1.json';path.write_text(json.dumps(request))
        return path,source

    def test_replay_rejects_modified_actions_and_source_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);path,source=self.fixture(folder);request,_=load_replay(path)
            request['actions'][0][0]=-.5;path.write_text(json.dumps(request))
            with self.assertRaises(ValueError):load_replay(path)
            path,source=self.fixture(folder);(folder/'source.json').write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError,'SHA'):load_replay(path)

    def test_same_archived_actions_are_used_under_three_caps_with_fingers_open(self):
        with tempfile.TemporaryDirectory() as temp:
            path,source=self.fixture(Path(temp));initial=source['initial'];collected=[]
            for case in [c for c in limit_cases() if c['limit_audit_role']=='replay' and c['source_prefix']==1]:
                seen=[];r=SimpleNamespace(state=lambda:copy.deepcopy(initial))
                settings={'execution_case':case,'limit_replay_path':str(path),**{k:source['settings'][k] for k in ('calibration','expected_joint_gains')}}
                def execute(target,grip,phase,fixed,action):seen.append((target,grip,action));return copy.deepcopy(initial)
                result=run_replay(r,settings,{},lambda:None,execute)
                self.assertEqual(result['outcome'],'archived_prefix_complete');self.assertEqual(len(seen),100)
                self.assertTrue(all(grip==1. for _,grip,_ in seen));collected.append([a for _,_,a in seen])
                delta=np.array(seen[0][0]['tcp97_target_world_m'])-initial['tcp_world_m']
                self.assertAlmostEqual(np.linalg.norm(delta),case['cap_m'])
            self.assertEqual(collected[0],collected[1]);self.assertEqual(collected[1],collected[2])

    def test_frozen_queries_use_identical_modified_inputs_and_preserve_failure_archive(self):
        from frozen_coordinate_probe import digest_observation
        for mutate in (False,True):
            with tempfile.TemporaryDirectory() as temp:
                case=next(c for c in limit_cases() if c['name']=='limits-frozen-wrist-blank')
                folder=Path(temp);state={'tcp_world_m':[0.,0.,.12],'object_pos_world_m':[.01,0.,.021]};step=[42]
                r=SimpleNamespace(output=folder,controls=0,state=lambda:copy.deepcopy(state),arm=None,tensor=None,
                    base=SimpleNamespace(sim=SimpleNamespace(get_physics_step_count=lambda:step[0])))
                obs={'prompt':case['prompt'],'observation/state':np.zeros(8,dtype=np.float32),
                    'observation/image':np.full((224,224,3),99,dtype=np.uint8),'observation/wrist_image':np.full((224,224,3),88,dtype=np.uint8)}
                report={};seen=[]
                def infer(client,observation,request,on_response):
                    seen.append(digest_observation(observation));self.assertFalse(observation['observation/wrist_image'].any())
                    self.assertTrue((observation['observation/image']==99).all())
                    chunk=[[0.]*7]*10;on_response({'actions':chunk},.1)
                    if mutate:step[0]+=1
                    return {**request,'raw_action_chunk':chunk},.1
                modules={'openpi_client':SimpleNamespace(),'openpi_client.websocket_client_policy':SimpleNamespace(WebsocketClientPolicy=lambda **kwargs:object())}
                with patch.dict('sys.modules',modules),patch('openpi_health.check_openpi_health',return_value={'status':'healthy'}),patch('openpi_step_probe.read_server_metadata',return_value={}),patch('replanning_probe.capture_round',return_value=({},42)),patch('online_registered_probe.prepare_capture_observation',return_value=(obs,{},{})),patch('online_registered_probe.query_once',side_effect=infer):
                    if mutate:
                        with self.assertRaisesRegex(RuntimeError,'physics changed'):run_frozen(r,{'execution_case':case,'calibration':{'world_from_isaac_rotation':np.eye(3).tolist()},'prompt':case['prompt'],'server_host':'127.0.0.1','server_port':8000},report,lambda:None)
                    else:
                        result=run_frozen(r,{'execution_case':case,'calibration':{'world_from_isaac_rotation':np.eye(3).tolist()},'prompt':case['prompt'],'server_host':'127.0.0.1','server_port':8000},report,lambda:None)
                        self.assertTrue(result['zero_action_controls']);self.assertEqual(len(seen),6);self.assertEqual(len(set(seen)),1)
                self.assertEqual(report['frozen_queries'][-1]['status'],'failed' if mutate else 'validated')
                with zipfile.ZipFile(folder/'frozen_query_bundle.zip') as archive:
                    self.assertIn('consumed_wrist.png',archive.namelist());self.assertIn('response_01.json',archive.namelist())


if __name__=='__main__':unittest.main()
