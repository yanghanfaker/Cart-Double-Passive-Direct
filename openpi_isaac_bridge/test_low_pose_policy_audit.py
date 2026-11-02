import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile
import numpy as np

import low_pose_policy_audit as audit
from test_online_registered_probe import capture
from online_registered_probe import prepare_capture_observation
from frozen_coordinate_probe import digest_observation
from kinematic_attribution import rotate


def item():
    return {'tcp_world_m':[.4,0,.1],'object_world_m':[.5,0,0],
        'hand_position_world_m':[.4,0,.197],'hand_quat_world_xyzw':[1,0,0,0],
        'world_rotation_L_from_I':[[0,-1,0],[1,0,0],[0,0,1]]}


def fixture(root,sensor_roundoff=0.):
    cal=None;cases=[]
    for repeat in (1,2,3):
        for mode in ('current','accumulated'):
            name=f'repeat-{repeat:02d}-{mode}';folder=root/name;folder.mkdir();queries=[];initial=None
            for height in (100,80,60):
                frame=folder/str(height);frame.mkdir();cal=capture(frame);g=json.loads((frame/'geometry_audit.json').read_bytes())
                hand={'position_m':[.4,0,height/1000+.097],'quat_xyzw':[1,0,0,0]};g['sensor_body_world']=hand
                g['sensor_tcp_world']['position_m']=(np.array(hand['position_m'])+rotate(hand['quat_xyzw'],[0,0,.1034])).tolist();g['object_root_world']['position_m']=[.5,0,0]
                g['sensor_tcp_world']['position_m'][0]+=sensor_roundoff
                (frame/'geometry_audit.json').write_text(json.dumps(g));obs,request,meta=prepare_capture_observation(frame,cal,'cube',g)
                response=[[0,-.1,-.1,0,0,0,1]]*10;request['raw_action_chunk']=response
                (frame/'online_replay_request.json').write_text(json.dumps(request));(frame/'online_model_response.json').write_text(json.dumps({'actions':response}))
                archive=folder/f'{height}.zip'
                with zipfile.ZipFile(archive,'w') as z:
                    for f in audit.FILES:z.write(frame/f,f)
                queries.append({'query':len(queries)+1,'input':meta,'archive':archive.name,'archive_sha256':audit.sha(archive.read_bytes()),
                    'executed_prefix_controls':5,'capture_physics_step':175,'camera_frames':{}})
                if initial is None:initial={'tcp_world_m':[.4,0,.1],'hand_quat_world_xyzw':[1,0,0,0],'joint_positions_rad':g['joint_positions_rad'],'object_pos_world_m':[.5,0,0]}
            d={'status':'completed','profile':'near-descent','reference_mode':mode,'queries_requested':120,'policy_queries':3,'inference_attempts':3,
                'action_controls':15,'hold_controls':10,'native_controls_executed':25,'boundary_reached':True,'stop_reason':'near_boundary_reached',
                'action_end_object_distance_m':.0599,'all_followup_inputs_changed':True,'queries':queries,'initial':initial,'first_input_sha256':queries[0]['input']['input_sha256']}
            payload=json.dumps(d).encode();(folder/'result.json').write_bytes(payload);(folder/'joint_gains.json').write_text('{}')
            (folder/'online_settings.json').write_text(json.dumps({'calibration':cal,'expected_joint_gains':{},'prompt':'cube'}))
            cases.append({'name':name,'reference_mode':mode,'repeat':repeat,'result':name+'/result.json','result_sha256':audit.sha(payload)})
    cb=json.dumps(cal).encode();(root/'world_registration.json').write_bytes(cb)
    m={'status':'completed','profile':'near-descent','queries_per_run':120,'repeats':3,'initial_inputs_identical':True,'all_followup_inputs_changed':True,
        'cases':cases,'world_registration_sha256':audit.sha(cb)}
    (root/'run.json').write_text(json.dumps(m));return root/'run.json'


class LowPoseTests(unittest.TestCase):
    def test_decode_frame_caps_direction_and_hypothetical_close(self):
        # LIBERO -Y maps to Isaac -X, away from a cube to the right.
        result=audit.analyze([[0,-.1,-.1,0,0,0,1]]*10,item())
        self.assertEqual(result['first_close_index'],1);self.assertEqual(result['first5']['close_requests'],5)
        self.assertAlmostEqual(result['first5']['decoded_pre_cap']['radial_outward_sum_mm'],25)
        self.assertAlmostEqual(result['first5']['bridge_capped']['radial_outward_sum_mm'],5*.5/np.sqrt(2))
        self.assertAlmostEqual(result['minimum_decoded_capped_direction_cosine'],1)
        self.assertAlmostEqual(result['first5']['first_close_hypothetical_before_action_height_mm']['bridge_capped'],100)
        self.assertEqual(result['first5']['translation_capped_actions'],5)
        toward=audit.analyze([[0,.001,0,0,0,0,-1]]*10,item())
        self.assertLess(toward['first5']['bridge_capped']['radial_outward_sum_mm'],0);self.assertIsNone(toward['first_close_index'])

    def test_reject_invalid_future_and_handle_zero_horizontal_offset(self):
        with self.assertRaises(ValueError):audit.analyze([[0]*7]*9,item())
        invalid=[[0]*7 for _ in range(10)];invalid[-1][-1]=float('nan')
        with self.assertRaises(ValueError):audit.analyze(invalid,item())
        s=item();s['object_world_m'][:2]=s['tcp_world_m'][:2]
        self.assertIsNone(audit.analyze([[0]*7]*10,s)['first10']['bridge_capped']['radial_outward_sum_mm'])
        s['world_rotation_L_from_I']=[[1,0,0],[0,1,0],[0,0,-1]]
        with self.assertRaises(ValueError):audit.analyze([[0]*7]*10,s)

    def test_reconstruct_18_exact_observations_and_reject_bad_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root,sensor_roundoff=1e-8);out=root/'out';out.mkdir();prepared=audit.prepare(source,out)
            self.assertEqual(len(prepared),18)
            for entry,obs,folder in prepared:
                self.assertEqual(digest_observation(obs),entry['input']['input_sha256']);self.assertAlmostEqual(entry['height_actual_mm'],entry['height_requested_mm'])
                self.assertAlmostEqual(entry['sensor_body_tcp_gap_m'],1e-8)
            with self.assertRaises(ValueError):audit.local_path(root,'../escape')
            original=(root/'repeat-01-current/100.zip').read_bytes();(root/'repeat-01-current/100.zip').write_bytes(original+b'corrupt')
            another=root/'another';another.mkdir()
            with self.assertRaisesRegex(ValueError,'archive SHA'):audit.prepare(source,another)

    def test_reject_geometric_gap_beyond_native_roundoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root,sensor_roundoff=1e-5);out=root/'out';out.mkdir()
            with self.assertRaisesRegex(ValueError,'capture height disagrees'):audit.prepare(source,out)

    def test_query_height_selection_uses_real_query_and_rejects_missing_height(self):
        q=lambda i,z:{'query':i,'input':{'state_provenance':{'source_sensor_position_world_m':[0,0,z-.0064],'source_hand_quat_xyzw':[1,0,0,0]}}}
        result={'initial':{'object_pos_world_m':[0,0,0]},'queries':[q(1,.101),q(2,.0998)]}
        self.assertEqual(audit.select_query(result,100)['query'],2)
        with self.assertRaises(ValueError):audit.select_query(result,60)

    def test_frozen_copies_alternating_order_and_full_future_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);obs={'observation/state':np.zeros(8,dtype=np.float32),'observation/image':np.zeros((224,224,3),dtype=np.uint8),
                'observation/wrist_image':np.zeros((224,224,3),dtype=np.uint8),'prompt':'cube'}
            prepared=[]
            for n in ('a','b'):
                folder=root/n;folder.mkdir();s={**item(),'name':n,'input':{'input_sha256':digest_observation(obs)},'queries':[]};prepared.append((s,copy.deepcopy(obs),folder))
            fingerprints=[]
            client=types.SimpleNamespace(infer=lambda o:(fingerprints.append(digest_observation(o)) or {'actions':np.zeros((10,7))}))
            report={'inference_attempts':0,'policy_queries':0};audit.run_queries(client,prepared,2,report,lambda:None)
            self.assertEqual(report['policy_queries'],4);self.assertEqual(len(set(fingerprints)),1)
            self.assertEqual([r['repeat'] for r in prepared[0][0]['queries']],[1,2])
            def mutated(o):o['observation/state'][0]=1;return {'actions':np.zeros((10,7))}
            with self.assertRaisesRegex(RuntimeError,'mutated'):audit.run_queries(types.SimpleNamespace(infer=mutated),prepared,1,report,lambda:None)
            self.assertEqual(obs['observation/state'][0],0)

    def test_summary_uses_snapshot_means_and_ignores_pending(self):
        items=[]
        for values in ([1,3],[9]):
            rows=[]
            for value in values:rows.append({'status':'completed','analysis':{f'first{k}':{c:{'radial_outward_sum_mm':value} for c in ('decoded_pre_cap','bridge_capped')} for k in (5,10)}})
            items.append({'reference_mode':'current','height_requested_mm':60,'queries':rows+[{'status':'querying'}]})
        s=audit.summarize(items)['current']['height_60mm'];self.assertEqual(s['completed_queries'],3)
        self.assertEqual(s['first5']['bridge_capped']['source_snapshot_means_mm']['mean'],5.5)

    def test_main_preserves_invalid_response_and_skips_incomplete_newest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source_root=root/'diagnostics/near_descent_reference_ab/0001';source_root.mkdir(parents=True);source=fixture(source_root)
            incomplete=source_root.parent/'0002';incomplete.mkdir();(incomplete/'run.json').write_text('{"status":"running"}')
            client=types.SimpleNamespace(infer=lambda o:{'actions':[[0]*7]*9})
            module=types.ModuleType('openpi_client.websocket_client_policy');module.WebsocketClientPolicy=lambda **kwargs:client
            args=['audit','--output-root',str(root/'out'),'--repeats','3']
            with patch.object(sys,'argv',args),patch.object(audit,'__file__',str(root/'openpi_isaac_bridge/low_pose_policy_audit.py')),patch.dict(sys.modules,{'openpi_client.websocket_client_policy':module}),patch.object(audit,'check_openpi_health',return_value={'status':'healthy'}):
                self.assertEqual(audit.main(),1)
            result=json.loads(next((root/'out').glob('*/report.json')).read_bytes());self.assertEqual(result['status'],'failed')
            self.assertEqual(result['policy_queries'],1);self.assertEqual(result['actions_executed'],0);self.assertEqual(result['snapshots'][0]['queries'][0]['status'],'failed')
            self.assertEqual(Path(result['source_manifest']).resolve(),source.resolve())
            self.assertTrue(list((root/'out').glob('*/*/response_01.json')))

    def test_main_completes_frozen_queries_without_simulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root);client=types.SimpleNamespace(infer=lambda o:{'actions':np.zeros((10,7))})
            module=types.ModuleType('openpi_client.websocket_client_policy');module.WebsocketClientPolicy=lambda **kwargs:client
            args=['audit','--source-manifest',str(source),'--output-root',str(root/'out'),'--repeats','3']
            with patch.object(sys,'argv',args),patch.dict(sys.modules,{'openpi_client.websocket_client_policy':module}),patch.object(audit,'check_openpi_health',return_value={'status':'healthy'}):
                self.assertEqual(audit.main(),0)
            result=json.loads(next((root/'out').glob('*/report.json')).read_bytes())
            self.assertEqual(result['status'],'completed');self.assertEqual(result['inference_attempts'],54);self.assertEqual(result['policy_queries'],54)
            self.assertEqual(result['actions_executed'],0);self.assertEqual(result['simulation_steps'],0)
            self.assertTrue(all(len(s['queries'])==3 for s in result['snapshots']))
            self.assertEqual(result['summary']['current']['height_60mm']['completed_queries'],9)


if __name__=='__main__':unittest.main()
