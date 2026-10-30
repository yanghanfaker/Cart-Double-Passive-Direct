import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch,MagicMock

import frozen_image_probe as probe


def geometry_fixture():
    import numpy as np
    from geometry_audit import pose_matrix
    theta=math.pi+.01;hand={'position_m':[.4,0,.3],'quat_xyzw':[math.sin(theta/2),0,0,math.cos(theta/2)]}
    H=np.array(pose_matrix(hand));sensor=H[:3,3]+H[:3,:3]@np.array([0,0,.1034])
    audit={'status':'completed','same_sensor_ik_body':True,'sensor_offset_closure':{'within_0p1mm_0p05deg':True},
        'sensor_body_offset':{'position_m':[0,0,.1034],'quat_xyzw':[0,0,0,1]},'sensor_body_world':hand,
        'sensor_tcp_world':{'position_m':sensor.tolist()},'finger_positions_isaac_m':[.04,.04],
        'object_root_world':{'position_m':[.5,.2,.021]},'cameras':{}}
    cal={'candidate_geometry_within_thresholds':True,'world_from_isaac_rotation':[[0,-1,0],[1,0,0],[0,0,1]],
        'world_from_isaac_translation_m':[-.66,0,.912]}
    return audit,cal


def source_fixture(root):
    import numpy as np
    from PIL import Image
    audit,cal=geometry_fixture();(root/'world_registration.json').write_text(json.dumps(cal))
    manifest={'status':'completed','all_geometry_checks_passed':True,'cases':[]}
    for case in probe.CASES:
        folder=root/case;folder.mkdir();geometry=copy.deepcopy(audit)
        result={'status':'completed','geometry_checks_passed':True,'calibration_markers_present':False,'cameras':{}}
        for view in ('external_camera','wrist_camera'):
            image=np.zeros((224,224,3),dtype=np.uint8);image[10:20,20:30]=[200,0,0]
            original=view+'.png';flipped=view+'_flip.png'
            Image.fromarray(image).save(folder/original);Image.fromarray(np.ascontiguousarray(image[:,::-1])).save(folder/flipped)
            geometry['cameras'][view]={'policy_png_sha256':hashlib.sha256((folder/original).read_bytes()).hexdigest()}
            result['cameras'][view]={'camera_parameters_passed':True,'frames':{'no_flip':original,'horizontal_flip':flipped}}
        (folder/'geometry_audit.json').write_text(json.dumps(geometry))
        (folder/'camera_alignment_setup.json').write_text(json.dumps({'scene_contains_markers':False,'cameras':{view:{'markers':[]} for view in result['cameras']}}))
        result_path=folder/'result.json';result_path.write_text(json.dumps(result))
        manifest['cases'].append({'name':case,'result':result_path.relative_to(root).as_posix(),'result_sha256':hashlib.sha256(result_path.read_bytes()).hexdigest()})
    source=root/'run.json';source.write_text(json.dumps(manifest));return source


class FrozenImageTests(unittest.TestCase):
    def test_maps_position_orientation_and_fingers_preserving_negative_w_branch(self):
        import numpy as np
        from geometry_audit import pose_matrix
        audit,cal=geometry_fixture();before=copy.deepcopy(audit)
        state,target,provenance=probe.registered_state(audit,cal)
        H=np.array(pose_matrix(audit['sensor_body_world']));R=np.array(cal['world_from_isaac_rotation']);t=np.array(cal['world_from_isaac_translation_m'])
        np.testing.assert_allclose(state[:3],R@(H[:3,3]+H[:3,:3]@np.array([0,0,.097]))+t,atol=1e-12)
        got=np.array(pose_matrix({'position_m':[0,0,0],'quat_xyzw':provenance['mapped_hand_quat_xyzw']}))[:3,:3]
        np.testing.assert_allclose(got,R@H[:3,:3],atol=1e-12)
        self.assertLess(provenance['mapped_hand_quat_xyzw'][3],0)
        self.assertGreater(math.hypot(*state[3:6]),math.pi)
        self.assertEqual(state[6:],[.04,-.04]);self.assertEqual(audit,before)
        np.testing.assert_allclose(target,R@np.array(audit['object_root_world']['position_m'])+t)

    def test_rejects_wrong_tcp_and_improper_world_registration(self):
        audit,cal=geometry_fixture();audit['sensor_body_offset']['position_m'][2]=.107
        with self.assertRaises(ValueError):probe.registered_state(audit,cal)
        audit,cal=geometry_fixture();cal['world_from_isaac_rotation']=[[1,0,0],[0,1,0],[0,0,-1]]
        with self.assertRaises(ValueError):probe.registered_state(audit,cal)

    def test_query_interleaves_and_flips_both_images_only(self):
        import numpy as np
        calls=[]
        class Client:
            def infer(self,obs):calls.append(copy.deepcopy(obs));return {'actions':np.zeros((10,7))}
        raw=np.zeros((224,224,3),dtype=np.uint8);raw[4:10,20:30]=[200,0,0]
        variants={c:{v:(raw.copy() if c=='no_flip' else raw[:,::-1].copy()) for v in ('external_camera','wrist_camera')} for c in probe.CONDITIONS}
        item={'name':'fixture','state_registered_float32':[0,0,0,0,0,0,.04,-.04],'diagnostic_cube_world_registered_m':[0,0,1]}
        rows=[];probe.run_queries(Client(),item,variants,'pick up the cube',3,rows.append)
        self.assertEqual([r['condition'] for r in rows],['no_flip','flip_x','flip_x','no_flip','no_flip','flip_x'])
        for obs,row in zip(calls,rows):
            np.testing.assert_array_equal(obs['observation/state'],np.array(item['state_registered_float32'],dtype=np.float32))
            for key,view in [('observation/image','external_camera'),('observation/wrist_image','wrist_camera')]:
                np.testing.assert_array_equal(obs[key],variants[row['condition']][view])
            self.assertEqual(set(obs),{'observation/state','observation/image','observation/wrist_image','prompt'})
            self.assertEqual(obs['prompt'],'pick up the cube');self.assertEqual(row['actions_executed'],0)
        summary=probe.summarize(rows);self.assertEqual(summary['flip_x']['first10']['queries'],3)
        self.assertEqual(summary['no_flip']['first10']['sample_std_mm'],0)
        for condition in probe.CONDITIONS:self.assertEqual(len({r['input_sha256'] for r in rows if r['condition']==condition}),1)

    def test_detects_mutated_input_short_chunk_and_invalid_future_action(self):
        import numpy as np
        raw=np.zeros((224,224,3),dtype=np.uint8)
        variants={c:{v:raw for v in ('external_camera','wrist_camera')} for c in probe.CONDITIONS}
        item={'name':'fixture','state_registered_float32':[0]*8,'diagnostic_cube_world_registered_m':[0,0,1]}
        for failure in ('mutate','short','future'):
            class Client:
                def infer(self,obs):
                    if failure=='mutate':obs['observation/image'][0,0,0]=1
                    chunk=[[0.]*7 for _ in range(9 if failure=='short' else 11)]
                    if failure=='future':chunk[-1][-1]=float('nan')
                    return {'actions':chunk}
            with self.subTest(failure=failure),self.assertRaises((RuntimeError,ValueError)):
                probe.run_queries(Client(),item,variants,'pick up the cube',1,lambda row:None)

    def test_prepares_all_sources_and_rejects_incorrect_flip(self):
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=source_fixture(root);out=root/'out';out.mkdir()
            prepared,sha=probe.prepare(source,out);self.assertEqual(len(prepared),2)
            self.assertEqual(sha,hashlib.sha256((root/'world_registration.json').read_bytes()).hexdigest())
            for item,variants in prepared:
                for view in variants['no_flip']:np.testing.assert_array_equal(variants['no_flip'][view][:,::-1],variants['flip_x'][view])
            path=root/probe.CASES[0]/'external_camera_flip.png';Image.fromarray(np.zeros((224,224,3),dtype=np.uint8)).save(path)
            with self.assertRaises(ValueError):probe.prepare(source,out)

    def test_full_cli_mock_writes_complete_report_with_zero_execution(self):
        import numpy as np
        class Client:
            def __init__(self,**kwargs):pass
            def get_server_metadata(self):return {}
            def infer(self,observation):return {'actions':np.zeros((10,7))}
        package=types.ModuleType('openpi_client');package.__path__=[]
        module=types.ModuleType('openpi_client.websocket_client_policy');module.WebsocketClientPolicy=Client
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=source_fixture(root);args=['probe','--source-manifest',str(source),'--output-root',str(root/'outputs'),'--repeats','5']
            with patch.object(sys,'argv',args),patch.dict(sys.modules,{'openpi_client':package,'openpi_client.websocket_client_policy':module}),patch.object(probe.socket,'create_connection',return_value=MagicMock()):
                probe.main()
            path=next((root/'outputs').glob('*/report.json'));report=json.loads(path.read_text())
            self.assertEqual(report['status'],'completed');self.assertEqual(report['policy_queries'],20);self.assertEqual(report['actions_executed'],0)
            self.assertEqual(report['inference_attempts'],20);self.assertNotIn('pending_query',report)
            for item in report['snapshots']:
                self.assertEqual(len(item['queries']),10)
                self.assertEqual(len({tuple(q['state_sent_float32']) for q in item['queries']}),1)


if __name__=='__main__':unittest.main()
