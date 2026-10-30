import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock,patch
import numpy as np
from PIL import Image

import online_registered_probe as probe
import run_online_registered_capture as runner
from test_frozen_image_probe import geometry_fixture
from test_clean_scene_probe import fixtures
from run_clean_scene_capture import prepare_cases
from test_registered_replay_probe import fixture
from frozen_coordinate_probe import digest_observation


def capture(root):
    audit,cal=geometry_fixture()
    audit['joint_names']=[f'panda_joint{k}' for k in range(1,8)]
    audit['joint_positions_rad']=[0,-.17,-.02,-2.45,-.006,2.226,.768]
    audit['cameras']={}
    for view in ('external_camera','wrist_camera'):
        rgb=np.zeros((224,224,3),dtype=np.uint8);rgb[10:20,30:40]=[150,20,10]
        plain=root/f'{view}_policy224.png';Image.fromarray(rgb).save(plain)
        Image.fromarray(np.ascontiguousarray(rgb[:,::-1])).save(root/f'{view}_policy224_flip_x.png')
        audit['cameras'][view]={'policy_png_sha256':hashlib.sha256(plain.read_bytes()).hexdigest()}
    (root/'geometry_audit.json').write_text(json.dumps(audit))
    (root/'clean_scene_result.json').write_text(json.dumps({'geometry_checks_passed':True,'calibration_markers_present':False}))
    return cal


class OnlineTests(unittest.TestCase):
    def test_current_capture_state_and_exact_flip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cal=capture(root);obs,request,meta=probe.prepare_observation(root,cal,'pick up the cube')
            self.assertEqual(obs['observation/state'].dtype,np.float32)
            self.assertEqual(request['source_input_sha256'],digest_observation(obs))
            self.assertEqual(meta['input_sha256'],digest_observation(obs))
            self.assertTrue(obs['observation/image'].flags.c_contiguous)
            self.assertEqual(obs['observation/image'][10,190].tolist(),[150,20,10])
            self.assertAlmostEqual(obs['observation/state'][7],-.04,places=7)
            # This source hand is beyond pi; preserve its signed branch after registration.
            self.assertLess(meta['state_provenance']['mapped_hand_quat_xyzw'][3],0)
            result,_=probe.query_once(types.SimpleNamespace(infer=lambda _: {'actions':np.zeros((10,7))}),obs,request)
            self.assertEqual(len(result['raw_action_chunk']),10)

    def test_reject_corrupt_flip_and_failed_clean_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cal=capture(root)
            Image.fromarray(np.zeros((224,224,3),dtype=np.uint8)).save(root/'wrist_camera_policy224_flip_x.png')
            with self.assertRaises(ValueError):probe.prepare_observation(root,cal,'cube')
            capture(root);(root/'clean_scene_result.json').write_text(json.dumps({'geometry_checks_passed':False,'calibration_markers_present':False}))
            with self.assertRaises(ValueError):probe.prepare_observation(root,cal,'cube')

    def test_entire_new_chunk_and_mutation_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);obs,request,_=probe.prepare_observation(root,capture(root),'cube')
            for mode in ('future_nan','short','mutation'):
                def infer(o):
                    chunk=np.zeros((10,7))
                    if mode=='future_nan':chunk[-1,-1]=float('nan')
                    if mode=='short':chunk=chunk[:5]
                    if mode=='mutation':o['observation/image'][0,0,0]=1
                    return {'actions':chunk}
                with self.subTest(mode=mode),self.assertRaises((ValueError,RuntimeError)):
                    probe.query_once(types.SimpleNamespace(infer=infer),copy.deepcopy(obs),request)

    def test_one_query_then_motion_and_stale_or_invalid_stop(self):
        for mode in ('success','stale','invalid'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);cal=capture(root)
                chunk=np.zeros((10,7))
                if mode=='invalid':chunk[9,6]=float('nan')
                client=MagicMock();client.infer.return_value={'actions':chunk};client.get_server_metadata.return_value={}
                module=types.ModuleType('openpi_client.websocket_client_policy');module.WebsocketClientPolicy=MagicMock(return_value=client)
                base=types.SimpleNamespace(sim=types.SimpleNamespace(get_physics_step_count=MagicMock(side_effect=[35,36 if mode=='stale' else 35])))
                motion=MagicMock(return_value={'final_target_reached':True,'completed_action_periods':5,'completed_settle_periods':10})
                settings={'calibration':cal,'prompt':'cube','server_host':'127.0.0.1','server_port':8000}
                with patch.dict(sys.modules,{'openpi_client.websocket_client_policy':module}),patch.object(probe.socket,'create_connection',MagicMock()),patch.object(probe,'run_registered_replay',motion):
                    if mode=='success':probe.run_online_registered(None,base,None,settings,root,None)
                    else:
                        with self.assertRaises((ValueError,RuntimeError)):probe.run_online_registered(None,base,None,settings,root,None)
                result=json.loads((root/'online_registered_result.json').read_bytes())
                self.assertEqual(client.infer.call_count,1);self.assertEqual(result['policy_queries'],1);self.assertEqual(result['inference_attempts'],1)
                self.assertEqual(motion.call_count,1 if mode=='success' else 0)
                self.assertEqual(result['status'],'completed' if mode=='success' else 'failed')
                if mode=='invalid':self.assertTrue((root/'invalid_model_response.txt').is_file())
                if mode!='success':self.assertEqual(result['completed_action_periods'],0)

    def test_server_failure_saved_before_inference_or_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cal=capture(root)
            with patch.object(probe.socket,'create_connection',side_effect=ConnectionRefusedError('offline')),patch.object(probe,'run_registered_replay') as motion:
                with self.assertRaises(ConnectionRefusedError):probe.run_online_registered(None,None,None,{'calibration':cal,'prompt':'cube','server_host':'127.0.0.1','server_port':8000},root,None)
                motion.assert_not_called()
            result=json.loads((root/'online_registered_result.json').read_bytes())
            self.assertEqual(result['inference_attempts'],0);self.assertEqual(result['status'],'failed')

    def test_two_scene_runner_and_failure_preservation(self):
        for fail in (False,True):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);previous=root/'previous';previous.mkdir();clean=root/'clean';clean.mkdir()
                cal=fixtures()[2];cal['candidate_geometry_within_thresholds']=True
                cb=json.dumps(cal).encode();(clean/'world_registration.json').write_bytes(cb)
                frozen={'source_manifest':str(clean/'run.json'),'world_registration_sha256':hashlib.sha256(cb).hexdigest(),'prompt':'pick up the cube'}
                fb=json.dumps(frozen).encode();(previous/'source_frozen_report.json').write_bytes(fb)
                prior={'status':'completed','all_final_targets_reached':True,'source_sha256':hashlib.sha256(fb).hexdigest(),'cases':[]}
                config=prepare_cases(*fixtures())[-1]
                for name in ('arm-only','arm-and-target-xy'):
                    out=previous/name;out.mkdir();r=json.dumps({'status':'completed','final_target_reached':True}).encode();(out/'result.json').write_bytes(r)
                    replay=fixture();replay['source_arm_positions_rad']=config['arm_positions_rad']
                    (out/'replay_request.json').write_text(json.dumps(replay));(out/'clean_scene_request.json').write_text(json.dumps(config))
                    prior['cases'].append({'name':name,'result':name+'/result.json','result_sha256':hashlib.sha256(r).hexdigest()})
                source=previous/'run.json';source.write_text(json.dumps(prior));commands=[]
                class Process:
                    def __init__(self,command,**kwargs):
                        commands.append((command,kwargs));self.stdout=io.StringIO('')
                        if not fail:
                            out=Path(command[command.index('--output-root')+1])/'capture';out.mkdir(parents=True)
                            (out/'online_registered_result.json').write_text(json.dumps({'status':'completed','policy_queries':1,'inference_attempts':1,
                                'completed_action_periods':5,'completed_settle_periods':10,'final_target_reached':False}))
                    def __enter__(self):return self
                    def __exit__(self,*args):pass
                    def wait(self):return 1 if fail else 0
                args=['online','--source-manifest',str(source),'--output-root',str(root/'out')]
                with patch.object(sys,'argv',args),patch.object(runner.socket,'create_connection',MagicMock()),patch.object(runner.subprocess,'Popen',Process):
                    self.assertEqual(runner.main(),1 if fail else 0)
                result=json.loads(next((root/'out').glob('*/run.json')).read_bytes())
                self.assertEqual(result['status'],'failed' if fail else 'completed')
                self.assertEqual(len(commands),1 if fail else 2)
                for command,kwargs in commands:
                    self.assertIn('--online-registered-json',command);self.assertNotIn('--registered-replay-json',command)
                    self.assertNotIn('VIRTUAL_ENV',kwargs['env'])
                if not fail:self.assertFalse(result['all_final_targets_reached'])


if __name__=='__main__':unittest.main()
