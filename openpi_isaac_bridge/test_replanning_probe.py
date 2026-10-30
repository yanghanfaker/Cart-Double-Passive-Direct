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

import replanning_probe as probe
import run_replanning_capture as runner
from registered_replay_probe import global_envelope
from test_online_registered_probe import capture
from test_clean_scene_probe import fixtures
from run_clean_scene_capture import prepare_cases


class ReplanningTests(unittest.TestCase):
    def test_shared_envelope_does_not_restart_at_chunk_boundary(self):
        reference={'tcp97_world_m':[.4,0,.3],'hand_quat_world_xyzw':[0,0,0,1],'object_position_world_m':[.5,.2,.02]}
        sample=copy.deepcopy(reference);sample['tcp97_world_m'][0]+=.009
        self.assertAlmostEqual(global_envelope(sample,reference)['tcp_displacement_m'],.009)
        sample['tcp97_world_m'][0]+=.002
        with self.assertRaises(RuntimeError):global_envelope(sample,reference)
        sample=copy.deepcopy(reference);sample['object_position_world_m'][1]+=.003
        with self.assertRaises(RuntimeError):global_envelope(sample,reference)
        sample=copy.deepcopy(reference);sample['hand_quat_world_xyzw']=[0,0,np.sin(np.radians(3)/2),np.cos(np.radians(3)/2)]
        with self.assertRaises(RuntimeError):global_envelope(sample,reference)

    def test_feedback_frame_checks_allow_identical_pixels(self):
        item={'capture_physics_step':175,'camera_frames':{'external_camera':2,'wrist_camera':3},
            'input':{'input_sha256':'a','state_sent_float32':[0]*8,'frames':{'image':{'array_sha256':'same'}}}}
        nxt=copy.deepcopy(item);nxt['capture_physics_step']=200;nxt['camera_frames']={'external_camera':7,'wrist_camera':8}
        nxt['input']['input_sha256']='b';nxt['input']['state_sent_float32'][0]=.002
        changed=probe.feedback_change(item,nxt)
        self.assertTrue(changed['state_changed']);self.assertTrue(changed['input_changed']);self.assertFalse(changed['image_array_changed']['image'])
        nxt['camera_frames']['wrist_camera']=3
        with self.assertRaises(RuntimeError):probe.feedback_change(item,nxt)
        nxt['camera_frames']['wrist_camera']=8;nxt['capture_physics_step']=201
        with self.assertRaises(RuntimeError):probe.feedback_change(item,nxt)

    def test_three_queries_no_intermediate_hold_and_preserve_partial_failures(self):
        for mode in ('success','invalid_second','stale_frame','physics_during_second_query'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);cal=capture(root);clock={'step':175,'queries':0};geometry=json.loads((root/'geometry_audit.json').read_bytes())
                base=types.SimpleNamespace(sim=types.SimpleNamespace(get_physics_step_count=lambda:clock['step']))
                def snapshot(base,arm,folder,initial,tensor,index):
                    capture(folder)
                    g=copy.deepcopy(geometry);shift=(index-1)*.002
                    g['sensor_body_world']['position_m'][0]+=shift;g['sensor_tcp_world']['position_m'][0]+=shift
                    for v in ('external_camera','wrist_camera'):
                        g['cameras'][v]['frame']=[2 if mode=='stale_frame' and index==2 else 2+(index-1)*5]
                    (folder/'geometry_audit.json').write_text(json.dumps(g));return g,clock['step']
                def infer(obs):
                    clock['queries']+=1;chunk=np.zeros((10,7))
                    if mode=='invalid_second' and clock['queries']==2:chunk[9,6]=float('nan')
                    if mode=='physics_during_second_query' and clock['queries']==2:clock['step']+=1
                    return {'actions':chunk}
                client=MagicMock();client.infer.side_effect=infer;client.get_server_metadata.return_value={}
                module=types.ModuleType('openpi_client.websocket_client_policy');module.WebsocketClientPolicy=MagicMock(return_value=client)
                motions=[]
                def motion(env,base,arm,request,folder,tensor,**kwargs):
                    motions.append((env,base,kwargs,request['source_state_float32']));hold=kwargs['settle_steps'];clock['step']+=(5+hold)*5
                    r={'status':'completed','completed_action_periods':5,'completed_settle_periods':hold,
                        'final_target_reached':hold==10,'final_tcp97_target_error_m':.00001,'final_rotation_error_deg':.001}
                    (folder/'registered_replay_result.json').write_text(json.dumps(r));return r
                settings={'calibration':cal,'prompt':'cube','server_host':'127.0.0.1','server_port':8000}
                env=object()
                with (patch.dict(sys.modules,{'openpi_client.websocket_client_policy':module}),patch.object(probe.socket,'create_connection',MagicMock()),
                        patch.object(probe,'capture_round',snapshot),patch.object(probe,'run_registered_replay',motion)):
                    if mode=='success':probe.run_replanning(env,base,None,settings,root,None)
                    else:
                        with self.assertRaises((ValueError,RuntimeError)):probe.run_replanning(env,base,None,settings,root,None)
                report=json.loads((root/'online_registered_result.json').read_bytes())
                if mode=='success':
                    self.assertEqual(report['completed_action_periods'],15);self.assertEqual(report['completed_settle_periods'],10)
                    self.assertEqual(report['policy_queries'],3);self.assertEqual(clock['step'],300)
                    self.assertEqual([m[2]['settle_steps'] for m in motions],[0,0,10]);self.assertTrue(report['all_followup_inputs_changed'])
                    self.assertTrue(report['final_target_reached'])
                    self.assertEqual([r['capture_physics_step'] for r in report['rounds']],[175,200,225])
                    self.assertNotEqual(motions[0][3],motions[1][3])
                    self.assertEqual(motions[0][2]['global_reference'],motions[2][2]['global_reference'])
                else:
                    self.assertEqual(report['status'],'failed');self.assertEqual(report['completed_action_periods'],5)
                    self.assertEqual(report['completed_settle_periods'],0);self.assertEqual(len(motions),1)
                    self.assertEqual(report['policy_queries'],1 if mode=='stale_frame' else 2)
                self.assertTrue(all(m[0] is env and m[1] is base for m in motions))

    def test_first_round_copies_current_capture_and_checks_step_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            initial=Path(tmp);cal=capture(initial)
            g=json.loads((initial/'geometry_audit.json').read_bytes());g['scene_setup']={'contains_nonphysical_camera_calibration_markers':False}
            g['physics_step_count_before']=g['physics_step_count_after']=175
            (initial/'geometry_audit.json').write_text(json.dumps(g));(initial/'camera_alignment_setup.json').write_text('{}')
            for name in ('external_camera','wrist_camera'):Image.fromarray(np.zeros((256,256,3),dtype=np.uint8)).save(initial/f'{name}.png')
            folder=initial/'first';folder.mkdir();base=types.SimpleNamespace(sim=types.SimpleNamespace(get_physics_step_count=lambda:175))
            result,step=probe.capture_round(base,None,folder,initial,None,1)
            self.assertEqual(step,175);self.assertEqual(result,g)
            with Image.open(folder/'wrist_camera_policy224.png') as a,Image.open(folder/'wrist_camera_policy224_flip_x.png') as b:
                np.testing.assert_array_equal(np.asarray(a)[:,::-1],np.asarray(b))

    def test_outer_runner_uses_single_arm_only_process_and_keeps_failure(self):
        for fail in (False,True):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);source=root/'source';source.mkdir();arm=source/'arm-only';arm.mkdir()
                config=prepare_cases(*fixtures())[1];(arm/'clean_scene_request.json').write_text(json.dumps(config))
                cal=fixtures()[2];cal['candidate_geometry_within_thresholds']=True;cb=json.dumps(cal).encode();(source/'world_registration.json').write_bytes(cb)
                prior={'status':'completed','final_target_reached':True,'policy_queries':1,'completed_action_periods':5,'completed_settle_periods':10,'input':{'prompt':'cube'}}
                pb=json.dumps(prior).encode();(arm/'result.json').write_bytes(pb)
                manifest={'status':'completed','all_final_targets_reached':True,'world_registration_sha256':hashlib.sha256(cb).hexdigest(),
                    'cases':[{'name':'arm-only','result':'arm-only/result.json','result_sha256':hashlib.sha256(pb).hexdigest()}]}
                source_manifest=source/'run.json';source_manifest.write_text(json.dumps(manifest));commands=[]
                class Process:
                    def __init__(self,command,**kwargs):
                        commands.append(command);self.stdout=io.StringIO('')
                        if not fail:
                            out=Path(command[command.index('--output-root')+1])/'capture';out.mkdir(parents=True)
                            (out/'online_registered_result.json').write_text(json.dumps({'status':'completed','policy_queries':3,'inference_attempts':3,
                                'completed_action_periods':15,'completed_settle_periods':10,'final_target_reached':False,'all_followup_inputs_changed':True}))
                    def __enter__(self):return self
                    def __exit__(self,*args):pass
                    def wait(self):return 1 if fail else 0
                with (patch.object(sys,'argv',['replan','--source-manifest',str(source_manifest),'--output-root',str(root/'out')]),
                        patch.object(runner.socket,'create_connection',MagicMock()),patch.object(runner.subprocess,'Popen',Process)):
                    self.assertEqual(runner.main(),1 if fail else 0)
                self.assertEqual(len(commands),1)
                report=json.loads(next((root/'out').glob('*/run.json')).read_bytes())
                self.assertEqual(report['status'],'failed' if fail else 'completed')
                settings=json.loads(Path(commands[0][commands[0].index('--online-registered-json')+1]).read_bytes())
                self.assertEqual(settings['mode'],'three-round-replan')
                if not fail:self.assertFalse(report['final_target_reached'])


if __name__=='__main__':unittest.main()
