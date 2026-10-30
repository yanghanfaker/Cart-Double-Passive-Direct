import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock,patch
import zipfile
import numpy as np

import online_reference_probe as probe
import run_online_reference_ab as runner
from test_online_registered_probe import capture
from test_clean_scene_probe import fixtures
from run_clean_scene_capture import prepare_cases
from test_target_reference_probe import state
from kinematic_attribution import rotate


def make_ab_source(root):
    config=prepare_cases(*fixtures())[1];cal=fixtures()[2];cal['candidate_geometry_within_thresholds']=True
    initial=state();initial['joint_positions_rad']=config['arm_positions_rad'];gains={'panda_joint1':{'stiffness':1125,'damping':25}}
    cb=json.dumps(cal).encode();(root/'world_registration.json').write_bytes(cb)
    gb=json.dumps({'queries':[{'input':{'prompt':'custom cube prompt'}}]}).encode();(root/'source_grasp_result.json').write_bytes(gb)
    gm=json.dumps({'result_sha256':runner.digest(gb)}).encode();(root/'source_grasp_manifest.json').write_bytes(gm)
    manifest={'status':'completed','policy_queries':0,'steps':100,'fixed_stream_sha256':'fixed','world_registration_sha256':runner.digest(cb),'source_sha256':runner.digest(gm),'cases':[]}
    for mode in ('current','accumulated'):
        folder=root/mode;folder.mkdir();(folder/'clean_scene_request.json').write_text(json.dumps(config))
        (folder/'joint_gains.json').write_text(json.dumps(gains));(folder/'reference_settings.json').write_text(json.dumps({'calibration':cal,'expected_joint_gains':gains}))
        rb=json.dumps({'status':'completed','reference_mode':mode,'action_controls':100,'hold_controls':10,'final_applied_target_reached':True,'fixed_stream_sha256':'fixed','initial':initial}).encode()
        (folder/'result.json').write_bytes(rb);manifest['cases'].append({'mode':mode,'result':mode+'/result.json','result_sha256':runner.digest(rb)})
    (root/'run.json').write_text(json.dumps(manifest));return manifest


class OnlineReferenceTests(unittest.TestCase):
    def test_initial_guard_rejects_mismatch_and_nonfinite(self):
        expected=state();self.assertEqual(probe.validate_initial(expected,expected)['tcp_error_m'],0)
        for key,delta in [('tcp_world_m',.001),('object_pos_world_m',.002),('joint_positions_rad',.002),('tcp_world_m',float('nan'))]:
            bad=copy.deepcopy(expected);bad[key][0]+=delta
            with self.subTest(key=key,delta=delta),self.assertRaises(RuntimeError):probe.validate_initial(bad,expected)

    def test_live_inputs_follow_actual_state_and_accumulation_spans_queries(self):
        for mode in ('current','accumulated','invalid_third','stale_second'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);cal=capture(root);geometry=json.loads((root/'geometry_audit.json').read_bytes());(root/'joint_gains.json').write_text('{}')
                q=geometry['sensor_body_world']['quat_xyzw'];hp=geometry['sensor_body_world']['position_m'];tcp=np.array(hp)+rotate(q,[0,0,.097]);clock={'step':175};holder={};observations=[]
                def initial_state():
                    return {'tcp_world_m':tcp.tolist(),'hand_position_world_m':list(hp),'hand_quat_world_xyzw':q,
                        'joint_positions_rad':geometry['joint_positions_rad'],'object_pos_world_m':geometry['object_root_world']['position_m'],
                        'finger_positions_m':[.04,.04]}
                class Runtime:
                    def __init__(self,env,base,arm,settings,output,tensor,report,save):
                        self.s=initial_state();self.controls=0;self.path=0.;self.trace=io.StringIO();self.report=report;holder['runtime']=self;(output/'queries').mkdir()
                    def state(self):return copy.deepcopy(self.s)
                    def step(self,target,grip,phase):
                        before=np.array(self.s['tcp_world_m']);new=before+.9*(np.array(target['tcp97_target_world_m'])-before)
                        self.path+=float(np.linalg.norm(new-before));self.controls+=1;clock['step']+=5
                        self.s['tcp_world_m']=new.tolist();self.s['hand_position_world_m']=(new-rotate(q,[0,0,.097])).tolist();self.s['hand_quat_world_xyzw']=target['hand_target_quat_world_xyzw']
                        self.report['native_controls_executed']=self.controls;return self.state()
                    def save_images(self,*args):pass
                def snapshot(base,arm,folder,initial,tensor,index):
                    capture(folder);g=copy.deepcopy(geometry);s=holder['runtime'].state();g['sensor_body_world']={'position_m':s['hand_position_world_m'],'quat_xyzw':s['hand_quat_world_xyzw']}
                    g['sensor_tcp_world']['position_m']=(np.array(s['hand_position_world_m'])+rotate(q,[0,0,.1034])).tolist()
                    g['cameras']['external_camera']['frame']=[index+1];g['cameras']['wrist_camera']['frame']=[index+2]
                    (folder/'geometry_audit.json').write_text(json.dumps(g));return g,clock['step']
                def infer(obs):
                    observations.append(copy.deepcopy(obs));chunk=np.array([[0,.01,0,0,0,0,-1.]]*10)
                    if mode=='invalid_third' and len(observations)==3:chunk[9,0]=float('nan')
                    if mode=='stale_second' and len(observations)==2:clock['step']+=1
                    return {'actions':chunk}
                client=MagicMock();client.infer.side_effect=infer;client.get_server_metadata.return_value={}
                module=types.ModuleType('openpi_client.websocket_client_policy');module.WebsocketClientPolicy=MagicMock(return_value=client)
                base=types.SimpleNamespace(sim=types.SimpleNamespace(get_physics_step_count=lambda:clock['step']))
                settings={'reference_mode':mode if mode in ('current','accumulated') else 'accumulated','queries':10,'calibration':cal,
                    'expected_initial':initial_state(),'expected_joint_gains':{},'prompt':'cube','server_host':'127.0.0.1','server_port':8000}
                with (patch.dict(sys.modules,{'openpi_client.websocket_client_policy':module}),patch('openpi_grasp_runtime.GraspRuntime',Runtime),
                        patch.object(probe,'check_openpi_health',return_value={'status':'healthy'}),patch.object(probe,'capture_round',snapshot)):
                    if mode in ('current','accumulated'):result=probe.run_online_reference(None,base,None,settings,root,None)
                    else:
                        with self.assertRaises((ValueError,RuntimeError)):probe.run_online_reference(None,base,None,settings,root,None)
                result=json.loads((root/'online_registered_result.json').read_bytes())
                if mode in ('current','accumulated'):
                    self.assertEqual(result['policy_queries'],10);self.assertEqual(result['native_controls_executed'],60)
                    self.assertEqual(result['action_controls'],50);self.assertEqual(result['hold_controls'],10);self.assertTrue(result['all_followup_inputs_changed'])
                    self.assertEqual(clock['step'],475)
                    self.assertGreater(np.linalg.norm(observations[1]['observation/state'][:3]-observations[0]['observation/state'][:3]),.002)
                    if mode=='current':self.assertGreater(result['final_ideal_position_error_m'],.0024)
                    else:self.assertLess(result['final_ideal_position_error_m'],1e-9)
                else:
                    self.assertEqual(result['status'],'failed');self.assertEqual(result['hold_controls'],0)
                    self.assertEqual(result['native_controls_executed'],10 if mode=='invalid_third' else 5)
                for row in result['queries']:
                    with zipfile.ZipFile(root/row['archive']) as z:
                        self.assertIn('geometry_audit.json',z.namelist());self.assertIn('query_record.json',z.namelist())

    def test_outer_repeats_alternate_order_and_keep_source_prompt(self):
        for failure in (False,True):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);source=root/'source';source.mkdir();make_ab_source(source);commands=[]
                class Process:
                    def __init__(self,command,**kwargs):
                        commands.append(command);self.n=len(commands);self.stdout=io.StringIO('')
                        if not(failure and self.n==2):
                            sf=Path(command[command.index('--online-registered-json')+1]);s=json.loads(sf.read_bytes());out=Path(command[command.index('--output-root')+1])/'capture';out.mkdir(parents=True)
                            r={'status':'completed','reference_mode':s['reference_mode'],'policy_queries':10,'inference_attempts':10,'action_controls':50,'hold_controls':10,'native_controls_executed':60,
                                'first_input_sha256':'same','all_followup_inputs_changed':True,'requested_close_actions':0,
                                **{key:.001 for key in ('distance_reduction_m','final_xy_offset_m','final_ideal_position_error_m','max_applied_backlog_m','max_pre_update_ideal_gap_m','max_applied_target_error_m')}}
                            (out/'online_registered_result.json').write_text(json.dumps(r))
                    def __enter__(self):return self
                    def __exit__(self,*args):pass
                    def wait(self):return 1 if failure and self.n==2 else 0
                args=['onlineab','--source-manifest',str(source/'run.json'),'--output-root',str(root/'out'),'--queries','10','--repeats','3']
                with patch.object(sys,'argv',args),patch.object(runner,'check_openpi_health',return_value={'status':'healthy'}),patch.object(runner.subprocess,'Popen',Process):
                    self.assertEqual(runner.main(),1 if failure else 0)
                settings=[json.loads(Path(c[c.index('--online-registered-json')+1]).read_bytes()) for c in commands]
                self.assertTrue(all(s['prompt']=='custom cube prompt' for s in settings))
                report=json.loads(next((root/'out').glob('*/run.json')).read_bytes())
                if failure:self.assertEqual(len(commands),2);self.assertEqual(report['status'],'failed');self.assertEqual(report['summary']['current']['completed_runs'],1)
                else:
                    self.assertEqual([s['reference_mode'] for s in settings],['current','accumulated','accumulated','current','current','accumulated'])
                    self.assertTrue(report['initial_inputs_identical']);self.assertEqual(report['summary']['accumulated']['completed_runs'],3)
                    self.assertEqual(report['summary']['current']['distance_reduction_m']['sample_std'],0)

    def test_source_tamper_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m=make_ab_source(root);runner.prepare_source(root,m)
            (root/'source_grasp_result.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'grasp result SHA'):runner.prepare_source(root,m)


if __name__=='__main__':unittest.main()
