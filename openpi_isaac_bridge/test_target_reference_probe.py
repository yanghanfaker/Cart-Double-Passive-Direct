import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np

import target_reference_probe as probe
import run_target_reference_ab as runner
from test_registered_replay_probe import fixture
from test_clean_scene_probe import fixtures
from run_clean_scene_capture import prepare_cases
from kinematic_attribution import rotate


def state(x=.4):
    q=[1,0,0,0];tcp=[x,0,.203]
    return {'tcp_world_m':tcp,'hand_position_world_m':(np.array(tcp)-rotate(q,[0,0,.097])).tolist(),
        'hand_quat_world_xyzw':q,'object_pos_world_m':[.5,0,.021],'joint_positions_rad':[0]*7,
        'joint_velocities_rad_s':[0]*7,'joint_position_targets_rad':None,'finger_positions_m':[.04,.04]}


def make_source(root):
    capture=root/'capture';capture.mkdir();(root/'queries').mkdir()
    config=prepare_cases(*fixtures())[1];cal=fixtures()[2];cal['candidate_geometry_within_thresholds']=True
    (root/'clean_scene_request.json').write_text(json.dumps(config));cb=json.dumps(cal).encode();(root/'world_registration.json').write_bytes(cb)
    (capture/'joint_gains.json').write_text(json.dumps({'panda_joint1':{'stiffness':1125,'damping':25}}))
    result={'status':'completed','mode':'openpi-grasp','native_controls_executed':100,'queries':[]};lines=[]
    for i in range(20):
        request=fixture();request['source_input_sha256']=f'input{i}';request['source_repeat']=i+1;request['source_arm_positions_rad']=config['arm_positions_rad']
        request['raw_action_chunk']=[[.01*(i+1),.001*j,0,0,0,0,-1] for j in range(10)]
        archive=root/'queries'/f'q{i}.zip'
        with zipfile.ZipFile(archive,'w') as z:
            z.writestr('online_replay_request.json',json.dumps(request));z.writestr('online_model_response.json',json.dumps({'actions':request['raw_action_chunk']}))
            z.writestr('geometry_audit.json',json.dumps({'object_root_world':{'position_m':[.5,0,.021]}}))
        result['queries'].append({'query':i+1,'status':'validated','archive':f'../queries/q{i}.zip','archive_sha256':runner.digest(archive.read_bytes()),'input':{'input_sha256':request['source_input_sha256']}})
        for j,a in enumerate(request['raw_action_chunk'][:5],1):
            n=i*5+j;lines.extend([{'event':'attempt','control':n,'phase':'approach','gripper_applied':1,
                'target':{'source_query':i+1,'source_prefix_index':j,'raw_model_action':a}},
                {'event':'result','control':n,'status':'checked','physics_substeps':5}])
    (capture/'motion.jsonl').write_text('\n'.join(json.dumps(x) for x in lines))
    rb=json.dumps(result).encode();(capture/'result.json').write_bytes(rb)
    manifest={'status':'completed','native_controls_executed':100,'result':'capture/result.json','result_sha256':runner.digest(rb),'world_registration_sha256':runner.digest(cb)}
    (root/'run.json').write_text(json.dumps(manifest));return manifest


class ReferenceTests(unittest.TestCase):
    def test_accumulated_target_keeps_previous_unfinished_increment(self):
        actual=state();current=probe.TargetReference(actual,np.eye(3));acc=probe.TargetReference(actual,np.eye(3));action=[.01,0,0,0,0,0,-1]
        first,_=current.advance(action,actual,'current');a1,_=acc.advance(action,actual,'accumulated')
        np.testing.assert_allclose(first['tcp97_target_world_m'],a1['tcp97_target_world_m'])
        lagged=state(.4004);c2,ideal=current.advance(action,lagged,'current');a2,_=acc.advance(action,lagged,'accumulated')
        self.assertAlmostEqual(c2['tcp97_target_world_m'][0],.4009);self.assertAlmostEqual(a2['tcp97_target_world_m'][0],.401)
        np.testing.assert_allclose(a2['tcp97_target_world_m'],ideal['tcp97_target_world_m'])

    def test_rotation_accumulation_uses_world_composition_and_rigid_point_shift(self):
        from geometry_audit import pose_matrix
        ref=probe.TargetReference(state(),np.eye(3));action=[0,0,0,1,0,0,-1]
        for _ in range(2):target,ideal=ref.advance(action,state(),'accumulated')
        theta=np.radians(.2);q=ideal['hand_target_quat_world_xyzw']
        R=np.array(pose_matrix({'position_m':[0,0,0],'quat_xyzw':q}))[:3,:3]
        expected=np.array([[1,0,0],[0,-np.cos(theta),np.sin(theta)],[0,-np.sin(theta),-np.cos(theta)]])
        np.testing.assert_allclose(R,expected,atol=1e-14)
        np.testing.assert_allclose(np.array(target['ik107_target_world_m'])-rotate(q,[0,0,.010]),state()['tcp_world_m'],atol=1e-14)
        with self.assertRaises(ValueError):ref.advance(action,state(),'unknown')

    def test_constant_tracking_loss_separates_ideal_and_applied_target(self):
        from openpi_grasp_runtime import GraspRuntime
        for mode in ('current','accumulated'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp);gains={};(out/'joint_gains.json').write_text('{}')
                request=fixture();request.update(source_state_float32=[-.26,0,1.115,0,0,0,.04,-.04],source_input_sha256='source',source_repeat=1,
                    raw_action_chunk=[[.01,0,0,0,0,0,-1] for _ in range(100)])
                settings={'reference_mode':mode,'replay_request':request,'steps':100,'expected_joint_gains':gains,'source_object_position_world_m':[.5,0,.021],
                    'fixed_stream_sha256':hashlib.sha256(json.dumps(request['raw_action_chunk'],separators=(',',':')).encode()).hexdigest()}
                class Runtime:
                    def __init__(self,env,base,arm,settings,output,tensor,report,save):self.s=state();self.path=0;self.trace=io.StringIO();self.report=report
                    def state(self):return copy.deepcopy(self.s)
                    def step(self,target,gripper,phase):
                        p=np.array(self.s['tcp_world_m']);new=p+.9*(np.array(target['tcp97_target_world_m'])-p);self.path+=np.linalg.norm(new-p)
                        self.s=state(float(new[0]));self.report['native_controls_executed']+=1;return self.state()
                    def save_images(self,*args):pass
                with patch('openpi_grasp_runtime.GraspRuntime',Runtime):result=probe.run_target_reference(None,None,None,settings,out,None)
                self.assertEqual(result['action_controls'],100);self.assertEqual(result['hold_controls'],10);self.assertEqual(result['native_controls_executed'],110)
                self.assertTrue(result['final_applied_target_reached'])
                if mode=='current':self.assertGreater(result['final_ideal_position_error_m'],.0049)
                else:self.assertLess(result['final_ideal_position_error_m'],1e-9)

    def test_extracts_only_executed_prefixes_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m=make_source(root);req,g,provenance,_,_=runner.prepare_stream(root,m,100)
            self.assertEqual(len(req['raw_action_chunk']),100);self.assertEqual(len(provenance),20)
            self.assertEqual(req['raw_action_chunk'][4][1],.004);self.assertEqual(req['raw_action_chunk'][5][1],0)
            (root/'queries/q1.zip').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'archive SHA'):runner.prepare_stream(root,m,100)

    def test_outer_two_modes_share_stream_and_preserve_failure(self):
        for fail in (False,True):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);source=root/'source';source.mkdir();make_source(source);commands=[]
                class Process:
                    def __init__(self,command,**kwargs):
                        commands.append(command);self.stdout=io.StringIO('')
                        if not fail:
                            sf=Path(command[command.index('--online-registered-json')+1]);settings=json.loads(sf.read_bytes())
                            out=Path(command[command.index('--output-root')+1])/'capture';out.mkdir(parents=True)
                            (out/'online_registered_result.json').write_text(json.dumps({'status':'completed','action_controls':100,'hold_controls':10,
                                'reference_mode':settings['reference_mode'],'fixed_stream_sha256':settings['fixed_stream_sha256'],
                                'final_ideal_position_error_m':.001,'final_applied_target_error_m':.00001}))
                    def __enter__(self):return self
                    def __exit__(self,*args):pass
                    def wait(self):return 1 if fail else 0
                args=['ab','--source-manifest',str(source/'run.json'),'--output-root',str(root/'out'),'--steps','100']
                with patch.object(sys,'argv',args),patch.object(runner.subprocess,'Popen',Process):self.assertEqual(runner.main(),1 if fail else 0)
                self.assertEqual(len(commands),1 if fail else 2)
                report=json.loads(next((root/'out').glob('*/run.json')).read_bytes());self.assertEqual(report['status'],'failed' if fail else 'completed')
                settings=[json.loads(Path(c[c.index('--online-registered-json')+1]).read_bytes()) for c in commands]
                if not fail:
                    self.assertEqual(settings[0]['replay_request'],settings[1]['replay_request'])
                    self.assertEqual([s['reference_mode'] for s in settings],['current','accumulated'])


if __name__=='__main__':unittest.main()
