import copy
import json
from pathlib import Path
import tempfile
import unittest
import sys
import io
import hashlib
from unittest.mock import patch
import run_clean_scene_capture as runner

from clean_scene_probe import validate_clean_scene,color_hint,finish_clean_capture
from run_clean_scene_capture import prepare_cases


def fixtures():
    q=[0,-.569,0,-2.81,0,3.037,.741]
    isaac={'joint_names':[f'panda_joint{k}' for k in range(1,8)],'joint_positions_rad':q,
        'object_root_world':{'position_m':[.5,0,.021]}}
    layout={'label':'benchmark_after10wait','objects_of_interest':['akita_black_bowl_1','plate_1'],
        'objects':{'akita_black_bowl_1':{'position_world_m':[-.06,.2,.898]}},
        'joint_positions':{**{f'robot0_joint{k}':v for k,v in enumerate([0,-.17,-.02,-2.45,-.006,2.226,.768],1)},'gripper0_finger_joint1':.039,'gripper0_finger_joint2':-.039},
        'eef_observation':{'position_world_m':[-.21,-.011,1.174]}}
    native={'status':'completed','benchmark_initialization_available':True,'marker_test':{'all_marker_views_passed':True},'layouts':[layout]}
    cal={'world_from_isaac_rotation':[[1,0,0],[0,1,0],[0,0,1]],'world_from_isaac_translation_m':[-.66,0,.912]}
    candidate={'status':'proposal_not_applied','cameras':{name:{'frame':frame,'position_m':p,'rotation_ros':[[1,0,0],[0,1,0],[0,0,1]],
        'K_256':[[200,0,128],[0,200,128],[0,0,1]]} for name,frame,p in [('external_camera','isaac_world',[0,0,0]),('wrist_camera','panda_hand',[.05,0,0])]}}
    return native,isaac,cal,candidate


class CleanSceneTests(unittest.TestCase):
    def test_factor_cases_match_arm_then_xy_and_do_not_copy_bowl_height(self):
        source=fixtures();before=copy.deepcopy(source);cases=prepare_cases(*source)
        self.assertEqual([c['name'] for c in cases],['baseline','arm-only','arm-and-target-xy'])
        self.assertEqual(cases[0]['arm_positions_rad'],source[1]['joint_positions_rad'])
        self.assertEqual(cases[1]['arm_positions_rad'],cases[2]['arm_positions_rad'])
        self.assertEqual(cases[0]['object_xy_world_m'],cases[1]['object_xy_world_m'])
        for a,b in zip(cases[2]['object_xy_world_m'],[.6,.2]):self.assertAlmostEqual(a,b)
        self.assertEqual(cases[2]['baseline_cube_rest_position_m'][2],.021)
        self.assertEqual(len(cases[2]['object_xy_world_m']),2)
        self.assertEqual(source,before)

    def test_rejects_incomplete_reference_and_unsafe_proxy_location(self):
        source=fixtures();source[0]['status']='failed'
        with self.assertRaises(ValueError):prepare_cases(*source)
        config=prepare_cases(*fixtures())[-1];config['object_xy_world_m']=[.6,.8]
        with self.assertRaises(ValueError):validate_clean_scene(config)
        config=prepare_cases(*fixtures())[-1];config['reference_world_rotation'][0][0]=float('nan')
        with self.assertRaises(ValueError):validate_clean_scene(config)

    def test_saturation_hint_does_not_claim_visibility(self):
        import numpy as np
        image=np.zeros((256,256,3),dtype=np.uint8);image[120:124,120:124]=[255,0,0]
        hint=color_hint(image,[128,128]);self.assertEqual(hint['colored_pixels'],16)
        self.assertIn('background',hint['note'])
        self.assertEqual(color_hint(image,[300,300])['colored_pixels'],0)

    def test_clean_capture_saves_exact_horizontal_flip_and_checks_support_height(self):
        import numpy as np
        from PIL import Image
        config=prepare_cases(*fixtures())[-1]
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);q=config['arm_positions_rad']
            audit={'joint_names':[f'panda_joint{k}' for k in range(1,8)],'joint_positions_rad':q,
                'object_root_world':{'position_m':[.6,.2,.021]},
                'sensor_body_world':{'position_m':[.4,0,.3],'quat_xyzw':[0,0,0,1]},
                'sensor_tcp_world':{'position_m':[.4,0,.4034]},'cameras':{}}
            for name,c in config['camera_candidate']['cameras'].items():
                p=c['position_m'] if c['frame']=='isaac_world' else [.45,0,.3]
                audit['cameras'][name]={'live_pose_world_ros':{'position_m':p,'quat_xyzw':[0,0,0,1]},'live_intrinsic_matrix':c['K_256'],
                    'projections':{'object_root':{'uv_px':[128,128]}}}
            setup={'scene_contains_markers':False,'cameras':{k:{'markers':[]} for k in audit['cameras']}}
            (out/'camera_alignment_setup.json').write_text(json.dumps(setup))
            image=np.zeros((224,224,3),dtype=np.uint8);image[50:70,10:20]=[255,0,0]
            for name in audit['cameras']:Image.fromarray(image).save(out/f'{name}_policy224.png')
            images={name:np.zeros((256,256,3),dtype=np.uint8) for name in audit['cameras']}
            def run():
                (out/'geometry_audit.json').write_text(json.dumps(audit))
                return finish_clean_capture(config,out,images)
            result=run();self.assertTrue(result['geometry_checks_passed'])
            for name in images:np.testing.assert_array_equal(np.array(Image.open(out/f'{name}_policy224_flip_x.png')),image[:,::-1])
            audit['object_root_world']['position_m'][2]=-.1
            self.assertFalse(run()['geometry_checks_passed'])
            setup['scene_contains_markers']=True;(out/'camera_alignment_setup.json').write_text(json.dumps(setup))
            with self.assertRaises(RuntimeError):run()

    def test_runner_executes_three_unmarked_cases_and_preserves_failure(self):
        for fail in (False,True):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);native,isaac,cal,candidate=fixtures();src=root/'native';src.mkdir();al=root/'alignment';al.mkdir()
                reference=src/'reference';reference.mkdir();native['layouts'][0]['image_folder']='reference';native['layouts'][0]['images']={}
                for view in ('agentview','robot0_eye_in_hand'):
                    name=view+'.png';content=b'previously validated clean image';(reference/name).write_bytes(content)
                    native['layouts'][0]['images'][view]={'policy224':{'file':name,'sha256':hashlib.sha256(content).hexdigest()}}
                report=src/'report.json';report.write_text(json.dumps(native))
                ip=src/'isaac_source.json';ip.write_text(json.dumps(isaac));cp=src/'world_registration.json';cp.write_text(json.dumps(cal))
                proposal=al/'candidate_camera_parameters.json';proposal.write_text(json.dumps(candidate))
                alignment=al/'run.json';alignment.write_text(json.dumps({'candidate_sha256':hashlib.sha256(proposal.read_bytes()).hexdigest()}))
                (src/'run.json').write_text(json.dumps({'isaac_source_sha256':hashlib.sha256(ip.read_bytes()).hexdigest(),
                    'world_registration_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'source_alignment':str(alignment)}))
                commands=[]
                class Process:
                    def __init__(self,command,**kwargs):
                        commands.append((command,kwargs));self.stdout=io.StringIO('')
                        if not fail:
                            out=Path(command[command.index('--output-root')+1])/'capture';out.mkdir(parents=True)
                            result={'status':'completed','geometry_checks_passed':True,'cameras':{view:{'object_center_projection_px':[128,128],
                                'frames':{'no_flip':'original.png','horizontal_flip':'flip.png'}} for view in ('external_camera','wrist_camera')}}
                            (out/'clean_scene_result.json').write_text(json.dumps(result))
                    def __enter__(self):return self
                    def __exit__(self,*args):pass
                    def wait(self):return 1 if fail else 0
                args=['capture','--native-report',str(report),'--output-root',str(root/'out')]
                with patch.object(sys,'argv',args),patch.object(runner.subprocess,'Popen',Process):
                    self.assertEqual(runner.main(),1 if fail else 0)
                result=json.loads(next((root/'out').glob('*/run.json')).read_text())
                self.assertEqual(result['status'],'failed' if fail else 'completed')
                self.assertEqual(len(commands),1 if fail else 3)
                for command,kwargs in commands:
                    self.assertIn('--clean-scene-json',command);self.assertNotIn('--query',command);self.assertNotIn('--camera-alignment-json',command)
                    self.assertNotIn('VIRTUAL_ENV',kwargs['env'])


if __name__=='__main__':unittest.main()
