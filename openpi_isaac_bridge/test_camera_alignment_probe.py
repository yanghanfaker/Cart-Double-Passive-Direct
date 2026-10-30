import copy
import math
from pathlib import Path
import unittest
import hashlib
import io
import json
import sys
import tempfile
from unittest.mock import patch
import run_camera_alignment as runner

from camera_alignment_probe import PIXELS,COLORS,marker_layout,quat_xyzw,analyze_markers,validate_candidate
from run_camera_alignment import summarize


class CameraAlignmentTests(unittest.TestCase):
    def candidate(self):
        return {'status':'proposal_not_applied','cameras':{name:{'frame':frame,'position_m':[0,0,0],
            'rotation_ros':[[1,0,0],[0,1,0],[0,0,1]],'K_256':[[200,0,128],[0,200,128],[0,0,1]]}
            for name,frame in [('external_camera','isaac_world'),('wrist_camera','panda_hand')]}}

    def test_candidate_rejects_malformed_rotation_and_intrinsics(self):
        validate_candidate(self.candidate())
        for field,value in [('rotation_ros',[[1,0,0],[0,1,0],[0,0,-1]]),('K_256',[[200,0,127],[0,200,128],[0,0,1]]),('position_m',[float('nan'),0,0])]:
            c=self.candidate();c['cameras']['wrist_camera'][field]=value
            with self.assertRaises(ValueError):validate_candidate(c)

    def test_rotation_quaternion_recovers_half_turn_and_general_rotation(self):
        import numpy as np
        from geometry_audit import pose_matrix
        for R in (np.diag([1,-1,-1]),np.diag([-1,1,-1]),np.diag([-1,-1,1]),np.array([[0,-1,0],[1,0,0],[0,0,1]]),np.eye(3)):
            got=np.array(pose_matrix({'position_m':[0,0,0],'quat_xyzw':quat_xyzw(R)}))[:3,:3]
            np.testing.assert_allclose(got,R,atol=1e-12)

    def test_marker_positions_project_to_asymmetric_expected_pixels(self):
        import numpy as np
        from camera_reference_audit import camera_from_world
        R=np.array([[0,-1,0],[1,0,0],[0,0,1]]);p=np.array([1.,2.,3.]);K=np.array([[300,0,128],[0,300,128],[0,0,1]])
        markers=marker_layout(p,R,K)
        for marker,pixel in zip(markers,PIXELS):
            np.testing.assert_allclose(camera_from_world(marker['world_position_m'],p,R,K),pixel,atol=1e-10)

    def test_render_like_colored_discs_identify_all_four_image_directions(self):
        import numpy as np
        yy,xx=np.indices((256,256));original=np.zeros((256,256,3),dtype=np.uint8)
        markers=marker_layout([0,0,0],np.eye(3),[[200,0,128],[0,200,128],[0,0,1]])
        for (x,y),color in zip(PIXELS,COLORS):original[(xx-x)**2+(yy-y)**2<=16]=np.array(color)*220
        for kind,image in [('identity',original),('flip_x',original[:,::-1]),('flip_y',original[::-1]),('rotate180',original[::-1,::-1])]:
            result=analyze_markers(image,markers)
            self.assertTrue(result['passed'],result);self.assertEqual(result['best_transform'],kind)
            self.assertAlmostEqual(result['best_max_pixel_error'],0)
        shifted=np.roll(original,6,axis=1)
        self.assertFalse(analyze_markers(shifted,markers)['passed'])
        self.assertFalse(analyze_markers(np.zeros_like(original),markers)['passed'])

    def test_aggregate_requires_consistent_mount_and_image_orientation(self):
        import numpy as np
        entry={'passed':True,'live_pose_in_requested_frame':{'position_m':[0,0,0],'rotation_ros':np.eye(3).tolist()},
            'landmarks':{'best_transform':'identity'}}
        results=[{'cameras':{name:copy.deepcopy(entry) for name in ('external_camera','wrist_camera')}} for _ in range(3)]
        self.assertTrue(summarize(results)['wrist_camera']['passed'])
        results[2]['cameras']['wrist_camera']['live_pose_in_requested_frame']['position_m'][0]=.001
        self.assertFalse(summarize(results)['wrist_camera']['passed'])
        results[2]['cameras']['external_camera']['landmarks']['best_transform']='rotate180'
        self.assertFalse(summarize(results)['external_camera']['passed'])

    def test_runner_preserves_three_results_and_stops_on_missing_capture(self):
        for missing in (False,True):
            with self.subTest(missing=missing),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);poses=[]
                for name in ('home','mixed34','mixed56'):
                    p=root/(name+'.json');p.write_text(json.dumps({'joint_names':[f'panda_joint{k}' for k in range(1,8)],
                        'joint_positions_rad':[0,-.569,0,-2.81,0,3.037,.741]}))
                    poses.append({'name':name,'isaac_audit':p.name,'isaac_audit_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
                source=root/'source.json';source.write_text(json.dumps({'status':'completed','poses':poses}))
                sha=hashlib.sha256(source.read_bytes()).hexdigest()
                audit=root/'report.json';audit.write_text(json.dumps({'status':'completed','source_manifest':str(source),'source_manifest_sha256':sha}))
                c=self.candidate();c['source_manifest_sha256']=sha
                (root/'candidate_camera_parameters.json').write_text(json.dumps(c));commands=[]
                class Child:
                    def __init__(self,command,**kwargs):
                        commands.append(command);self.stdout=io.StringIO('')
                        if not missing:
                            out=Path(command[command.index('--output-root')+1])/'capture';out.mkdir(parents=True)
                            entry={'passed':True,'live_pose_in_requested_frame':{'position_m':[0,0,0],'rotation_ros':[[1,0,0],[0,1,0],[0,0,1]]},'landmarks':{'best_transform':'identity'}}
                            (out/'camera_alignment_result.json').write_text(json.dumps({'status':'completed','all_cameras_passed':True,
                                'cameras':{k:entry for k in ('external_camera','wrist_camera')}}))
                    def __enter__(self):return self
                    def __exit__(self,*args):pass
                    def wait(self):return 0
                args=['probe','--camera-audit',str(audit),'--output-root',str(root/'out')]
                with patch.object(sys,'argv',args),patch.object(runner.subprocess,'Popen',Child):
                    self.assertEqual(runner.main(),1 if missing else 0)
                result=json.loads(next((root/'out').glob('*/run.json')).read_text())
                self.assertEqual(result['status'],'failed' if missing else 'completed')
                self.assertEqual(len(commands),1 if missing else 3)
                self.assertIn('--camera-alignment-json',commands[0]);self.assertNotIn('--query',commands[0])


if __name__=='__main__':unittest.main()
