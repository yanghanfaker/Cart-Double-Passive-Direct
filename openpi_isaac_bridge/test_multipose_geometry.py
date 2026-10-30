import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
import sys
import io
from unittest.mock import patch

from multipose_geometry import design_poses,validate_arm_pose,fit_rigid,analyze
from run_multipose_geometry import unique_completed
import run_multipose_geometry as runner


class MultiPoseTests(unittest.TestCase):
    def test_design_splits_and_conservative_joint_limits(self):
        home=[0,-.569,0,-2.81,0,3.037,.741];before=copy.deepcopy(home)
        poses=design_poses(home)
        self.assertEqual(len(poses),11)
        self.assertEqual(sum(p['split']=='fit' for p in poses),7)
        self.assertEqual(sum(p['split']=='validation' for p in poses),4)
        self.assertEqual(len({tuple(p['requested_joint_positions_rad']) for p in poses}),11)
        self.assertEqual(home,before)

    def test_invalid_angles_and_incomplete_reports_rejected(self):
        for q in ([0]*7,[True]*7,[float('nan')]*7,[0]*6):
            with self.assertRaises(ValueError):validate_arm_pose(q)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):unique_completed(Path(tmp),'report.json')
            folder=Path(tmp)/'case';folder.mkdir();(folder/'report.json').write_text('{"status":"failed"}')
            with self.assertRaises(RuntimeError):unique_completed(Path(tmp),'report.json')

    def test_registration_recovers_known_rotation_and_translation(self):
        import numpy as np
        points=np.array([[0,0,0],[.1,0,0],[0,.1,0],[0,0,.1],[.05,.02,.03]])
        angle=.3;R=np.array([[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1]])
        t=np.array([-.66,0,.912]);got,offset,_=fit_rigid(points,points@R.T+t)
        np.testing.assert_allclose(got,R,atol=1e-12);np.testing.assert_allclose(offset,t,atol=1e-12)
        with self.assertRaises(ValueError):fit_rigid([[i,0,0] for i in range(5)],[[i,0,0] for i in range(5)])

    def test_synthetic_pose_registration_and_heldout_error(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);poses=design_poses([0,-.569,0,-2.81,0,3.037,.741]);t=np.array([-.66,0,.912])
            for index,p in enumerate(poses):
                angle=.1*index;q=[0,math.sin(angle/2),0,math.cos(angle/2)]
                R=np.array([[math.cos(angle),0,math.sin(angle)],[0,1,0],[-math.sin(angle),0,math.cos(angle)]])
                hand=np.array([.4+.03*math.sin(index),.05*math.cos(index),.4+.02*math.sin(2*index)])
                names=[f'panda_joint{i}' for i in range(1,8)];values=p['requested_joint_positions_rad']
                isaac={'status':'completed','joint_names':names,'joint_positions_rad':values,
                    'finger_positions_isaac_m':[.04,.04],'physics_step_count_before':175,'physics_step_count_after':175,
                    'sensor_body_world':{'position_m':hand.tolist(),'quat_xyzw':q},
                    'robot_root_world':{'position_m':[0,0,0],'quat_xyzw':[0,0,0,1]},
                    'sensor_tcp_world':{'position_m':(hand+R@np.array([0,0,.1034])).tolist()},
                    'sensor_body_offset':{'position_m':[0,0,.1034]},'same_sensor_ik_body':True,
                    'sensor_offset_closure':{'within_0p1mm_0p05deg':True}}
                ref={'status':'completed','joint_assignments':[{'actual':v} for v in values+[.04,-.04]],
                    'sim_time_before_forward':0,'sim_time_after_capture':0,
                    'robot_root':{'rotation_world_matrix':np.eye(3).tolist(),'position_world_m':t.tolist()},
                    'robot_eef_body':{'position_world_m':(hand+t).tolist(),'rotation_world_matrix':R.tolist()},
                    'eef_site':{'position_world_m':(hand+t+R@np.array([0,0,.097])).tolist()}}
                p['status']='completed';p['isaac_audit']=f'i{index}.json';p['reference']=f'l{index}.json'
                (folder/p['isaac_audit']).write_text(json.dumps(isaac));(folder/p['reference']).write_text(json.dumps(ref))
            (folder/'run.json').write_text(json.dumps({'poses':poses}))
            result=analyze(folder)
            self.assertTrue(result['candidate_geometry_within_thresholds'])
            np.testing.assert_allclose(result['fitted_tcp_delta_hand_m'],[0,0,-.0064],atol=1e-12)
            Rbefore=copy.deepcopy(result['world_from_isaac_rotation'])
            path=folder/poses[-1]['reference'];ref=json.loads(path.read_text());ref['eef_site']['position_world_m'][0]+=.005;path.write_text(json.dumps(ref))
            result=analyze(folder)
            self.assertFalse(result['candidate_geometry_within_thresholds'])
            self.assertGreater(result['summary']['validation']['registered_corrected_tcp_error_mm']['max'],4.9)
            self.assertEqual(result['world_from_isaac_rotation'],Rbefore)

    def test_runner_uses_new_capture_paths_and_stops_on_missing_reference(self):
        for missing in (False,True):
            with self.subTest(missing=missing),tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp);source=folder/'source.json'
                source.write_text(json.dumps({'status':'completed','joint_names':[f'panda_joint{i}' for i in range(1,8)],
                    'joint_positions_rad':[0,-.569,0,-2.81,0,3.037,.741],'finger_positions_isaac_m':[.04,.04]}))
                commands=[]
                class Child:
                    def __init__(self,command,**kwargs):
                        commands.append(command);self.stdout=io.StringIO('')
                        if 'run_geometry_audit.py' in command[1]:
                            out=Path(command[command.index('--output-root')+1])/'capture';out.mkdir(parents=True)
                            (out/'geometry_audit.json').write_text('{"status":"completed"}')
                        elif 'libero_reference_audit.py' in command[1]:
                            out=Path(command[command.index('--output-root')+1])/'capture';out.mkdir(parents=True)
                            if not missing:
                                import hashlib
                                src=Path(command[command.index('--isaac-audit')+1])
                                (out/'reference.json').write_text('{"status":"completed"}')
                                (out/'run.json').write_text(json.dumps({'status':'completed','isaac_source_sha256':hashlib.sha256(src.read_bytes()).hexdigest()}))
                        else:
                            (Path(command[-1])/'analysis.json').write_text('{"status":"completed","candidate_geometry_within_thresholds":false}')
                    def __enter__(self):return self
                    def __exit__(self,*args):pass
                    def wait(self):return 0
                args=['probe','--source-audit',str(source),'--reference-python',sys.executable,'--output-root',str(folder/'out')]
                with patch.object(sys,'argv',args),patch.object(runner.subprocess,'Popen',Child):
                    self.assertEqual(runner.main(),1 if missing else 0)
                manifest=json.loads(next((folder/'out').glob('*/run.json')).read_text())
                self.assertEqual(manifest['status'],'failed' if missing else 'completed')
                self.assertEqual(len(commands),2 if missing else 23)
                self.assertNotIn('--query',commands[0])
                self.assertIn('--geometry-joints-json',commands[0])


if __name__=='__main__':unittest.main()
