import base64
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from geometry_audit import compose,closure,multiply,inverse,pose_matrix,project,write_overlay
import run_geometry_audit


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.pose={'position_m':[0,0,0],'quat_xyzw':[0,0,0,1]}
        self.K=[[100,0,128],[0,100,128],[0,0,1]]

    def test_ros_projection_right_down_forward(self):
        self.assertEqual(project([0,0,1],self.pose,self.K,256,256)['uv_px'],[128,128])
        self.assertEqual(project([.1,.2,1],self.pose,self.K,256,256)['uv_px'],[138,148])
        self.assertFalse(project([0,0,-1],self.pose,self.K,256,256)['in_front'])
        self.assertFalse(project([5,0,1],self.pose,self.K,256,256)['inside_image'])

    def test_rotated_translated_camera_projection(self):
        q=[0,math.sin(math.pi/4),0,math.cos(math.pi/4)]
        camera={'position_m':[1,2,3],'quat_xyzw':q}
        result=project([2,2,3],camera,self.K,256,256)
        self.assertAlmostEqual(result['uv_px'][0],128)
        self.assertAlmostEqual(result['uv_px'][1],128)
        self.assertAlmostEqual(result['camera_ros_xyz_m'][2],1)

    def test_pose_matrix_inverse_and_rotated_offset_closure(self):
        q=[0,math.sin(math.pi/4),0,math.cos(math.pi/4)]
        parent={'position_m':[1,2,3],'quat_xyzw':q}
        offset={'position_m':[0,0,.1],'quat_xyzw':[0,0,0,1]}
        expected=compose(parent,offset)
        for a,b in zip(expected['position_m'],[1.1,2,3]):self.assertAlmostEqual(a,b)
        self.assertTrue(closure(expected,expected)['within_0p1mm_0p05deg'])
        a=pose_matrix(parent);b=pose_matrix(parent,True)
        for i in range(4):
            for j in range(4):self.assertAlmostEqual(sum(a[i][k]*b[k][j] for k in range(4)),int(i==j))

    def test_quaternion_order_and_inverse(self):
        q=[.5,.5,.5,.5];identity=multiply(q,inverse(q))
        for a,b in zip(identity,[0,0,0,1]):self.assertAlmostEqual(a,b)

    def test_invalid_intrinsics_and_nonfinite_points_rejected(self):
        with self.assertRaises(ValueError):project([float('nan'),0,1],self.pose,self.K,256,256)
        with self.assertRaises(ValueError):project([0,0,1],self.pose,[[0,0,0],[0,100,0],[0,0,1]],256,256)

    def test_overlay_embeds_unchanged_png_and_does_not_draw_hidden_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'overlay.svg';payload=b'original PNG fixture'
            write_overlay(path,payload,{'sensor_tcp':{'uv_px':[128,128],'inside_image':True},'object_root':{'uv_px':None,'inside_image':False}},256,256)
            text=path.read_text();self.assertIn(base64.b64encode(payload).decode(),text)
            self.assertEqual(text.count('<circle '),1)

    def test_runner_launches_static_mode_and_keeps_outputs_in_new_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);commands=[]
            class Process:
                def __init__(self,command,**kwargs):
                    commands.append(command);folder=Path(command[command.index('--output')+1])
                    (folder/'geometry_audit.json').write_text(json.dumps({'status':'completed'}));self.stdout=io.StringIO('')
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def wait(self):return 0
            argv=['audit','--isaaclab-root',str(root),'--output-root',str(root/'results')]
            with patch.object(sys,'argv',argv),patch.object(run_geometry_audit.subprocess,'Popen',Process):
                self.assertEqual(run_geometry_audit.main(),0)
            self.assertEqual(len(commands),1);command=commands[0]
            self.assertEqual(command[command.index('--motion-test')+1],'geometry-audit')
            self.assertNotIn('--query',command);self.assertNotIn('--record-physics',command)
            folder=Path(command[command.index('--output')+1]);self.assertEqual(json.loads((folder/'run.json').read_text())['status'],'completed')
            self.assertEqual(json.loads((folder/'joint_gains.json').read_text())['panda_joint2'],{'stiffness':5400,'damping':120})


if __name__=='__main__':unittest.main()
