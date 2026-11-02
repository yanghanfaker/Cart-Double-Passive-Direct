import copy,unittest
import numpy as np
from native_pose_refresh import analyze,requested_geometry

class PoseRefreshTests(unittest.TestCase):
 def data(self):
  names=[f'panda_joint{i}' for i in range(1,8)]
  g={'status':'completed','joint_names':names,'joint_positions_rad':[0]*7,'finger_positions_isaac_m':[.04,.04],'sensor_body_world':{'position_m':[0,0,0],'quat_xyzw':[0,0,0,1]}}
  s={'calibration':{'world_from_isaac_rotation':np.eye(3).tolist(),'world_from_isaac_translation_m':[0,0,0]},'native_initial':{'robot0_joint_pos':[0]*7,'robot0_gripper_qpos':[.04,-.04],'robot0_eef_pos':[.0005,0,.097],'robot0_eef_quat':[0,0,0,1]}}
  r={'status':'completed','sim_time_before_forward':1,'sim_time_after_capture':1,'joint_assignments':[{'joint':f'robot0_joint{i}','actual':0} for i in range(1,8)],'observation':{'robot0_eef_pos':[.0001,0,.097],'robot0_eef_quat':[0,0,0,1]}}
  return g,s,r
 def test_error_decomposition(self):
  g,s,r=self.data();a=analyze(g,s,r)
  self.assertAlmostEqual(a['isaac_minus_dynamic']['norm_mm'],.5)
  self.assertAlmostEqual(a['isaac_minus_static']['norm_mm'],.1)
  np.testing.assert_allclose(np.array(a['isaac_minus_static']['vector_mm'])+a['static_minus_dynamic']['vector_mm'],a['isaac_minus_dynamic']['vector_mm'])
 def test_no_time_advance_and_joint_match(self):
  g,s,r=self.data();r['sim_time_after_capture']=1.01
  with self.assertRaises(ValueError):analyze(g,s,r)
  r['sim_time_after_capture']=1;r['joint_assignments'][0]['actual']=.001
  with self.assertRaises(ValueError):analyze(g,s,r)
 def test_requested_geometry_preserves_original(self):
  g,s,r=self.data();original=copy.deepcopy(g);s['native_initial']['robot0_joint_pos'][0]=.2
  a=requested_geometry(g,s['native_initial']);self.assertEqual(a['joint_positions_rad'][0],.2);self.assertEqual(g,original)

if __name__=='__main__':unittest.main()
