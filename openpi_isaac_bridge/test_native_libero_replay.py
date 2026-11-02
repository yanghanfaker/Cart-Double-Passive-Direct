import unittest
import numpy as np
from frozen_coordinate_probe import digest_observation
from native_libero_replay import ReplayClient,compare_trace

class ReplayTests(unittest.TestCase):
 def obs(self):return {'observation/state':np.zeros(8),'observation/image':np.zeros((2,2,3),dtype=np.uint8),'observation/wrist_image':np.zeros((2,2,3),dtype=np.uint8),'prompt':'task'}
 def test_exact_input_and_budget(self):
  o=self.obs();c=ReplayClient([{'input_sha256':digest_observation(o),'actions':[[0]*7]*5}]);self.assertEqual(c.infer(o)['actions'].shape,(5,7))
  with self.assertRaises(ValueError):c.infer(o)
 def test_changed_input_stops_before_consuming(self):
  o=self.obs();c=ReplayClient([{'input_sha256':digest_observation(o),'actions':[[0]*7]*5}]);o['prompt']='changed'
  with self.assertRaises(ValueError):c.infer(o)
  self.assertEqual(c.index,0);self.assertFalse(c.checks[0]['exact_match'])
 def test_trajectory_tolerance_and_success(self):
  def r(x):return {'status':'completed','success':True,'trace':[{'step':1,'action':[0]*7,'task_success':True,'after':{'robot0_eef_pos':[x,0,0],'robot0_gripper_qpos':[0,0],'robot0_joint_pos':[0]*7}}]}
  self.assertEqual(compare_trace(r(0),r(0))['position_m'],0)
  with self.assertRaises(ValueError):compare_trace(r(0),r(.001))
  q=r(0);q['success']=False
  with self.assertRaises(ValueError):compare_trace(r(0),q)

if __name__=='__main__':unittest.main()
