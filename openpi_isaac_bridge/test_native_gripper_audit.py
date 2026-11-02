import types,unittest
import numpy as np
from native_gripper_audit import NativeGripperClient

class Grip:
 speed=.01
 actuators=['finger1','finger2']
 def __init__(self):self.current_action=np.array([1.,-1.])
 def format_action(self,a):
  self.current_action=np.clip(self.current_action+np.array([-1.,1.])*self.speed*np.sign(a),-1,1)
  return self.current_action.copy()

class GripperAuditTests(unittest.TestCase):
 def env(self):
  model=types.SimpleNamespace(actuator_name2id=lambda n:['finger1','finger2'].index(n),actuator_ctrlrange=np.array([[0,.04],[-.04,0]]),actuator_trntype=np.array([0,0]),actuator_gainprm=np.ones((2,10)),actuator_biasprm=np.zeros((2,10)))
  return types.SimpleNamespace(robots=[types.SimpleNamespace(gripper=Grip())],sim=types.SimpleNamespace(model=model,data=types.SimpleNamespace(ctrl=np.array([.03,-.03]))))
 def test_counting_preserves_original_outputs(self):
  env=self.env();client=NativeGripperClient([]);expected=Grip();client.before_step(env)
  for _ in range(25):np.testing.assert_array_equal(env.robots[0].gripper.format_action([1]),expected.format_action([1]))
  r=client.observe_step(env);self.assertEqual(r['format_action_calls_this_control'],25);self.assertEqual(r['actuators'][0]['ctrl'],.03)
 def test_no_double_wrap_and_metadata_once(self):
  env=self.env();c=NativeGripperClient([]);c.before_step(env);env.robots[0].gripper.format_action([-1]);r=c.observe_step(env);self.assertIn('implementation',r)
  c.before_step(env);env.robots[0].gripper.format_action([1]);r=c.observe_step(env);self.assertEqual(r['format_action_calls_this_control'],1);self.assertEqual(r['format_action_calls_total'],2);self.assertNotIn('implementation',r)
 def test_nonfinite_control_rejected(self):
  env=self.env();c=NativeGripperClient([]);c.before_step(env);env.sim.data.ctrl[0]=np.nan
  with self.assertRaises(RuntimeError):c.observe_step(env)

if __name__=='__main__':unittest.main()
