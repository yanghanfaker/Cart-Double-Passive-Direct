import unittest
import sys
import types
from unittest.mock import patch
import numpy as np
from execution_alignment_probe import cases,validate_case,ramp,ramp_schedule,FingerOverride
from registered_replay_probe import target_from_action


class ExecutionAlignmentTests(unittest.TestCase):
    def test_native_adapter_writes_after_original_and_restores(self):
        class Term:
            def apply_actions(self):writes.append('original')
        class Commands:
            def set_position_index(self,**kw):writes.append(kw['value'].copy())
            def set_velocity_index(self,**kw):pass
        writes=[];term=Term()
        fake=types.SimpleNamespace(full=lambda shape,value,**kw:np.full(shape,value),zeros_like=np.zeros_like,float32=None)
        runtime=types.SimpleNamespace(base=types.SimpleNamespace(action_manager=types.SimpleNamespace(get_term=lambda name:term),physics_dt=.01,device='cpu'),
            robot=types.SimpleNamespace(actuators=types.SimpleNamespace(target_command=Commands())),fingers=[7,8])
        with patch.dict(sys.modules,torch=fake):
            hook=FingerOverride(runtime,'substep')
            try:
                hook.begin(1)
                for _ in range(5):term.apply_actions()
                self.assertEqual(hook.calls,5)
                self.assertAlmostEqual(hook.value,.035)
                self.assertEqual(writes[0],'original')
                np.testing.assert_allclose(writes[-1],[[.035,.035]])
            finally:hook.close()
        self.assertNotIn('apply_actions',term.__dict__)
    def test_physical_ramp_units_and_saturation(self):
        value=.04
        for _ in range(8):value=ramp(value,1,.05)
        self.assertAlmostEqual(value,0)
        self.assertEqual(ramp(value,1,.05),0)
        for _ in range(8):value=ramp(value,-1,.05)
        self.assertAlmostEqual(value,.04)

    def test_period_and_substep_endpoints_agree_and_reverse(self):
        a=b=.04
        for sign in ramp_schedule(True):
            a=ramp(a,sign,.05)
            for _ in range(5):b=ramp(b,sign,.01)
            self.assertAlmostEqual(a,b)
        self.assertAlmostEqual(b,.04)

    def test_only_declared_local_envelopes(self):
        self.assertEqual(len(cases()),8)
        for case in cases():validate_case(case)
        with self.assertRaises(ValueError):validate_case({**cases()[-1],'cap_m':.05})
        with self.assertRaises(ValueError):ramp(.05,1,.01)

    def test_target_translation_cap_and_rotation_preserved(self):
        action=[1,.3,-.2,.2,0,0,-1]
        a=target_from_action([0,0,0],[0,0,0,1],np.eye(3),action)
        b=target_from_action([0,0,0],[0,0,0,1],np.eye(3),action,translation_cap=.005)
        self.assertAlmostEqual(np.linalg.norm(a['bounded_delta_L_m']),.0005)
        self.assertAlmostEqual(np.linalg.norm(b['bounded_delta_L_m']),.005)
        self.assertEqual(a['hand_target_quat_world_xyzw'],b['hand_target_quat_world_xyzw'])
        self.assertTrue(b['translation_cap_triggered'])
        np.testing.assert_allclose(b['decoded_delta_L_m'],[.05,.015,-.01])
        for cap in (float('nan'),0,.006):
            with self.assertRaises(ValueError):target_from_action([0,0,0],[0,0,0,1],np.eye(3),action,translation_cap=cap)


if __name__=='__main__':unittest.main()
