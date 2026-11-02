import copy
import unittest
import numpy as np
from execution_focus_cases import focus_cases,linearized_tcp
from execution_alignment_probe import validate_case,motion_plan,next_target
from execution_coverage_cases import configure_finger_profile
import test_execution_coverage as coverage_fixture


class FocusTests(unittest.TestCase):
    def test_case_count_and_dependencies(self):
        seen=set();self.assertEqual(len(focus_cases()),19)
        for case in focus_cases():
            validate_case(case);self.assertNotIn(case['name'],seen);self.assertTrue(set(case['requires'])<=seen);seen.add(case['name'])

    def test_factorial_separates_motion_and_keeps_gripper_open(self):
        source=[[.1,.2,.3,.4,.5,.6,1]]*5;original=copy.deepcopy(source)
        for case in focus_cases():
            if not case['name'].startswith('factor-'):continue
            plan=motion_plan(case,source)
            for _,a in plan[:5]:
                self.assertEqual(a[6],-1)
                if case['kind']=='translation':self.assertEqual(a[3:6],[0,0,0])
                if case['kind']=='recorded-rotation':self.assertEqual(a[:3],[0,0,0])
        self.assertEqual(source,original)

    def test_armature_changes_both_inertias_without_passive_drive(self):
        config=coverage_fixture.CoverageTests().config()
        for actuator in config.values():actuator.armature=.1
        configure_finger_profile(config,'kp1000-kd100',armature=1.)
        self.assertEqual(config['active'].armature,1)
        self.assertEqual(config['passive'].armature,1)
        self.assertEqual((config['passive'].stiffness,config['passive'].damping),(0,0))
        self.assertEqual(config['arm'].armature,.1)
        self.assertEqual(config['active'].joint_effort_limit,200)

    def test_missing_armature_support_fails_without_mutating_gains(self):
        config=coverage_fixture.CoverageTests().config()
        with self.assertRaises(ValueError):configure_finger_profile(config,'kp1000-kd100',armature=1.)
        self.assertIsNone(config['active'].stiffness)

    def test_anchored_tcp_rejects_measured_position_drift(self):
        case=next(c for c in focus_cases() if c.get('reference_mode')=='anchored_tcp')
        initial={'hand_position_world_m':[0,0,0],'hand_quat_world_xyzw':[0,0,0,1]}
        anchor=next_target(initial,None,np.eye(3),[0]*7,case)
        drifted={**initial,'hand_position_world_m':[.001,.002,0]}
        actual=next_target(drifted,anchor,np.eye(3),[0,0,0,1,0,0,-1],case)
        np.testing.assert_allclose(actual['tcp97_target_world_m'],[0,0,.097],atol=1e-12)
        current=next_target(drifted,anchor,np.eye(3),[0,0,0,1,0,0,-1],{**case,'reference_mode':'current'})
        np.testing.assert_allclose(current['tcp97_target_world_m'],[.001,.002,.097],atol=1e-12)

    def test_jacobian_point_shift_sign_and_frame(self):
        J=np.zeros((6,7));J[3,0]=1
        result=linearized_tcp(J,[.01,0,0,0,0,0,0],[0,0,0,1],[0,0,0,1])
        np.testing.assert_allclose(result['tcp97_predicted_world_m'],[0,.0001,0])
        # A root-frame x translation rotated 90deg about world z becomes y.
        J=np.zeros((6,7));J[0,0]=1
        result=linearized_tcp(J,[.001,0,0,0,0,0,0],[0,0,2**-.5,2**-.5],[0,0,0,1])
        np.testing.assert_allclose(result['tcp97_predicted_world_m'],[0,.001,0],atol=1e-12)
        with self.assertRaises(ValueError):linearized_tcp(np.zeros((6,6)),[0]*7,[0,0,0,1],[0,0,0,1])


if __name__=='__main__':unittest.main()
