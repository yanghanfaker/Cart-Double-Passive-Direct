import copy
import math
import types
import unittest
import numpy as np
from execution_coverage_cases import coverage_cases,FINGER_PROFILES,configure_finger_profile
from execution_alignment_probe import validate_case,motion_plan,next_target
from registered_replay_probe import target_from_action


class CoverageTests(unittest.TestCase):
    def test_declared_cases_and_prerequisite_order(self):
        seen=set();cases=coverage_cases()
        self.assertEqual(len(cases),22)
        for case in cases:
            validate_case(case);self.assertTrue(set(case['requires'])<=seen);self.assertNotIn(case['name'],seen);seen.add(case['name'])
        with self.assertRaises(ValueError):validate_case({**cases[0],'finger_profile':'unsafe'})

    def config(self):
        implicit=type('Implicit',(),{'is_implicit_model':True})
        def actuator(pattern,kp,kd):return types.SimpleNamespace(joint_names_expr=[pattern],stiffness=kp,damping=kd,class_type=implicit,joint_effort_limit=200)
        return {'arm':actuator('panda_joint[1-7]',400,80),'active':actuator('panda_finger_joint1',None,None),'passive':actuator('panda_finger_joint2',0,0)}

    def test_active_only_gains_preserve_passive_arm_and_limits(self):
        for name,values in FINGER_PROFILES.items():
            config=self.config();configure_finger_profile(config,name)
            self.assertEqual((config['active'].stiffness,config['active'].damping),values or (None,None))
            self.assertEqual((config['passive'].stiffness,config['passive'].damping),(0,0))
            self.assertEqual((config['arm'].stiffness,config['arm'].damping),(400,80))
            self.assertEqual(config['active'].joint_effort_limit,200)

    def test_reject_mixed_or_nonpassive_before_mutation(self):
        for field,value in [('stiffness',1),('damping',1)]:
            config=self.config();setattr(config['passive'],field,value)
            with self.assertRaises(ValueError):configure_finger_profile(config,'kp1000-kd10')
            self.assertIsNone(config['active'].stiffness)
        config=self.config();config['active'].joint_names_expr=['panda_.*']
        with self.assertRaises(ValueError):configure_finger_profile(config,'kp1000-kd10')

    def test_rotation_axes_sign_and_caps(self):
        for case in coverage_cases():
            if case['kind']!='rotation':continue
            plan=motion_plan(case,[[.2]*7]*5);self.assertEqual(len(plan),25)
            raw=plan[0][1];np.testing.assert_array_equal(raw[:3],[0,0,0])
            target=target_from_action([0,0,0],[0,0,0,1],np.eye(3),raw,rotation_cap=math.radians(case['rotation_cap_deg']))
            expected=np.zeros(3);expected[case['axis']]=case['sign']*math.radians(case['rotation_cap_deg'])
            np.testing.assert_allclose(target['bounded_rotation_L_rad'],expected)
        with self.assertRaises(ValueError):target_from_action([0,0,0],[0,0,0,1],np.eye(3),[0]*7,rotation_cap=math.radians(.6))

    def test_combined_preserves_source_and_holds_target(self):
        source=[[.1,.2,.3,.4,.5,.6,s] for s in [-1,-1,-1,-1,1]];original=copy.deepcopy(source)
        case=next(c for c in coverage_cases() if c['name']=='combined-2mm-substep')
        plan=motion_plan(case,source)
        self.assertEqual([x[1] for x in plan[:5]],source);self.assertEqual(plan[5:],[('hold',None)]*20)
        self.assertEqual(source,original)

    def test_preconditioning_is_explicit_and_open(self):
        source=[[.1]*7]*5;case=coverage_cases()[-1];plan=motion_plan(case,source)
        self.assertEqual(len(plan),30)
        for phase,raw in plan[:5]:self.assertEqual(phase,'prelude');self.assertEqual(raw[3:],[0,0,0,-1])
        self.assertEqual(plan[5][0],'command')

    def test_accumulated_reference_retains_previous_target(self):
        case=next(c for c in coverage_cases() if c.get('reference_mode')=='accumulated')
        before={'hand_position_world_m':[0,0,0],'hand_quat_world_xyzw':[0,0,0,1]}
        a=next_target(before,None,np.eye(3),[1,0,0,0,0,0,-1],case)
        b=next_target(before,a,np.eye(3),[1,0,0,0,0,0,-1],case)
        self.assertAlmostEqual(b['tcp97_target_world_m'][0],.004)
        c=next_target(before,a,np.eye(3),[1,0,0,0,0,0,-1],{**case,'reference_mode':'current'})
        self.assertAlmostEqual(c['tcp97_target_world_m'][0],.002)


if __name__=='__main__':unittest.main()
