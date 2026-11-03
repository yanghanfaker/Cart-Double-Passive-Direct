import copy
from types import SimpleNamespace
import unittest
import numpy as np
from collision_geometry import corners,live_bound,rigid_scale,box_relation,support_candidates,geometry_summary,CollisionGeometry
from execution_online_caps import online_caps_cases
from execution_alignment_probe import validate_case
from execution_hybrid import run_policy_approach


class OnlineCapsTests(unittest.TestCase):
    def test_matrix_changes_only_cap_prefix_repeat_and_declared_observation(self):
        cases=online_caps_cases();self.assertEqual(len(cases),12)
        self.assertEqual(len({x['name'] for x in cases}),12)
        for c in cases:
            validate_case(c);self.assertTrue(c['collision_geometry'])
            self.assertEqual(c['arm_profile'],'k1687p5-d50');self.assertEqual(c['finger_profile'],'kp350-kd50')
        policies=[c for c in cases if c.get('hybrid_role')=='policy']
        self.assertEqual({(c['cap_m'],c['prefix_length'],c['repeat']) for c in policies},
            {(cap,p,r) for cap in (.0005,.001) for p in (1,5) for r in (1,2)})
        first=[(c['cap_m'],c['prefix_length']) for c in policies if c['repeat']==1]
        second=[(c['cap_m'],c['prefix_length']) for c in policies if c['repeat']==2]
        self.assertEqual(second,list(reversed(first)))
        for c in policies:
            self.assertEqual(c['handoff_mode'],'geometry');self.assertEqual(c['requires_success'],['online-caps-reference-start'])
        replays=[c for c in cases if c.get('limit_audit_role')=='replay']
        self.assertEqual({(c['source_prefix'],c['cap_m']) for c in replays},{(1,.002),(5,.002)})

    def test_new_case_uses_existing_policy_gate_without_executing_close_action(self):
        from test_execution_hybrid import HybridTests
        state=HybridTests().state(near=True);case=online_caps_cases()[1]
        result=run_policy_approach(lambda:copy.deepcopy(state),lambda i:{'raw_action_chunk':[[0.]*6+[1.]]*10},
            lambda *args:(_ for _ in ()).throw(AssertionError('should hand off first')),case)
        self.assertTrue(result['handoff']);self.assertEqual(result['controls'],0)

    def test_positive_scale_is_baked_and_reflection_or_shear_is_rejected(self):
        np.testing.assert_allclose(rigid_scale(np.diag([.8,.8,.8])),[.8,.8,.8])
        for matrix in (np.diag([-1,1,1]),[[1,.4,0],[0,1,0],[0,0,1]],np.zeros((3,3)),[1,2,3]):
            with self.assertRaises(ValueError):rigid_scale(matrix)

    def test_local_bounds_follow_live_pose_instead_of_authored_world_pose(self):
        points=corners([0,0,0],[.01,.02,.03]);q=[0,0,np.sin(np.pi/4),np.cos(np.pi/4)]
        result=live_bound(points,[1,2,3],q)
        np.testing.assert_allclose(result['minimum_m'],[.98,2,3],atol=1e-12)
        np.testing.assert_allclose(result['maximum_m'],[1,2.01,3.03],atol=1e-12)
        shifted=live_bound(points,[1,2,4],q)
        self.assertAlmostEqual(shifted['minimum_m'][2]-result['minimum_m'][2],1.)

    def test_aabb_overlap_is_separate_from_signed_table_top_gap(self):
        support={'minimum_m':[-1,-1,-.1],'maximum_m':[1,1,0.]}
        finger={'minimum_m':[0,0,.002],'maximum_m':[.01,.01,.03]}
        relation=box_relation(finger,support)
        self.assertFalse(relation['aabb_overlap']);self.assertAlmostEqual(relation['aabb_distance_m'],.002)
        self.assertAlmostEqual(relation['vertical_gap_to_box_top_m'],.002)
        finger['minimum_m'][2]=-.001
        self.assertTrue(box_relation(finger,support)['aabb_overlap'])
        self.assertEqual(box_relation(finger,support)['aabb_distance_m'],0.)
        self.assertLess(box_relation(finger,support)['vertical_gap_to_box_top_m'],0.)

    def test_support_selection_requires_initial_xy_and_height_agreement(self):
        a={'path':'table','minimum_m':[0,0,-.5],'maximum_m':[1,1,0.]}
        b={'path':'other','minimum_m':[2,2,-.5],'maximum_m':[3,3,0.]}
        c={'path':'overhead','minimum_m':[0,0,.5],'maximum_m':[1,1,1.]}
        self.assertEqual(support_candidates([a,b,c],[.5,.5,.021]),[a])

    def test_first_open_deflection_keeps_partial_geometry_status_explicit(self):
        row={'control_index':8,'physics_substep':2,'finger_target_m':[.04,.04],
            'finger_position_m':[.037,.0375],'tcp_world_m':[0,0,.02],
            'collision_geometry':{'status':'partial','fingers':[{'support_pairs':[],'object_pairs':[]}]}}
        summary=geometry_summary([row])
        self.assertEqual(summary['partial_samples'],1);self.assertEqual(summary['fully_available_samples'],0)
        self.assertIsNone(summary['minimum_support_aabb_distance_m'])
        self.assertEqual(summary['first_open_target_deflection']['control_index'],8)
        row['finger_target_m']=[.02,.02]
        self.assertIsNone(geometry_summary([row])['first_open_target_deflection'])

    def test_geometry_setup_failure_is_explicit_not_a_fabricated_contact(self):
        from unittest.mock import patch
        with patch.object(CollisionGeometry,'_prepare',side_effect=RuntimeError('no compatible USD stage')):
            probe=CollisionGeometry(SimpleNamespace())
        self.assertEqual(probe.model['status'],'unavailable')
        self.assertIn('no compatible USD stage',probe.sample()['reason'])
        self.assertEqual(probe.model['contact_forces']['status'],'unavailable')


if __name__=='__main__':unittest.main()
