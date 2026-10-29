import copy
import math
from pathlib import Path
import tempfile
import unittest

from kinematic_attribution import JacobianTap, shift_jacobian, enrich_kinematic_row, summarize_kinematics
from physics_trace import PhysicsJointRecorder


def sample_row():
    jac = [[0.0]*7 for _ in range(6)]
    jac[1][0], jac[1][2] = 1.0, -1.0
    return {"phase":"replay", "position_before_rad":[0.0]*7,
            "position_rad":[.008,0,.006,0,0,0,0], "position_target_rad":[.01,0,.01,0,0,0,0],
            "sensor_target_pos_root_m":[0,0,0],
            "kinematics_before":{"joint_pos_rad":[0.0]*7, "actual_ik_jacobian_root":jac,
                "sensor_pos_root_m":[0,0,0], "ik_pos_root_m":[0,0,0],
                "ik_quat_root_xyzw":[0,0,0,1], "ik_target_pos_root_m":[0,0,0],
                "ik_target_quat_root_xyzw":[0,0,0,1]},
            "kinematics_after":{"sensor_pos_root_m":[0,.002,0]}}


class KinematicsTests(unittest.TestCase):
    def test_shift_uses_angular_cross_lever_arm_in_root_axes(self):
        jac = [[0.0]*7 for _ in range(6)]
        jac[5][0] = 1
        shifted = shift_jacobian(jac,[.1,0,0])
        self.assertAlmostEqual(shifted[1][0],.1)
        self.assertEqual(shifted[0][0],0)
        self.assertEqual(shifted[5][0],1)
        self.assertEqual(jac[1][0],0)

    def test_opposing_joint_contributions_and_exact_local_motion(self):
        result = enrich_kinematic_row(sample_row())["kinematic_attribution"]
        self.assertAlmostEqual(result["joint_tracking_contributions_xyz_m"][1][0],.002)
        self.assertAlmostEqual(result["joint_tracking_contributions_xyz_m"][1][2],-.004)
        self.assertAlmostEqual(result["joint_tracking_sum_xyz_m"][1],-.002)
        self.assertAlmostEqual(result["measured_sensor_target_error_xyz_m"][1],-.002)
        self.assertLess(math.hypot(*result["accounting_closure_error_xyz_m"]),1e-14)
        self.assertLess(math.hypot(*result["displacement_linearization_residual_xyz_m"]),1e-14)

    def test_rigid_offset_goal_mapping_tracks_desired_orientation(self):
        row=sample_row()
        row["kinematics_before"]["sensor_pos_root_m"]=[.0036,0,0]
        row["kinematics_before"]["ik_target_quat_root_xyzw"]=[0,0,math.sqrt(.5),math.sqrt(.5)]
        row["sensor_target_pos_root_m"]=[.0036,0,0]
        result=enrich_kinematic_row(row)["kinematic_attribution"]
        self.assertAlmostEqual(result["goal_point_mapping_error_xyz_m"][0],.0036)
        self.assertAlmostEqual(result["goal_point_mapping_error_xyz_m"][1],-.0036)

    def test_rejects_misaligned_pre_step_joint_state(self):
        row=sample_row();row["kinematics_before"]["joint_pos_rad"][0]=.001
        with self.assertRaises(RuntimeError):
            enrich_kinematic_row(row)

    def test_rejects_bad_jacobian_and_nonfinite_position(self):
        with self.assertRaises(ValueError):
            shift_jacobian([[0]*6]*6,[0,0,0])
        row=sample_row();row["kinematics_after"]["sensor_pos_root_m"][0]=float("nan")
        with self.assertRaises(ValueError):
            enrich_kinematic_row(row)

    def test_summary_keeps_joint_and_sum_rms_separate(self):
        row=sample_row();row.update(enrich_kinematic_row(row))
        summary=summarize_kinematics([row])["phases"]["replay"]
        self.assertAlmostEqual(summary["per_joint_y_contribution_rms_m"][0],.002)
        self.assertAlmostEqual(summary["per_joint_y_contribution_rms_m"][2],.004)
        self.assertAlmostEqual(summary["rms_xyz_m"]["joint_tracking_sum_xyz_m"][1],.002)

    def test_tap_does_not_add_original_calls_and_restores_method(self):
        class Action:
            calls=0
            def _compute_frame_jacobian(self):
                self.calls+=1
                return [[self.calls]]
        action=Action();tap=JacobianTap(action,copy.deepcopy)
        try:
            action._compute_frame_jacobian()
            tap.active=True
            result=action._compute_frame_jacobian()
            result[0][0]=999
            self.assertEqual(tap.take(),[[2]])
            self.assertEqual(action.calls,2)
            with self.assertRaises(RuntimeError): tap.take()
            action._compute_frame_jacobian()
            with self.assertRaises(RuntimeError): action._compute_frame_jacobian()
        finally:tap.close()
        self.assertNotIn("_compute_frame_jacobian",action.__dict__)

    def test_enrichment_uses_existing_substeps_without_adding_updates(self):
        class Scene:
            calls=0
            def write_data_to_sim(self):pass
            def update(self,dt):self.calls+=1
        scene=Scene();row=sample_row()
        with tempfile.TemporaryDirectory() as tmp:
            recorder=PhysicsJointRecorder(scene,.01,.05,
                lambda:{k:copy.deepcopy(v) for k,v in row.items() if k!="kinematics_after"},
                lambda:{"kinematics_after":copy.deepcopy(row["kinematics_after"])},
                Path(tmp)/"trace.jsonl",{},enrich_row=enrich_kinematic_row)
            try:
                recorder.begin_control({})
                for _ in range(5):
                    scene.write_data_to_sim();scene.update(.01)
                recorder.end_control()
                self.assertEqual(scene.calls,5)
                self.assertEqual(len(recorder.rows),5)
                self.assertIn("kinematic_attribution",recorder.rows[-1])
            finally:recorder.close()


if __name__=="__main__":unittest.main()
