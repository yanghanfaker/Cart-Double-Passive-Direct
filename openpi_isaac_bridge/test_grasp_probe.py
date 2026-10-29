import copy
import math
import unittest

from grasp_probe import run_grasp_probe, angle_deg, GraspAbort, down_quat_from_object


class Plant:
    """Synthetic state provider for sequencing tests, not a physics model."""
    def __init__(self, mode="held"):
        self.mode = mode
        self.ee = [0.5, 0.0, 0.385]
        self.obj = [0.5, 0.0, 0.025]
        self.q = [1, 0, 0, 0]
        self.fingers = [0.04, 0.04]
        self.phases = []
    def read(self):
        return copy.deepcopy({"ee_pos_root_m": self.ee, "ee_quat_root": self.q,
            "object_pos_root_m": self.obj, "object_quat_root": [0,0,0,1],
            "finger_positions_m": self.fingers, "object_in_ee_pos_m": [a-b for a,b in zip(self.obj,self.ee)],
            "object_in_ee_quat": [1,0,0,0]})
    def step(self, target, quat, gripper, phase):
        self.phases.append(phase)
        delta = [a-b for a,b in zip(target,self.ee)]
        size = math.hypot(*delta)
        self.ee = [q+d*min(1,0.001/size) for q,d in zip(self.ee,delta)] if size else list(self.ee)
        self.q = list(quat)
        self.fingers = [0.04,0.04] if gripper>0 else ([0,0] if self.mode=="empty" else [0.02,0.02])
        if phase in ("lift","hold") and self.mode=="held":
            self.obj = [self.ee[0],self.ee[1],self.ee[2]-0.005]
        if phase=="close" and self.mode=="thrown":
            self.obj[0] += 0.05


class GraspProbeTests(unittest.TestCase):
    def test_measured_object_yaw_sets_top_down_grasp_in_xyzw(self):
        for degrees in (-20,0,20):
            yaw=math.radians(degrees)
            obj=[0,0,math.sin(yaw/2),math.cos(yaw/2)]
            q,measured=down_quat_from_object(obj)
            self.assertAlmostEqual(measured,yaw)
            self.assertAlmostEqual(q[0],math.cos(yaw/2))
            self.assertAlmostEqual(q[1],math.sin(yaw/2))
            # R(q) applied to local +Z must point straight down.
            x,y,z,w=q
            self.assertAlmostEqual(1-2*(x*x+y*y),-1)
            other,_=down_quat_from_object([-v for v in obj])
            self.assertAlmostEqual(angle_deg(q,other),0)

    def test_slower_orientation_preserves_grasp_checks(self):
        plant=Plant()
        report=run_grasp_probe(plant.read,plant.step,announce=lambda _:None,orient_step_deg=.25)
        self.assertTrue(report["success"])
        self.assertEqual(report["orient_step_deg"],.25)

    def run_probe(self, plant):
        return run_grasp_probe(plant.read, plant.step, announce=lambda _: None)

    def test_sustained_lift_and_relative_pose_required(self):
        plant = Plant()
        report = self.run_probe(plant)
        self.assertTrue(report["success"], report.get("failure_reason"))
        self.assertGreaterEqual(report["hold_check"]["min_object_lift_m"], 0.035)
        self.assertEqual(sum(s["phase"]=="hold" for s in report["trace"]), 40)

    def test_empty_close_never_attempts_lift(self):
        plant = Plant("empty")
        report = self.run_probe(plant)
        self.assertFalse(report["success"])
        self.assertNotIn("lift", plant.phases)
        self.assertFalse(report["close_check"]["passed"])

    def test_finger_obstruction_without_object_lift_is_not_success(self):
        report = self.run_probe(Plant("not-held"))
        self.assertTrue(report["close_check"]["passed"])
        self.assertFalse(report["success"])
        self.assertEqual(report["hold_check"]["min_object_lift_m"], 0)

    def test_object_ejection_stops_before_lift_and_preserves_trace(self):
        plant = Plant("thrown")
        report = self.run_probe(plant)
        self.assertFalse(report["success"])
        self.assertIn("0.5 m/s", report["failure_reason"])
        self.assertNotIn("lift", plant.phases)
        self.assertTrue(report["trace"])

    def test_quaternion_sign_and_invalid_norm(self):
        self.assertAlmostEqual(angle_deg([1,0,0,0],[-1,0,0,0]), 0)
        with self.assertRaises(GraspAbort):
            angle_deg([0,0,0,0], [1,0,0,0])


if __name__ == "__main__":
    unittest.main()
