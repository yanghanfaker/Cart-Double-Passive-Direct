"""Read-only Jacobian tap and local error accounting in robot-root axes."""
import math
import operator


def resolved_body_identity(sensor_ids, sensor_names, ik_ids, ik_names, used_ik_index):
    """Compare resolved articulation indices, not path regex or private names."""
    sensor_ids = [operator.index(value) for value in sensor_ids]
    ik_ids = [operator.index(value) for value in ik_ids]
    used_ik_index = operator.index(used_ik_index)
    details = {"sensor_body_ids":sensor_ids,"sensor_body_names":list(sensor_names),
               "ik_body_ids":ik_ids,"ik_body_names":list(ik_names),"used_ik_body_index":used_ik_index}
    if (len(sensor_ids)!=1 or len(sensor_names)!=1 or len(ik_ids)!=1 or len(ik_names)!=1
            or sensor_ids[0]!=ik_ids[0] or ik_ids[0]!=used_ik_index):
        raise RuntimeError(f"Sensor/IK resolved body mismatch: {details}")
    return details


def vector(values, size):
    if len(values) != size or not all(math.isfinite(v) for v in values):
        raise ValueError(f"expected finite vector of length {size}")
    return list(values)


def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def add(a, b):
    return [x+y for x,y in zip(a,b)]


def sub(a, b):
    return [x-y for x,y in zip(a,b)]


def normalized_quaternion(q):
    q = vector(q,4)
    length = math.hypot(*q)
    if abs(length-1) > .001:
        raise ValueError("expected approximately unit xyzw quaternion")
    return [x/length for x in q]


def rotate(q, v):
    q = normalized_quaternion(q)
    first = cross(q[:3],v)
    second = cross(q[:3],first)
    return [v[i]+2*q[3]*first[i]+2*second[i] for i in range(3)]


def shift_jacobian(jacobian, offset_root):
    """Geometric velocities in root axes: v_sensor = v_ik + omega x r."""
    if len(jacobian) != 6:
        raise ValueError("expected 6 by 7 Jacobian")
    jacobian = [vector(row,7) for row in jacobian]
    offset_root = vector(offset_root,3)
    result = [row[:] for row in jacobian]
    for joint in range(7):
        correction = cross([jacobian[i][joint] for i in range(3,6)],offset_root)
        for axis in range(3):
            result[axis][joint] += correction[axis]
    return result


class JacobianTap:
    """Call the original IK function exactly once; capture a detached copy."""
    def __init__(self, action, capture):
        self.action, self.capture = action, capture
        self.name = "_compute_frame_jacobian"
        self.had_method = self.name in action.__dict__
        self.original = getattr(action,self.name)
        self.pending = None
        self.active = False
        setattr(action,self.name,self.wrapped)

    def wrapped(self, *args, **kwargs):
        result = self.original(*args,**kwargs)
        if self.active:
            if self.pending is not None:
                raise RuntimeError("multiple IK Jacobians before command capture")
            self.pending = self.capture(result)
        return result

    def take(self):
        if self.pending is None:
            raise RuntimeError("no actual IK Jacobian captured for this physics step")
        result, self.pending = self.pending, None
        return result

    def close(self):
        if self.had_method:
            setattr(self.action,self.name,self.original)
        else:
            delattr(self.action,self.name)


def enrich_kinematic_row(row):
    before = row["kinematics_before"]
    after = row["kinematics_after"]
    q0 = vector(row["position_before_rad"],7)
    q1 = vector(row["position_rad"],7)
    qt = vector(row["position_target_rad"],7)
    if math.dist(q0,vector(before["joint_pos_rad"],7)) > 1e-6:
        raise RuntimeError("Jacobian and pre-integration joint state are not aligned")
    p0 = vector(before["sensor_pos_root_m"],3)
    p1 = vector(after["sensor_pos_root_m"],3)
    goal = vector(row["sensor_target_pos_root_m"],3)
    r = sub(p0,vector(before["ik_pos_root_m"],3))
    jac = shift_jacobian(before["actual_ik_jacobian_root"],r)
    def project(dq):
        return [sum(jac[i][j]*dq[j] for j in range(7)) for i in range(3)]
    contribution = [[jac[i][j]*(qt[j]-q1[j]) for j in range(7)] for i in range(3)]
    tracking = [sum(values) for values in contribution]
    actual = sub(p1,p0)
    prediction = project(sub(q1,q0))
    residual = sub(actual,prediction)
    ikq = normalized_quaternion(before["ik_quat_root_xyzw"])
    local_offset = rotate([-ikq[0],-ikq[1],-ikq[2],ikq[3]],r)
    implied_goal = add(vector(before["ik_target_pos_root_m"],3),
                       rotate(before["ik_target_quat_root_xyzw"],local_offset))
    mapping_error = sub(goal,implied_goal)
    ik_residual = sub(sub(implied_goal,p0),project(sub(qt,q0)))
    measured = sub(goal,p1)
    reconstructed = sub(add(add(mapping_error,ik_residual),tracking),residual)
    return {"sensor_jacobian_root":jac,
            "kinematic_attribution":{
                "joint_tracking_contributions_xyz_m":contribution,
                "joint_tracking_sum_xyz_m":tracking,
                "goal_point_mapping_error_xyz_m":mapping_error,
                "ik_target_linearization_residual_xyz_m":ik_residual,
                "predicted_actual_displacement_xyz_m":prediction,
                "measured_actual_displacement_xyz_m":actual,
                "displacement_linearization_residual_xyz_m":residual,
                "measured_sensor_target_error_xyz_m":measured,
                "accounting_closure_error_xyz_m":sub(measured,reconstructed)}}


def summarize_kinematics(rows):
    if not rows or "kinematic_attribution" not in rows[0]:
        return None
    if any("kinematic_attribution" not in row for row in rows):
        raise ValueError("incomplete kinematic attribution")
    phases = {}
    for phase in dict.fromkeys(row["phase"] for row in rows):
        group = [r["kinematic_attribution"] for r in rows if r["phase"]==phase]
        rms = lambda values:math.sqrt(sum(v*v for v in values)/len(values))
        fields = ("joint_tracking_sum_xyz_m", "goal_point_mapping_error_xyz_m",
                  "ik_target_linearization_residual_xyz_m", "displacement_linearization_residual_xyz_m",
                  "measured_sensor_target_error_xyz_m", "accounting_closure_error_xyz_m")
        phases[phase] = {"samples":len(group),"rms_xyz_m":{
            field:[rms([r[field][i] for r in group]) for i in range(3)] for field in fields},
            "per_joint_y_contribution_rms_m":[rms([r["joint_tracking_contributions_xyz_m"][1][j] for r in group]) for j in range(7)]}
    return {"phases":phases,
        "scope":"pre-integration Jacobian times local joint errors; first-order accounting, not exact causal or torque decomposition; closure is algebraic",
        "frame":"all geometric Jacobian rows and XYZ vectors expressed in robot root axes"}
