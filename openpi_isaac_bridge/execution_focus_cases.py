"""Follow-up cases separating gripper lag, J6 saturation and rotation/TCP drift."""


def focus_cases():
    result=[]
    for profile in ('inherited','kp1000-kd50','kp1000-kd100','kp350-kd50','kp350-kd100'):
        result.append({'name':'lag-'+profile,'kind':'gripper','gripper_mode':'substep','cap_m':.0005,
            'finger_profile':profile,'requires':[] if profile=='inherited' else ['lag-inherited']})
    result.append({**result[2],'name':'lag-kp1000-kd100-armature1','finger_armature':1.,'requires':['lag-kp1000-kd100']})
    result.append({'name':'factor-translation5','kind':'translation','gripper_mode':'binary','cap_m':.005,'requires':[]})
    for deg in (.1,.5):
        result.append({'name':f'factor-rotation{deg:g}','kind':'recorded-rotation','gripper_mode':'binary',
            'cap_m':.0005,'rotation_cap_deg':deg,'requires':[] if deg==.1 else ['factor-rotation0.1']})
    for mm,deg in ((2,.1),(5,.1),(2,.5),(5,.5)):
        required=['factor-translation5',f'factor-rotation{deg:g}']
        if (mm,deg)==(5,.5):required+=['factor-t5-r0.1','factor-t2-r0.5']
        result.append({'name':f'factor-t{mm}-r{deg:g}','kind':'combined','cap_m':mm/1000,
            'rotation_cap_deg':deg,'gripper_mode':'binary','force_open':True,'requires':required})
    for mode in ('current','accumulated','anchored_tcp'):
        for sign in (-1,1):
            name=f"drift-rx-{'negative' if sign<0 else 'positive'}-{mode.replace('_','-')}"
            result.append({'name':name,'kind':'rotation','axis':0,'sign':sign,'rotation_cap_deg':.5,
                'reference_mode':mode,'cap_m':.0005,'gripper_mode':'binary',
                'requires':[] if mode=='current' else [f"drift-rx-{'negative' if sign<0 else 'positive'}-current"]})
    return result


def linearized_tcp(jacobian,dq,root_quat,hand_quat):
    """Actual IK107 Jacobian -> TCP97 first-order displacement, in world axes."""
    import numpy as np
    from kinematic_attribution import rotate
    J=np.asarray(jacobian);dq=np.asarray(dq)
    if J.shape!=(6,7) or dq.shape!=(7,) or not np.isfinite(J).all() or not np.isfinite(dq).all():
        raise ValueError('expected finite 6x7 Jacobian and seven joint increments')
    delta=J@dq
    linear=np.array(rotate(root_quat,delta[:3].tolist()));angular=np.array(rotate(root_quat,delta[3:].tolist()))
    lever=np.array(rotate(hand_quat,[0,0,-.010]))
    shift=np.cross(angular,lever)
    return {'ik107_linear_world_m':linear.tolist(),'angular_world_rad':angular.tolist(),
        'tcp97_lever_correction_world_m':shift.tolist(),'tcp97_predicted_world_m':(linear+shift).tolist(),
        'scope':'first-order actual IK107 Jacobian and rigid 10mm point shift; not a dynamics prediction'}
