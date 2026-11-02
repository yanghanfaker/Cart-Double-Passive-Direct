"""J1/J3 response comparisons; all other arm gains and effort limits stay fixed."""
import copy
import math

ARM_PROFILES={
    'baseline':{},
    'd50':{'panda_joint1':{'damping':50.},'panda_joint3':{'damping':50.}},
    'k1687p5':{'panda_joint1':{'stiffness':1687.5},'panda_joint3':{'stiffness':1687.5}},
    'k1687p5-d50':{'panda_joint1':{'stiffness':1687.5,'damping':50.},'panda_joint3':{'stiffness':1687.5,'damping':50.}},
    'j1-d50':{'panda_joint1':{'damping':50.}},
    'j3-d50':{'panda_joint3':{'damping':50.}},
}


def sync_cases():
    def rx(name,profile,deg,sign=1,reference='anchored_tcp'):
        return {'name':name,'kind':'rotation','axis':0,'sign':sign,'rotation_cap_deg':deg,
            'reference_mode':reference,'arm_profile':profile,'cap_m':.0005,'gripper_mode':'binary','requires':[]}
    result=[rx('sync-rx-baseline-0.1','baseline',.1),rx('sync-rx-current-0.25','baseline',.25,reference='current')]
    for profile in ARM_PROFILES:result.append(rx('sync-rx-'+profile+'-positive',profile,.25))
    for profile in ('baseline','d50','k1687p5-d50'):result.append(rx('sync-rx-'+profile+'-negative',profile,.25,-1))
    result.append(rx('sync-rx-baseline-0.5','baseline',.5))
    for profile in ('baseline','d50','k1687p5-d50'):
        for deg in (.1,.25):
            result.append({'name':f'sync-combined-{profile}-{deg:g}','kind':'combined','cap_m':.002,
                'rotation_cap_deg':deg,'arm_profile':profile,'gripper_mode':'binary','force_open':True,
                'requires':[] if deg==.1 else [f'sync-combined-{profile}-0.1']})
    for profile,armature in (('kp350-kd50',None),('kp1000-kd100',1.)):
        case={'name':'sync-closure-'+profile,'kind':'combined','cap_m':.002,'rotation_cap_deg':.1,
            'arm_profile':'baseline','finger_profile':profile,'gripper_mode':'substep','requires':['sync-combined-baseline-0.1']}
        if armature is not None:case['finger_armature']=armature
        result.append(case)
    return result


def gain_plan(source_overrides,source_audit,profile):
    if profile not in ARM_PROFILES:raise ValueError('unknown J1/J3 profile')
    names=[f'panda_joint{i}' for i in range(1,8)]
    if source_audit['joint_names']!=names:raise ValueError('source arm order differs')
    actual={key:source_audit['runtime']['joint_'+key]['values'] for key in ('stiffness','damping')}
    if any(len(v)!=7 or any(not math.isfinite(x) or x<=0 for x in v) for v in actual.values()):raise ValueError('invalid source runtime gains')
    for joint in ('panda_joint1','panda_joint3'):
        i=names.index(joint)
        if source_overrides.get(joint)!={'stiffness':1125,'damping':25} or actual['stiffness'][i]!=1125 or actual['damping'][i]!=25:
            raise ValueError('expected recorded J1/J3 baseline 1125/25')
    overrides=copy.deepcopy(source_overrides);expected=copy.deepcopy(actual)
    for joint,changes in ARM_PROFILES[profile].items():
        overrides[joint].update(changes)
        for key,value in changes.items():expected[key][names.index(joint)]=value
    return overrides,expected
