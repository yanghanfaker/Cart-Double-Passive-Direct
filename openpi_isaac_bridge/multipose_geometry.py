"""Multi-pose geometry design and held-out registration, independent of controllers."""
import json
import math
from pathlib import Path

LIMITS = [(-2.8973,2.8973),(-1.7628,1.7628),(-2.8973,2.8973),(-3.0718,-.0698),
          (-2.8973,2.8973),(-.0175,3.7525),(-2.8973,2.8973)]


def validate_arm_pose(values):
    if not isinstance(values,list) or len(values)!=7:
        raise ValueError('expected seven arm angles')
    for value,(lo,hi) in zip(values,LIMITS):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not lo+.01<=value<=hi-.01:
            raise ValueError('nonfinite joint angle or outside conservative Panda limits')
    return list(values)


def design_poses(home):
    home=validate_arm_pose(home)
    variations=[('home','fit',{}),('j1-positive','fit',{0:.20}),('j2-positive','fit',{1:.18}),
        ('j3-negative','fit',{2:-.20}),('j4-positive','fit',{3:.18}),
        ('j5-positive','fit',{4:.20}),('j6-negative','fit',{5:-.15}),
        ('mixed12','validation',{0:-.15,1:-.12}),('mixed34','validation',{2:.16,3:.12}),
        ('mixed56','validation',{4:-.18,5:.12}),('mixed147','validation',{0:.12,3:.10,6:.18})]
    result=[]
    for name,split,changes in variations:
        q=list(home)
        for i,v in changes.items():q[i]+=v
        result.append({'name':name,'split':split,'requested_joint_positions_rad':validate_arm_pose(q)})
    return result


def fit_rigid(x,y):
    import numpy as np
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    if x.shape!=y.shape or x.ndim!=2 or x.shape[1]!=3 or len(x)<4 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('need at least four finite paired xyz positions')
    singular=np.linalg.svd(x-x.mean(0),compute_uv=False)
    if singular[1]<.005:
        raise ValueError('insufficient position spread for rigid registration: second singular value <5 mm')
    u,s,vt=np.linalg.svd((x-x.mean(0)).T@(y-y.mean(0)))
    correction=np.eye(3);correction[2,2]=np.linalg.det(vt.T@u.T)
    rotation=vt.T@correction@u.T
    return rotation,y.mean(0)-rotation@x.mean(0),singular


def rotation_angle(rotation):
    import numpy as np
    return math.degrees(math.acos(float(np.clip((np.trace(rotation)-1)/2,-1,1))))


def analyze(folder):
    import numpy as np
    from geometry_audit import pose_matrix
    folder=Path(folder);manifest=json.loads((folder/'run.json').read_text())
    if len(manifest['poses'])!=11 or any(p.get('status')!='completed' for p in manifest['poses']):
        raise ValueError('all eleven pose captures required')
    pairs=[]
    for item in manifest['poses']:
        isaac=json.loads((folder/item['isaac_audit']).read_text())
        ref=json.loads((folder/item['reference']).read_text())
        if isaac['status']!='completed' or ref['status']!='completed':raise ValueError('incomplete paired capture')
        iq=dict(zip(isaac['joint_names'],isaac['joint_positions_rad']))
        assigned=ref['joint_assignments']
        actual=[q['actual'] for q in assigned]
        expected=[iq[f'panda_joint{i}'] for i in range(1,8)]+isaac['finger_positions_isaac_m'][:1]+[-isaac['finger_positions_isaac_m'][1]]
        error=max(abs(a-b) for a,b in zip(actual,expected))
        if len(actual)!=9 or error>1e-7:raise ValueError('actual joint pose mismatch')
        if isaac['physics_step_count_before']!=isaac['physics_step_count_after'] or ref['sim_time_before_forward']!=ref['sim_time_after_capture']:
            raise ValueError('capture advanced simulation')
        hand=np.array(pose_matrix(isaac['sensor_body_world']))
        ir=np.array(pose_matrix(isaac['robot_root_world']))
        lr=np.array(ref['robot_root']['rotation_world_matrix'])
        lp=np.array(ref['robot_root']['position_world_m'])
        sensor=np.array(isaac['sensor_tcp_world']['position_m'])
        offset=np.array(isaac['sensor_body_offset']['position_m'])
        if np.linalg.norm(offset-[0,0,.1034])>1e-8:raise ValueError('unexpected source TCP offset')
        if not isaac['same_sensor_ik_body'] or not isaac['sensor_offset_closure']['within_0p1mm_0p05deg']:
            raise ValueError('Isaac internal frame closure failed')
        pairs.append({'item':item,'hand':hand[:3,3],'hand_R':hand[:3,:3],
            'ref_hand':np.array(ref['robot_eef_body']['position_world_m']),
            'ref_hand_R':np.array(ref['robot_eef_body']['rotation_world_matrix']),
            'sensor':sensor,'corrected':sensor+hand[:3,:3]@np.array([0,0,-.0064]),
            'site':np.array(ref['eef_site']['position_world_m']),
            'nominal_R':lr@ir[:3,:3].T,'nominal_t':lp-lr@ir[:3,:3].T@ir[:3,3],
            'actual_joint_match_max_rad_or_m':error,
            'requested_actual_max_rad':max(abs(iq[f'panda_joint{i+1}']-v) for i,v in enumerate(item['requested_joint_positions_rad']))})
    training=[p for p in pairs if p['item']['split']=='fit']
    rotation,translation,spread=fit_rigid([p['hand'] for p in training],[p['ref_hand'] for p in training])
    # World transform anchored with hand origins first; estimate local TCP correction only on fit poses.
    matrix=np.vstack([rotation@p['hand_R'] for p in training])
    rhs=np.concatenate([p['site']-(rotation@p['sensor']+translation) for p in training])
    fitted_offset=np.linalg.lstsq(matrix,rhs,rcond=None)[0]
    rows=[]
    for p in pairs:
        def residual(point,R=rotation,t=translation):return float(np.linalg.norm(R@p[point]+t-p['site'])*1000)
        row={k:p['item'][k] for k in ('name','split')}
        row.update(hand_error_mm=float(np.linalg.norm(rotation@p['hand']+translation-p['ref_hand'])*1000),
            hand_rotation_error_deg=rotation_angle(p['ref_hand_R'].T@rotation@p['hand_R']),
            nominal_base_corrected_tcp_error_mm=residual('corrected',p['nominal_R'],p['nominal_t']),
            registered_raw_tcp_error_mm=residual('sensor'),registered_corrected_tcp_error_mm=residual('corrected'),
            fitted_tcp_error_mm=float(np.linalg.norm(rotation@(p['sensor']+p['hand_R']@fitted_offset)+translation-p['site'])*1000),
            actual_joint_match_max_rad_or_m=p['actual_joint_match_max_rad_or_m'],
            requested_actual_max_rad=p['requested_actual_max_rad'])
        rows.append(row)
    summaries={}
    for split in ('fit','validation'):
        subset=[r for r in rows if r['split']==split]
        summaries[split]={key:{'rms':math.sqrt(sum(r[key]**2 for r in subset)/len(subset)),
                               'max':max(r[key] for r in subset)} for key in rows[0] if key.endswith(('_mm','_deg'))}
    result={'status':'completed','policy_queries':0,'poses':rows,
        'world_from_isaac_rotation':rotation.tolist(),'world_from_isaac_translation_m':translation.tolist(),
        'fit_position_spread_singular_values_m':spread.tolist(),'fitted_tcp_delta_hand_m':fitted_offset.tolist(),
        'fixed_tcp_delta_hand_m':[0,0,-.0064],'summary':summaries,
        'validation_criteria':{'corrected_tcp_max_mm':1.0,'hand_rotation_max_deg':.1,'validation_poses':4},
        'candidate_geometry_within_thresholds':summaries['validation']['registered_corrected_tcp_error_mm']['max']<=1.0 and summaries['validation']['hand_rotation_error_deg']['max']<=.1,
        'scope':'fit uses seven hand origins; four held-out poses never used in fitting; geometry only, no camera alignment, controller equivalence or grasp validation'}
    (folder/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('[Multi-pose] Validation: '+json.dumps(summaries['validation']),flush=True)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    analyze(parser.parse_args().folder)
