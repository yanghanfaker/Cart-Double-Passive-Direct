"""Unmarked proxy scene capture; geometry and image diagnostics only."""
import json
import math
from pathlib import Path


def validate_clean_scene(config):
    import numpy as np
    from multipose_geometry import validate_arm_pose
    from camera_alignment_probe import validate_candidate
    validate_arm_pose(config['arm_positions_rad']);validate_candidate(config['camera_candidate'])
    xy=config['object_xy_world_m']
    if len(xy)!=2 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in xy):raise ValueError('invalid requested object XY')
    if not 0.25<=xy[0]<=.8 or abs(xy[1])>.3:raise ValueError('proxy XY outside diagnostic workspace')
    if config.get('cube_z_policy')!='preserve_existing_spawn_height_and_check_rest_height':raise ValueError('must preserve cube support height')
    for key,length in [('baseline_cube_rest_position_m',3),('reference_eef_position_world_m',3),('reference_world_translation_m',3)]:
        v=config[key]
        if len(v)!=length or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in v):raise ValueError('invalid '+key)
    R=np.asarray(config['reference_world_rotation'])
    if R.shape!=(3,3) or not np.isfinite(R).all() or not np.allclose(R.T@R,np.eye(3),atol=1e-5) or abs(np.linalg.det(R)-1)>1e-5:raise ValueError('invalid reference rotation')
    return config


def color_hint(image,uv):
    """Local saturation hint only; does not establish object segmentation/visibility."""
    import numpy as np
    if uv is None:return {'colored_pixels':0,'note':'projection behind camera'}
    u,v=uv;x0=max(0,int(u)-18);x1=min(256,int(u)+19);y0=max(0,int(v)-18);y1=min(256,int(v)+19)
    if x0>=x1 or y0>=y1:return {'colored_pixels':0,'roi_xyxy':[x0,y0,x1,y1]}
    region=image[y0:y1,x0:x1].astype(float);maximum=region.max(axis=2);minimum=region.min(axis=2)
    mask=(maximum>90)&(maximum-minimum>60)
    return {'colored_pixels':int(np.sum(mask)),'roi_xyxy':[x0,y0,x1,y1],
        'note':'heuristic saturation count near projected cube center; background may contribute, occlusion not ruled out'}


def finish_clean_capture(config,output,images):
    import numpy as np
    from PIL import Image
    from geometry_audit import pose_matrix
    output=Path(output);audit=json.loads((output/'geometry_audit.json').read_text())
    setup=json.loads((output/'camera_alignment_setup.json').read_text())
    if setup.get('scene_contains_markers') or any(c['markers'] for c in setup['cameras'].values()):raise RuntimeError('clean scene unexpectedly contains markers')
    joint=dict(zip(audit['joint_names'],audit['joint_positions_rad']))
    error=max(abs(joint[f'panda_joint{k+1}']-v) for k,v in enumerate(config['arm_positions_rad']))
    obj=np.array(audit['object_root_world']['position_m']);xy_error=float(np.linalg.norm(obj[:2]-config['object_xy_world_m']))
    z_change=abs(float(obj[2])-config['baseline_cube_rest_position_m'][2])
    hand=np.array(pose_matrix(audit['sensor_body_world']))
    corrected=np.array(audit['sensor_tcp_world']['position_m'])+hand[:3,:3]@np.array([0,0,-.0064])
    registered=np.array(config['reference_world_rotation'])@corrected+config['reference_world_translation_m']
    result={'status':'completed','case':config['name'],'policy_queries':0,'model_actions_executed':0,'calibration_markers_present':False,
        'arm_max_position_error_rad':error,'cube_requested_xy_world_m':config['object_xy_world_m'],'cube_actual_world_m':obj.tolist(),
        'cube_xy_error_m':xy_error,'cube_rest_z_change_from_baseline_m':z_change,
        'mapped_tcp_position_difference_from_benchmark_m':float(np.linalg.norm(registered-config['reference_eef_position_world_m'])),
        'geometry_checks_passed':error<=.001 and xy_error<=.001 and z_change<=.002,
        'thresholds':{'arm_rad':.001,'object_xy_m':.001,'rest_height_change_m':.002},
        'scope':'arm initialization and cube proxy XY only; cube height rests on original support; not full LIBERO scene or object reproduction','cameras':{}}
    for name,image in images.items():
        uv=audit['cameras'][name]['projections']['object_root']['uv_px']
        margin=min(uv[0],uv[1],255-uv[0],255-uv[1]) if uv else None
        raw=np.array(Image.open(output/f'{name}_policy224.png'))
        Image.fromarray(np.ascontiguousarray(raw[:,::-1])).save(output/f'{name}_policy224_flip_x.png')
        c=config['camera_candidate']['cameras'][name];actual=np.array(pose_matrix(audit['cameras'][name]['live_pose_world_ros']))
        local=actual if c['frame']=='isaac_world' else np.linalg.inv(hand)@actual
        position_error=float(np.linalg.norm(local[:3,3]-c['position_m']))
        angle=math.degrees(math.acos(float(np.clip((np.trace(np.array(c['rotation_ros']).T@local[:3,:3])-1)/2,-1,1))))
        K_error=float(np.max(np.abs(np.array(audit['cameras'][name]['live_intrinsic_matrix'])-c['K_256'])))
        camera_passed=position_error<=.0001 and angle<=.05 and K_error<=.05
        result['cameras'][name]={'object_center_projection_px':uv,'center_margin_px':margin,
            'camera_parameters_passed':camera_passed,'camera_position_error_m':position_error,'camera_rotation_error_deg':angle,'K_max_abs_error_px':K_error,
            'center_at_least_8px_inside':margin is not None and margin>=8,
            'color_hint':color_hint(image,uv),
            'frames':{'no_flip':f'{name}_policy224.png','horizontal_flip':f'{name}_policy224_flip_x.png'}}
    result['geometry_checks_passed']=result['geometry_checks_passed'] and all(v['camera_parameters_passed'] for v in result['cameras'].values())
    (output/'clean_scene_result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('[Clean scene] '+json.dumps({'case':config['name'],'geometry':result['geometry_checks_passed'],
        'projections':{k:v['object_center_projection_px'] for k,v in result['cameras'].items()}}),flush=True)
    return result
