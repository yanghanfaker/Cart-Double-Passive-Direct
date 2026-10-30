"""Static pinhole/pose audit. No policy queries, target changes or extra physics steps."""
import base64
import hashlib
import inspect
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

from kinematic_attribution import normalized_quaternion, rotate, resolve_sensor_body_path
from openpi_step_probe import libero_state_vector
from replay_probe import quaternion_distance_deg


def multiply(a,b):
    x,y,z,w=normalized_quaternion(a); X,Y,Z,W=normalized_quaternion(b)
    return [w*X+x*W+y*Z-z*Y,w*Y-x*Z+y*W+z*X,w*Z+x*Y-y*X+z*W,w*W-x*X-y*Y-z*Z]


def inverse(q):
    x,y,z,w=normalized_quaternion(q)
    return [-x,-y,-z,w]


def compose(parent,offset):
    translated=rotate(parent["quat_xyzw"],offset["position_m"])
    return {"position_m":[a+b for a,b in zip(parent["position_m"],translated)],
            "quat_xyzw":multiply(parent["quat_xyzw"],offset["quat_xyzw"])}


def pose_matrix(pose,invert=False):
    q=pose["quat_xyzw"];p=pose["position_m"]
    if invert:
        q=inverse(q);p=rotate(q,[-v for v in p])
    columns=[rotate(q,[1,0,0]),rotate(q,[0,1,0]),rotate(q,[0,0,1])]
    return [[columns[j][i] for j in range(3)]+[p[i]] for i in range(3)]+[[0,0,0,1]]


def project(point,pose,K,width,height):
    if len(point)!=3 or not all(math.isfinite(v) for v in point):raise ValueError("invalid world point")
    if len(K)!=3 or any(len(row)!=3 or not all(math.isfinite(v) for v in row) for row in K):raise ValueError("invalid K")
    if K[0][0]<=0 or K[1][1]<=0 or K[2]!=[0,0,1]:raise ValueError("expected pinhole K with positive focal lengths")
    xyz=rotate(inverse(pose["quat_xyzw"]),[a-b for a,b in zip(point,pose["position_m"])])
    if xyz[2]<=1e-8:return {"camera_ros_xyz_m":xyz,"uv_px":None,"in_front":False,"inside_image":False}
    uv=[sum(K[i][j]*xyz[j] for j in range(3))/xyz[2] for i in range(2)]
    return {"camera_ros_xyz_m":xyz,"uv_px":uv,"in_front":True,"inside_image":0<=uv[0]<width and 0<=uv[1]<height}


def closure(expected,measured):
    pos=math.dist(expected["position_m"],measured["position_m"])
    angle=quaternion_distance_deg(expected["quat_xyzw"],measured["quat_xyzw"])
    return {"position_error_m":pos,"rotation_error_deg":angle,"within_0p1mm_0p05deg":pos<=.0001 and angle<=.05}


def write_overlay(path,image_bytes,projections,width,height):
    # Vector overlay embeds original bytes; it never changes the saved RGB PNG.
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height+96}" viewBox="0 0 {width} {height+96}">',
           '<rect width="100%" height="100%" fill="#161b22"/>',
           f'<image width="{width}" height="{height}" href="data:image/png;base64,{base64.b64encode(image_bytes).decode()}"/>']
    colors={"object_root":"#ffe600","sensor_tcp":"#ff5555","ik_tcp":"#00e5ff","sensor_body":"#ffffff","robot_root":"#62ff7a"}
    for i,(name,p) in enumerate(projections.items()):
        color=colors.get(name,"#ffaaff");uv=p["uv_px"]
        if p["inside_image"]:
            u,v=uv;parts.append(f'<circle cx="{u}" cy="{v}" r="4" fill="none" stroke="{color}" stroke-width="1.3"/>')
            parts.append(f'<text x="{min(width-85,max(1,u+5))}" y="{min(height-3,max(12,v-5))}" font-size="9" fill="{color}">{escape(name)}</text>')
        label=f'{name}: '+(f'({uv[0]:.1f},{uv[1]:.1f})' if uv else 'behind camera')
        parts.append(f'<text x="4" y="{height+14+i*15}" font-size="10" fill="{color}">{escape(label)}</text>')
    parts.append('</svg>');Path(path).write_text('\n'.join(parts),encoding='utf-8')


def capture_geometry(base,cfg,arm_term,images,output,tensor,warmup_controls):
    import numpy as np
    from PIL import Image
    from pxr import UsdPhysics
    import isaaclab.sim as sim_utils
    from openpi_client import image_tools
    output=Path(output)
    report={"status":"running","diagnostic_motion_steps":0,"policy_queries":0,
            "warmup_hold_controls":warmup_controls,"quaternion_order":"xyzw",
            "scope":"Isaac internal geometry/coordinate audit; not LIBERO registration or grasp validation",
            "training_tcp_registration":"unverified: training model attachment and frame registration not supplied",
            "references":{"camera_data":"https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/sensors/camera/camera_data.py",
                          "libero_input":"https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py"}}
    def save():
        (output/'geometry_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    def arr(value):
        if hasattr(value,"detach"):return value.detach().clone().cpu().tolist()
        if hasattr(value,"torch"):return value.torch.detach().clone().cpu().tolist()
        if hasattr(value,"tolist"):return value.tolist()
        return value
    def safe(value,depth=0):
        if value is None or isinstance(value,(str,bool,int,float)):return value
        if depth>12:return "<depth limit>"
        if isinstance(value,dict):return {str(k):safe(v,depth+1) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [safe(v,depth+1) for v in value]
        converted=arr(value)
        return safe(converted,depth+1) if converted is not value else f'<{type(value).__name__}>'
    def pose(position,quaternion):return {"position_m":position,"quat_xyzw":quaternion}
    save()
    try:
        before=int(base.sim.get_physics_step_count())
        robot=base.scene['robot'];ee=base.scene['ee_frame'];obj=base.scene['object']
        report['scene_setup']={'seed':cfg.seed,'object_pose_range':cfg.events.reset_object_position.params['pose_range'],
            'robot_gravity_disabled':cfg.scene.robot.spawn.rigid_props.disable_gravity,
            'control_dt_s':float(base.step_dt),'physics_dt_s':float(cfg.sim.dt),
            'wrist_initial_aim':'fixed scene reference [0.5,0,0.04]; no live object re-aiming'}
        root=pose(arr(robot.data.root_pos_w)[0],arr(robot.data.root_quat_w)[0])
        sensor=pose(arr(ee.data.target_pos_w)[0][0],arr(ee.data.target_quat_w)[0][0])
        ikp,ikq=arm_term._compute_frame_pose()
        ik_root=pose(arr(ikp)[0],arr(ikq)[0]);ik=compose(root,ik_root)
        stage=sim_utils.get_current_stage()
        if stage is None:raise RuntimeError('no active USD stage')
        rigid_paths=[str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        sensor_cfg=cfg.scene.ee_frame.target_frames[0]
        resolution=resolve_sensor_body_path(sensor_cfg.prim_path,robot.cfg.prim_path,base.scene.env_regex_ns,rigid_paths,list(robot.body_names))
        report['body_resolution']=resolution
        sensor_id=resolution['sensor_body_ids'][0];ik_id=int(arm_term._body_idx)
        body_positions=arr(robot.data.body_pos_w)[0];body_quats=arr(robot.data.body_quat_w)[0]
        hand=pose(body_positions[sensor_id],body_quats[sensor_id]);ik_body=pose(body_positions[ik_id],body_quats[ik_id])
        sensor_offset=pose(list(sensor_cfg.offset.pos),list(sensor_cfg.offset.rot))
        ik_offset=pose(list(arm_term.cfg.body_offset.pos),list(arm_term.cfg.body_offset.rot))
        fingers,finger_names=robot.find_joints('panda_finger.*')
        if list(finger_names)!=['panda_finger_joint1','panda_finger_joint2']:raise RuntimeError('unexpected finger ordering')
        finger_positions=tensor(robot.data.joint_pos)[0,fingers].detach().cpu().tolist()
        state=libero_state_vector(sensor['position_m'],sensor['quat_xyzw'],finger_positions)
        report.update(robot_root_world=root,sensor_body_world=hand,ik_body_world=ik_body,
            sensor_tcp_world=sensor,ik_tcp_root=ik_root,ik_tcp_world=ik,
            sensor_body_offset=sensor_offset,ik_body_offset=ik_offset,
            same_sensor_ik_body=sensor_id==ik_id,ik_body_index=ik_id,ik_body_name=robot.body_names[ik_id],
            sensor_offset_closure=closure(compose(hand,sensor_offset),sensor),
            ik_offset_closure=closure(compose(ik_body,ik_offset),ik),
            sensor_ik_distance_m=math.dist(sensor['position_m'],ik['position_m']),
            policy_state_world_float32=np.asarray(state,dtype=np.float32).tolist(),
            policy_state_layout='world TCP xyz; signed rotation vector; Isaac q1 and negated q2',
            joint_names=list(robot.joint_names),joint_positions_rad=arr(robot.data.joint_pos)[0],joint_velocities_rad_s=arr(robot.data.joint_vel)[0],
            finger_positions_isaac_m=finger_positions,
            finger_body_origins_world={name:body_positions[i] for i,name in enumerate(robot.body_names) if 'finger' in name},
            finger_origin_note='rigid body origins, not measured fingertip contact points',
            object_root_world=pose(arr(obj.data.root_pos_w)[0],arr(obj.data.root_quat_w)[0]))
        points={'object_root':report['object_root_world']['position_m'],'sensor_tcp':sensor['position_m'],
                'ik_tcp':ik['position_m'],'sensor_body':hand['position_m'],'robot_root':root['position_m']}
        report['cameras']={};implementation=[]
        for name in ('external_camera','wrist_camera'):
            camera=base.scene[name];data=camera.data
            if not np.array_equal(tensor(data.output['rgb'])[0,...,:3].detach().cpu().numpy(),images[name]):
                raise RuntimeError('camera buffer changed since saved raw image')
            current=pose(arr(data.pos_w)[0],arr(data.quat_w_ros)[0])
            live_world_q=arr(data.quat_w_world)[0];K=arr(data.intrinsic_matrices)[0]
            info=safe(data.info);rgb_info=info.get('rgb') if isinstance(info,dict) else None
            capture=rgb_info.get('capture') if isinstance(rgb_info,dict) else None
            if capture is None and isinstance(info,dict):capture=info.get('capture')
            projection_pose=current;projection_K=K;binding='live CameraData; capture-time binding unavailable'
            if isinstance(capture,dict) and all(k in capture for k in ('pos_w','quat_w_world','intrinsic_matrices')):
                p=capture['pos_w'][0];q=capture['quat_w_world'][0]
                convention=multiply(inverse(live_world_q),current['quat_xyzw'])
                projection_pose=pose(p,multiply(q,convention));projection_K=capture['intrinsic_matrices'][0]
                binding='capture metadata tied to RGB; projected body points remain current static world poses'
            height,width=images[name].shape[:2]
            projections={label:project(point,projection_pose,projection_K,width,height) for label,point in points.items()}
            raw=(output/f'{name}.png').read_bytes()
            processed=image_tools.resize_with_pad(images[name],224,224)
            if processed.dtype!=np.uint8 or processed.shape!=(224,224,3):raise RuntimeError('invalid policy RGB shape')
            Image.fromarray(processed).save(output/f'{name}_policy224.png')
            write_overlay(output/f'{name}_projection.svg',raw,projections,width,height)
            report['cameras'][name]={'prim_path_pattern':camera.cfg.prim_path,'image_shape_hw':[height,width],
                'live_pose_world_ros':current,'live_quat_world_convention_xyzw':live_world_q,
                'live_quat_opengl_xyzw':arr(data.quat_w_opengl)[0],
                'live_intrinsic_matrix':K,'projection_pose_world_ros':projection_pose,'projection_intrinsic_matrix':projection_K,
                'world_from_camera_ros':pose_matrix(projection_pose),'camera_ros_from_world':pose_matrix(projection_pose,True),
                'frame':arr(camera.frame),'camera_data_info':info,'projection_binding':binding,
                'projection_convention':'+Z forward, +X right, +Y down; homogeneous pinhole K; raw image pixels',
                'projections':projections,'raw_png_sha256':hashlib.sha256(raw).hexdigest(),
                'policy_png_sha256':hashlib.sha256((output/f'{name}_policy224.png').read_bytes()).hexdigest(),
                'policy_preprocessing':'existing openpi_client resize_with_pad to 224; RGB; no flip',
                'projection_note':'no independently measured pixel correspondences; visibility flags ignore occlusion and do not establish reprojection accuracy',
                'spawn_parameters':{k:getattr(camera.cfg.spawn,k,None) for k in ('focal_length','horizontal_aperture','vertical_aperture','clipping_range')}}
            for prop in ('pos_w','quat_w_world','quat_w_ros','intrinsic_matrices'):
                try:implementation.append(f'\n{name}: {prop}\n'+inspect.getsource(inspect.getattr_static(type(data),prop).fget))
                except (OSError,TypeError,AttributeError):implementation.append(f'{name}: source unavailable for {prop}')
        (output/'camera_implementation.txt').write_text('\n'.join(implementation),encoding='utf-8')
        report['physics_step_count_before']=before;report['physics_step_count_after']=int(base.sim.get_physics_step_count())
        if report['physics_step_count_after']!=before:raise RuntimeError('audit unexpectedly advanced physics')
        report['status']='completed'
        print('[Geometry audit] sensor closure='+str(report['sensor_offset_closure'])+'; IK closure='+str(report['ik_offset_closure']),flush=True)
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        save();print(f'[Geometry audit] Report: {output/"geometry_audit.json"}',flush=True)
