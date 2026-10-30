"""Static rendered landmark test for proposed cameras; never used by a policy."""
import json
import math
from pathlib import Path

PIXELS=((58.,67.),(184.,91.),(103.,192.))
COLORS=((1.,0.,1.),(0.,1.,1.),(1.,1.,0.))


def validate_candidate(data):
    import numpy as np
    if data.get('status')!='proposal_not_applied':raise ValueError('expected camera proposal')
    cameras=data['cameras']
    for name,frame in [('external_camera','isaac_world'),('wrist_camera','panda_hand')]:
        c=cameras[name];p=np.array(c['position_m']);R=np.array(c['rotation_ros']);K=np.array(c['K_256'])
        if c['frame']!=frame or p.shape!=(3,) or R.shape!=(3,3) or K.shape!=(3,3):raise ValueError('invalid camera frame or shape')
        if not all(np.isfinite(v).all() for v in (p,R,K)):raise ValueError('nonfinite camera parameters')
        if np.max(np.abs(R.T@R-np.eye(3)))>1e-5 or abs(np.linalg.det(R)-1)>1e-5:raise ValueError('invalid proper rotation')
        if K[0,0]<=0 or not np.allclose(K,[[K[0,0],0,128],[0,K[0,0],128],[0,0,1]],atol=1e-6):raise ValueError('expected centered square pixels at 256 resolution')
    return data


def quat_xyzw(R):
    """Convert a proper rotation matrix to a normalized xyzw quaternion."""
    import numpy as np
    R=np.array(R);tr=float(np.trace(R))
    if tr>0:
        s=math.sqrt(tr+1)*2;q=[(R[2,1]-R[1,2])/s,(R[0,2]-R[2,0])/s,(R[1,0]-R[0,1])/s,s/4]
    else:
        i=int(np.argmax(np.diag(R)));j=(i+1)%3;k=(i+2)%3
        s=math.sqrt(max(0,1+R[i,i]-R[j,j]-R[k,k]))*2
        q=[0.,0.,0.,0.];q[i]=s/4;q[j]=(R[j,i]+R[i,j])/s;q[k]=(R[k,i]+R[i,k])/s;q[3]=(R[k,j]-R[j,k])/s
    q=np.array(q);return (q/np.linalg.norm(q)).tolist()


def marker_layout(position,rotation,K):
    import numpy as np
    p=np.array(position);R=np.array(rotation);K=np.array(K);markers=[]
    for index,(pixel,color) in enumerate(zip(PIXELS,COLORS)):
        depth=.14+index*.02
        camera=np.linalg.solve(K,np.array([*pixel,1.]))*depth
        markers.append({'id':index,'expected_ros_pixel':list(pixel),'world_position_m':(p+R@camera).tolist(),
            'radius_m':float(depth*4/K[0,0]),'emissive_rgb':color})
    return markers


def image_point(uv,kind):
    x,y=uv
    return [255-x if kind in ('flip_x','rotate180') else x,255-y if kind in ('flip_y','rotate180') else y]


def colored_components(image,color):
    import numpy as np
    image=np.asarray(image).astype(float);high=np.min(image[:,:,[i for i,c in enumerate(color) if c]],axis=2)
    low=image[:,:,[i for i,c in enumerate(color) if not c][0]]
    mask=(high>100)&(low<high*.65);seen=np.zeros(mask.shape,dtype=bool);components=[]
    for y,x in zip(*np.where(mask)):
        if seen[y,x]:continue
        stack=[(int(y),int(x))];seen[y,x]=True;points=[]
        while stack:
            a,b=stack.pop();points.append((a,b))
            for c,d in ((a-1,b),(a+1,b),(a,b-1),(a,b+1)):
                if 0<=c<mask.shape[0] and 0<=d<mask.shape[1] and mask[c,d] and not seen[c,d]:
                    seen[c,d]=True;stack.append((c,d))
        if 8<=len(points)<=300:
            coords=np.array(points);components.append({'pixels':len(points),'centroid_xy':coords[:,::-1].mean(0).tolist()})
    return components


def analyze_markers(image,markers):
    import numpy as np
    if image.shape!=(256,256,3) or image.dtype!=np.uint8:raise ValueError('expected native 256 RGB')
    components=[colored_components(image,m['emissive_rgb']) for m in markers]
    hypotheses=[]
    for kind in ('identity','flip_x','flip_y','rotate180'):
        matched=[]
        for marker,parts in zip(markers,components):
            choices=[(math.dist(image_point(c['centroid_xy'],kind),marker['expected_ros_pixel']),c) for c in parts]
            if not choices:break
            error,part=min(choices,key=lambda v:v[0]);matched.append({'id':marker['id'],'centroid_raw_xy':part['centroid_xy'],
                'centroid_after_transform_xy':image_point(part['centroid_xy'],kind),'pixel_error':error,'component_pixels':part['pixels']})
        if len(matched)==3:
            hypotheses.append({'transform':kind,'rms_pixel_error':math.sqrt(sum(m['pixel_error']**2 for m in matched)/3),
                'max_pixel_error':max(m['pixel_error'] for m in matched),'matches':matched})
    hypotheses.sort(key=lambda h:h['rms_pixel_error'])
    if not hypotheses:return {'passed':False,'reason':'not all three colored markers detected','components':components,'hypotheses':[]}
    best=hypotheses[0];gap=hypotheses[1]['rms_pixel_error']-best['rms_pixel_error']
    return {'passed':best['max_pixel_error']<=2 and gap>=10,'best_transform':best['transform'],
        'best_max_pixel_error':best['max_pixel_error'],'best_rms_pixel_error':best['rms_pixel_error'],
        'runner_up_rms_gap_px':gap,'components':components,'hypotheses':hypotheses,
        'scope':'known colored marker centroids only; no object recognition, distortion fit or policy validation'}


def apply_and_spawn(base,candidate,output,tensor,spawn_markers=True):
    import numpy as np
    import torch
    import inspect
    from pxr import UsdGeom,UsdShade,Sdf,Gf
    import isaaclab.sim as sim_utils
    from geometry_audit import pose_matrix
    from kinematic_attribution import resolve_sensor_body_path
    from pxr import UsdPhysics
    candidate=validate_candidate(candidate);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    stage=sim_utils.get_current_stage();robot=base.scene['robot']
    # Reuse the proven full-path resolver, rather than interpreting a USD path as regex.
    rigid=[str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    cfg=base.cfg.scene.ee_frame.target_frames[0]
    resolved=resolve_sensor_body_path(cfg.prim_path,robot.cfg.prim_path,base.scene.env_regex_ns,rigid,list(robot.body_names))
    index=resolved['sensor_body_ids'][0]
    hand={'position_m':tensor(robot.data.body_pos_w)[0,index].cpu().tolist(),'quat_xyzw':tensor(robot.data.body_quat_w)[0,index].cpu().tolist()}
    H=np.array(pose_matrix(hand));setup={'status':'applied_to_static_test_only','policy_queries':0,
        'markers_have_physics':False,'scene_contains_markers':spawn_markers,
        'quaternion_order':'xyzw','hand_pose_at_application':hand,'cameras':{}}
    impl=[]
    for name,c in candidate['cameras'].items():
        p=np.array(c['position_m']);R=np.array(c['rotation_ros'])
        if c['frame']=='panda_hand':p=H[:3,3]+H[:3,:3]@p;R=H[:3,:3]@R
        camera=base.scene[name];device=tensor(camera.data.pos_w).device
        camera.set_world_poses(positions=torch.tensor([p.tolist()],device=device,dtype=torch.float32),
            orientations=torch.tensor([quat_xyzw(R)],device=device,dtype=torch.float32),convention='ros')
        camera.set_intrinsic_matrices(torch.tensor([c['K_256']],device=device,dtype=torch.float32))
        markers=marker_layout(p,R,c['K_256']) if spawn_markers else []
        for marker in markers:
            path=f'/World/CameraAlignmentMarkers/{name}/marker{marker["id"]}'
            sphere=UsdGeom.Sphere.Define(stage,path);sphere.CreateRadiusAttr(marker['radius_m'])
            sphere.AddTranslateOp().Set(Gf.Vec3d(*marker['world_position_m']))
            sphere.CreateDisplayColorAttr([Gf.Vec3f(*marker['emissive_rgb'])])
            material=UsdShade.Material.Define(stage,path+'/Material');shader=UsdShade.Shader.Define(stage,path+'/Material/Shader')
            shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0,0,0))
            shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*marker['emissive_rgb']))
            shader.CreateOutput('surface',Sdf.ValueTypeNames.Token)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
            UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material)
            marker['prim_path']=path
        setup['cameras'][name]={'requested':c,'world_at_application':{'position_m':p.tolist(),'rotation_ros':R.tolist()},'markers':markers}
        for method in (camera.set_world_poses,camera.set_intrinsic_matrices):
            try:impl.append(inspect.getsource(method))
            except (OSError,TypeError):impl.append('source unavailable')
    (output/'camera_alignment_setup.json').write_text(json.dumps(setup,indent=2),encoding='utf-8')
    (output/'camera_setter_implementation.txt').write_text('\n\n'.join(impl),encoding='utf-8')
    return setup


def finish_capture(base,setup,images,output,tensor):
    import numpy as np
    from geometry_audit import pose_matrix
    from camera_reference_audit import overlay
    from PIL import Image
    from multipose_geometry import rotation_angle
    output=Path(output);audit=json.loads((output/'geometry_audit.json').read_text())
    H=np.array(pose_matrix(audit['sensor_body_world']))
    report={'status':'completed','policy_queries':0,'camera_parameters_applied_to_static_test':True,
        'scene_contains_calibration_markers':True,'camera_motion_during_capture':'five existing hold controls after one-time installation',
        'thresholds':{'position_m':.0001,'rotation_deg':.05,'K_max_abs_px':.05,'landmark_max_px':2,'orientation_rms_gap_px':10},'cameras':{}}
    for name,s in setup['cameras'].items():
        camera=base.scene[name];actual=np.array(pose_matrix(audit['cameras'][name]['live_pose_world_ros']))
        requested=s['requested'];p=np.array(requested['position_m']);R=np.array(requested['rotation_ros'])
        observed=actual if requested['frame']=='isaac_world' else np.linalg.inv(H)@actual
        pos=float(np.linalg.norm(observed[:3,3]-p));rot=rotation_angle(R.T@observed[:3,:3])
        K=float(np.max(np.abs(np.array(audit['cameras'][name]['live_intrinsic_matrix'])-requested['K_256'])))
        detection=analyze_markers(images[name],s['markers'])
        for kind in ('identity','flip_x','flip_y','rotate180'):
            arr=images[name]
            if kind in ('flip_x','rotate180'):arr=arr[:,::-1]
            if kind in ('flip_y','rotate180'):arr=arr[::-1]
            Image.fromarray(np.ascontiguousarray(arr)).save(output/f'{name}_{kind}.png')
        points={f'marker{m["id"]}':m['expected_ros_pixel'] for m in s['markers']}
        # Expected locations in untransformed raw image depend on measured orientation.
        kind=detection.get('best_transform','identity')
        rawpoints={k:image_point(v,kind) for k,v in points.items()}
        overlay(output/f'{name}_expected_raw.svg',(output/f'{name}.png').read_bytes(),rawpoints)
        report['cameras'][name]={'position_error_m':pos,'rotation_error_deg':rot,'K_max_abs_error_px':K,
            'live_pose_in_requested_frame':{'position_m':observed[:3,3].tolist(),'rotation_ros':observed[:3,:3].tolist()},
            'metadata_passed':pos<=.0001 and rot<=.05 and K<=.05,'landmarks':detection,
            'passed':pos<=.0001 and rot<=.05 and K<=.05 and detection['passed']}
    report['all_cameras_passed']=all(v['passed'] for v in report['cameras'].values())
    report['scope']='Isaac static camera and colored-marker rendering only; no LIBERO framebuffer marker capture or policy/scene-domain validation'
    (output/'camera_alignment_result.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('[Camera alignment] '+json.dumps({k:{'metadata':v['metadata_passed'],'landmarks':v['landmarks']['passed'],'transform':v['landmarks'].get('best_transform')} for k,v in report['cameras'].items()}),flush=True)
    return report
