"""Audit saved cameras across paired poses; no simulation, inference or deployment."""
import argparse
from datetime import datetime,timezone
import hashlib
import html
import json
import math
from pathlib import Path


def camera_from_world(point,position,rotation,K):
    import numpy as np
    xyz=np.asarray(rotation).T@(np.asarray(point)-np.asarray(position))
    if xyz[2]<=1e-8:return None
    pixel=np.asarray(K)@xyz
    return (pixel[:2]/pixel[2]).tolist()


def rotate_pixel_180(uv,width,height):
    return None if uv is None else [width-1-uv[0],height-1-uv[1]]


def pose_parts(pose):
    import numpy as np
    from geometry_audit import pose_matrix
    matrix=np.array(pose_matrix(pose))
    return matrix[:3,3],matrix[:3,:3]


def angle(R):
    import numpy as np
    return math.degrees(math.acos(float(np.clip((np.trace(R)-1)/2,-1,1))))


def local_mount(parent_p,parent_R,camera_p,camera_R):
    import numpy as np
    return parent_R.T@(np.asarray(camera_p)-parent_p),parent_R.T@camera_R


def overlay(path,image,points,rotated=False):
    import base64
    encoded=base64.b64encode(image).decode()
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256">',
        f'<image width="256" height="256" href="data:image/png;base64,{encoded}"/>']
    for name,uv in points.items():
        if rotated:uv=rotate_pixel_180(uv,256,256)
        if uv is None:continue
        u,v=uv
        if 0<=u<256 and 0<=v<256:
            parts.append(f'<circle cx="{u}" cy="{v}" r="4" fill="none" stroke="#ff2b2b"/>')
            parts.append(f'<text x="{min(180,max(2,u+5))}" y="{min(250,max(12,v-5))}" font-size="10" fill="#ff2b2b">{html.escape(name)}</text>')
    parts.append('</svg>');path.write_text('\n'.join(parts),encoding='utf-8')


def audit(source,output_root):
    import numpy as np
    from PIL import Image
    source=Path(source).absolute();root=source.parent
    payload=source.read_bytes();manifest=json.loads(payload)
    if manifest.get('status')!='completed' or len(manifest.get('poses',[]))!=11:
        raise ValueError('need completed eleven-pose manifest')
    calibration_bytes=(root/'analysis.json').read_bytes();calibration=json.loads(calibration_bytes)
    if calibration.get('status')!='completed' or not calibration.get('candidate_geometry_within_thresholds'):
        raise ValueError('held-out geometry validation must pass first')
    world_R=np.array(calibration['world_from_isaac_rotation']);world_t=np.array(calibration['world_from_isaac_translation_m'])
    out=Path(output_root).absolute()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True,exist_ok=False)
    report={'status':'running','policy_queries':0,'simulation_launches':0,'camera_parameters_applied':False,
        'source_manifest':str(source),'source_manifest_sha256':hashlib.sha256(payload).hexdigest(),
        'calibration_sha256':hashlib.sha256(calibration_bytes).hexdigest(),'poses':[],
        'scope':'camera metadata and image provenance audit; projected labels are not independently measured pixel correspondences',
        'pixel_convention':'ROS +Z forward, +X right, +Y down. MuJoCo framebuffer orientation not assumed verified.',
        'reference_preprocessing':'saved LIBERO raw[::-1,::-1] then resize224; not automatically applied to Isaac',
        'external_api_reference':'https://isaac-sim.github.io/IsaacLab/v3.0.0-beta/source/api/lab/isaaclab.sensors.html'}
    def save():(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    save();mounts={'isaac':[],'libero':[]};cards=[];candidate={}
    try:
        for index,item in enumerate(manifest['poses']):
            if item.get('status')!='completed':raise ValueError('incomplete pose')
            ip=root/item['isaac_audit'];lp=root/item['reference']
            for path,key in [(ip,'isaac_audit_sha256'),(lp,'reference_sha256')]:
                if hashlib.sha256(path.read_bytes()).hexdigest()!=item[key]:raise ValueError('paired report SHA mismatch')
            i=json.loads(ip.read_text());l=json.loads(lp.read_text())
            ih,iH=pose_parts(i['sensor_body_world']);lh=np.array(l['robot_eef_body']['position_world_m']);lH=np.array(l['robot_eef_body']['rotation_world_matrix'])
            row={'name':item['name'],'cameras':{}}
            sub=out/item['name'];sub.mkdir()
            for view,refname in [('external_camera','agentview'),('wrist_camera','robot0_eye_in_hand')]:
                ic=i['cameras'][view];lc=l['cameras'][refname]
                cp,cR=pose_parts(ic['live_pose_world_ros']);rp=np.array(lc['position_world_m']);rR=np.array(lc['rotation_world_ros_matrix'])
                K=np.array(ic['live_intrinsic_matrix']);rK=np.array(lc['pinhole_intrinsic_256'])
                if ic['image_shape_hw']!=[256,256]:raise ValueError('expected square 256 images')
                if view=='wrist_camera':
                    imount=local_mount(ih,iH,cp,cR);lmount=local_mount(lh,lH,rp,rR)
                    mounts['isaac'].append(imount);mounts['libero'].append(lmount)
                else:imount=lmount=None
                native_sensor=np.array(i['sensor_tcp_world']['position_m'])
                corrected=native_sensor+iH@np.array([0,0,-.0064])
                source_points={'base':i['robot_root_world']['position_m'],'hand':ih,'tcp97':corrected}
                ref_points={'base':l['robot_root']['position_world_m'],'hand':lh,'tcp97':l['eef_site']['position_world_m']}
                ipixels={k:camera_from_world(v,cp,cR,K) for k,v in source_points.items()}
                rpixels={k:camera_from_world(v,rp,rR,rK) for k,v in ref_points.items()}
                # Test geometry agreement in the SAME reference camera, without claiming image feature detection.
                shared={k:camera_from_world(world_R@np.asarray(v)+world_t,rp,rR,rK) for k,v in source_points.items()}
                pixel_diff={k:float(np.linalg.norm(np.array(shared[k])-rpixels[k])) if shared[k] is not None and rpixels[k] is not None else None for k in shared}
                details={'camera_center_difference_after_world_registration_mm':float(np.linalg.norm(world_R@cp+world_t-rp)*1000),
                    'full_orientation_difference_deg':angle(rR.T@world_R@cR),
                    'optical_forward_difference_deg':math.degrees(math.acos(float(np.clip(np.dot(world_R@cR[:,2],rR[:,2]),-1,1)))),
                    'isaac_vertical_fov_deg':math.degrees(2*math.atan(128/K[1,1])),
                    'libero_vertical_fov_deg':lc['vertical_fov_deg'],
                    'isaac_K':K.tolist(),'libero_K':rK.tolist(),
                    'isaac_projected_pixels':ipixels,'libero_projected_pixels':rpixels,
                    'registered_geometry_in_reference_camera_pixel_difference':pixel_diff,
                    'pixel_measurement_status':'not measured: mathematical projections only; occlusion and framebuffer orientation unresolved'}
                if imount:
                    details['isaac_hand_local_mount']={'position_m':imount[0].tolist(),'rotation_ros':imount[1].tolist()}
                    details['libero_hand_local_mount']={'position_m':lmount[0].tolist(),'rotation_ros':lmount[1].tolist()}
                row['cameras'][view]=details
                # Verify and copy actual original imagery, never overwrite capture files.
                image_entries=[('isaac_raw',ip.parent/f'{view}.png',ic['raw_png_sha256']),
                    ('libero_raw',lp.parent/lc['files']['observation256']['file'],lc['files']['observation256']['sha256']),
                    ('libero_rotated',lp.parent/lc['files']['rotated256']['file'],lc['files']['rotated256']['sha256'])]
                loaded={}
                for label,path,sha in image_entries:
                    data=path.read_bytes()
                    if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('image SHA mismatch')
                    with Image.open(path) as pic:arr=np.array(pic)
                    if arr.shape!=(256,256,3) or arr.dtype!=np.uint8:raise ValueError('expected 256 RGB')
                    loaded[label]=arr;(sub/f'{view}_{label}.png').write_bytes(data)
                if not np.array_equal(loaded['libero_raw'][::-1,::-1],loaded['libero_rotated']):raise ValueError('reference 180 degree preprocessing mismatch')
                Image.fromarray(np.ascontiguousarray(loaded['isaac_raw'][::-1,::-1])).save(sub/f'{view}_isaac_rotated.png')
                overlay(sub/f'{view}_isaac_projection.svg',(sub/f'{view}_isaac_raw.png').read_bytes(),ipixels)
                overlay(sub/f'{view}_libero_projection.svg',(sub/f'{view}_libero_raw.png').read_bytes(),rpixels)
                overlay(sub/f'{view}_libero_rotated_projection.svg',(sub/f'{view}_libero_rotated.png').read_bytes(),rpixels,True)
                cards.append(f'<h3>{html.escape(item["name"])} / {view}</h3><div class="row">'+''.join(
                    f'<figure><img src="{item["name"]}/{view}_{label}.png"><figcaption>{label}</figcaption></figure>' for label in ['isaac_raw','isaac_rotated','libero_raw','libero_rotated'])+'</div>'+f'<p><a href="{item["name"]}/{view}_isaac_projection.svg">Isaac投影</a> · <a href="{item["name"]}/{view}_libero_projection.svg">LIBERO投影假设</a> · <a href="{item["name"]}/{view}_libero_rotated_projection.svg">旋转后投影假设</a></p>')
                if index==0:
                    if view=='external_camera':
                        candidate[view]={'frame':'isaac_world','position_m':(world_R.T@(rp-world_t)).tolist(),'rotation_ros':(world_R.T@rR).tolist(),'K_256':rK.tolist()}
                    else:
                        candidate[view]={'frame':'panda_hand','position_m':lmount[0].tolist(),'rotation_ros':lmount[1].tolist(),'K_256':rK.tolist()}
            report['poses'].append(row);save()
        report['wrist_mount_variation_from_home']={}
        for name,values in mounts.items():
            p0,R0=values[0]
            report['wrist_mount_variation_from_home'][name]={
                'max_translation_mm':max(float(np.linalg.norm(p-p0)*1000) for p,R in values),
                'max_rotation_deg':max(angle(R0.T@R) for p,R in values)}
        (out/'candidate_camera_parameters.json').write_text(json.dumps({'status':'proposal_not_applied','source_manifest_sha256':report['source_manifest_sha256'],
            'cameras':candidate,'note':'raw optical geometry proposal only; framebuffer/policy orientation requires rendered landmark verification; no change to live task config'},indent=2),encoding='utf-8')
        (out/'comparison.html').write_text('<!doctype html><meta charset="utf-8"><title>Camera reference audit</title><style>body{font:16px sans-serif;background:#eee;padding:20px}.row{display:flex;flex-wrap:wrap}figure{margin:8px}img{width:256px;height:256px}figcaption{text-align:center}</style><h1>相机图像与投影对照</h1><p>原图与180度旋转图仅供方向检查；没有自动选择最佳图像方向。红圈为仿真坐标计算的投影，未做像素检测，可能被遮挡。不同场景的物体并不对应。</p>'+''.join(cards),encoding='utf-8')
        report.update(status='completed',candidate_parameters='candidate_camera_parameters.json',gallery='comparison.html',
            calibration_verdict='not calibrated: no independently measured 2D correspondences, no aligned render validation')
        home=report['poses'][0]['cameras'];variation=report['wrist_mount_variation_from_home']
        lines=['# 相机静态对照报告','',
            '已完成11个姿态的元数据及图片校验；没有启动仿真、调用模型或应用相机参数。',
            '## 原始姿态的相机差异','',
            '|相机|配准后的光心距离/mm|光轴夹角/°|Isaac垂直FOV/°|LIBERO垂直FOV/°|',
            '|---|---:|---:|---:|---:|']
        for view,c in home.items():
            lines.append(f'|{view}|{c["camera_center_difference_after_world_registration_mm"]:.3f}|{c["optical_forward_difference_deg"]:.3f}|{c["isaac_vertical_fov_deg"]:.3f}|{c["libero_vertical_fov_deg"]:.3f}|')
        lines+=['','## 手腕安装一致性','',
            f'相对home姿态，Isaac手腕相机安装平移最大变化{variation["isaac"]["max_translation_mm"]:.6f}mm，安装朝向最大变化{variation["isaac"]["max_rotation_deg"]:.3f}°；LIBERO分别为{variation["libero"]["max_translation_mm"]:.6f}mm和{variation["libero"]["max_rotation_deg"]:.6f}°。',
            '现有Isaac脚本每次独立初始化后向固定世界点重新对准手腕相机，所以不同初始姿态不共享同一个安装朝向。这不是同一回合内相机抖动的证据；同初始姿态的旧策略试验可能仍使用相同安装关系，不能据此断言它导致了停滞。','',
            '## 图像方向与下一步','',
            'comparison.html并排展示两边原图及180度旋转图，SVG叠加基座、手部原点、97mm TCP的仿真投影。投影未做独立像素检测，可能被遮挡；LIBERO原始帧缓冲的朝向仍需用可见标记核验，不能把这些计算结果当作重投影精度。',
            'candidate_camera_parameters.json只提出光学几何参数：外部相机在Isaac世界中的位姿、手腕相机在手部局部的固定安装，以及参考内参。尚未应用，也没有自动选择Isaac是否旋转180度。',
            '建议后续仅在静态采集模式应用候选位置/朝向/FOV，取消手腕朝世界目标重新对准，再渲染已知可见标记核对像素位置和方向；先不要接入模型动作。',
            '不同场景和物体外观仍存在域差异，相机对齐不会使黑柜方块场景变成LIBERO任务。','',
            '[图像对照画廊](comparison.html) · [详细JSON](report.json) · [候选参数](candidate_camera_parameters.json)']
        (out/'analysis_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print('[Camera audit] Wrist mount variation: '+json.dumps(report['wrist_mount_variation_from_home']),flush=True)
        return out
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        save();print(f'[Camera audit] Report: {out/"report.json"}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);repo=Path(__file__).resolve().parent.parent
    parser.add_argument('--source-manifest',type=Path)
    parser.add_argument('--output-root',type=Path,default=repo/'diagnostics/camera_reference_audit')
    args=parser.parse_args()
    sources=sorted((repo/'diagnostics/multipose_geometry').glob('*/run.json'))
    source=args.source_manifest or next((p for p in reversed(sources) if json.loads(p.read_text()).get('status')=='completed'),None)
    if source is None:parser.error('no completed multi-pose capture')
    audit(source,args.output_root)


if __name__=='__main__':main()
