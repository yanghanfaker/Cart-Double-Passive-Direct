"""Capture three unmarked proxy scenes; no model requests or automatic control."""
import argparse
from datetime import datetime,timezone
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys


def prepare_cases(native,isaac,calibration,candidate):
    import numpy as np
    from libero_reference_audit import ordered_joints
    from clean_scene_probe import validate_clean_scene
    if native.get('status')!='completed' or not native.get('benchmark_initialization_available') or not native['marker_test']['all_marker_views_passed']:
        raise ValueError('completed native initialization and marker test required')
    layout=next(s for s in native['layouts'] if s['label']=='benchmark_after10wait')
    interest=[n for n in layout['objects_of_interest'] if 'bowl' in n]
    if len(interest)!=1:raise ValueError('expected one named target bowl')
    target=layout['objects'][interest[0]]['position_world_m']
    R=np.array(calibration['world_from_isaac_rotation']);t=np.array(calibration['world_from_isaac_translation_m'])
    proxy=R.T@(np.array(target)-t)
    source_q=dict(zip(isaac['joint_names'],isaac['joint_positions_rad']))
    home=[source_q[f'panda_joint{k}'] for k in range(1,8)]
    names=ordered_joints(list(layout['joint_positions']),'arm');arm=[layout['joint_positions'][name] for name in names]
    cube=isaac['object_root_world']['position_m']
    cases=[]
    for name,angles,xy in [('baseline',home,cube[:2]),('arm-only',arm,cube[:2]),('arm-and-target-xy',arm,proxy[:2].tolist())]:
        config={'name':name,'arm_positions_rad':angles,'object_xy_world_m':list(xy),'camera_candidate':candidate,
            'cube_z_policy':'preserve_existing_spawn_height_and_check_rest_height','baseline_cube_rest_position_m':cube,
            'reference_world_rotation':R.tolist(),'reference_world_translation_m':t.tolist(),
            'reference_eef_position_world_m':layout['eef_observation']['position_world_m'],
            'reference_target_name':interest[0],'reference_target_body_origin_world_m':target,
            'object_role':'existing cube proxy; match horizontal location only; keep existing cube spawn height and support'}
        cases.append(validate_clean_scene(config))
    return cases


def main():
    parser=argparse.ArgumentParser(description=__doc__);repo=Path(__file__).resolve().parent.parent
    parser.add_argument('--native-report',type=Path)
    parser.add_argument('--isaaclab-root',type=Path,default=Path('/home/test/Robot/IsaacLab'))
    parser.add_argument('--output-root',type=Path,default=repo/'diagnostics/clean_scene_capture')
    args=parser.parse_args()
    sources=sorted((repo/'diagnostics/libero_scene_probe').glob('*/report.json'))
    source=args.native_report or next((p for p in reversed(sources) if json.loads(p.read_text()).get('status')=='completed'),None)
    if source is None:parser.error('no completed native LIBERO scene probe')
    source=source.absolute();parent=json.loads((source.parent/'run.json').read_text())
    native=json.loads(source.read_text());isaac_bytes=(source.parent/'isaac_source.json').read_bytes();cal_bytes=(source.parent/'world_registration.json').read_bytes()
    if hashlib.sha256(isaac_bytes).hexdigest()!=parent['isaac_source_sha256'] or hashlib.sha256(cal_bytes).hexdigest()!=parent['world_registration_sha256']:
        parser.error('native source SHA mismatch')
    alignment=Path(parent['source_alignment']);al=json.loads(alignment.read_text());proposal=(alignment.parent/'candidate_camera_parameters.json').read_bytes()
    if hashlib.sha256(proposal).hexdigest()!=al['candidate_sha256']:parser.error('camera proposal SHA mismatch')
    cases=prepare_cases(native,json.loads(isaac_bytes),json.loads(cal_bytes),json.loads(proposal))
    folder=args.output_root.absolute()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');folder.mkdir(parents=True,exist_ok=False)
    for name,content in [('native_report.json',source.read_bytes()),('isaac_source.json',isaac_bytes),('world_registration.json',cal_bytes),('camera_candidate.json',proposal)]:
        (folder/name).write_bytes(content)
    manifest={'status':'running','policy_queries':0,'source_native_report':str(source),
        'source_native_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'cases':[],
        'scope':'three independent no-marker Isaac scenes; arm pose and cube XY proxy only; no full LIBERO scene reproduction'}
    def save():(folder/'run.json').write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
    save();print(f'[Clean scene runner] Output: {folder}',flush=True)
    try:
        for config in cases:
            target=folder/config['name'];target.mkdir();request=target/'request.json';request.write_text(json.dumps(config,indent=2),encoding='utf-8')
            command=[sys.executable,str(Path(__file__).with_name('run_geometry_audit.py')),
                '--isaaclab-root',str(args.isaaclab_root.absolute()),'--output-root',str(target/'capture'),'--clean-scene-json',str(request)]
            row={'name':config['name'],'status':'running','command':command};manifest['cases'].append(row);save()
            env=os.environ.copy();env.pop('VIRTUAL_ENV',None);env.pop('PYTHONPATH',None)
            print(f'[Clean scene runner] START {config["name"]}',flush=True)
            with (target/'runner.log').open('w',encoding='utf-8') as log:
                with subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace') as process:
                    for line in process.stdout:
                        log.write(line);log.flush()
                        if '[Clean scene]' in line or '[Geometry runner]' in line:print(line,end='',flush=True)
                    if process.wait()!=0:raise RuntimeError(f'capture failed; inspect {target/"runner.log"}')
            files=list((target/'capture').glob('*/clean_scene_result.json'))
            if len(files)!=1:raise RuntimeError('expected one clean scene result')
            result=json.loads(files[0].read_text())
            if result['status']!='completed':raise RuntimeError('incomplete capture result')
            row.update(status='completed',result=files[0].relative_to(folder).as_posix(),
                result_sha256=hashlib.sha256(files[0].read_bytes()).hexdigest(),geometry_checks_passed=result['geometry_checks_passed'],
                camera_projections={k:v['object_center_projection_px'] for k,v in result['cameras'].items()});save()
        reference=next(s for s in native['layouts'] if s['label']=='benchmark_after10wait')
        cards=[]
        for view,ref in [('external_camera','agentview'),('wrist_camera','robot0_eye_in_hand')]:
            entry=reference['images'][ref]['policy224'];origin=source.parent/reference['image_folder']/entry['file']
            content=origin.read_bytes()
            if hashlib.sha256(content).hexdigest()!=entry['sha256']:raise ValueError('clean reference RGB SHA mismatch')
            (folder/(ref+'_reference_policy224.png')).write_bytes(content)
            cards.append(f'<h2>{view}</h2><img src="{ref}_reference_policy224.png"><p>LIBERO官方预处理参考图；物体/场景不同，仅作布局和方向参考。</p>')
            for row in manifest['cases']:
                result_path=folder/row['result'];data=json.loads(result_path.read_text());images=data['cameras'][view]['frames']
                cards.append(f'<h3>{html.escape(row["name"])}</h3>'+''.join(f'<figure><img src="{(result_path.parent/f).relative_to(folder).as_posix()}"><figcaption>{label}</figcaption></figure>' for label,f in images.items()))
        (folder/'comparison.html').write_text('<!doctype html><meta charset="utf-8"><title>Clean proxy scene comparison</title><style>body{font:16px sans-serif;padding:20px;background:#eee}figure{display:inline-block;margin:8px}img{width:224px;height:224px}</style><h1>无标记代理场景对照</h1><p>不调用模型；左右翻转仅为对照图，不自动部署。</p>'+''.join(cards),encoding='utf-8')
        manifest['status']='completed';manifest['all_geometry_checks_passed']=all(r['geometry_checks_passed'] for r in manifest['cases'])
        return 0
    except Exception as error:
        manifest.update(status='failed',error_type=type(error).__name__,error=str(error))
        for row in manifest['cases']:
            if row['status']=='running':row.update(status='failed',error=str(error))
        print(f'[Clean scene runner] STOP: {error}',flush=True);return 1
    finally:
        save();print(f'[Clean scene runner] Manifest: {folder/"run.json"}',flush=True)


if __name__=='__main__':raise SystemExit(main())
