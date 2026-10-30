"""Three-pose static candidate camera render test with colored landmarks."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


def summarize(results):
    output={}
    for name in ('external_camera','wrist_camera'):
        rows=[r['cameras'][name] for r in results]
        first=rows[0]['live_pose_in_requested_frame'];poses=[r['live_pose_in_requested_frame'] for r in rows]
        pos=max(math.dist(first['position_m'],p['position_m']) for p in poses)
        angles=[]
        for p in poses:
            trace=sum(a*b for aa,bb in zip(first['rotation_ros'],p['rotation_ros']) for a,b in zip(aa,bb))
            angles.append(math.degrees(math.acos(max(-1.,min(1.,(trace-1)/2)))))
        orientation=[r['landmarks'].get('best_transform') for r in rows]
        output[name]={'all_pose_results_passed':all(r['passed'] for r in rows),
            'raw_to_ros_image_transforms':orientation,'image_orientation_consistent':len(set(orientation))==1 and None not in orientation,
            'pose_frame_translation_variation_m':pos,'pose_frame_rotation_variation_deg':max(angles),
            'scope':'world frame for external camera; hand-local frame for wrist camera'}
        output[name]['passed']=output[name]['all_pose_results_passed'] and output[name]['image_orientation_consistent'] and pos<=.0001 and max(angles)<=.05
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__);repo=Path(__file__).resolve().parent.parent
    parser.add_argument('--camera-audit',type=Path)
    parser.add_argument('--isaaclab-root',type=Path,default=Path('/home/test/Robot/IsaacLab'))
    parser.add_argument('--output-root',type=Path,default=repo/'diagnostics/camera_alignment')
    args=parser.parse_args()
    sources=sorted((repo/'diagnostics/camera_reference_audit').glob('*/report.json'))
    audit=args.camera_audit or next((p for p in reversed(sources) if json.loads(p.read_text()).get('status')=='completed'),None)
    if audit is None:parser.error('no completed camera audit')
    audit=audit.absolute();camera_audit=json.loads(audit.read_text())
    if camera_audit.get('status')!='completed':parser.error('source camera audit incomplete')
    source=Path(camera_audit['source_manifest']);payload=source.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=camera_audit['source_manifest_sha256']:parser.error('source manifest SHA mismatch')
    original=json.loads(payload)
    if original['status']!='completed':parser.error('source multi-pose run incomplete')
    proposal=(audit.parent/'candidate_camera_parameters.json').read_bytes();proposed=json.loads(proposal)
    if proposed['source_manifest_sha256']!=camera_audit['source_manifest_sha256']:parser.error('candidate source SHA mismatch')
    folder=args.output_root.absolute()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');folder.mkdir(parents=True,exist_ok=False)
    candidate=folder/'candidate_camera_parameters.json';candidate.write_bytes(proposal)
    report={'status':'running','policy_queries':0,'source_camera_audit':str(audit),
        'source_camera_audit_sha256':hashlib.sha256(audit.read_bytes()).hexdigest(),
        'candidate_sha256':hashlib.sha256(proposal).hexdigest(),'poses':[],
        'scope':'three independent static scenes with nonphysical colored calibration spheres; no policy inputs or actions'}
    def save():(folder/'run.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    save();print(f'[Camera alignment runner] Output: {folder}',flush=True)
    try:
        results=[]
        for name in ('home','mixed34','mixed56'):
            p=next(p for p in original['poses'] if p['name']==name)
            origin=(source.parent/p['isaac_audit']).read_bytes()
            if hashlib.sha256(origin).hexdigest()!=p['isaac_audit_sha256']:raise ValueError('original Isaac pose SHA mismatch')
            i=json.loads(origin);joints=dict(zip(i['joint_names'],i['joint_positions_rad']))
            target=folder/name;target.mkdir()
            qfile=target/'initial_joint_positions.json';qfile.write_text(json.dumps([joints[f'panda_joint{k}'] for k in range(1,8)]),encoding='utf-8')
            command=[sys.executable,str(Path(__file__).with_name('run_geometry_audit.py')),
                '--isaaclab-root',str(args.isaaclab_root.absolute()),'--output-root',str(target/'capture'),
                '--geometry-joints-json',str(qfile),'--camera-alignment-json',str(candidate)]
            row={'name':name,'status':'running','command':command};report['poses'].append(row);save()
            print(f'[Camera alignment runner] START {name}',flush=True)
            with (target/'runner.log').open('w',encoding='utf-8') as log:
                with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',stdin=subprocess.DEVNULL) as child:
                    for line in child.stdout:
                        log.write(line);log.flush()
                        if '[Geometry runner]' in line or '[Camera alignment]' in line:print(line,end='',flush=True)
                    if child.wait()!=0:raise RuntimeError(f'capture failed; inspect {target/"runner.log"}')
            paths=list((target/'capture').glob('*/camera_alignment_result.json'))
            if len(paths)!=1:raise RuntimeError('expected one new alignment result')
            result=json.loads(paths[0].read_text())
            if result['status']!='completed':raise RuntimeError('incomplete marker analysis')
            row.update(status='completed',result=paths[0].relative_to(folder).as_posix(),
                result_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),all_cameras_passed=result['all_cameras_passed'])
            results.append(result);save()
        report['camera_summary']=summarize(results)
        report['all_camera_tests_passed']=all(c['passed'] for c in report['camera_summary'].values())
        report['status']='completed'
        report['limitations']='Only Isaac raw-to-ROS framebuffer direction tested. LIBERO raw-to-policy preprocessing remains separate; no policy image flip is deployed.'
        print('[Camera alignment runner] Summary: '+json.dumps(report['camera_summary']),flush=True)
        return 0
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        for row in report['poses']:
            if row['status']=='running':row.update(status='failed',error=str(error))
        print(f'[Camera alignment runner] STOP: {error}',flush=True);return 1
    finally:
        save();print(f'[Camera alignment runner] Manifest: {folder/"run.json"}',flush=True)


if __name__=='__main__':raise SystemExit(main())
