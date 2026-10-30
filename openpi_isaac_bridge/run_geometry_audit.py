"""Run one static geometry audit in the existing source-built Isaac environment."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
from gain_cases import validate_gain_cases


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--isaaclab-root',type=Path,default=Path('/home/test/Robot/IsaacLab'))
    parser.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parent.parent/'diagnostics/geometry_audit')
    parser.add_argument('--geometry-joints-json',type=Path)
    parser.add_argument('--camera-alignment-json',type=Path)
    parser.add_argument('--clean-scene-json',type=Path)
    args=parser.parse_args()
    if args.clean_scene_json and (args.geometry_joints_json or args.camera_alignment_json):
        parser.error('clean scene input cannot combine pose or marker-camera inputs')
    root=args.isaaclab_root.expanduser().resolve()
    if not root.is_dir():parser.error('Isaac Lab directory not found')
    script=Path(__file__).with_name('check_cameras.py').resolve()
    cases=validate_gain_cases(json.loads(script.with_name('openpi_j13_candidate_cases.json').read_text()))
    if len(cases)!=1:parser.error('geometry audit expects exactly one candidate gain case')
    case=cases[0]
    folder=args.output_root.expanduser().resolve()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'source_case.json').write_text(json.dumps(case,indent=2),encoding='utf-8')
    gains=folder/'joint_gains.json';gains.write_text(json.dumps(case['joint_overrides'],indent=2),encoding='utf-8')
    command=['uv','run','--extra','viser','--extra','openpi','isaaclab','-p',str(script),'--viz','viser',
             '--motion-test','geometry-audit','--gain-profile',case['profile'],'--joint-gains-json',str(gains),'--output',str(folder)]
    if args.geometry_joints_json:
        from multipose_geometry import validate_arm_pose
        payload=args.geometry_joints_json.expanduser().resolve().read_bytes()
        validate_arm_pose(json.loads(payload))
        (folder/'requested_arm_pose.json').write_bytes(payload)
        command += ['--geometry-joints-json',str(folder/'requested_arm_pose.json')]
    if args.camera_alignment_json:
        payload=args.camera_alignment_json.expanduser().absolute().read_bytes()
        json.loads(payload)
        (folder/'camera_candidate.json').write_bytes(payload)
        command += ['--camera-alignment-json',str(folder/'camera_candidate.json')]
    if args.clean_scene_json:
        payload=args.clean_scene_json.expanduser().absolute().read_bytes();json.loads(payload)
        (folder/'clean_scene_request.json').write_bytes(payload)
        command += ['--clean-scene-json',str(folder/'clean_scene_request.json')]
    manifest={'status':'running','command':command,'output':str(folder),
              'scope':'static geometry capture after hold warmup; no model inference or diagnostic motion'}
    def save():(folder/'run.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    save(); print(f'[Geometry runner] Output: {folder}',flush=True)
    try:
        with (folder/'console.log').open('w',encoding='utf-8') as log:
            with subprocess.Popen(command,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace') as process:
                for line in process.stdout:
                    log.write(line)
                    if any(tag in line for tag in ('[Geometry audit]','[Actuator audit]','[Camera alignment]','[Clean scene]')):print(line,end='',flush=True)
                manifest['exit_code']=process.wait()
        report_path=folder/'geometry_audit.json'
        if manifest['exit_code']!=0 or not report_path.is_file() or json.loads(report_path.read_text())['status']!='completed':
            manifest['status']='failed';print(f'[Geometry runner] Inspect {folder/"console.log"}',file=sys.stderr);return 1
        if args.camera_alignment_json and not (folder/'camera_alignment_result.json').is_file():
            manifest['status']='failed';return 1
        if args.clean_scene_json and not (folder/'clean_scene_result.json').is_file():
            manifest['status']='failed';return 1
        manifest['status']='completed';return 0
    except Exception as error:
        manifest.update(status='failed',error=str(error));raise
    finally:
        save();print(f'[Geometry runner] Manifest: {folder/"run.json"}',flush=True)


if __name__=='__main__':raise SystemExit(main())
