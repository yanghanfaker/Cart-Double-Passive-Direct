"""Replay representative frozen horizontal-flip outputs in two clean scenes."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from registered_replay_probe import validate_replay


def representative(snapshot):
    queries=[q for q in snapshot['queries'] if q['condition']=='flip_x']
    if len(queries)!=10:raise ValueError('expected ten horizontal-flip source queries')
    mean=statistics.mean(q['first5']['bounded_local_approach_projection_sum_mm'] for q in queries)
    return min(queries,key=lambda q:(abs(q['first5']['bounded_local_approach_projection_sum_mm']-mean),q['repeat']))


def main():
    parser=argparse.ArgumentParser(description=__doc__);repo=Path(__file__).resolve().parent.parent
    parser.add_argument('--source-report',type=Path)
    parser.add_argument('--isaaclab-root',type=Path,default=Path('/home/test/Robot/IsaacLab'))
    parser.add_argument('--output-root',type=Path,default=repo/'diagnostics/registered_replay')
    args=parser.parse_args()
    sources=sorted((repo/'diagnostics/openpi_image_ab').glob('*/report.json'))
    source=args.source_report or next((p for p in reversed(sources) if json.loads(p.read_text()).get('status')=='completed'),None)
    if source is None:parser.error('no complete frozen image A/B report')
    source=source.absolute();payload=source.read_bytes();data=json.loads(payload)
    if data['status']!='completed' or data['actions_executed']!=0:parser.error('invalid frozen source report')
    original=Path(data['source_manifest']);original_payload=original.read_bytes()
    if hashlib.sha256(original_payload).hexdigest()!=data['source_manifest_sha256']:parser.error('clean source manifest SHA mismatch')
    folder=args.output_root.absolute()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');folder.mkdir(parents=True,exist_ok=False)
    (folder/'source_frozen_report.json').write_bytes(payload)
    manifest={'status':'running','source_report':str(source),'source_sha256':hashlib.sha256(payload).hexdigest(),
        'policy_queries':0,'selection':'flip_x query nearest its case mean first5 projection, tie by repeat; not best-query selection',
        'cases':[],'scope':'two independent 5-action replay + 10 fixed-target settling periods; 0.5mm/0.1deg caps, gripper open; no new inference'}
    def save():(folder/'run.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    save();print(f'[Registered replay runner] Output: {folder}',flush=True)
    try:
        for name in ('arm-only','arm-and-target-xy'):
            snapshot=next(s for s in data['snapshots'] if s['name']==name);query=representative(snapshot)
            gb=(source.parent/(name+'_source_geometry.json')).read_bytes()
            if hashlib.sha256(gb).hexdigest()!=snapshot['source_geometry_sha256']:raise ValueError('source geometry SHA mismatch')
            geometry=json.loads(gb);q=dict(zip(geometry['joint_names'],geometry['joint_positions_rad']))
            request=json.loads((original.parent/name/'request.json').read_text())
            request['arm_positions_rad']=[q[f'panda_joint{k}'] for k in range(1,8)]
            provenance=snapshot['state_provenance']
            replay=validate_replay({'condition':'flip_x','raw_action_chunk':query['raw_action_chunk'],
                'source_repeat':query['repeat'],'source_input_sha256':query['input_sha256'],
                'source_state_float32':query['state_sent_float32'],
                'source_mapped_hand_quat_xyzw':provenance['mapped_hand_quat_xyzw'],
                'registration_quat_xyzw':provenance['registration_quat_xyzw'],
                'world_rotation_L_from_I':provenance['registration_rotation'],
                'world_translation_L_from_I_m':provenance['registration_translation_m'],
                'source_arm_positions_rad':request['arm_positions_rad']})
            target=folder/name;target.mkdir()
            config=target/'clean_scene_request.json';config.write_text(json.dumps(request,indent=2),encoding='utf-8')
            replay_file=target/'replay_request.json';replay_file.write_text(json.dumps(replay,indent=2),encoding='utf-8')
            (target/'selected_source_query.json').write_text(json.dumps(query,indent=2),encoding='utf-8')
            command=[sys.executable,str(Path(__file__).with_name('run_geometry_audit.py')),
                '--isaaclab-root',str(args.isaaclab_root.absolute()),'--output-root',str(target/'capture'),
                '--clean-scene-json',str(config),'--registered-replay-json',str(replay_file)]
            row={'name':name,'status':'running','source_repeat':query['repeat'],'source_input_sha256':query['input_sha256'],'command':command}
            manifest['cases'].append(row);save();print(f'[Registered replay runner] START {name}, recorded repeat={query["repeat"]}',flush=True)
            env=os.environ.copy();env.pop('VIRTUAL_ENV',None);env.pop('PYTHONPATH',None)
            with (target/'runner.log').open('w',encoding='utf-8') as log:
                with subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace') as child:
                    for line in child.stdout:
                        log.write(line);log.flush()
                        if '[Registered replay]' in line or '[Geometry runner]' in line:print(line,end='',flush=True)
                    if child.wait()!=0:raise RuntimeError(f'replay stopped; inspect {target/"runner.log"}')
            results=list((target/'capture').glob('*/registered_replay_result.json'))
            if len(results)!=1:raise RuntimeError('expected one replay result')
            result=json.loads(results[0].read_text())
            if result['status']!='completed':raise RuntimeError('replay incomplete')
            row.update(status='completed',result=results[0].relative_to(folder).as_posix(),
                result_sha256=hashlib.sha256(results[0].read_bytes()).hexdigest(),
                actual_initial_direction_approach_mm=result['actual_initial_direction_approach_mm'],
                commanded_initial_direction_approach_mm=result['command_initial_direction_approach_mm'],
                final_target_reached=result['final_target_reached']);save()
        manifest.update(status='completed',all_final_targets_reached=all(c['final_target_reached'] for c in manifest['cases']))
        return 0
    except Exception as error:
        manifest.update(status='failed',error_type=type(error).__name__,error=str(error))
        for row in manifest['cases']:
            if row['status']=='running':row.update(status='failed',error=str(error))
        print(f'[Registered replay runner] STOP: {error}',flush=True);return 1
    finally:
        save();print(f'[Registered replay runner] Manifest: {folder/"run.json"}',flush=True)


if __name__=='__main__':raise SystemExit(main())
