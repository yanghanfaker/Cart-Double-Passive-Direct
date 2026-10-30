"""Run one current-observation query and short motion in each proven clean scene."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import numpy as np

from clean_scene_probe import validate_clean_scene
from registered_replay_probe import validate_replay


def main():
    repo=Path(__file__).resolve().parent.parent;parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest',type=Path)
    parser.add_argument('--isaaclab-root',type=Path,default=Path('/home/test/Robot/IsaacLab'))
    parser.add_argument('--output-root',type=Path,default=repo/'diagnostics/online_registered')
    parser.add_argument('--server-host',default='127.0.0.1');parser.add_argument('--server-port',type=int,default=8000)
    parser.add_argument('--prompt')
    args=parser.parse_args()
    sources=sorted((repo/'diagnostics/registered_replay').glob('*/run.json'))
    source=args.source_manifest or next((p for p in reversed(sources) if json.loads(p.read_bytes()).get('all_final_targets_reached')),None)
    if source is None:parser.error('no completed registered replay with both final targets reached')
    source=source.expanduser().absolute();data=json.loads(source.read_bytes())
    if data['status']!='completed' or not data['all_final_targets_reached']:parser.error('need successful previous registered replay')
    folder=args.output_root.expanduser().absolute()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');folder.mkdir(parents=True,exist_ok=False)
    (folder/'source_replay_manifest.json').write_bytes(source.read_bytes())
    manifest={'status':'preparing','source_manifest':str(source),'source_manifest_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'cases':[],'scope':'two independent current-observation one-query probes; each 5 capped actions + 10 fixed-target holds; gripper open'}
    def save():(folder/'run.json').write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
    save();print(f'[Online runner] Output: {folder}',flush=True)
    try:
        frozen_payload=(source.parent/'source_frozen_report.json').read_bytes()
        if hashlib.sha256(frozen_payload).hexdigest()!=data['source_sha256']:raise ValueError('frozen provenance SHA mismatch')
        frozen=json.loads(frozen_payload)
        cp=Path(frozen['source_manifest']).parent/'world_registration.json';cal_bytes=cp.read_bytes()
        if hashlib.sha256(cal_bytes).hexdigest()!=frozen['world_registration_sha256']:raise ValueError('world registration SHA mismatch')
        calibration=json.loads(cal_bytes)
        if not calibration.get('candidate_geometry_within_thresholds'):raise ValueError('validated world registration required')
        (folder/'world_registration.json').write_bytes(cal_bytes)
        manifest['world_registration_sha256']=hashlib.sha256(cal_bytes).hexdigest()
        prompt=args.prompt or frozen['prompt']
        settings={'calibration':calibration,'prompt':prompt,'server_host':args.server_host,'server_port':args.server_port}
        manifest.update(status='running',prompt=prompt,server_host=args.server_host,server_port=args.server_port);save()
        with socket.create_connection((args.server_host,args.server_port),timeout=3):pass
        for name in ('arm-only','arm-and-target-xy'):
            prior=next(c for c in data['cases'] if c['name']==name)
            result_bytes=(source.parent/prior['result']).read_bytes()
            if hashlib.sha256(result_bytes).hexdigest()!=prior['result_sha256']:raise ValueError('previous motion result SHA mismatch')
            result=json.loads(result_bytes)
            if result['status']!='completed' or not result['final_target_reached']:raise ValueError('previous case did not converge')
            replay=validate_replay(json.loads((source.parent/name/'replay_request.json').read_bytes()))
            config=validate_clean_scene(json.loads((source.parent/name/'clean_scene_request.json').read_bytes()))
            if config['arm_positions_rad']!=replay['source_arm_positions_rad']:raise ValueError('clean arm configuration does not match previous proof')
            for config_key,replay_key,cal_key in [('reference_world_rotation','world_rotation_L_from_I','world_from_isaac_rotation'),
                    ('reference_world_translation_m','world_translation_L_from_I_m','world_from_isaac_translation_m')]:
                if not np.allclose(config[config_key],calibration[cal_key],atol=1e-12,rtol=0) or not np.allclose(replay[replay_key],calibration[cal_key],atol=1e-12,rtol=0):
                    raise ValueError('scene and replay world registration disagree with calibration')
            target=folder/name;target.mkdir()
            (target/'source_previous_result.json').write_bytes(result_bytes)
            config_file=target/'clean_scene_request.json';config_file.write_text(json.dumps(config,indent=2),encoding='utf-8')
            settings_file=target/'online_settings.json';settings_file.write_text(json.dumps(settings,indent=2),encoding='utf-8')
            command=[sys.executable,str(Path(__file__).with_name('run_geometry_audit.py')),'--isaaclab-root',str(args.isaaclab_root.absolute()),
                '--output-root',str(target/'capture'),'--clean-scene-json',str(config_file),'--online-registered-json',str(settings_file)]
            row={'name':name,'status':'running','command':command};manifest['cases'].append(row);save()
            print(f'[Online runner] START {name}; one fresh query',flush=True)
            env=os.environ.copy();env.pop('VIRTUAL_ENV',None);env.pop('PYTHONPATH',None)
            with (target/'runner.log').open('w',encoding='utf-8') as log:
                with subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                        text=True,encoding='utf-8',errors='replace') as child:
                    for line in child.stdout:
                        log.write(line);log.flush()
                        if any(tag in line for tag in ('[Geometry runner]','[Registered replay]','[Online registered]')):print(line,end='',flush=True)
                    if child.wait()!=0:raise RuntimeError(f'live probe stopped; inspect {target/"runner.log"}')
            outputs=list((target/'capture').glob('*/online_registered_result.json'))
            if len(outputs)!=1:raise RuntimeError('expected one live result')
            online=json.loads(outputs[0].read_bytes())
            if online['status']!='completed' or online['policy_queries']!=1 or online['inference_attempts']!=1 or online['completed_action_periods']!=5 or online['completed_settle_periods']!=10:
                raise RuntimeError('live query/motion count incomplete')
            row.update(status='completed',result=outputs[0].relative_to(folder).as_posix(),
                result_sha256=hashlib.sha256(outputs[0].read_bytes()).hexdigest(),final_target_reached=online['final_target_reached']);save()
        manifest.update(status='completed',all_final_targets_reached=all(c['final_target_reached'] for c in manifest['cases']))
        return 0
    except Exception as error:
        manifest.update(status='failed',error_type=type(error).__name__,error=str(error))
        for row in manifest['cases']:
            if row['status']=='running':row.update(status='failed',error=str(error))
        print(f'[Online runner] STOP: {error}',flush=True);return 1
    finally:
        save();print(f'[Online runner] Manifest: {folder/"run.json"}',flush=True)


if __name__=='__main__':raise SystemExit(main())
