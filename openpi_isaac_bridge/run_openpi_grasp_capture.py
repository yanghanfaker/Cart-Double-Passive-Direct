"""Run the OpenPI cube grasp attempt after a successful registered replan test."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

from clean_scene_probe import validate_clean_scene
from openpi_health import check_openpi_health


def main():
    repo=Path(__file__).resolve().parent.parent;parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest',type=Path)
    parser.add_argument('--output-root',type=Path,default=repo/'diagnostics/openpi_grasp')
    parser.add_argument('--isaaclab-root',type=Path,default=Path('/home/test/Robot/IsaacLab'))
    parser.add_argument('--server-host',default='127.0.0.1');parser.add_argument('--server-port',type=int,default=8000)
    parser.add_argument('--max-queries',type=int,default=200)
    args=parser.parse_args()
    if not 1<=args.max_queries<=200:parser.error('--max-queries must be 1..200')
    sources=sorted((repo/'diagnostics/registered_replanning').glob('*/run.json'))
    source=args.source_manifest or next((p for p in reversed(sources) if json.loads(p.read_bytes()).get('final_target_reached') and json.loads(p.read_bytes()).get('all_followup_inputs_changed')),None)
    if source is None:parser.error('need a completed three-round registered replanning test')
    source=source.expanduser().absolute();data=json.loads(source.read_bytes())
    if data['status']!='completed' or not data['final_target_reached'] or not data['all_followup_inputs_changed']:parser.error('source replan did not pass')
    folder=args.output_root.expanduser().absolute()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');folder.mkdir(parents=True,exist_ok=False)
    (folder/'source_replanning_manifest.json').write_bytes(source.read_bytes())
    report={'status':'preparing','success':False,'source_manifest':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'case':'arm-only','max_queries':args.max_queries,'scope':'OpenPI approach and closure request; program-gated closure and lift/hold verification'}
    def save():(folder/'run.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    save();print(f'[OpenPI grasp runner] Output: {folder}',flush=True)
    try:
        prior_bytes=(source.parent/data['result']).read_bytes()
        if hashlib.sha256(prior_bytes).hexdigest()!=data['result_sha256']:raise ValueError('source result SHA mismatch')
        prior=json.loads(prior_bytes)
        if prior['status']!='completed' or prior['mode']!='three-round-replan' or prior['completed_action_periods']!=15 or prior['policy_queries']!=3 or not prior['final_target_reached']:
            raise ValueError('source is not a complete three-round probe')
        config=validate_clean_scene(json.loads((source.parent/'clean_scene_request.json').read_bytes()))
        if config['name']!='arm-only':raise ValueError('first grasp attempt uses the centered arm-only scene')
        calibration_bytes=(source.parent/'world_registration.json').read_bytes()
        if hashlib.sha256(calibration_bytes).hexdigest()!=data['world_registration_sha256']:raise ValueError('calibration SHA mismatch')
        calibration=json.loads(calibration_bytes)
        if not calibration.get('candidate_geometry_within_thresholds'):raise ValueError('validated calibration required')
        for key,ckey in [('reference_world_rotation','world_from_isaac_rotation'),('reference_world_translation_m','world_from_isaac_translation_m')]:
            if not np.allclose(config[key],calibration[ckey],atol=1e-12,rtol=0):raise ValueError('scene and calibration disagree')
        settings={'mode':'openpi-grasp','calibration':calibration,'prompt':prior['rounds'][0]['input']['prompt'],
            'server_host':args.server_host,'server_port':args.server_port,'max_queries':args.max_queries}
        (folder/'source_replanning_result.json').write_bytes(prior_bytes);(folder/'world_registration.json').write_bytes(calibration_bytes)
        cf=folder/'clean_scene_request.json';cf.write_text(json.dumps(config,indent=2),encoding='utf-8')
        sf=folder/'grasp_settings.json';sf.write_text(json.dumps(settings,indent=2),encoding='utf-8')
        command=[sys.executable,str(Path(__file__).with_name('run_geometry_audit.py')),'--isaaclab-root',str(args.isaaclab_root.absolute()),
            '--output-root',str(folder/'capture'),'--clean-scene-json',str(cf),'--online-registered-json',str(sf)]
        report.update(status='running',command=command,world_registration_sha256=hashlib.sha256(calibration_bytes).hexdigest());save()
        report['server_health']=check_openpi_health(args.server_host,args.server_port);save()
        env=os.environ.copy();env.pop('VIRTUAL_ENV',None);env.pop('PYTHONPATH',None)
        with (folder/'runner.log').open('w',encoding='utf-8') as log:
            with subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace') as child:
                for line in child.stdout:
                    log.write(line);log.flush()
                    if '[Geometry runner]' in line or '[OpenPI grasp]' in line:print(line,end='',flush=True)
                if child.wait()!=0:raise RuntimeError(f'grasp probe stopped; inspect {folder/"runner.log"}')
        outputs=list((folder/'capture').glob('*/online_registered_result.json'))
        if len(outputs)!=1:raise RuntimeError('expected one grasp experiment result')
        result=json.loads(outputs[0].read_bytes())
        if result['status']!='completed' or result['mode']!='openpi-grasp':raise RuntimeError('grasp experiment incomplete')
        report.update(status='completed',success=result['success'],outcome=result['outcome'],policy_queries=result['policy_queries'],
            native_controls_executed=result['native_controls_executed'],result=outputs[0].relative_to(folder).as_posix(),
            result_sha256=hashlib.sha256(outputs[0].read_bytes()).hexdigest())
        print(f'[OpenPI grasp runner] success={result["success"]}; outcome={result["outcome"]}',flush=True)
        return 0
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));print(f'[OpenPI grasp runner] STOP: {error}',flush=True);return 1
    finally:
        save();print(f'[OpenPI grasp runner] Manifest: {folder/"run.json"}',flush=True)


if __name__=='__main__':raise SystemExit(main())
