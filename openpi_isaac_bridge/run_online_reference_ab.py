"""Repeated live-observation A/B of target reference conventions."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import numpy as np
from clean_scene_probe import validate_clean_scene
from online_reference_probe import validate_initial
from openpi_health import check_openpi_health


def digest(b):return hashlib.sha256(b).hexdigest()


def prepare_source(root,data):
    if data['status']!='completed' or data['policy_queries']!=0:raise ValueError('need completed fixed-stream A/B source')
    results={};configs={};gains={};payloads={};captures={}
    for mode in ('current','accumulated'):
        matches=[c for c in data['cases'] if c['mode']==mode]
        if len(matches)!=1:raise ValueError('missing or duplicated source mode')
        entry=matches[0];p=root/entry['result'];b=p.read_bytes();r=json.loads(b)
        if digest(b)!=entry['result_sha256']:raise ValueError('source result SHA mismatch')
        if r['status']!='completed' or r['reference_mode']!=mode or r['action_controls']!=data['steps'] or r['hold_controls']!=10 or not r['final_applied_target_reached'] or r['fixed_stream_sha256']!=data['fixed_stream_sha256']:
            raise ValueError('source reference experiment incomplete')
        results[mode]=r;payloads[mode]=b;captures[mode]=p.parent
        configs[mode]=validate_clean_scene(json.loads((root/mode/'clean_scene_request.json').read_bytes()))
        gains[mode]=json.loads((p.parent/'joint_gains.json').read_bytes())
    validate_initial(results['current']['initial'],results['accumulated']['initial'])
    if configs['current']!=configs['accumulated'] or gains['current']!=gains['accumulated']:raise ValueError('source configs or gains differ')
    cb=(root/'world_registration.json').read_bytes()
    if digest(cb)!=data['world_registration_sha256']:raise ValueError('source calibration SHA mismatch')
    cal=json.loads(cb)
    if not cal.get('candidate_geometry_within_thresholds'):raise ValueError('validated calibration required')
    old_settings=json.loads((root/'current/reference_settings.json').read_bytes())
    if old_settings['calibration']!=cal or old_settings['expected_joint_gains']!=gains['current']:raise ValueError('source calibration/gains settings disagree')
    for key,ckey in [('reference_world_rotation','world_from_isaac_rotation'),('reference_world_translation_m','world_from_isaac_translation_m')]:
        if not np.allclose(configs['current'][key],cal[ckey],atol=1e-12,rtol=0):raise ValueError('scene and calibration disagree')
    grasp_manifest_bytes=(root/'source_grasp_manifest.json').read_bytes()
    if digest(grasp_manifest_bytes)!=data['source_sha256']:raise ValueError('source grasp manifest SHA mismatch')
    grasp_bytes=(root/'source_grasp_result.json').read_bytes()
    if digest(grasp_bytes)!=json.loads(grasp_manifest_bytes)['result_sha256']:raise ValueError('source grasp result SHA mismatch')
    prompt=json.loads(grasp_bytes)['queries'][0]['input']['prompt']
    if not isinstance(prompt,str) or not prompt.strip():raise ValueError('invalid source prompt')
    return configs['current'],gains['current'],results['current']['initial'],cb,payloads,prompt


def summarize(cases):
    summary={}
    for mode in ('current','accumulated'):
        group=[c for c in cases if c['reference_mode']==mode and c['status']=='completed'];summary[mode]={'completed_runs':len(group)}
        for key in ('distance_reduction_m','final_xy_offset_m','final_ideal_position_error_m','max_applied_backlog_m','max_pre_update_ideal_gap_m','max_applied_target_error_m'):
            values=[c[key] for c in group]
            if values:summary[mode][key]={'mean':statistics.mean(values),'sample_std':statistics.stdev(values) if len(values)>1 else None,'min':min(values),'max':max(values)}
    return summary


def main():
    repo=Path(__file__).resolve().parent.parent;parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest',type=Path)
    parser.add_argument('--output-root',type=Path,default=repo/'diagnostics/online_reference_ab')
    parser.add_argument('--isaaclab-root',type=Path,default=Path('/home/test/Robot/IsaacLab'))
    parser.add_argument('--server-host',default='127.0.0.1');parser.add_argument('--server-port',type=int,default=8000)
    parser.add_argument('--queries',type=int,choices=(10,20),default=20);parser.add_argument('--repeats',type=int,choices=(1,3),default=3)
    args=parser.parse_args();sources=sorted((repo/'diagnostics/target_reference_ab').glob('*/run.json'))
    source=args.source_manifest or next((p for p in reversed(sources) if json.loads(p.read_bytes()).get('status')=='completed'),None)
    if source is None:parser.error('need completed fixed-stream target reference comparison')
    source=source.expanduser().absolute();data=json.loads(source.read_bytes())
    folder=args.output_root.expanduser().absolute()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');folder.mkdir(parents=True,exist_ok=False)
    (folder/'source_reference_manifest.json').write_bytes(source.read_bytes())
    report={'status':'preparing','source_manifest':str(source),'source_sha256':digest(source.read_bytes()),'queries_per_run':args.queries,'repeats':args.repeats,'cases':[],
        'scope':'independent live observations and model outputs for each reference condition; open gripper',
        'sampling':'unpaired stochastic inference, no shared model-noise seed; condition order alternates per repeat'}
    def save():
        report['summary']=summarize(report['cases']);(folder/'run.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    save();print(f'[Online AB runner] Output: {folder}',flush=True)
    try:
        config,gains,initial,cb,payloads,prompt=prepare_source(source.parent,data)
        (folder/'world_registration.json').write_bytes(cb)
        for mode,b in payloads.items():(folder/f'source_{mode}_result.json').write_bytes(b)
        report['world_registration_sha256']=digest(cb);report['server_health']=check_openpi_health(args.server_host,args.server_port)
        report['status']='running';save()
        for repeat in range(1,args.repeats+1):
            order=('current','accumulated') if repeat%2 else ('accumulated','current')
            for mode in order:
                name=f'repeat-{repeat:02d}-{mode}';target=folder/name;target.mkdir()
                cf=target/'clean_scene_request.json';cf.write_text(json.dumps(config,indent=2),encoding='utf-8')
                settings={'mode':'online-reference-ab','reference_mode':mode,'queries':args.queries,'calibration':json.loads(cb),
                    'expected_joint_gains':gains,'expected_initial':initial,'prompt':prompt,
                    'server_host':args.server_host,'server_port':args.server_port}
                sf=target/'online_settings.json';sf.write_text(json.dumps(settings,indent=2),encoding='utf-8')
                command=[sys.executable,str(Path(__file__).with_name('run_geometry_audit.py')),'--isaaclab-root',str(args.isaaclab_root.absolute()),
                    '--output-root',str(target/'capture'),'--clean-scene-json',str(cf),'--online-registered-json',str(sf)]
                row={'name':name,'repeat':repeat,'reference_mode':mode,'status':'running','command':command};report['cases'].append(row);save()
                print(f'[Online AB runner] START {name}',flush=True)
                env=os.environ.copy();env.pop('VIRTUAL_ENV',None);env.pop('PYTHONPATH',None)
                with (target/'runner.log').open('w',encoding='utf-8') as log:
                    with subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace') as child:
                        for line in child.stdout:
                            log.write(line);log.flush()
                            if '[Online reference]' in line or '[Geometry runner]' in line:print(line,end='',flush=True)
                        if child.wait()!=0:raise RuntimeError(f'online comparison stopped; inspect {target/"runner.log"}')
                outputs=list((target/'capture').glob('*/online_registered_result.json'))
                if len(outputs)!=1:raise RuntimeError('expected one live comparison report')
                r=json.loads(outputs[0].read_bytes())
                if r['status']!='completed' or r['reference_mode']!=mode or r['policy_queries']!=args.queries or r['inference_attempts']!=args.queries or r['action_controls']!=5*args.queries or r['hold_controls']!=10 or r['native_controls_executed']!=5*args.queries+10:
                    raise RuntimeError('incomplete online comparison')
                row.update(status='completed',result=outputs[0].relative_to(folder).as_posix(),result_sha256=digest(outputs[0].read_bytes()),first_input_sha256=r['first_input_sha256'],
                    all_followup_inputs_changed=r['all_followup_inputs_changed'],requested_close_actions=r['requested_close_actions'])
                for key in ('distance_reduction_m','final_xy_offset_m','final_ideal_position_error_m','max_applied_backlog_m','max_pre_update_ideal_gap_m','max_applied_target_error_m'):row[key]=r[key]
                save()
        report.update(status='completed',initial_inputs_identical=len({c['first_input_sha256'] for c in report['cases']})==1,
            all_followup_inputs_changed=all(c['all_followup_inputs_changed'] for c in report['cases']))
        return 0
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        for row in report['cases']:
            if row['status']=='running':row['status']='failed'
        print(f'[Online AB runner] STOP: {error}',flush=True);return 1
    finally:
        save();print(f'[Online AB runner] Manifest: {folder/"run.json"}',flush=True)


if __name__=='__main__':raise SystemExit(main())
