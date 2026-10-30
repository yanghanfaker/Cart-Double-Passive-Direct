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
from online_reference_probe import validate_initial,comparison_limits
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


def prepare_online_source(root,data,expected_profile='short'):
    budget=20 if expected_profile=='short' else 100
    if expected_profile not in ('short','descent'):raise ValueError('unsupported source profile')
    if data['status']!='completed' or data.get('profile','short')!=expected_profile or not data['initial_inputs_identical'] or not data['all_followup_inputs_changed']:
        raise ValueError('source must be a completed expected online profile with matching/updating inputs')
    if len(data['cases'])!=6 or data['queries_per_run']!=budget:raise ValueError('source requires six completed runs at expected budget')
    if sorted(c['reference_mode'] for c in data['cases'])!=['accumulated']*3+['current']*3:raise ValueError('descent source requires three runs per mode')
    configs=[];gains=[];initials=[];payloads={};settings_list=[];first_inputs=set()
    for case in data['cases']:
        p=root/case['result'];b=p.read_bytes();r=json.loads(b)
        if digest(b)!=case['result_sha256']:raise ValueError('source live result SHA mismatch')
        validate_run_result(r,case['reference_mode'],budget,expected_profile)
        first_inputs.add(r['first_input_sha256'])
        if expected_profile=='descent' and (not r['boundary_reached'] or r['stop_reason']!='near_boundary_reached' or not .08<=r['action_end_object_distance_m']<=.09):raise ValueError('near descent requires every source run to reach 90mm boundary')
        if not r['all_followup_inputs_changed']:raise ValueError('source inputs did not update')
        configs.append(validate_clean_scene(json.loads((root/case['name']/'clean_scene_request.json').read_bytes())))
        gains.append(json.loads((p.parent/'joint_gains.json').read_bytes()));initials.append(r['initial'])
        settings_list.append(json.loads((root/case['name']/'online_settings.json').read_bytes()))
        payloads[case['name']]=b
    if len(first_inputs)!=1:raise ValueError('source first inputs differ')
    if not all(c==configs[0] for c in configs) or not all(g==gains[0] for g in gains):raise ValueError('source live configs/gains differ')
    for initial in initials:validate_initial(initial,initials[0])
    cb=(root/'world_registration.json').read_bytes()
    if digest(cb)!=data['world_registration_sha256']:raise ValueError('source live calibration SHA mismatch')
    cal=json.loads(cb)
    if not cal.get('candidate_geometry_within_thresholds'):raise ValueError('validated calibration required')
    prompt=settings_list[0]['prompt']
    for settings in settings_list:
        if settings['prompt']!=prompt or settings['calibration']!=cal or settings['expected_joint_gains']!=gains[0]:raise ValueError('source live settings differ')
    if not isinstance(prompt,str) or not prompt.strip():raise ValueError('invalid source prompt')
    for key,ckey in [('reference_world_rotation','world_from_isaac_rotation'),('reference_world_translation_m','world_from_isaac_translation_m')]:
        if not np.allclose(configs[0][key],cal[ckey],atol=1e-12,rtol=0):raise ValueError('source scene/calibration mismatch')
    return configs[0],gains[0],initials[0],cb,payloads,prompt


def validate_run_result(r,mode,budget,profile):
    if r['status']!='completed' or r['reference_mode']!=mode or r.get('profile','short')!=profile or r['hold_controls']!=10 or r['policy_queries']!=r['inference_attempts'] or r['native_controls_executed']!=r['action_controls']+10:
        raise RuntimeError('incomplete online comparison')
    if profile=='short':
        if r['policy_queries']!=budget or r['action_controls']!=5*budget:raise RuntimeError('incomplete short comparison')
    else:
        q=r['policy_queries'];a=r['action_controls'];reason=r['stop_reason']
        if r['queries_requested']!=budget or not 1<=q<=budget or not 5*(q-1)<a<=5*q:raise RuntimeError('invalid descent query/control counts')
        if reason not in ('near_boundary_reached','descent_height_floor_reached','distance_increased_from_best_over_10mm','progress_below_2mm_per100_controls','query_budget_exhausted'):raise RuntimeError('unknown descent stop reason')
        if r['boundary_reached']!=(reason=='near_boundary_reached'):raise RuntimeError('boundary flag and reason disagree')
        if reason=='query_budget_exhausted' and (q!=budget or a!=5*budget):raise RuntimeError('budget exhausted before full budget executed')


def numeric_summary(values):
    return {'mean':statistics.mean(values),'sample_std':statistics.stdev(values) if len(values)>1 else None,'min':min(values),'max':max(values)}


def summarize(cases):
    summary={}
    for mode in ('current','accumulated'):
        group=[c for c in cases if c['reference_mode']==mode and c['status']=='completed'];summary[mode]={'completed_runs':len(group)}
        for key in ('distance_reduction_m','final_xy_offset_m','final_ideal_position_error_m','max_applied_backlog_m','max_pre_update_ideal_gap_m','max_applied_target_error_m'):
            values=[c[key] for c in group]
            if values:summary[mode][key]=numeric_summary(values)
        reached=[c for c in group if c.get('boundary_reached')]
        summary[mode]['boundary_reached_runs']=len(reached)
        if reached:summary[mode]['action_controls_to_boundary']=numeric_summary([c['action_controls'] for c in reached])
        summary[mode]['stop_reasons']={reason:sum(c.get('stop_reason')==reason for c in group) for reason in sorted({c.get('stop_reason','query_budget_exhausted') for c in group})}
        for mm in (200,150,100,80,70,60):
            key=f'height_{mm}mm';points=[c.get('height_checkpoints',{}).get(key) for c in group];points=[p for p in points if p is not None]
            summary[mode][key]={'reached_runs':len(points)}
            for metric in ('xy_offset_m','object_distance_m','action_control'):
                if points:summary[mode][key][metric]=numeric_summary([p[metric] for p in points])
        close_cases=[c['first_close_request'] for c in group if c.get('first_close_request') is not None]
        summary[mode]['runs_with_close_request']=len(close_cases)
        summary[mode]['requested_close_actions']=sum(c.get('requested_close_actions',0) for c in group)
        summary[mode]['first_close_request']={'observed_runs':len(close_cases)}
        for metric in ('height_m','xy_offset_m','object_distance_m','action_control'):
            if close_cases:summary[mode]['first_close_request'][metric]=numeric_summary([c[metric] for c in close_cases])
    return summary


def main():
    repo=Path(__file__).resolve().parent.parent;parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest',type=Path)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--isaaclab-root',type=Path,default=Path('/home/test/Robot/IsaacLab'))
    parser.add_argument('--server-host',default='127.0.0.1');parser.add_argument('--server-port',type=int,default=8000)
    parser.add_argument('--profile',choices=('short','descent','near-descent'),default='short')
    parser.add_argument('--queries',type=int);parser.add_argument('--repeats',type=int,choices=(1,3),default=3)
    args=parser.parse_args()
    if args.queries is None:args.queries={'short':20,'descent':100,'near-descent':120}[args.profile]
    if args.queries not in comparison_limits(args.profile)['queries']:parser.error('queries must be 120/140 for near-descent, 80/100 for descent, or 10/20 for short')
    source_category={'short':'target_reference_ab','descent':'online_reference_ab','near-descent':'descent_reference_ab'}[args.profile]
    output_category={'short':'online_reference_ab','descent':'descent_reference_ab','near-descent':'near_descent_reference_ab'}[args.profile]
    if args.output_root is None:args.output_root=repo/'diagnostics'/output_category
    sources=sorted((repo/'diagnostics'/source_category).glob('*/run.json'))
    source=args.source_manifest or next((p for p in reversed(sources) if json.loads(p.read_bytes()).get('status')=='completed'),None)
    if source is None:parser.error('no completed reference comparison source for this profile')
    source=source.expanduser().absolute();data=json.loads(source.read_bytes())
    folder=args.output_root.expanduser().absolute()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');folder.mkdir(parents=True,exist_ok=False)
    (folder/'source_reference_manifest.json').write_bytes(source.read_bytes())
    report={'status':'preparing','source_manifest':str(source),'source_sha256':digest(source.read_bytes()),'queries_per_run':args.queries,'repeats':args.repeats,'cases':[],
        'profile':args.profile,
        'scope':'independent live observations and model outputs for each reference condition; open gripper',
        'sampling':'unpaired stochastic inference, no shared model-noise seed; condition order alternates per repeat'}
    def save():
        report['summary']=summarize(report['cases']);(folder/'run.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    save();print(f'[Online AB runner] Output: {folder}',flush=True)
    try:
        if args.profile=='short':prepared=prepare_source(source.parent,data)
        else:prepared=prepare_online_source(source.parent,data,'descent' if args.profile=='near-descent' else 'short')
        config,gains,initial,cb,payloads,prompt=prepared
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
                    'profile':args.profile,
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
                validate_run_result(r,mode,args.queries,args.profile)
                row.update(status='completed',result=outputs[0].relative_to(folder).as_posix(),result_sha256=digest(outputs[0].read_bytes()),first_input_sha256=r['first_input_sha256'],
                    all_followup_inputs_changed=r['all_followup_inputs_changed'],requested_close_actions=r['requested_close_actions'])
                row.update(stop_reason=r.get('stop_reason','query_budget_exhausted'),boundary_reached=r.get('boundary_reached',False),
                    action_controls=r['action_controls'],queries_used=r['policy_queries'],height_checkpoints=r.get('height_checkpoints',{}),
                    first_close_request=r.get('first_close_request'))
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
