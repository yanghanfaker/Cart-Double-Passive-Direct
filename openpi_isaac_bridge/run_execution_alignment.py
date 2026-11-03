"""Run declared execution or deterministic contact diagnostics in fresh Isaac scenes."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from execution_alignment_probe import cases
from execution_coverage_cases import coverage_cases
from execution_focus_cases import focus_cases
from execution_sync_cases import sync_cases
from execution_generalization import generalization_cases
from execution_contact import contact_cases
from execution_hybrid import hybrid_cases,validate_contact_source
from execution_limit_audit import limit_cases,prepare_replay_sources
from execution_online_caps import online_caps_cases
from native_libero_rollout import save_json


def missing_prerequisites(case,rows):
    done={x['case']['name']:x for x in rows}
    missing=[name for name in case.get('requires',[]) if done.get(name,{}).get('status')!='completed']
    for name in case.get('requires_success',[]):
        prior=done.get(name,{})
        if not (prior.get('summary',{}).get('grasp_result') or {}).get('success',False):missing.append(name+' grasp success')
    return missing


def summarize(result):
    r=json.loads(result.read_bytes());trace_path=result.parent/'physics_joint_trace.jsonl'
    compressed=trace_path.with_suffix('.jsonl.gz')
    if not trace_path.exists() and not compressed.exists():
        if r['status']=='completed':raise RuntimeError('completed case missing physics trace')
        return {'status':r['status'],'error':r.get('error'),'controls':len(r.get('trace',[])),'physics_samples':0,
            'policy_queries':r.get('policy_queries',0),'inference_attempts':r.get('inference_attempts',0),
            'result_sha256':hashlib.sha256(result.read_bytes()).hexdigest()}
    if trace_path.exists():lines=trace_path.read_text().splitlines()
    else:
        import gzip
        with gzip.open(compressed,'rt',encoding='utf-8') as source:lines=source.read().splitlines()
    rows=[json.loads(line) for line in lines]
    if r['status']=='completed' and len(rows)!=5*len(r['trace']):raise RuntimeError('completed case has incomplete physics trace')
    errors=np.array([np.array(x['position_target_rad'])-x['position_rad'] for x in rows])
    effort_rows=[x for x in rows if all(x[k]['status']=='available' for k in ('computed_torque','applied_torque','effort_limits'))]
    saturation={'sample_count':len(effort_rows),'clipped_samples':int(sum(np.max(np.abs(np.array(x['computed_torque']['value'])-x['applied_torque']['value']))>1e-5 for x in effort_rows)),
        'peak_requested_fraction_per_joint':np.max([np.abs(x['computed_torque']['value'])/np.array(x['effort_limits']['value']) for x in effort_rows],axis=0).tolist() if effort_rows else None,
        'scope':'computed versus applied buffer difference; not an independent solver-force measurement'}
    return {'status':r['status'],'error':r.get('error'),'controls':len(r['trace']),'physics_samples':len(rows),
        'policy_queries':r.get('policy_queries',0),'inference_attempts':r.get('inference_attempts',0),
        'saturation':saturation,
        'max_tcp_linearization_residual_m':max((x['kinematic_decomposition']['actual_linearization_residual_m'] for x in rows),default=None),
        'initial_tcp_error_m':r['initial_tcp_error_m'],
        'joint_max_abs_tracking_error_rad':np.max(np.abs(errors),axis=0).tolist() if len(rows) else None,
        'joint_rms_tracking_error_rad':np.sqrt(np.mean(errors**2,axis=0)).tolist() if len(rows) else None,
        'peak_tcp_distance_from_start_m':max((float(np.linalg.norm(np.array(x['tcp_world_m'])-r['initial']['tcp_world_m'])) for x in rows),default=None),
        'peak_abs_joint_velocity_rad_s':np.max(np.abs([x['velocity_rad_s'] for x in rows]),axis=0).tolist() if rows else None,
        'min_joint_limit_margin_rad':np.min([x['joint_limit_margin_rad'] for x in rows],axis=0).tolist() if rows else None,
        'controls_with_translation_cap':sum(x['target']['translation_cap_triggered'] for x in r['trace'] if x['raw_action'] is not None),
        'controls_with_rotation_cap':sum(x['target']['rotation_cap_triggered'] for x in r['trace'] if x['raw_action'] is not None),
        'max_period_end_tcp_error_m':max((x['tcp97_target_error_m'] for x in map(json.loads,(result.parent/'motion.jsonl').read_text().splitlines()) if x['event']=='result'),default=None),
        'stream_initial':r.get('stream_initial'),'initial_velocity_comparison':r.get('initial_velocity_comparison'),
        'finger_dynamics_audit':json.loads((result.parent/'finger_dynamics_audit.json').read_bytes()) if (result.parent/'finger_dynamics_audit.json').exists() else None,
        'grasp_result':r.get('grasp_result'),
        'frozen_queries':r.get('frozen_queries'),'frozen_input':r.get('frozen_input'),'replay_result':r.get('replay_result'),
        'actual_tcp_delta_m':(np.array(r.get('final',r.get('last_state',r['initial']))['tcp_world_m'])-r['initial']['tcp_world_m']).tolist(),
        'finger_controls':[{'step':x['step'],'sign':x['native_gripper_sign'],
            'ramp_target_m':x['ramp_target_m'],'finger_positions_m':x['after']['finger_positions_m']} for x in r['trace']],
        'result_sha256':hashlib.sha256(result.read_bytes()).hexdigest()}


def main():
    repo=Path(__file__).resolve().parent.parent;p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--isaaclab-root',type=Path,default=Path('/home/test/Robot/IsaacLab'))
    p.add_argument('--suite',choices=['alignment','coverage','focus','sync','generalization','contact','hybrid','limits','online_caps'],default='alignment')
    p.add_argument('--output-root',type=Path)
    p.add_argument('--only',choices=[c['name'] for c in cases()+coverage_cases()+focus_cases()+sync_cases()+generalization_cases()+contact_cases()+hybrid_cases()+limit_cases()+online_caps_cases()],nargs='+')
    args=p.parse_args();selected={'alignment':cases,'coverage':coverage_cases,'focus':focus_cases,'sync':sync_cases,'generalization':generalization_cases,'contact':contact_cases,'hybrid':hybrid_cases,'limits':limit_cases,'online_caps':online_caps_cases}[args.suite]()
    if args.only:
        wanted=set(args.only);declared={c['name']:c for c in selected}
        if wanted-set(declared):p.error('--only must belong to selected suite')
        def add_dependencies(name):
            for dependency in declared[name].get('requires',[]):
                if dependency not in wanted:wanted.add(dependency);add_dependencies(dependency)
        for name in list(wanted):add_dependencies(name)
        selected=[c for c in selected if c['name'] in wanted]
    args.output_root=args.output_root or repo/f'diagnostics/execution_{args.suite}'
    folder=args.output_root.absolute()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');folder.mkdir(parents=True)
    report={'status':'running','new_policy_queries':0,'cases':[],
        'suite':args.suite,'scope':'independent fresh no-contact scenes; prerequisite failures skip dependent cases; synthetic preconditioning changes pose and velocity, not native dynamic restoration; no grasp success claim'}
    report['code_sha256']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('run_execution_alignment.py','execution_alignment_probe.py','execution_coverage_cases.py','execution_focus_cases.py','execution_sync_cases.py','execution_generalization.py','execution_contact.py','execution_hybrid.py','execution_limit_audit.py','execution_online_caps.py','collision_geometry.py','live_grasp_query.py','online_registered_probe.py','replanning_probe.py','openpi_health.py','grasp_probe.py','openpi_grasp_logic.py','libero_reference_audit.py','run_multipose_geometry.py','check_cameras.py','isaac_native_segment.py','run_geometry_audit.py','openpi_grasp_runtime.py','registered_replay_probe.py','physics_trace.py')}
    if args.suite=='generalization':
        report.update(pose_reference_root='pose_references',scope='three static native-FK checked starting poses; same archived actions reused counterfactually; no new inference, contact, or grasp success claim')
    if args.suite=='contact':report['scope']='ten fresh deterministic cube approach/closure/lift diagnostics; two repeats per height/finger and open negative controls; success reported separately from completed execution; no VLA inference or measured contact force'
    if args.suite=='hybrid':report['scope']='eight live cube-proxy policy approaches plus two program references; 1/5 action prefixes and geometry/model-close handoff; program-assisted refinement and grasp, not autonomous VLA or native LIBERO evaluation'
    if args.suite=='limits':report['scope']='eight frozen input variants, six matched first100-action cap replays and two program references; not online policy grasp or native benchmark evaluation'
    if args.suite=='online_caps':report['scope']='eight online cap/prefix cases, two program references and two archived 2mm deflection reproductions; optional authored-bound geometry diagnostics, no measured contact force'
    def save():
        report['new_policy_queries']=sum(x.get('summary',{}).get('policy_queries',0) for x in report['cases'])
        save_json(folder/'run.json',report)
    save();translation_failed=False
    env=os.environ.copy();env.pop('VIRTUAL_ENV',None);env.pop('PYTHONPATH',None)
    replay_sources={}
    try:
        if args.suite=='limits' or (args.suite=='online_caps' and any(c.get('limit_audit_role')=='replay' for c in selected)):
            try:
                source=next((p for p in sorted((repo/'diagnostics/execution_hybrid').glob('*/run.json'),reverse=True) if json.loads(p.read_bytes()).get('status')=='completed'),None)
                if source is None:raise ValueError('completed prior hybrid suite required')
                replay_sources=prepare_replay_sources(source,folder/'replay_inputs')
                report['source_hybrid_manifest_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
                if any(c.get('limit_audit_role')=='frozen' for c in selected):
                    from openpi_health import check_openpi_health
                    report['server_health']=check_openpi_health('127.0.0.1',8000)
            except Exception as error:
                report.update(status='preflight-failed',error=str(error));print('[Execution alignment] Preflight failed: '+str(error),flush=True);return 1
        if args.suite in ('hybrid','online_caps'):
            try:
                source=next((p for p in sorted((repo/'diagnostics/execution_contact').glob('*/run.json'),reverse=True) if json.loads(p.read_bytes()).get('status')=='completed'),None)
                if source is None:raise ValueError('completed prior contact suite required')
                report['validated_contact_source']=validate_contact_source(source)
                (folder/'source_contact_run.json').write_bytes(source.read_bytes())
                if any(c.get('hybrid_role')=='policy' for c in selected):
                    from openpi_health import check_openpi_health
                    report['server_health']=check_openpi_health('127.0.0.1',8000)
            except Exception as error:
                report.update(status='preflight-failed',error=str(error));print('[Execution alignment] Preflight failed: '+str(error),flush=True);return 1
        for case in selected:
            row={'case':case,'status':'pending'};report['cases'].append(row)
            if 'pose_variant' in case:
                cache_manifest=folder/'pose_references'/case['pose_variant']/'run.json'
                if cache_manifest.exists() and json.loads(cache_manifest.read_bytes()).get('status')=='failed':
                    row.update(status='skipped',reason='native pose reference preparation failed',pose_manifest=cache_manifest.relative_to(folder).as_posix());save();continue
            missing=missing_prerequisites(case,report['cases'])
            if missing:
                row.update(status='skipped',reason='prerequisites not completed',prerequisites=missing);save();continue
            if case['kind']=='translation' and translation_failed:
                row.update(status='skipped',reason='earlier translation level failed');save();continue
            dest=folder/case['name'];dest.mkdir();request=dest/'case.json';save_json(request,case)
            command=[sys.executable,str(Path(__file__).with_name('isaac_native_segment.py')),
                '--execution-case-json',str(request),'--isaaclab-root',str(args.isaaclab_root.absolute()),'--output-root',str(dest/'runs')]
            if 'pose_variant' in case:command+=['--pose-cache-root',str(folder/'pose_references')]
            if case.get('limit_audit_role')=='replay':command+=['--limit-replay-json',str(replay_sources[case['source_prefix']])]
            row.update(status='running',command=command);save();print('[Execution alignment] START '+case['name'],flush=True)
            with (dest/'console.log').open('w',encoding='utf-8') as log:
                code=subprocess.call(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            row['exit_code']=code;results=list((dest/'runs').glob('*/capture/*/online_registered_result.json'))
            try:
                if len(results)!=1:raise RuntimeError('expected one result; inspect console.log')
                result=results[0];row['result']=result.relative_to(folder).as_posix()
                row['summary']=summarize(result)
                if code or row['summary']['status']!='completed':raise RuntimeError(row['summary'].get('error') or 'case failed; inspect native result and physics trace')
                if case.get('hybrid_role')=='reference' and not (row['summary'].get('grasp_result') or {}).get('success',False):
                    raise RuntimeError('program reference grasp did not pass')
                if case.get('limit_audit_role')=='replay' and case['cap_m']==.0005:
                    if not (row['summary'].get('replay_result') or {}).get('baseline_replay_matches_source',False):raise RuntimeError('0.5mm archived replay did not match source')
                row['status']='completed'
            except Exception as error:
                row.update(status='failed',error=str(error))
                if case['kind']=='translation':translation_failed=True
            save();print('[Execution alignment] '+case['name']+': '+row['status'],flush=True)
        report['status']='completed' if all(x['status']=='completed' for x in report['cases']) else 'needs-inspection'
        return 0 if report['status']=='completed' else 1
    finally:save();print('[Execution alignment] Summary: '+str(folder/'run.json'),flush=True)


if __name__=='__main__':raise SystemExit(main())
