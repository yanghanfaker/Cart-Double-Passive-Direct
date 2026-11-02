"""Run isolated gripper and progressive translation diagnostics in fresh Isaac scenes."""
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
from native_libero_rollout import save_json


def summarize(result):
    r=json.loads(result.read_bytes());rows=[json.loads(line) for line in (result.parent/'physics_joint_trace.jsonl').read_text().splitlines()]
    if r['status']=='completed' and len(rows)!=5*len(r['trace']):raise RuntimeError('completed case has incomplete physics trace')
    errors=np.array([np.array(x['position_target_rad'])-x['position_rad'] for x in rows])
    return {'status':r['status'],'error':r.get('error'),'controls':len(r['trace']),'physics_samples':len(rows),
        'initial_tcp_error_m':r['initial_tcp_error_m'],
        'joint_max_abs_tracking_error_rad':np.max(np.abs(errors),axis=0).tolist() if len(rows) else None,
        'peak_abs_joint_velocity_rad_s':np.max(np.abs([x['velocity_rad_s'] for x in rows]),axis=0).tolist() if rows else None,
        'min_joint_limit_margin_rad':np.min([x['joint_limit_margin_rad'] for x in rows],axis=0).tolist() if rows else None,
        'controls_with_translation_cap':sum(x['target']['translation_cap_triggered'] for x in r['trace'] if x['raw_action'] is not None),
        'actual_tcp_delta_m':(np.array((r['final'] if 'final' in r else r['last_state'])['tcp_world_m'])-r['initial']['tcp_world_m']).tolist(),
        'finger_controls':[{'step':x['step'],'sign':x['native_gripper_sign'],
            'ramp_target_m':x['ramp_target_m'],'finger_positions_m':x['after']['finger_positions_m']} for x in r['trace']],
        'result_sha256':hashlib.sha256(result.read_bytes()).hexdigest()}


def main():
    repo=Path(__file__).resolve().parent.parent;p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--isaaclab-root',type=Path,default=Path('/home/test/Robot/IsaacLab'))
    p.add_argument('--output-root',type=Path,default=repo/'diagnostics/execution_alignment')
    p.add_argument('--only',choices=[c['name'] for c in cases()],nargs='+')
    args=p.parse_args();folder=args.output_root.absolute()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');folder.mkdir(parents=True)
    report={'status':'running','new_policy_queries':0,'cases':[],
        'scope':'four isolated gripper cases and four translation-only cap levels; independent fresh scenes; later translation levels skipped after failure; no grasp success claim'}
    report['code_sha256']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('run_execution_alignment.py','execution_alignment_probe.py','isaac_native_segment.py','openpi_grasp_runtime.py','registered_replay_probe.py','physics_trace.py')}
    def save():save_json(folder/'run.json',report)
    save();translation_failed=False
    env=os.environ.copy();env.pop('VIRTUAL_ENV',None);env.pop('PYTHONPATH',None)
    try:
        for case in cases():
            if args.only and case['name'] not in args.only:continue
            row={'case':case,'status':'pending'};report['cases'].append(row)
            if case['kind']=='translation' and translation_failed:
                row.update(status='skipped',reason='earlier translation level failed');save();continue
            dest=folder/case['name'];dest.mkdir();request=dest/'case.json';save_json(request,case)
            command=[sys.executable,str(Path(__file__).with_name('isaac_native_segment.py')),
                '--execution-case-json',str(request),'--isaaclab-root',str(args.isaaclab_root.absolute()),'--output-root',str(dest/'runs')]
            row.update(status='running',command=command);save();print('[Execution alignment] START '+case['name'],flush=True)
            with (dest/'console.log').open('w',encoding='utf-8') as log:
                code=subprocess.call(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            row['exit_code']=code;results=list((dest/'runs').glob('*/capture/*/online_registered_result.json'))
            try:
                if len(results)!=1:raise RuntimeError('expected one result; inspect console.log')
                result=results[0];row['result']=result.relative_to(folder).as_posix()
                row['summary']=summarize(result)
                if code or row['summary']['status']!='completed':raise RuntimeError('case failed; inspect native result and physics trace')
                row['status']='completed'
            except Exception as error:
                row.update(status='failed',error=str(error))
                if case['kind']=='translation':translation_failed=True
            save();print('[Execution alignment] '+case['name']+': '+row['status'],flush=True)
        report['status']='completed' if all(x['status']=='completed' for x in report['cases']) else 'needs-inspection'
        return 0 if report['status']=='completed' else 1
    finally:save();print('[Execution alignment] Summary: '+str(folder/'run.json'),flush=True)


if __name__=='__main__':raise SystemExit(main())
