"""Matched native OSC replay: isolate bridge-sized caps and delayed gripper commands."""
import argparse
from datetime import datetime,timezone
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import numpy as np

from native_gripper_comparison import read,sha,load_responses,worker as native_worker
from native_libero_rollout import save_json
from openpi_step_probe import validate_action_chunk


def execution_cases():
    profiles=[('identity',None,None,0,False),('open',None,None,0,True),
        ('delay5',None,None,5,False),('delay10',None,None,10,False),
        ('translation0p5',.0005,None,0,False),('translation1',.001,None,0,False),
        ('translation2',.002,None,0,False),('rotation0p1',None,.1,0,False),
        ('both0p5',.0005,.1,0,False),('both0p5-delay5',.0005,.1,5,False)]
    return [dict(name=f't{task}-s{state}-{name}-replay',task_id=task,state_index=state,
        mode=name+'-replay',seed=7,prefix_steps=5,max_controls=220,
        translation_cap_m=cap,rotation_cap_deg=rotation,close_delay_controls=delay,force_open=opened)
        for task in (0,2) for state in (0,1,2) for name,cap,rotation,delay,opened in profiles]


class Intervention:
    """Stateful only for consecutive positive gripper requests; never extends source."""
    def __init__(self,case):self.case=case;self.close_streak=0
    def __call__(self,action):
        validate_action_chunk([list(action)]);a=list(action)
        # Identity / delay-only paths preserve pose values byte-for-value, including
        # values outside [-1,1]; native OSC retains responsibility for input clipping.
        for start,scale,cap in ((0,.05,self.case['translation_cap_m']),
                                (3,.5,None if self.case['rotation_cap_deg'] is None else math.radians(self.case['rotation_cap_deg']))):
            if cap is None:continue
            v=np.clip(np.array(action[start:start+3],dtype=float),-1,1)*scale
            n=float(np.linalg.norm(v))
            if n>cap:v*=cap/n
            a[start:start+3]=(v/scale).tolist()
        self.close_streak=self.close_streak+1 if action[6]>0 else 0
        if self.case['force_open'] or (0<self.close_streak<=self.case['close_delay_controls']):a[6]=-1.
        return a


def state_difference(a,b):
    if set(a)!=set(b):raise ValueError('snapshot fields differ')
    values=[]
    for key in a:
        if isinstance(a[key],dict):values.append(state_difference(a[key],b[key]))
        else:
            x=np.asarray(a[key],dtype=float);y=np.asarray(b[key],dtype=float)
            if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('invalid snapshot')
            values.append(float(np.max(np.abs(x-y))))
    return max(values,default=0.)


def compare_execution(source,result,case):
    if result['status']!='completed' or result['query_origin']!='archive' or result['new_policy_queries']!=0:
        raise ValueError('completed offline replay required')
    rows=result['trace'];original=source['trace']
    if not rows or len(rows)>len(original) or len(rows)!=result['action_steps']:raise ValueError('invalid replay horizon')
    if len(rows)<len(original) and result['stop_reason'] not in ('task_success','done_without_success'):
        raise ValueError('short replay without terminal event')
    transform=Intervention(case);pose_changed=grip_changed=0;cap_changes=0
    initial_error=state_difference(source['initial'],result['initial'])
    eef_max=0.;orientation_max=0.;state_max=initial_error;tracking=[]
    for a,b in zip(original,rows):
        if a['raw_action']!=b['raw_action'] or (a['step'],a['query'],a['prefix_index'])!=(b['step'],b['query'],b['prefix_index']):
            raise ValueError('replay raw source/order mismatch')
        expected=transform(a['raw_action'])
        if not np.allclose(expected,b['action'],atol=1e-15,rtol=0):raise ValueError('undeclared action intervention')
        pose_changed+=int(expected[:6]!=a['action'][:6]);grip_changed+=int(expected[6]!=a['action'][6])
        cap_changes+=int(np.linalg.norm(np.clip(a['raw_action'][:3],-1,1)*.05-np.clip(expected[:3],-1,1)*.05)>1e-12)
        if abs(b['after']['sim_time_s']-b['before']['sim_time_s']-.05)>1e-9:raise ValueError('not a 50ms control')
        eef_max=max(eef_max,float(np.linalg.norm(np.array(a['after']['robot0_eef_pos'])-b['after']['robot0_eef_pos'])))
        qa=np.array(a['after']['robot0_eef_quat']);qb=np.array(b['after']['robot0_eef_quat'])
        orientation_max=max(orientation_max,math.degrees(2*math.acos(float(np.clip(abs(np.dot(qa,qb))/(np.linalg.norm(qa)*np.linalg.norm(qb)),0,1)))))
        state_max=max(state_max,state_difference(a['after'],b['after']))
        t=b.get('controller_tracking',{})
        if t.get('status')=='available':tracking.append(t['position_error_m'])
    identity=case['mode']=='identity-replay'
    return {'source_controls':len(original),'common_controls':len(rows),'early_terminal':len(rows)<len(original),
        'paired_initial_max_abs_difference':initial_error,'common_max_state_abs_difference':state_max,
        'max_eef_difference_m':eef_max,'max_orientation_difference_deg':orientation_max,
        'pose_changed_controls':pose_changed,'gripper_changed_controls':grip_changed,'translation_clipped_controls':cap_changes,
        'controller_samples':len(tracking),'max_last_controller_position_error_m':max(tracking,default=None),
        'identity_reproduction_passed':len(rows)==len(original) and state_max<=1e-7 if identity else None,
        'source_success':source['success'],'replay_success':result['success'],
        'scope':'common archived-action prefix; altered-policy finite horizon, not an online success-rate estimate; controller buffers precede final integration'}


def validate_source(path):
    path=Path(path);run=read(path)
    if run.get('status')!='completed' or len(run.get('pairs',[]))!=6 or not all(p['physical_reproduction_passed'] for p in run['pairs']):
        raise ValueError('completed six-pair native gripper comparison required')
    sources={}
    for task in (0,2):
        for state in (0,1,2):
            name=f't{task}-s{state}-model-live'
            rows=[r for r in run['cases'] if r['case']['name']==name]
            if len(rows)!=1 or rows[0]['status']!='completed':raise ValueError('missing live source: '+name)
            source=path.parent/rows[0]['result']
            if sha(source.read_bytes())!=rows[0]['result_sha256']:raise ValueError('source worker digest differs')
            report=read(source)
            if report['case']['task_id']!=task or report['case']['state_index']!=state or report['case']['seed']!=7:
                raise ValueError('source task/state/seed differs')
            episode=source.parent/report['episode']
            if sha(episode.read_bytes())!=report['episode_sha256']:raise ValueError('source episode digest differs')
            content,_=load_responses(episode)
            if not content['success']:raise ValueError('successful native source required')
            sources[(task,state)]=source.parent
    return sources


def main():
    repo=Path(__file__).resolve().parent.parent;p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--openpi-root',type=Path,default=Path('/home/test/Robot/openpi'))
    p.add_argument('--source-run',type=Path)
    p.add_argument('--output-root',type=Path,default=repo/'diagnostics/native_execution_comparison')
    p.add_argument('--worker',type=Path,help=argparse.SUPPRESS);p.add_argument('--source-report',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.worker:
        case=read(args.worker/'case.json')
        if case not in execution_cases():raise ValueError('undeclared execution case')
        return native_worker(args.worker,args,declared_cases=execution_cases(),action_transform=Intervention(case),pose_intervention=True)
    folder=args.output_root.absolute()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');folder.mkdir(parents=True)
    report={'status':'preflight','cases':[],'new_policy_queries':0,
        'scope':'six successful native sources x ten archived interventions; OSC, gains, scene and physics unchanged; no Isaac execution or online inference',
        'code_sha256':{n:sha(Path(__file__).with_name(n).read_bytes()) for n in ('native_execution_comparison.py','native_gripper_comparison.py','native_libero_rollout.py','official_libero_launcher.py','official_libero_initial_states.json')}}
    def save():save_json(folder/'run.json',report)
    save()
    try:
        source=args.source_run or next((f for f in sorted((repo/'diagnostics/native_gripper_comparison').glob('*/run.json'),reverse=True) if read(f).get('status')=='completed'),None)
        if source is None:raise ValueError('completed native_gripper_comparison source missing')
        source=source.absolute();sources=validate_source(source)
        report['source_run_sha256']=sha(source.read_bytes());shutil.copyfile(source,folder/'source_run.json')
        from official_libero_launcher import verify_init_bytes
        manifest=read(Path(__file__).with_name('official_libero_initial_states.json'))
        for name,expected in manifest['files'].items():
            target=folder/'verified_init_states'/name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(verify_init_bytes((source.parent/'verified_init_states'/name).read_bytes(),expected))
        inputs={}
        for key,home in sources.items():
            target=folder/'source_inputs'/home.name;target.mkdir(parents=True)
            for relative in ('report.json','case.json','initial_state.npy','task.bddl','scene.xml','controller_implementation.py','gripper_implementation.py','episode/report.json','episode/queries.zip'):
                dest=target/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(home/relative,dest)
            inputs[key]=target
        python=(args.openpi_root/'examples/libero/.venv/bin/python').absolute()
        from run_multipose_geometry import preflight_reference
        report['runtime_preflight']=preflight_reference(python,args.openpi_root.absolute(),folder)
        env=os.environ.copy()
        for name in ('VIRTUAL_ENV','PYTHONHOME','TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'):env.pop(name,None)
        env.update(PYTHONPATH=str(args.openpi_root.absolute()/'third_party/libero'),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYNPUT_BACKEND='dummy',TORCH_FORCE_WEIGHTS_ONLY_LOAD='1')
        report['status']='running';save();identities={}
        for case in execution_cases():
            key=(case['task_id'],case['state_index']);row={'case':case,'status':'pending'};report['cases'].append(row);save()
            if case['mode']!='identity-replay' and not identities.get(key):
                row.update(status='skipped',reason='identity replay not reproduced');save();continue
            out=folder/case['name'];out.mkdir();save_json(out/'case.json',case)
            cmd=[str(python),str(Path(__file__).absolute()),'--worker',str(out),'--openpi-root',str(args.openpi_root.absolute()),
                 '--source-report',str(inputs[key]/'episode/report.json')]
            row['command']=cmd;save();print('[Native execution]',case['name'],flush=True)
            with (out/'worker.log').open('w',encoding='utf-8') as log:
                child=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=child.wait(timeout=900)
                except (subprocess.TimeoutExpired,KeyboardInterrupt):
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    raise
            try:
                worker=read(out/'report.json');row.update(result=(out/'report.json').relative_to(folder).as_posix(),result_sha256=sha((out/'report.json').read_bytes()),exit_code=code)
                if code or worker['status']!='completed':raise ValueError(worker.get('error','worker failed'))
                source_config=read(inputs[key]/'report.json')
                fields=('prompt','init_sha256','bddl_sha256','initial_state_sha256','target','controller','control_frequency_hz')
                if any(worker[f]!=source_config[f] for f in fields):raise ValueError('source environment/controller configuration changed')
                if worker['controller']['name']!='OSC_POSE' or worker['controller']['output_max']!=[.05,.05,.05,.5,.5,.5]:raise ValueError('unexpected native scaling')
                episode_path=out/'episode/report.json'
                if sha(episode_path.read_bytes())!=worker['episode_sha256']:raise ValueError('worker episode digest differs')
                episode=read(episode_path)
                row['comparison']=compare_execution(read(inputs[key]/'episode/report.json'),episode,case)
                checks=worker['replay_input_checks'];row['all_query_inputs_match_source']=all(c['exact_input_match'] for c in checks)
                if case['mode']=='identity-replay':
                    identities[key]=row['comparison']['identity_reproduction_passed'] and row['all_query_inputs_match_source']
                    if not identities[key]:raise ValueError('identity did not reproduce states and query inputs')
                if row['comparison']['paired_initial_max_abs_difference']>1e-7:raise ValueError('paired initial state differs')
                if row['comparison']['controller_samples']!=episode['action_steps']:raise ValueError('controller diagnostics unavailable')
                row.update(status='completed',metrics=worker['metrics'])
                report['new_policy_queries']+=episode['new_policy_queries']
            except Exception as error:row.update(status='failed',error=type(error).__name__+': '+str(error))
            save()
        report['status']='completed' if len(report['cases'])==60 and all(r['status']=='completed' for r in report['cases']) else 'needs-inspection'
    except KeyboardInterrupt:report.update(status='interrupted',error='KeyboardInterrupt')
    except Exception as error:report.update(status='failed',error=type(error).__name__+': '+str(error))
    finally:save();print('[Native execution]',report['status'],folder/'run.json',flush=True)
    return 0 if report['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
