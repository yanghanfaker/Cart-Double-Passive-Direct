"""Replay archived native policy responses, rejecting input or trajectory drift."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import zipfile
import numpy as np
from frozen_coordinate_probe import digest_observation
from low_pose_policy_audit import local_path
from native_libero_rollout import save_json, worker


def sha(b): return hashlib.sha256(b).hexdigest()
def read(p): return json.loads(Path(p).read_bytes())


def load_source(path):
    path=Path(path);m=read(path)
    if m['status']!='completed' or m['successes']!=3 or m['completed_episodes']!=3 or [e['episode_index'] for e in m['episodes']]!=[0,1,2]:
        raise ValueError('complete three-success native source required')
    responses=[];reports=[]
    for e in m['episodes']:
        p=local_path(path.parent,e['result']);r=read(p);reports.append(r)
        if not r['success'] or r['status']!='completed' or len(r['trace'])!=r['action_steps'] or sum(q['executed_prefix_steps'] for q in r['queries'])!=r['action_steps']:
            raise ValueError('incomplete source trace')
        archive=p.parent/'queries.zip'
        if sha(archive.read_bytes())!=r['query_archive_sha256']:raise ValueError('source archive SHA mismatch')
        with zipfile.ZipFile(archive) as z:
            if len(z.namelist())!=len(set(z.namelist())):raise ValueError('duplicate archive entries')
            for q in r['queries']:
                b=z.read(f"query-{q['query']:03d}/response.json")
                if sha(b)!=q['response_sha256']:raise ValueError('source response SHA mismatch')
                a=json.loads(b)['actions'];actual=[x for x in r['trace'] if x['query']==q['query']]
                if len(actual)!=q['executed_prefix_steps'] or any(x['action']!=a[k] for k,x in enumerate(actual)):
                    raise ValueError('response does not match source execution')
                responses.append({'input_sha256':q['input_sha256'],'actions':a,'episode':e['episode_index'],'query':q['query']})
    return m,reports,responses


class ReplayClient:
    def __init__(self,responses):self.responses=responses;self.index=0;self.checks=[]
    def get_server_metadata(self):return {'mode':'archived_responses','new_policy_queries':0}
    def infer(self,obs):
        if self.index>=len(self.responses):raise ValueError('replay requested extra response')
        q=self.responses[self.index];actual=digest_observation(obs)
        self.checks.append({'response_index':self.index,'expected_input_sha256':q['input_sha256'],'actual_input_sha256':actual,'exact_match':actual==q['input_sha256']})
        # Small renderer/numerical differences can alter exact image hashes. Do not
        # silently feed the next response after any changed observation.
        if actual!=q['input_sha256']:raise ValueError('native replay input differs from recorded source; inspect replay_checks.json')
        self.index+=1
        return {'actions':np.array(q['actions'],dtype=float)}


def compare_trace(original,replay):
    if replay['status']!='completed' or not replay['success'] or len(original['trace'])!=len(replay['trace']):raise ValueError('native replay outcome/count differs')
    maxima={'position_m':0.,'fingers_m':0.,'joints_rad':0.}
    for a,b in zip(original['trace'],replay['trace']):
        if a['step']!=b['step'] or a['action']!=b['action'] or a['task_success']!=b['task_success']:raise ValueError('native replay action/success differs')
        for key,field in (('position_m','robot0_eef_pos'),('fingers_m','robot0_gripper_qpos'),('joints_rad','robot0_joint_pos')):
            delta=float(np.max(np.abs(np.array(a['after'][field])-b['after'][field])))
            if not np.isfinite(delta) or delta>1e-7:raise ValueError('native replay trajectory differs by >1e-7')
            maxima[key]=max(maxima[key],delta)
    return maxima


def run_worker(folder,args):
    m,original,responses=load_source(args.source_report)
    client=ReplayClient(responses);report={'status':'running','new_policy_queries':0,'source_report_sha256':sha(args.source_report.read_bytes())}
    (folder/'source_report.json').write_bytes(args.source_report.read_bytes())
    args.task_id=m['task_id'];args.seed=m['seed'];args.server_host='127.0.0.1';args.server_port=8000
    try:
        code=worker(folder,args,client_factory=lambda:client)
        if code:raise RuntimeError('native replay failed; inspect report.json')
        current=read(folder/'report.json')
        for key in ('init_sha256','bddl_sha256','prompt','controller'):
            if current[key]!=m[key]:raise ValueError('native source configuration changed: '+key)
        comparisons=[]
        for old,e in zip(original,current['episodes']):comparisons.append(compare_trace(old,read(folder/e['result'])))
        if client.index!=len(responses):raise ValueError('not all archived responses consumed')
        report.update(status='completed',archived_responses_consumed=client.index,trajectory_maxima=comparisons,successes=current['successes'])
        return 0
    except Exception as error:report.update(status='failed',error=str(error));return 1
    finally:
        report['input_checks']=client.checks;save_json(folder/'replay_checks.json',report)


def main():
    repo=Path(__file__).resolve().parent.parent;p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-report',type=Path);p.add_argument('--openpi-root',type=Path,default=Path('/home/test/Robot/openpi'))
    p.add_argument('--output-root',type=Path,default=repo/'diagnostics/native_libero_replay');p.add_argument('--worker',type=Path,help=argparse.SUPPRESS);args=p.parse_args()
    if args.worker:return run_worker(args.worker,args)
    args.source_report=args.source_report or next((f for f in sorted((repo/'diagnostics/native_libero_rollout').glob('*/report.json'),reverse=True) if read(f).get('status')=='completed'),None)
    if args.source_report is None:p.error('no completed native rollout source')
    load_source(args.source_report)
    folder=args.output_root.absolute()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');folder.mkdir(parents=True)
    report={'status':'preflight','new_policy_queries':0}
    try:
        from run_multipose_geometry import preflight_reference
        python=(args.openpi_root/'examples/libero/.venv/bin/python').absolute();report['preflight']=preflight_reference(python,args.openpi_root.absolute(),folder)
        command=[str(python),str(Path(__file__).absolute()),'--worker',str(folder),'--source-report',str(args.source_report.absolute()),'--openpi-root',str(args.openpi_root.absolute())]
        env=os.environ.copy();env.pop('VIRTUAL_ENV',None);env.update(PYTHONPATH=str(args.openpi_root/'third_party/libero'),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYNPUT_BACKEND='dummy')
        report.update(status='running',command=command);save_json(folder/'run.json',report)
        print(f'[Native replay] Output: {folder}',flush=True)
        with (folder/'worker.log').open('w',encoding='utf-8') as log:
            with subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True) as child:
                try:code=child.wait(timeout=1800)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait();raise RuntimeError('replay timeout')
        result=read(folder/'replay_checks.json')
        if code or result['status']!='completed':raise RuntimeError('replay differs or failed; inspect replay_checks.json and worker.log')
        report.update(status='completed',successes=result['successes']);return 0
    except Exception as error:report.update(status='failed',error=str(error));return 1
    finally:save_json(folder/'run.json',report);print(f"[Native replay] {report['status']}: {folder/'run.json'}",flush=True)


if __name__=='__main__':raise SystemExit(main())
