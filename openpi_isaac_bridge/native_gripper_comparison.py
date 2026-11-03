"""Native LIBERO gripper intervention matrix: live feedback versus frozen replay."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import zipfile
import numpy as np
from native_libero_rollout import rollout,save_json,snapshot,controller_record
from frozen_coordinate_probe import digest_observation
from official_libero_launcher import verify_init_bytes,allowed_numpy_types
from openpi_health import check_openpi_health
from openpi_step_probe import read_server_metadata,validate_action_chunk


read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda b:hashlib.sha256(b).hexdigest()


def comparison_cases():
    result=[]
    for task in (0,2):
        for state in (0,1,2):
            prefix=f't{task}-s{state}'
            modes=('model-live','model-replay','open-replay','open-live') if state%2==0 else ('model-live','open-live','open-replay','model-replay')
            for mode in modes:
                result.append({'name':prefix+'-'+mode,'task_id':task,'state_index':state,'mode':mode,'seed':7,
                               'prefix_steps':5,'max_controls':220,'source_name':prefix+'-model-live'})
    return result


def force_open(action):
    validate_action_chunk([list(action)])
    return list(action[:6])+[-1.]


def load_responses(path):
    path=Path(path);report=read(path)
    if report['status']!='completed' or report['query_origin']!='live':raise ValueError('completed live model source required')
    if report['action_steps']!=len(report['trace']):raise ValueError('source trace count differs')
    archive=path.parent/'queries.zip'
    if sha(archive.read_bytes())!=report['query_archive_sha256']:raise ValueError('source archive digest mismatch')
    responses=[]
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist())!=len(set(z.namelist())):raise ValueError('duplicate query archive entries')
        for query in report['queries']:
            if query['status']!='validated' or abs(query['sim_time_before_infer_s']-query['sim_time_after_infer_s'])>1e-12:
                raise ValueError('source query not validated or physics advanced')
            name=f"query-{query['query']:03d}"
            payload=z.read(name+'/response.json')
            if sha(payload)!=query['response_sha256']:raise ValueError('source response digest mismatch')
            actions=validate_action_chunk(json.loads(payload)['actions'])
            traces=[x for x in report['trace'] if x['query']==query['query']]
            if [x['raw_action'] for x in traces]!=actions[:query['executed_prefix_steps']]:raise ValueError('source response/execution mismatch')
            if any(x['action']!=x['raw_action'] for x in traces):raise ValueError('source must use model gripper')
            responses.append({'actions':actions,'input_sha256':query['input_sha256']})
    return report,responses


class FrozenClient:
    def __init__(self,responses):self.responses=responses;self.index=0;self.checks=[]
    def get_server_metadata(self):return {'mode':'frozen_counterfactual','new_policy_queries':0}
    def infer(self,observation):
        if self.index>=len(self.responses):raise ValueError('no padding or extra source responses allowed')
        source=self.responses[self.index];actual=digest_observation(observation)
        self.checks.append({'query':self.index+1,'source_input_sha256':source['input_sha256'],
                            'actual_input_sha256':actual,'exact_input_match':actual==source['input_sha256']})
        self.index+=1
        # Changed observations are intentional in open replay; the normal replay
        # is separately checked for physical and input equivalence after rollout.
        return {'actions':np.array(source['actions'],dtype=float)}


def episode_metrics(report,target_key):
    states=[report['initial']]+[x['after'] for x in report['trace']]
    target=lambda s:np.array(s['object_observations'][target_key])
    distance=lambda s:float(np.linalg.norm(np.array(s['robot0_eef_pos'])-target(s)))
    traces=report['trace'];first_close=next((x['step'] for x in traces if x['raw_action'][6]>0),None)
    return {'controls':report['action_steps'],'success':report['success'],'stop_reason':report['stop_reason'],
        'new_policy_queries':report['new_policy_queries'],'response_requests':report['policy_queries'],
        'first_model_close_control':first_close,'model_close_requests':sum(x['raw_action'][6]>0 for x in traces),
        'executed_close_controls':sum(x['action'][6]>0 for x in traces),
        'minimum_eef_to_target_body_origin_m':min(map(distance,states)),
        'final_eef_minus_target_body_origin_m':(np.array(states[-1]['robot0_eef_pos'])-target(states[-1])).tolist(),
        'target_max_displacement_m':max(float(np.linalg.norm(target(s)-target(states[0]))) for s in states),
        'first100_eef_delta_m':(np.array(states[min(100,len(traces))]['robot0_eef_pos'])-states[0]['robot0_eef_pos']).tolist(),
        'first100_raw_translation_sum_controller_units':np.sum([x['raw_action'][:3] for x in traces[:100]],axis=0).tolist() if traces else [0.,0.,0.],
        'gripper_command_available_controls':sum(x.get('native_gripper_command',{}).get('status')=='available' for x in traces),
        'distance_scope':'EEF to target rigid-body origin, not fingertip/grasp-surface distance or a close gate'}


def compare_pair(source,normal,opened):
    rows=source['trace']
    if len(rows)!=len(normal['trace']) or len(rows)!=len(opened['trace']):raise ValueError('frozen replay must execute exactly source control count')
    keys=('robot0_eef_pos','robot0_gripper_qpos','robot0_joint_pos','robot0_joint_vel')
    def delta(a,b):
        values=[float(np.max(np.abs(np.array(a[k])-b[k]))) for k in keys if k in a and k in b]
        for k in a['object_observations']:
            values.append(float(np.max(np.abs(np.array(a['object_observations'][k])-b['object_observations'][k]))))
        return max(values,default=0.)
    initial=max(delta(source['initial'],normal['initial']),delta(source['initial'],opened['initial']))
    normal_error=0.;pre_close_error=0.;position_effect=0.;first_close=None
    for a,b,c in zip(rows,normal['trace'],opened['trace']):
        if not (a['raw_action']==b['raw_action']==c['raw_action'] and a['action']==b['action'] and c['action']==force_open(a['action'])):
            raise ValueError('frozen intervention changed pose actions or source response')
        normal_error=max(normal_error,delta(a['after'],b['after']))
        if first_close is None and a['raw_action'][6]>0:first_close=a['step']
        if first_close is None:pre_close_error=max(pre_close_error,delta(b['after'],c['after']))
        position_effect=max(position_effect,float(np.linalg.norm(np.array(b['after']['robot0_eef_pos'])-c['after']['robot0_eef_pos'])))
    return {'source_replay_max_state_abs_difference':normal_error,'paired_initial_max_abs_difference':initial,
            'before_first_blocked_close_max_state_abs_difference':pre_close_error,'first_blocked_close_control':first_close,
            'maximum_open_vs_normal_eef_distance_m':position_effect,
            'physical_reproduction_passed':initial<=1e-7 and normal_error<=1e-7 and pre_close_error<=1e-7,
            'intervention_observed':first_close is not None,
            'scope':'fixed archived pose actions; physical comparison valid only if normal replay reproduces source; live model outputs are not same-random-seed pairs'}


def baseline_source(path):
    path=Path(path);r=read(path)
    expected_client=read(Path(__file__).with_name('official_libero_manifest.json'))
    if r.get('source',{}).get('sha256')!=expected_client['sha256'] or sha((path.parent/'official_main.py').read_bytes())!=expected_client['sha256']:
        raise ValueError('baseline upstream client differs from pinned official source')
    if r.get('returncode')!=0 or (r.get('metrics') or {}).get('episodes')!=10 or r.get('video_files')!=10:
        raise ValueError('ten-episode official native baseline required')
    if r.get('init_state_rollout',{}).get('status')!='official-client-returned':raise ValueError('official compatibility launcher did not return')
    text=(path.parent/'console.log').read_text(errors='replace')
    phase=text.split('INFO:root:Total episodes: 10',1)
    if len(phase)!=2 or 'Caught exception:' in text or 'Traceback (most recent call last)' in phase[0]:
        raise ValueError('baseline contains episode errors or no final summary')
    for entry in r['artifacts']:
        p=path.parent/entry['path']
        if not p.is_file() and entry['bytes']==0 and entry['sha256']==sha(b''):continue
        data=p.read_bytes()
        if len(data)!=entry['bytes'] or sha(data)!=entry['sha256']:raise ValueError('baseline artifact mismatch: '+entry['path'])
    return r


def worker(folder,args):
    case=read(folder/'case.json')
    if case not in comparison_cases():raise ValueError('undeclared comparison case')
    report={'status':'starting','case':case};env=None;client=None
    def save():save_json(folder/'report.json',report)
    save()
    try:
        spec=importlib.util.find_spec('libero')
        root=next(Path(p)/'libero' for p in spec.submodule_search_locations if (Path(p)/'libero/assets').is_dir())
        config=folder/'libero_config';config.mkdir()
        save_json(config/'config.yaml',{'benchmark_root':str(root),'bddl_files':str(root/'bddl_files'),'assets':str(root/'assets'),
            'init_states':str(folder.parent/'verified_init_states'),'datasets':str(root.parent/'datasets')})
        os.environ['LIBERO_CONFIG_PATH']=str(config)
        import torch
        from libero.libero import benchmark,get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from openpi_client import image_tools
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        from libero_scene_probe import select_state
        np.random.seed(case['seed'])
        suite=benchmark.get_benchmark_dict()['libero_spatial']();task=suite.get_task(case['task_id'])
        init=Path(get_libero_path('init_states'))/task.problem_folder/task.init_states_file
        manifest=read(Path(__file__).with_name('official_libero_initial_states.json'))
        relative=task.problem_folder+'/'+task.init_states_file
        data=verify_init_bytes(init.read_bytes(),manifest['files'][relative])
        with torch.serialization.safe_globals(allowed_numpy_types(np)):states=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
        bddl=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
        report.update(prompt=task.language,init_sha256=sha(data),bddl_sha256=sha(bddl.read_bytes()))
        (folder/'task.bddl').write_bytes(bddl.read_bytes())
        env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256);env.seed(case['seed']);env.reset()
        state=select_state(states,case['state_index'],len(env.get_sim_state()))
        np.save(folder/'initial_state.npy',state,allow_pickle=False)
        report['initial_state_sha256']=sha((folder/'initial_state.npy').read_bytes())
        obs=env.set_init_state(state)
        for _ in range(10):
            obs,_,done,_=env.step([0.]*6+[-1.])
            if done or env.check_success():raise RuntimeError('already terminal during dummy wait')
        interests=list(env.obj_of_interest)
        targets=[n for n in interests if 'black_bowl' in n]
        if len(targets)!=1 or targets[0]+'_pos' not in obs:raise RuntimeError('target bowl not uniquely resolved from objects of interest: '+str(interests))
        report['target']={'name':targets[0],'observation_key':targets[0]+'_pos','objects_of_interest':interests}
        report['controller']=controller_record(env.robots[0].controller)
        report['control_frequency_hz']=float(env.env.control_freq)
        if not math.isclose(report['control_frequency_hz'],20.):raise RuntimeError('expected native 20Hz controller')
        import inspect
        (folder/'controller_implementation.py').write_text(inspect.getsource(type(env.robots[0].controller)),encoding='utf-8')
        (folder/'gripper_implementation.py').write_text(inspect.getsource(type(env.robots[0].gripper)),encoding='utf-8')
        (folder/'scene.xml').write_text(env.env.model.get_xml(),encoding='utf-8')
        replay=case['mode'].endswith('replay');budget=case['max_controls']
        if replay:
            source,responses=load_responses(args.source_report)
            client=FrozenClient(responses);budget=source['action_steps']
            report.update(source_report_sha256=sha(args.source_report.read_bytes()),source_controls=budget)
        else:
            report['server_health']=check_openpi_health(args.server_host,args.server_port)
            client=WebsocketClientPolicy(host=args.server_host,port=args.server_port)
        report['server_metadata']=read_server_metadata(client);report['status']='running';save()
        result=rollout(env,client,obs,str(task.language),folder/'episode',image_tools,max_steps=budget,
            action_transform=force_open if case['mode'].startswith('open') else None,
            record_gripper=True,query_origin='archive' if replay else 'live')
        report['episode']='episode/report.json'
        report['metrics']=episode_metrics(result,report['target']['observation_key'])
        if replay:
            report['replay_input_checks']=client.checks
            if client.index!=len(responses) or result['action_steps']!=budget:raise RuntimeError('frozen replay ended before exact source control count')
        report['status']='completed';return 0
    except Exception as error:
        report.update(status='failed',error=type(error).__name__+': '+str(error));print(report['error'],flush=True);return 1
    finally:
        if env is not None:
            try:env.close()
            except Exception as error:report['cleanup_error']=type(error).__name__+': '+str(error)
        episode=folder/'episode/report.json'
        if episode.is_file():
            partial=read(episode)
            report.update(episode='episode/report.json',episode_sha256=sha(episode.read_bytes()),
                          new_policy_queries=partial['new_policy_queries'],inference_attempts=partial['inference_attempts'])
        save()


def main():
    repo=Path(__file__).resolve().parent.parent;p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--openpi-root',type=Path,default=Path('/home/test/Robot/openpi'))
    p.add_argument('--baseline-report',type=Path)
    p.add_argument('--output-root',type=Path,default=repo/'diagnostics/native_gripper_comparison')
    p.add_argument('--server-host',default='127.0.0.1');p.add_argument('--server-port',type=int,default=8000)
    p.add_argument('--worker',type=Path,help=argparse.SUPPRESS);p.add_argument('--source-report',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.worker:return worker(args.worker,args)
    folder=args.output_root.absolute()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');folder.mkdir(parents=True)
    report={'status':'preflight','cases':[],'pairs':[],'new_policy_queries':0,
        'scope':'recorded native intervention experiment; not unmodified official evaluation; two tasks x three states x four modes',
        'hypothesis':'separate physical effects of blocking closure from model-feedback effects; always-open success rate alone is not the test',
        'code_sha256':{name:sha(Path(__file__).with_name(name).read_bytes()) for name in ('native_gripper_comparison.py','native_libero_rollout.py','official_libero_initial_states.json','official_libero_launcher.py','frozen_coordinate_probe.py')}}
    def save():save_json(folder/'run.json',report)
    save()
    try:
        source=args.baseline_report or next((f for f in sorted((repo/'diagnostics/official_libero_baseline').glob('*/run.json'),reverse=True)
             if read(f).get('init_state_rollout',{}).get('status')=='official-client-returned' and (read(f).get('metrics') or {}).get('episodes')==10),None)
        if source is None:raise RuntimeError('completed ten-task native baseline required')
        baseline_source(source)
        report['baseline_source_sha256']=sha(source.read_bytes());(folder/'baseline_source_run.json').write_bytes(source.read_bytes())
        manifest=read(Path(__file__).with_name('official_libero_initial_states.json'))
        for name,expected in manifest['files'].items():
            data=verify_init_bytes((source.parent/'verified_init_states'/name).read_bytes(),expected)
            target=folder/'verified_init_states'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        python=(args.openpi_root/'examples/libero/.venv/bin/python').absolute()
        from run_multipose_geometry import preflight_reference
        report['runtime_preflight']=preflight_reference(python,args.openpi_root.absolute(),folder)
        report['server_health']=check_openpi_health(args.server_host,args.server_port)
        env=os.environ.copy();env.pop('VIRTUAL_ENV',None);env.pop('PYTHONHOME',None);env.pop('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD',None)
        env.update(PYTHONPATH=str(args.openpi_root/'third_party/libero'),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYNPUT_BACKEND='dummy',TORCH_FORCE_WEIGHTS_ONLY_LOAD='1')
        report['status']='running';save()
        completed={}
        for case in comparison_cases():
            row={'case':case,'status':'pending'};report['cases'].append(row);save()
            if case['mode']!='model-live' and case['source_name'] not in completed:
                row.update(status='skipped',reason='live source failed');save();continue
            out=folder/case['name'];out.mkdir();save_json(out/'case.json',case)
            cmd=[str(python),str(Path(__file__).absolute()),'--worker',str(out),'--openpi-root',str(args.openpi_root.absolute()),
                 '--server-host',args.server_host,'--server-port',str(args.server_port)]
            if case['mode'].endswith('replay'):cmd+=['--source-report',str(completed[case['source_name']]/'episode/report.json')]
            row['command']=cmd;save();print('[Native gripper]',case['name'],flush=True)
            with (out/'worker.log').open('w',encoding='utf-8') as log:
                child=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=child.wait(timeout=900)
                except (subprocess.TimeoutExpired,KeyboardInterrupt):
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    raise
            result=read(out/'report.json') if (out/'report.json').is_file() else {'status':'failed','error':'worker report missing'}
            row.update(status='completed' if code==0 and result['status']=='completed' else 'failed',exit_code=code,
                       result=(out/'report.json').relative_to(folder).as_posix(),result_sha256=sha((out/'report.json').read_bytes()) if (out/'report.json').is_file() else None,
                       metrics=result.get('metrics'),error=result.get('error'))
            report['new_policy_queries']+=result.get('new_policy_queries',0)
            if row['status']=='completed':completed[case['name']]=out
            save()
        for task in (0,2):
            for state in (0,1,2):
                prefix=f't{task}-s{state}';names=[prefix+'-'+x for x in ('model-live','model-replay','open-replay')]
                if all(name in completed for name in names):
                    results=[read(completed[name]/'episode/report.json') for name in names]
                    pair=compare_pair(*results);pair['name']=prefix
                    configurations=[read(completed[name]/'report.json') for name in names]
                    pair['source_configuration_equal']=all(all(c[key]==configurations[0][key] for key in
                        ('prompt','init_sha256','bddl_sha256','initial_state_sha256','target','controller','control_frequency_hz')) for c in configurations[1:])
                    pair['physical_reproduction_passed'] &= pair['source_configuration_equal']
                    pair['normal_replay_all_query_inputs_equal']=all(x['exact_input_match'] for x in read(completed[names[1]]/'report.json')['replay_input_checks'])
                    report['pairs'].append(pair)
        passed=all(x['status']=='completed' and x['metrics']['gripper_command_available_controls']==x['metrics']['controls'] for x in report['cases']) and len(report['pairs'])==6 and all(x['physical_reproduction_passed'] for x in report['pairs'])
        report['status']='completed' if passed else 'needs-inspection'
    except KeyboardInterrupt:report.update(status='interrupted',error='KeyboardInterrupt')
    except Exception as error:report.update(status='failed',error=type(error).__name__+': '+str(error))
    finally:save();print('[Native gripper]',report['status'],folder/'run.json',flush=True)
    return 0 if report['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
