"""Three native LIBERO policy rollouts, with exact query and action provenance."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import zipfile

import numpy as np
from PIL import Image
from frozen_coordinate_probe import digest_observation
from openpi_health import check_openpi_health
from openpi_step_probe import read_server_metadata, validate_action_chunk


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def policy_observation(obs, prompt, image_tools):
    """Official example conventions; retain native state dtype and signed fingers."""
    q = np.asarray(obs['robot0_eef_quat']).copy()
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > .001:
        raise ValueError('invalid native xyzw quaternion')
    q[3] = np.clip(q[3], -1, 1)
    den = np.sqrt(1-q[3]*q[3])
    rv = np.zeros(3) if math.isclose(den, 0.) else q[:3]*2*math.acos(q[3])/den
    state = np.concatenate((obs['robot0_eef_pos'], rv, obs['robot0_gripper_qpos']))
    if state.shape != (8,) or not np.isfinite(state).all(): raise ValueError('invalid native state')
    result = {'observation/state': state, 'prompt': prompt}
    for source, target in (('agentview_image', 'observation/image'), ('robot0_eye_in_hand_image', 'observation/wrist_image')):
        raw = np.asarray(obs[source])
        if raw.shape != (256, 256, 3) or raw.dtype != np.uint8: raise ValueError('expected 256 RGB')
        result[target] = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(raw[::-1, ::-1]), 224, 224))
    return result


def snapshot(obs, env):
    keys = ('robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_joint_pos', 'robot0_joint_vel')
    data = {k: np.asarray(obs[k]).tolist() for k in keys if k in obs}
    # Use returned observables without an extra forward()/refresh, matching example timing.
    data['object_observations'] = {k: np.asarray(v).tolist() for k, v in obs.items()
                                   if not k.startswith('robot') and (k.endswith('_pos') or k.endswith('_quat'))}
    data['sim_time_s'] = float(env.sim.data.time)
    if not np.isfinite(np.asarray(data['robot0_eef_pos'])).all(): raise ValueError('nonfinite pose')
    return data


def controller_record(controller):
    # robosuite versions do not expose all constructor arguments as attributes.
    return {k: np.asarray(getattr(controller, k, None)).tolist() for k in
            ('name', 'input_min', 'input_max', 'output_min', 'output_max', 'kp', 'kd',
             'damping_ratio', 'control_delta', 'use_delta', 'policy_freq')}


def rollout(env, client, obs, prompt, folder, image_tools, max_steps=220):
    folder.mkdir(); frames = []
    report = {'status': 'running', 'success': False, 'stop_reason': None, 'policy_queries': 0,
              'inference_attempts': 0, 'action_steps': 0, 'step_attempts': 0, 'queries': [], 'trace': [],
              'initial': snapshot(obs, env), 'replan_steps': 5, 'budget': max_steps,
              'action_semantics': 'raw server actions passed directly to native env.step; native controller clipping/scaling applies; no Isaac bridge caps or forced-open gripper'}
    def save(): save_json(folder/'report.json', report)
    save(); chunk = []; query = None
    try:
        with zipfile.ZipFile(folder/'queries.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for step in range(max_steps):
                if not chunk:
                    element = policy_observation(obs, prompt, image_tools)
                    number = report['policy_queries']+1; prefix = f'query-{number:03d}'
                    fingerprint = digest_observation(element)
                    for key, name in (('observation/image', 'external'), ('observation/wrist_image', 'wrist')):
                        buf = io.BytesIO(); Image.fromarray(element[key]).save(buf, format='PNG')
                        archive.writestr(f'{prefix}/{name}.png', buf.getvalue())
                    buf = io.BytesIO(); np.save(buf, element['observation/state'], allow_pickle=False)
                    archive.writestr(f'{prefix}/state.npy', buf.getvalue())
                    request = {'input_sha256': fingerprint, 'state_dtype': str(element['observation/state'].dtype),
                               'state': element['observation/state'].tolist(), 'prompt': prompt, 'before_step': step+1}
                    archive.writestr(f'{prefix}/request.json', json.dumps(request, allow_nan=False))
                    query = dict(request, query=number, status='querying', executed_prefix_steps=0)
                    report['queries'].append(query); report['inference_attempts'] += 1; save()
                    start = time.perf_counter(); response = client.infer(element)
                    query['latency_s'] = time.perf_counter()-start; report['policy_queries'] += 1
                    actions = response['actions']; actions = actions.tolist() if hasattr(actions, 'tolist') else actions
                    archive.writestr(f'{prefix}/response.txt', repr(actions))
                    validate_action_chunk(actions)
                    if len(actions) < 5: raise ValueError('native prefix requires >=5 actions')
                    if digest_observation(element) != fingerprint: raise ValueError('client mutated input')
                    payload = json.dumps({'actions': actions}, allow_nan=False).encode()
                    archive.writestr(f'{prefix}/response.json', payload)
                    query.update(status='validated', response_sha256=hashlib.sha256(payload).hexdigest(), predicted_steps=len(actions))
                    chunk = [list(a) for a in actions[:5]]; save()
                    print(f'[Native LIBERO] {folder.name} query={number}, action step={step+1}', flush=True)
                before = snapshot(obs, env); action = chunk.pop(0)
                report['step_attempts'] += 1; report['pending_action'] = {'step': step+1, 'action': action}; save()
                before_hook=getattr(client,'before_step',None)
                if callable(before_hook):before_hook(env)
                obs, reward, done, _ = env.step(action)
                report['action_steps'] += 1; query['executed_prefix_steps'] += 1; report.pop('pending_action')
                success = bool(env.check_success())
                report['trace'].append({'step': step+1, 'query': query['query'], 'prefix_index': query['executed_prefix_steps'],
                    'action': action, 'close_requested': action[6] > 0, 'before': before, 'after': snapshot(obs, env),
                    'reward': float(reward), 'done': bool(done), 'task_success': success})
                after_hook=getattr(client,'observe_step',None)
                if callable(after_hook):report['trace'][-1]['gripper_diagnostic']=after_hook(env)
                if step % 2 == 0 or success or done:
                    external = np.asarray(obs['agentview_image'])[::-1, ::-1]
                    wrist = np.asarray(obs['robot0_eye_in_hand_image'])[::-1, ::-1]
                    frames.append(Image.fromarray(np.concatenate((external, wrist), axis=1)).resize((448, 224)))
                save()
                if success or done:
                    report.update(success=success, stop_reason='task_success' if success else 'done_without_success'); break
            else: report['stop_reason'] = 'budget_exhausted'
        report['status'] = 'completed'
    except Exception as error:
        if query and query['status'] == 'querying': query.update(status='failed', error=str(error))
        report.update(status='failed', error_type=type(error).__name__, error=str(error)); raise
    finally:
        try: report['final'] = snapshot(obs, env)
        except Exception as error: report['final_snapshot_error'] = str(error)
        if (folder/'queries.zip').is_file(): report['query_archive_sha256'] = hashlib.sha256((folder/'queries.zip').read_bytes()).hexdigest()
        try:
            if frames:
                frames[0].save(folder/'preview.gif', save_all=True, append_images=frames[1:], duration=100, loop=0)
                report['preview'] = 'preview.gif'
                report['preview_note'] = 'two views, every second 20Hz control step, nominal 10fps; last success/done frame may be closer'
        except Exception as error: report['preview_error'] = str(error)
        save()
    return report


def worker(folder, args, client_factory=None):
    report = {'status': 'starting', 'episodes': [], 'suite': 'libero_spatial', 'task_id': args.task_id,
              'seed': args.seed, 'scope': 'native task baseline, not matched Isaac cube scene or full benchmark; local seed does not seed server inference'}
    def save(): save_json(folder/'report.json', report)
    save(); env = None
    try:
        spec = importlib.util.find_spec('libero')
        root = next(Path(p)/'libero' for p in spec.submodule_search_locations if (Path(p)/'libero/assets').is_dir())
        config = folder/'libero_config'; config.mkdir()
        save_json(config/'config.yaml', {'benchmark_root': str(root), 'bddl_files': str(root/'bddl_files'),
            'assets': str(root/'assets'), 'init_states': str(root/'init_files'), 'datasets': str(root.parent/'datasets')})
        os.environ['LIBERO_CONFIG_PATH'] = str(config)
        import torch
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from openpi_client import image_tools
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        from libero_scene_probe import select_state
        np.random.seed(args.seed)
        suite = benchmark.get_benchmark_dict()['libero_spatial']()
        if not 0 <= args.task_id < suite.n_tasks: raise ValueError('task id outside suite')
        task = suite.get_task(args.task_id)
        bddl = Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
        init = Path(get_libero_path('init_states'))/task.problem_folder/task.init_states_file
        report.update(prompt=str(task.language), bddl_sha256=hashlib.sha256(bddl.read_bytes()).hexdigest(),
                      init_sha256=hashlib.sha256(init.read_bytes()).hexdigest())
        (folder/'task.bddl').write_bytes(bddl.read_bytes())
        allowed = [np.core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype('float64')), type(np.dtype('float32'))]
        with torch.serialization.safe_globals(allowed): states = torch.load(init, map_location='cpu', weights_only=True)
        if len(states) < 3: raise ValueError('need at least three benchmark initial states')
        report['initialization'] = 'benchmark states 0,1,2; restricted torch loader; ten dummy wait steps each'
        report['server_health'] = check_openpi_health(args.server_host, args.server_port) if client_factory is None else {'status': 'offline_replay_no_network'}
        client = WebsocketClientPolicy(host=args.server_host, port=args.server_port) if client_factory is None else client_factory()
        report['server_metadata'] = read_server_metadata(client)
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        env.seed(args.seed)
        if not math.isclose(float(env.env.control_freq), 20.): raise ValueError('expected native 20 Hz controller')
        controller = env.robots[0].controller
        report['controller'] = controller_record(controller)
        import inspect
        (folder/'controller_implementation.py').write_text(inspect.getsource(type(controller)), encoding='utf-8')
        report['control_frequency_hz'] = float(env.env.control_freq)
        (folder/'scene.xml').write_text(env.env.model.get_xml(), encoding='utf-8')
        official = args.openpi_root/'examples/libero/main.py'
        if official.is_file():
            data = official.read_bytes(); (folder/'installed_openpi_libero_example.py').write_bytes(data)
            report['installed_example_sha256'] = hashlib.sha256(data).hexdigest()
        report['status'] = 'running'; save()
        for index in range(3):
            row = {'episode_index': index, 'status': 'initializing', 'dummy_wait_steps': 0}; report['episodes'].append(row); save()
            env.reset(); initial = select_state(states, index, len(env.get_sim_state()))
            np.save(folder/f'initial_state_{index}.npy', initial, allow_pickle=False)
            obs = env.set_init_state(initial)
            for _ in range(10):
                obs, _, done, _ = env.step([0.]*6+[-1.]); row['dummy_wait_steps'] += 1
                if done or env.check_success(): raise RuntimeError('task already terminal during initial wait')
            row.update(status='running', result=f'episode-{index:02d}/report.json'); save()
            result = rollout(env, client, obs, str(task.language), folder/f'episode-{index:02d}', image_tools)
            row.update(status=result['status'], success=result['success'], action_steps=result['action_steps'], policy_queries=result['policy_queries'], stop_reason=result['stop_reason']); save()
        report.update(status='completed', successes=sum(e['success'] for e in report['episodes']), completed_episodes=3)
        return 0
    except Exception as error:
        for row in report['episodes']:
            if row['status'] in ('initializing', 'running'): row.update(status='failed', error=str(error))
        report.update(status='failed', error_type=type(error).__name__, error=str(error)); print(f'[Native LIBERO] STOP: {error}', flush=True); return 1
    finally:
        save()
        if env is not None: env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--openpi-root', type=Path, default=Path('/home/test/Robot/openpi'))
    p.add_argument('--output-root', type=Path, default=Path(__file__).resolve().parent.parent/'diagnostics/native_libero_rollout')
    p.add_argument('--server-host', default='127.0.0.1'); p.add_argument('--server-port', type=int, default=8000)
    p.add_argument('--task-id', type=int, default=0); p.add_argument('--seed', type=int, default=7)
    p.add_argument('--worker', type=Path, help=argparse.SUPPRESS); args = p.parse_args()
    if args.worker: return worker(args.worker, args)
    folder = args.output_root.absolute()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'); folder.mkdir(parents=True)
    report = {'status': 'preflight'}
    try:
        from run_multipose_geometry import preflight_reference
        python = (args.openpi_root/'examples/libero/.venv/bin/python').absolute()
        report['preflight'] = preflight_reference(python, args.openpi_root.absolute(), folder)
        command = [str(python), str(Path(__file__).absolute()), '--worker', str(folder), '--openpi-root', str(args.openpi_root.absolute()),
                   '--server-host', args.server_host, '--server-port', str(args.server_port), '--task-id', str(args.task_id), '--seed', str(args.seed)]
        report.update(status='running', command=command); save_json(folder/'run.json', report)
        env = os.environ.copy(); env.pop('VIRTUAL_ENV', None)
        env.update(PYTHONPATH=str(args.openpi_root/'third_party/libero'), MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', PYNPUT_BACKEND='dummy')
        print(f'[Native LIBERO] Output: {folder}; monitor worker.log', flush=True)
        with (folder/'worker.log').open('w', encoding='utf-8') as log:
            with subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True) as process:
                try: code = process.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait(); raise RuntimeError('native worker exceeded 30 minutes')
        report['worker_exit_code'] = code
        result = json.loads((folder/'report.json').read_bytes())
        if code or result['status'] != 'completed': raise RuntimeError('native rollout failed; inspect report.json and worker.log')
        report.update(status='completed', successes=result['successes'], completed_episodes=result['completed_episodes']); return 0
    except Exception as error:
        report.update(status='failed', error=str(error)); return 1
    finally:
        save_json(folder/'run.json', report); print(f"[Native LIBERO] {report['status']}: {folder/'run.json'}", flush=True)


if __name__ == '__main__': raise SystemExit(main())
