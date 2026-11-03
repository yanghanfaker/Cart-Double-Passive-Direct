"""Launch a SHA-verified, unmodified upstream LIBERO evaluation on Spark.

Uses the existing LIBERO client venv and existing pi05_libero server. No Isaac
bridge, forced-open fingers, custom motion caps, or patched upstream rollout.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import urllib.request
from official_arena_preflight import command
from openpi_health import check_openpi_health


MANIFEST = Path(__file__).with_name('official_libero_manifest.json')
RUNTIME_PROBE = r'''
import importlib.metadata,importlib.util,json
from pathlib import Path
import numpy,imageio,tyro,tqdm,mujoco
from openpi_client import image_tools,websocket_client_policy
spec=importlib.util.find_spec('libero')
if spec is None:raise RuntimeError('LIBERO package missing in this Python')
root=next(Path(p)/'libero' for p in spec.submodule_search_locations if (Path(p)/'libero/assets').is_dir())
versions={}
for name in ('numpy','mujoco','robosuite','libero','openpi-client','tyro','imageio','torch','websockets'):
    try:versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:versions[name]=None
print(json.dumps({'libero_root':str(root),'versions':versions,'openpi_client_module':str(Path(image_tools.__file__))}))
'''


def verified_source(data, manifest):
    if hashlib.sha256(data).hexdigest() != manifest['sha256']:
        raise ValueError('Official example SHA-256 mismatch; refusing to run changed source')
    return data


def parse_log(text, expected_episodes=10):
    errors = [line for line in text.splitlines() if 'Caught exception:' in line or 'Traceback (most recent call last)' in line]
    flags = re.findall(r'^INFO:root:Success: (True|False)\s*$', text, re.MULTILINE)
    totals = re.findall(r'^INFO:root:Total episodes: (\d+)\s*$', text, re.MULTILINE)
    rates = re.findall(r'^INFO:root:Total success rate: ([0-9.eE+-]+)\s*$', text, re.MULTILINE)
    success_count = flags.count('True')
    rate = success_count / len(flags) if flags else None
    consistent = (len(totals) == 1 and int(totals[0]) == expected_episodes
                  and len(flags) == expected_episodes and len(rates) == 1
                  and rate is not None and abs(float(rates[0]) - rate) < 1e-10)
    return {'episode_flags': [x == 'True' for x in flags], 'episodes': len(flags),
            'successes': success_count, 'success_rate': rate, 'caught_errors': errors,
            'complete': bool(consistent and not errors),
            'scope': 'upstream done-based success reports; ten-episode smoke test, not full benchmark reproduction'}


def child_command(python, source, output, seed=7):
    return [str(python), '-u', str(source), '--args.host', '127.0.0.1', '--args.port', '8000',
            '--args.task-suite-name', 'libero_spatial', '--args.num-trials-per-task', '1',
            '--args.replan-steps', '5', '--args.seed', str(seed), '--args.video-out-path', str(output)]


def server_candidates():
    result=[]
    proc=Path('/proc')
    if not proc.is_dir():return result
    for path in proc.iterdir():
        if not path.name.isdigit():continue
        try:
            if path.stat().st_uid != os.getuid():continue
            args=(path/'cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
            if any('serve_policy.py' in x or x == 'openpi.serving.websocket_policy_server' for x in args):
                result.append({'pid':int(path.name),'argv':args})
        except (OSError,ProcessLookupError):pass
    return result


def inventory(folder):
    result=[]
    for path in sorted(folder.rglob('*')):
        if path.is_file() and path.name != 'run.json':
            digest_object=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):digest_object.update(block)
            digest=digest_object.hexdigest()
            result.append({'path':path.relative_to(folder).as_posix(),'bytes':path.stat().st_size,'sha256':digest})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--openpi-root',type=Path,default=Path('/home/test/Robot/openpi'))
    parser.add_argument('--client-python',type=Path)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--timeout',type=int,default=3600)
    args=parser.parse_args()
    if args.timeout < 60:parser.error('--timeout must be at least 60 seconds')
    root=args.openpi_root.absolute()
    # Do not resolve this symlink: the venv's Python must preserve its environment.
    python=args.client_python or root/'examples/libero/.venv/bin/python'
    output=args.output_root or Path(__file__).resolve().parent.parent/'diagnostics/official_libero_baseline'
    folder=output.absolute()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(MANIFEST.read_bytes())
    report={'status':'preflight','source':manifest,'platform':{'system':platform.system(),'machine':platform.machine()},
            'seed':args.seed,'server':{'host':'127.0.0.1','port':8000,'expected_config':'pi05_libero',
            'identity_status':'unverified; health and endpoint are not proof of loaded weights',
            'candidate_processes':server_candidates()},'simulation_launch_attempted':False,'metrics':None,
            'scope':'unmodified official LIBERO client on existing ARM-compatible dependencies; not official Arena/Isaac baseline',
            'comparison_limit':'different scene and physics from Isaac cube proxy; success would establish native model operation, not isolate a specific bridge bug',
            'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def save():
        (folder/'run.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    save()
    try:
        if platform.system()!='Linux':raise RuntimeError('Run on the Linux Spark host with the existing LIBERO venv')
        if not python.is_file():raise RuntimeError('Missing LIBERO client Python: '+str(python))
        report['openpi_checkout']={'head':command(['git','rev-parse','HEAD'],root),
                                   'tracked_changes':command(['git','status','--porcelain','--untracked-files=no'],root)}
        report['server']['health']=check_openpi_health()
        env=os.environ.copy();env.pop('VIRTUAL_ENV',None);env.pop('PYTHONHOME',None)
        env['PYTHONPATH']=str(root/'third_party/libero')
        env['MUJOCO_GL']=env.get('MUJOCO_GL','egl');env['PYTHONUNBUFFERED']='1'
        report['MUJOCO_GL']=env['MUJOCO_GL']
        probe=subprocess.run([str(python),'-c',RUNTIME_PROBE],cwd=root,env=env,text=True,
                             stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
        (folder/'client_probe.log').write_text(probe.stdout,encoding='utf-8')
        if probe.returncode:raise RuntimeError('Client dependencies failed; inspect client_probe.log')
        runtime=json.loads(probe.stdout.strip().splitlines()[-1]);report['runtime']=runtime
        libero=Path(runtime['libero_root'])
        report['libero_checkout']={'head':command(['git','rev-parse','HEAD'],libero),
                                  'tracked_changes':command(['git','status','--porcelain','--untracked-files=no'],libero)}
        report['libero_checkout']['reference_submodule_commit']=manifest['reference_libero_submodule_commit']
        report['libero_checkout']['matches_reference_revision']=report['libero_checkout']['head'].get('output','').strip()==manifest['reference_libero_submodule_commit']
        for directory in ('assets','bddl_files','init_files'):
            if not (libero/directory).is_dir():raise RuntimeError('LIBERO data directory missing: '+directory)
        config=folder/'libero_config';config.mkdir()
        (config/'config.yaml').write_text(json.dumps({'benchmark_root':str(libero),'bddl_files':str(libero/'bddl_files'),
            'assets':str(libero/'assets'),'init_states':str(libero/'init_files'),'datasets':str(libero.parent/'datasets')},indent=2)+'\n',encoding='utf-8')
        env['LIBERO_CONFIG_PATH']=str(config)
        with urllib.request.urlopen(manifest['source_url'],timeout=45) as response:
            data=verified_source(response.read(),manifest)
        source=folder/'official_main.py';source.write_bytes(data)
        report['command']=child_command(python,source,folder/'videos',args.seed)
        report.update(status='running',simulation_launch_attempted=True);save()
        print('[Official LIBERO] Ten native tasks, one episode each, prefix=5; log:',folder/'console.log',flush=True)
        with (folder/'console.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen(report['command'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,text=True)
            try:report['returncode']=process.wait(timeout=args.timeout)
            except (subprocess.TimeoutExpired,KeyboardInterrupt):
                process.terminate()
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                raise
        report['metrics']=parse_log((folder/'console.log').read_text(encoding='utf-8',errors='replace'))
        videos=list((folder/'videos').glob('*.mp4'))
        report['video_files']=len(videos)
        valid_videos=len(videos)==manifest['expected_episodes'] and all(p.stat().st_size>0 for p in videos)
        report['status']='completed' if report['returncode']==0 and report['metrics']['complete'] and valid_videos else 'needs-inspection'
        # Even a completed zero-success run is a valid negative baseline, not a successful grasp.
        report['model_inference_count']='unavailable; unmodified upstream client does not archive queries'
    except KeyboardInterrupt:report.update(status='interrupted',error='KeyboardInterrupt')
    except Exception as error:report.update(status='failed',error=type(error).__name__+': '+str(error))
    finally:
        report['artifacts']=inventory(folder)
        report['large_files_over_95MiB']=[x['path'] for x in report['artifacts'] if x['bytes']>95*1024*1024]
        save()
    print('[Official LIBERO]',report['status'],'metrics:',report['metrics'],flush=True)
    print('Report:',folder/'run.json',flush=True)
    return 0 if report['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
