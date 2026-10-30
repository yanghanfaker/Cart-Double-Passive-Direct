"""Create a separate static LIBERO reference venv, then run its geometry capture."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

SNAPSHOT="""import importlib.metadata,json,platform,sys
print(json.dumps({'python':sys.executable,'version':list(sys.version_info[:3]),'machine':platform.machine(),'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']}}))
"""


def environment_paths(root):
    root=Path(root).expanduser().resolve()
    base=root/'.venv';parent=root/'examples/libero';target=parent/'.venv'
    if target.resolve()==base.resolve() or target.resolve().parent!=parent.resolve():
        raise ValueError('reference venv must remain under openpi/examples/libero and differ from model venv')
    marker=target/'.reference_audit_env.json'
    if target.exists() and not marker.is_file():raise ValueError(f'existing unmanaged reference environment: {target}; no files changed')
    return base/'bin/python',target,target/'bin/python',marker


def install_plan(root,python,requirements):
    root=Path(root)
    return [
      ['uv','pip','install','--python',str(python),'--requirement',str(requirements)],
      ['uv','pip','install','--python',str(python),'--constraint',str(requirements),'--editable',str(root/'packages/openpi-client')],
      ['uv','pip','install','--python',str(python),'--no-deps','--editable',str(root/'third_party/libero')]]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--openpi-root',type=Path,default=Path('/home/test/Robot/openpi'))
    parser.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parent.parent/'diagnostics/libero_reference_setup')
    parser.add_argument('--skip-capture',action='store_true')
    args=parser.parse_args();root=args.openpi_root.expanduser().resolve()
    base,target,python,marker=environment_paths(root)
    if not base.is_file():parser.error('existing OpenPI Python not found')
    for path in (root/'third_party/libero/libero/libero/__init__.py',root/'packages/openpi-client/pyproject.toml'):
        if not path.is_file():parser.error(f'required existing source not found: {path}')
    requirements=Path(__file__).with_name('libero_reference_runtime.txt').resolve()
    folder=args.output_root.expanduser().resolve()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');folder.mkdir(parents=True,exist_ok=False)
    (folder/'runtime_requirements.txt').write_bytes(requirements.read_bytes())
    env=os.environ.copy();env.pop('VIRTUAL_ENV',None);env.pop('PYTHONPATH',None)
    env['PYNPUT_BACKEND']='dummy';env['MUJOCO_GL']='egl';env['PYOPENGL_PLATFORM']='egl'
    report={'status':'starting','base_python':str(base),'reference_environment':str(target),'commands':[],
            'requirements_sha256':hashlib.sha256(requirements.read_bytes()).hexdigest(),
            'scope':'new venv only; pinned direct runtime dependencies, not a fully hashed training lockfile; no installs target model Python'}
    def save():(folder/'setup.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    def snapshot(path):
        result=subprocess.run([str(base),'-c',SNAPSHOT],env=env,capture_output=True,text=True,check=True)
        value=json.loads(result.stdout);path.write_text(json.dumps(value,indent=2),encoding='utf-8');return value
    def command(args,label):
        item={'label':label,'command':args};report['commands'].append(item);save()
        print(f'[LIBERO setup] {label}',flush=True)
        with (folder/f'{label}.log').open('w',encoding='utf-8') as log:
            with subprocess.Popen(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace') as child:
                for line in child.stdout:log.write(line);print(line,end='',flush=True)
                item['exit_code']=child.wait()
        save();return item['exit_code']
    save();before=None
    try:
        before=snapshot(folder/'model_environment_before.json')
        if before['version'][:2]!=[3,11]:raise RuntimeError('this reference recipe requires the observed Python 3.11 base')
        if not target.exists():
            if command(['uv','venv','--python',str(base),str(target)],'create_venv')!=0:raise RuntimeError('venv creation failed')
            marker.write_text(json.dumps({'purpose':'static LIBERO reference','base_python':str(base)},indent=2),encoding='utf-8')
        for index,args_list in enumerate(install_plan(root,python,requirements),1):
            if command(args_list,f'install_{index}')!=0:raise RuntimeError(f'install stage {index} failed; inspect logs')
        report['status']='environment_ready';save()
        if not args.skip_capture:
            capture_root=folder/'capture'
            code=command([sys.executable,str(Path(__file__).with_name('libero_reference_audit.py')),
                          '--openpi-root',str(root),'--python',str(python),'--output-root',str(capture_root)],'capture_reference')
            reports=sorted(capture_root.glob('*/run.json'))
            report['reference_run_report']=str(reports[-1]) if reports else None
            report['reference_status']=json.loads(reports[-1].read_text()).get('status') if reports else 'missing_report'
            success=code==0 and report['reference_status']=='completed'
            report['status']='completed' if success else 'environment_ready_capture_failed'
            return 0 if success else 1
        return 0
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));return 1
    finally:
        if before is not None:
            try:
                after=snapshot(folder/'model_environment_after.json')
                report['model_environment_versions_unchanged']=before==after
            except Exception as error:report['model_environment_comparison_error']=str(error)
        save();print(f'[LIBERO setup] {report["status"]}; report: {folder/"setup.json"}',flush=True)


if __name__=='__main__':raise SystemExit(main())
