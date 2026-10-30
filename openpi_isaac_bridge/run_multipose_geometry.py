"""Capture eleven paired static Isaac/LIBERO poses and test held-out registration."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from libero_reference_audit import source_targets
from multipose_geometry import design_poses


def unique_completed(root,filename):
    files=list(root.glob('*/'+filename))
    if len(files)!=1:raise RuntimeError(f'expected one new {filename} under {root}')
    path=files[0]
    if json.loads(path.read_text())['status']!='completed':raise RuntimeError(f'incomplete report: {path}')
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    repo=Path(__file__).resolve().parent.parent
    parser.add_argument('--isaaclab-root',type=Path,default=Path('/home/test/Robot/IsaacLab'))
    parser.add_argument('--openpi-root',type=Path,default=Path('/home/test/Robot/openpi'))
    parser.add_argument('--source-audit',type=Path)
    parser.add_argument('--reference-python',type=Path)
    parser.add_argument('--output-root',type=Path,default=repo/'diagnostics/multipose_geometry')
    args=parser.parse_args()
    candidates=sorted((repo/'diagnostics/geometry_audit').glob('*/geometry_audit.json'))
    source=args.source_audit or next((p for p in reversed(candidates) if json.loads(p.read_text()).get('status')=='completed'),None)
    if source is None:parser.error('no completed original geometry audit; pass --source-audit')
    payload=source.read_bytes();home,_=source_targets(json.loads(payload));poses=design_poses(home)
    reference_python=(args.reference_python or args.openpi_root/'examples/libero/.venv/bin/python').resolve()
    if not reference_python.is_file():parser.error('existing isolated LIBERO interpreter missing')
    folder=args.output_root.resolve()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'design_source.json').write_bytes(payload)
    manifest={'status':'running','source_sha256':hashlib.sha256(payload).hexdigest(),
        'source_path':str(source),'policy_queries':0,'reference_python':str(reference_python),
        'poses':poses,'scope':'independent initializations; hold warmup in Isaac; matched actual qpos + forward only in LIBERO; no model actions or grasp tests'}
    def save():(folder/'run.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    def run(command,log):
        with log.open('w',encoding='utf-8') as stream:
            with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',stdin=subprocess.DEVNULL) as child:
                for line in child.stdout:
                    stream.write(line);stream.flush()
                    if any(tag in line for tag in ('[Geometry runner]','[LIBERO reference]','[Multi-pose]')):print(line,end='',flush=True)
                if child.wait()!=0:raise RuntimeError(f'subprocess failed; inspect {log}')
    save();print(f'[Multi-pose] Output: {folder}',flush=True)
    try:
        for item in poses:
            item['status']='running';save()
            target=folder/item['name'];target.mkdir()
            qfile=target/'requested_joint_positions.json';qfile.write_text(json.dumps(item['requested_joint_positions_rad']),encoding='utf-8')
            command=[sys.executable,str(Path(__file__).with_name('run_geometry_audit.py')),
                '--isaaclab-root',str(args.isaaclab_root.resolve()),'--output-root',str(target/'isaac'),
                '--geometry-joints-json',str(qfile)]
            item['isaac_command']=command
            print(f'[Multi-pose] {item["name"]} ({item["split"]}) Isaac capture',flush=True);save()
            run(command,target/'isaac_runner.log')
            audit=unique_completed(target/'isaac','geometry_audit.json')
            item['isaac_audit']=audit.relative_to(folder).as_posix()
            item['isaac_audit_sha256']=hashlib.sha256(audit.read_bytes()).hexdigest()
            command=[sys.executable,str(Path(__file__).with_name('libero_reference_audit.py')),
                '--openpi-root',str(args.openpi_root.resolve()),'--python',str(reference_python),
                '--isaac-audit',str(audit),'--output-root',str(target/'libero')]
            item['libero_command']=command;save()
            print(f'[Multi-pose] {item["name"]} LIBERO matched actual qpos',flush=True)
            run(command,target/'libero_runner.log')
            reference=unique_completed(target/'libero','reference.json')
            ref_run=json.loads((reference.parent/'run.json').read_text())
            if ref_run['status']!='completed' or ref_run['isaac_source_sha256']!=item['isaac_audit_sha256']:
                raise RuntimeError('reference missing or source hash mismatch')
            item['reference']=reference.relative_to(folder).as_posix()
            item['reference_sha256']=hashlib.sha256(reference.read_bytes()).hexdigest()
            item['status']='completed';save()
        run([str(reference_python),str(Path(__file__).with_name('multipose_geometry.py')),str(folder)],folder/'analysis.log')
        result=json.loads((folder/'analysis.json').read_text())
        if result['status']!='completed':raise RuntimeError('analysis incomplete')
        manifest['status']='completed'
        manifest['candidate_geometry_within_thresholds']=result['candidate_geometry_within_thresholds']
        return 0
    except Exception as error:
        manifest.update(status='failed',error_type=type(error).__name__,error=str(error))
        print(f'[Multi-pose] STOP: {error}',flush=True)
        return 1
    finally:
        save();print(f'[Multi-pose] Manifest: {folder/"run.json"}',flush=True)


if __name__=='__main__':raise SystemExit(main())
