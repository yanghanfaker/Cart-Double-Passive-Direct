"""Run the byte-identical official client with verified NumPy init-state support.

Keep PyTorch weights_only=True. Only the existing NumPy array/dtype allowlist is
scoped around preflight and the upstream entrypoint; no torch.load monkeypatch.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import runpy
import sys


STATE_MANIFEST = Path(__file__).with_name('official_libero_initial_states.json')
CLIENT_MANIFEST = Path(__file__).with_name('official_libero_manifest.json')


def verify_init_bytes(data, expected):
    if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
        raise ValueError('Official LIBERO initial-state bytes differ from pinned reference')
    return data


def snapshot_states(libero_root, destination, manifest):
    if len(manifest['files']) != 10: raise ValueError('Exactly ten spatial initial-state files required')
    records=[]
    for name, expected in sorted(manifest['files'].items()):
        relative=PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or len(relative.parts)!=2 or relative.parts[0]!='libero_spatial' or relative.suffix!='.pruned_init':
            raise ValueError('Invalid initial-state manifest path')
        source=Path(libero_root)/'init_files'/name
        data=verify_init_bytes(source.read_bytes(),expected)
        target=Path(destination)/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        records.append({'path':name,**expected})
    return records


def allowed_numpy_types(np):
    return [np.core.multiarray._reconstruct,np.ndarray,np.dtype,type(np.dtype('float64')),type(np.dtype('float32'))]


def check_states(destination, records, torch, np):
    loaded=[]
    with torch.serialization.safe_globals(allowed_numpy_types(np)):
        for record in records:
            data=verify_init_bytes((Path(destination)/record['path']).read_bytes(),record)
            try:states=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
            except Exception as error:raise RuntimeError('Cannot load '+record['path']+': '+type(error).__name__+': '+str(error)) from error
            array=np.asarray(states)
            if array.ndim!=2 or array.shape[0]<1 or array.shape[1]<1 or array.dtype.kind!='f' or not np.isfinite(array).all():
                raise ValueError('Expected finite 2D floating-point benchmark states: '+record['path'])
            loaded.append({**record,'shape':list(array.shape),'dtype':str(array.dtype),
                           'first_state_sha256':hashlib.sha256(np.ascontiguousarray(array[0]).tobytes()).hexdigest()})
    return loaded


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--libero-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('official_args',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    report={'status':'starting','preflight_only':args.preflight_only,'weights_only':True,
            'scope':'byte-identical upstream client with scoped NumPy init-state compatibility; original actions/controller untouched',
            'launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def save():
        report['updated_utc']=datetime.now(timezone.utc).isoformat()
        (args.output/'init_state_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    save()
    try:
        os.environ.pop('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD',None)
        os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD']='1'
        manifest=json.loads(STATE_MANIFEST.read_bytes())
        report['manifest_sha256']=hashlib.sha256(STATE_MANIFEST.read_bytes()).hexdigest()
        report['reference_commit']=manifest['commit']
        destination=args.output/'verified_init_states'
        records=snapshot_states(args.libero_root,destination,manifest)
        import numpy as np
        import torch
        report['torch_version']=torch.__version__
        report['allowed_globals']=[x.__module__+'.'+x.__name__ for x in allowed_numpy_types(np)]
        report['files']=check_states(destination,records,torch,np)
        config_dir=Path(os.environ['LIBERO_CONFIG_PATH'])
        config_path=config_dir/'config.yaml'
        config=json.loads(config_path.read_bytes())
        config['init_states']=str(destination)
        config_path.write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
        report['config_init_states']=str(destination)
        report['status']='states-validated';save()
        if args.preflight_only:return 0
        if args.source is None:raise ValueError('--source required for official rollout')
        client=json.loads(CLIENT_MANIFEST.read_bytes())
        if hashlib.sha256(args.source.read_bytes()).hexdigest()!=client['sha256']:
            raise ValueError('Official entrypoint hash mismatch')
        extra=args.official_args
        if extra and extra[0]=='--':extra=extra[1:]
        sys.argv=[str(args.source),*extra]
        report['status']='official-client-running';save()
        with torch.serialization.safe_globals(allowed_numpy_types(np)):
            runpy.run_path(str(args.source),run_name='__main__')
        report['status']='official-client-returned';save()
        return 0
    except BaseException as error:
        # Upstream Tyro may use SystemExit; distinguish a normal zero exit.
        if isinstance(error,SystemExit) and error.code in (None,0):
            report['status']='official-client-returned';save();return 0
        report.update(status='failed',error=type(error).__name__+': '+str(error));save();raise


if __name__=='__main__':raise SystemExit(main())
