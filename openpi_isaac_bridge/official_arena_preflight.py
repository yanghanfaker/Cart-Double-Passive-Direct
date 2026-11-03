"""Record prerequisites for the pinned, unmodified official Arena baseline.

This command neither installs dependencies nor launches a simulation or a model.
An ARM port is deliberately not labeled an official baseline.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys


MANIFEST = Path(__file__).with_name('official_arena_manifest.json')


def platform_supported(system, machine):
    return system == 'Linux' and machine.lower() == 'x86_64'


def command(args, cwd=None):
    try:
        completed = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, timeout=45, check=False)
        return {'status': 'completed', 'returncode': completed.returncode,
                'command': [str(x) for x in args], 'output': completed.stdout[-24000:]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'status': 'unavailable', 'command': [str(x) for x in args],
                'reason': type(error).__name__ + ': ' + str(error)}


def verify_source(root, manifest):
    root = Path(root)
    result = {'root': str(root), 'expected_commit': manifest['commit'], 'files': {}, 'problems': []}
    if not root.is_dir():
        result['problems'].append('Arena checkout is missing')
        return result
    head = command(['git', 'rev-parse', 'HEAD'], root)
    result['git_head'] = head
    if head.get('returncode') != 0 or head.get('output', '').strip() != manifest['commit']:
        result['problems'].append('Arena HEAD differs from pinned official commit')
    dirty = command(['git', 'status', '--porcelain', '--untracked-files=no'], root)
    result['tracked_changes'] = dirty
    if dirty.get('returncode') != 0 or dirty.get('output', '').strip():
        result['problems'].append('Tracked Arena source must be unchanged')
    for name, expected in manifest['files'].items():
        path = root / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        result['files'][name] = {'sha256': actual, 'matches': actual == expected['sha256']}
        if actual != expected['sha256']:
            result['problems'].append('Source file missing or modified: ' + name)
    result['submodules'] = command(['git', 'submodule', 'status', '--recursive'], root)
    submodules = result['submodules']
    if submodules.get('returncode') != 0 or any(line.startswith(('-', '+', 'U')) for line in submodules.get('output', '').splitlines()):
        result['problems'].append('Initialize submodules at their pinned revisions')
    return result


def build_report(arena_root=None):
    manifest = json.loads(MANIFEST.read_bytes())
    supported = platform_supported(platform.system(), platform.machine())
    report = {'status': 'environment-inspection', 'simulation_executed': False,
              'model_inference_executed': False, 'grasp_success': None,
              'platform': {'system': platform.system(), 'machine': platform.machine(),
                           'supported_by_pinned_arena': supported},
              'python': {'executable': sys.executable, 'version': platform.python_version()},
              'official_source': manifest, 'problems': [],
              'disk_free_bytes': shutil.disk_usage(Path.cwd()).free,
              'scope': 'prerequisites only; not runtime, inference or grasp validation'}
    report['tools'] = {name: shutil.which(name) for name in ('git', 'uv', 'docker', 'nvidia-smi')}
    for tool, args in (('nvidia-smi', ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader']),
                       ('uv', ['uv', '--version']), ('docker', ['docker', 'version', '--format', '{{.Server.Version}}'])):
        report[tool] = command(args) if report['tools'][tool] else {'status': 'unavailable', 'reason': 'not on PATH'}
    report['installed_versions_in_this_python'] = {}
    for name in ('isaaclab', 'isaacsim', 'isaaclab_arena', 'torch', 'openpi-client'):
        try: report['installed_versions_in_this_python'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: report['installed_versions_in_this_python'][name] = None
    if not supported:
        report['problems'].append('Pinned official Arena supports Linux x86_64 only; do not uv sync or emulate its GPU stack on this host')
        report['status'] = 'blocked-platform'
    if arena_root is not None:
        report['checkout'] = verify_source(arena_root, manifest)
        report['problems'].extend(report['checkout']['problems'])
    else:
        report['problems'].append('Arena checkout not inspected; pass --arena-root after cloning the pinned revision')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arena-root', type=Path)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    output = args.output_root or Path(__file__).resolve().parent.parent / 'diagnostics/official_arena_baseline'
    folder = output / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder.mkdir(parents=True, exist_ok=False)
    report = build_report(args.arena_root)
    report['preflight_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['manifest_sha256'] = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    (folder / 'preflight.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('[Official Arena]', report['status'], report['platform']['system'], report['platform']['machine'])
    for issue in report['problems']: print(' -', issue)
    print('Report:', folder / 'preflight.json')
    return 2 if report['problems'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
