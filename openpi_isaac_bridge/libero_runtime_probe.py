"""Inspect every official LIBERO client dependency before creating an environment."""
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess


MODULES = ('numpy', 'imageio', 'imageio_ffmpeg', 'tyro', 'tqdm', 'mujoco',
           'openpi_client.image_tools', 'openpi_client.websocket_client_policy')


def inspect_runtime(importer=importlib.import_module):
    modules = {}; failures = {}; loaded = {}
    for name in MODULES:
        try:
            module = importer(name); loaded[name] = module
            modules[name] = {'status': 'available', 'file': str(getattr(module, '__file__', 'unknown'))}
        except Exception as error:
            reason = type(error).__name__ + ': ' + str(error)
            modules[name] = {'status': 'unavailable', 'reason': reason}; failures[name] = reason
    result = {'modules': modules, 'failures': failures, 'versions': {}, 'libero_root': None}
    for name in ('numpy', 'mujoco', 'robosuite', 'libero', 'openpi-client', 'tyro', 'imageio', 'imageio-ffmpeg', 'torch', 'websockets'):
        try: result['versions'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result['versions'][name] = None
    try:
        spec = importlib.util.find_spec('libero')
        if spec is None: raise RuntimeError('LIBERO package missing in this Python')
        root = next((Path(p)/'libero' for p in spec.submodule_search_locations if (Path(p)/'libero/assets').is_dir()), None)
        if root is None: raise RuntimeError('LIBERO asset root could not be located')
        result['libero_root'] = str(root)
    except Exception as error: failures['libero_assets'] = type(error).__name__ + ': ' + str(error)
    if 'openpi_client.image_tools' in loaded:
        result['openpi_client_module'] = str(loaded['openpi_client.image_tools'].__file__)
    if 'imageio_ffmpeg' in loaded:
        try:
            executable = loaded['imageio_ffmpeg'].get_ffmpeg_exe()
            response = subprocess.run([executable, '-version'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, timeout=10, check=False)
            if response.returncode: raise RuntimeError('ffmpeg -version returned '+str(response.returncode))
            result['ffmpeg'] = {'executable': executable, 'version': response.stdout.splitlines()[0]}
        except Exception as error: failures['ffmpeg_executable'] = type(error).__name__ + ': ' + str(error)
    return result


if __name__ == '__main__':
    print(json.dumps(inspect_runtime(), allow_nan=False))
