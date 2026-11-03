import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from official_arena_preflight import MANIFEST, platform_supported, verify_source


class OfficialArenaPreflightTests(unittest.TestCase):
    def test_platform_gate(self):
        self.assertTrue(platform_supported('Linux', 'x86_64'))
        for system, machine in [('Linux', 'aarch64'), ('Linux', 'arm64'), ('Windows', 'AMD64'), ('Darwin', 'arm64')]:
            self.assertFalse(platform_supported(system, machine))

    def test_manifest_pins_model_and_sources(self):
        manifest = json.loads(MANIFEST.read_bytes())
        self.assertEqual(len(manifest['commit']), 40)
        self.assertEqual(manifest['openpi_commit'], 'c23745b5ad24e98f66967ea795a07b2588ed6c79')
        self.assertIn('isaaclab_arena_openpi/policy/droid_adapter.py', manifest['files'])
        self.assertIn('isaaclab_arena_environments/experiment_configs/droid_pnp_openpi_experiment.yaml', manifest['files'])

    def test_missing_checkout_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            result = verify_source(Path(folder) / 'missing', {'commit': 'a' * 40})
        self.assertIn('Arena checkout is missing', result['problems'])

    def test_source_and_submodule_integrity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.yaml'; path.write_bytes(b'official')
            manifest = {'commit': 'a' * 40, 'files': {'config.yaml': {'sha256': hashlib.sha256(b'official').hexdigest()}}}
            with patch('official_arena_preflight.command', side_effect=[{'returncode': 0, 'output': 'a' * 40}, {'returncode': 0, 'output': ''}, {'returncode': 0, 'output': ' abc submodule'}]):
                self.assertFalse(verify_source(folder, manifest)['problems'])
            path.write_bytes(b'modified')
            with patch('official_arena_preflight.command', side_effect=[{'returncode': 0, 'output': 'b' * 40}, {'returncode': 0, 'output': ' M config.yaml'}, {'returncode': 0, 'output': '-abc submodule'}]):
                result = verify_source(folder, manifest)
            self.assertEqual(len(result['problems']), 4)


if __name__ == '__main__': unittest.main()
