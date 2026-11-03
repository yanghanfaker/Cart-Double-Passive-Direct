from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock
import numpy as np
from official_libero_launcher import STATE_MANIFEST, allowed_numpy_types, check_states, snapshot_states, verify_init_bytes


def fixture():
    data=b'known reference state bytes'
    return data,{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


class OfficialLiberoLauncherTests(unittest.TestCase):
    def test_reference_manifest_has_ten_pinned_files(self):
        manifest=json.loads(STATE_MANIFEST.read_bytes())
        self.assertEqual(manifest['commit'],'f78abd68ee283de9f9be3c8f7e2a9ad60246e95c')
        self.assertEqual(len(manifest['files']),10)
        self.assertTrue(all(x.endswith('.pruned_init') for x in manifest['files']))

    def test_mismatched_bytes_rejected(self):
        data,expected=fixture()
        self.assertEqual(verify_init_bytes(data,expected),data)
        with self.assertRaises(ValueError):verify_init_bytes(data+b'x',expected)

    def test_snapshot_is_byte_identical_and_rejects_path_escape(self):
        data,expected=fixture()
        files={f'libero_spatial/task{i}.pruned_init':expected for i in range(10)}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'source';out=Path(folder)/'output'
            for name in files:
                p=root/'init_files'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
            records=snapshot_states(root,out,{'files':files})
            self.assertEqual(len(records),10)
            self.assertTrue(all((out/name).read_bytes()==data for name in files))
            bad=dict(files);bad['../escape.pruned_init']=bad.pop(next(iter(bad)))
            with self.assertRaises(ValueError):snapshot_states(root,out,{'files':bad})

    def test_loader_uses_explicit_restricted_mode_and_checks_finite_state(self):
        data,expected=fixture()
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'state.pruned_init';p.write_bytes(data)
            records=[{'path':p.name,**expected}]
            torch=SimpleNamespace(serialization=SimpleNamespace(safe_globals=Mock(return_value=nullcontext())),
                                  load=Mock(return_value=np.ones((50,6),dtype=np.float64)))
            rows=check_states(folder,records,torch,np)
            self.assertEqual(rows[0]['shape'],[50,6])
            self.assertEqual(torch.load.call_args.kwargs,{'map_location':'cpu','weights_only':True})
            self.assertEqual(torch.serialization.safe_globals.call_count,1)
            torch.load.return_value=np.array([[np.nan]])
            with self.assertRaises(ValueError):check_states(folder,records,torch,np)

    def test_unverified_file_rejected_before_load(self):
        data,expected=fixture()
        torch=SimpleNamespace(serialization=SimpleNamespace(safe_globals=lambda allowed:nullcontext()),load=Mock())
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'state.pruned_init').write_bytes(b'changed')
            with self.assertRaises(ValueError):check_states(folder,[{'path':'state.pruned_init',**expected}],torch,np)
        torch.load.assert_not_called()

    def test_allowlist_contains_only_array_and_float_dtype_support(self):
        allowed=allowed_numpy_types(np)
        self.assertEqual(len(allowed),5)
        self.assertIn(np.ndarray,allowed)
        self.assertIn(type(np.dtype('float64')),allowed)
        self.assertIn(type(np.dtype('float32')),allowed)


if __name__=='__main__':unittest.main()
