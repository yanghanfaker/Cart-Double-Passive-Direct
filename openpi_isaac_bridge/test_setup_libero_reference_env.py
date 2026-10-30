import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import setup_libero_reference_env as setup


class SetupTests(unittest.TestCase):
    def test_install_plan_targets_only_reference_python(self):
        base,target,python,marker=setup.environment_paths(Path('/example/openpi'))
        commands=setup.install_plan('/example/openpi',python,'requirements.txt')
        self.assertNotEqual(base,python)
        for command in commands:self.assertEqual(command[command.index('--python')+1],str(python))
        self.assertIn('--no-deps',commands[-1])

    def test_existing_unmanaged_environment_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/'examples/libero/.venv';target.mkdir(parents=True)
            sentinel=target/'keep';sentinel.write_text('user environment')
            with self.assertRaises(ValueError):setup.environment_paths(root)
            self.assertEqual(sentinel.read_text(),'user environment')

    def run_mocked(self,capture_status):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'.venv/bin/python';base.parent.mkdir(parents=True);base.write_text('fixture')
            for file in ('third_party/libero/libero/libero/__init__.py','packages/openpi-client/pyproject.toml'):
                path=root/file;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture')
            commands=[]
            class Child:
                def __init__(self,command,**kwargs):
                    commands.append(command);self.stdout=io.StringIO('')
                    if command[:2]==['uv','venv']:
                        target=Path(command[-1]);(target/'bin').mkdir(parents=True);(target/'bin/python').write_text('fixture')
                    elif '--output-root' in command:
                        out=Path(command[command.index('--output-root')+1])/'fixture';out.mkdir(parents=True)
                        (out/'run.json').write_text(json.dumps({'status':capture_status}))
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def wait(self):return 0
            response=subprocess.CompletedProcess([],0,json.dumps({'version':[3,11,17],'packages':{'numpy':'1.26.4'}}),'')
            with patch.object(sys,'argv',['setup','--openpi-root',str(root),'--output-root',str(root/'results')]),patch.object(setup.subprocess,'run',return_value=response),patch.object(setup.subprocess,'Popen',Child):
                code=setup.main()
            report=json.loads(next((root/'results').glob('*/setup.json')).read_text())
            self.assertTrue(report['model_environment_versions_unchanged'])
            self.assertTrue((root/'examples/libero/.venv/.reference_audit_env.json').is_file())
            self.assertEqual(len(commands),5)
            for command in commands:
                if command[:3]==['uv','pip','install']:
                    self.assertEqual(Path(command[command.index('--python')+1]).resolve(),(root/'examples/libero/.venv/bin/python').resolve())
            return code,report

    def test_setup_and_capture_complete_without_modifying_model_versions(self):
        code,report=self.run_mocked('completed')
        self.assertEqual(code,0);self.assertEqual(report['status'],'completed')

    def test_missing_capture_dependencies_are_not_called_success(self):
        code,report=self.run_mocked('dependency_missing')
        self.assertEqual(code,1);self.assertEqual(report['status'],'environment_ready_capture_failed')


if __name__=='__main__':unittest.main()
