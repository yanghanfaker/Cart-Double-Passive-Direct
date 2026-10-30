import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import libero_reference_audit as audit


def source_fixture():
    return {'status':'completed','joint_names':[f'panda_joint{i}' for i in range(1,8)]+['panda_finger_joint1','panda_finger_joint2'],
            'joint_positions_rad':[0,-.569,0,-2.81,0,3.037,.741,.04,.04],'finger_positions_isaac_m':[.04,.04]}


class ReferenceTests(unittest.TestCase):
    def test_targets_preserve_arm_pose_and_reference_finger_signs(self):
        arm,fingers=audit.source_targets(source_fixture())
        self.assertEqual(arm,[0,-.569,0,-2.81,0,3.037,.741]);self.assertEqual(fingers,[.04,-.04])

    def test_invalid_source_is_rejected(self):
        d=source_fixture();d['status']='failed'
        with self.assertRaises(ValueError):audit.source_targets(d)
        d=source_fixture();d['joint_positions_rad'][0]=float('nan')
        with self.assertRaises(ValueError):audit.source_targets(d)

    def test_reference_joint_mapping_by_suffix_not_list_order(self):
        names=[f'robot0_joint{i}' for i in reversed(range(1,8))]
        self.assertEqual(audit.ordered_joints(names,'arm'),list(reversed(names)))
        self.assertEqual(audit.ordered_joints(['gripper0_finger_joint2','gripper0_finger_joint1'],'finger'),['gripper0_finger_joint1','gripper0_finger_joint2'])

    def test_missing_or_ambiguous_joint_mapping_rejected(self):
        with self.assertRaises(ValueError):audit.ordered_joints(['robot0_joint1'],'arm')
        with self.assertRaises(ValueError):audit.ordered_joints(['a_finger_joint1','b_finger_joint1','b_finger_joint2'],'finger')

    def test_preflight_runs_without_importing_libero(self):
        result=subprocess.run([sys.executable,'-c',audit.PREFLIGHT],capture_output=True,text=True,check=True)
        report=json.loads(result.stdout)
        self.assertEqual(set(report['modules']),set(audit.MODULES))
        self.assertIn('machine',report);self.assertIn('versions',report)

    def test_missing_dependencies_are_saved_without_launching_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'isaac.json';source.write_text(json.dumps(source_fixture()))
            fake={'modules':{name:{'available':False} for name in audit.MODULES}}
            result=subprocess.CompletedProcess([],0,json.dumps(fake),'')
            args=['audit','--isaac-audit',str(source),'--python',sys.executable,'--output-root',str(root/'out')]
            with patch.object(sys,'argv',args),patch.object(audit.subprocess,'run',return_value=result),patch.object(audit.subprocess,'Popen') as popen:
                self.assertEqual(audit.main(),0);popen.assert_not_called()
            saved=json.loads(next((root/'out').glob('*/run.json')).read_text())
            self.assertEqual(saved['status'],'dependency_missing');self.assertEqual(saved['package_installs'],0)

    def test_available_dependencies_launch_isolated_capture_and_keep_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'isaac.json';source.write_text(json.dumps(source_fixture()))
            fake={'modules':{name:{'available':True} for name in audit.MODULES}}
            class Worker:
                def __init__(self,command,**kwargs):
                    self.kwargs=kwargs;folder=Path(command[command.index('--worker')+1])
                    (folder/'reference.json').write_text(json.dumps({'status':'completed'}))
                    assert kwargs['stdin']==subprocess.DEVNULL and kwargs['start_new_session']
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def wait(self,timeout=None):return 0
            args=['audit','--isaac-audit',str(source),'--python',sys.executable,'--output-root',str(root/'out')]
            with patch.object(sys,'argv',args),patch.object(audit.subprocess,'run',return_value=subprocess.CompletedProcess([],0,json.dumps(fake),'')),patch.object(audit.subprocess,'Popen',Worker):
                self.assertEqual(audit.main(),0)
            saved=json.loads(next((root/'out').glob('*/run.json')).read_text())
            self.assertEqual(saved['status'],'completed')
            self.assertEqual(next((root/'out').glob('*/isaac_source.json')).read_bytes(),source.read_bytes())


if __name__=='__main__':unittest.main()
