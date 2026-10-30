import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import libero_scene_probe as probe


class LiberoSceneTests(unittest.TestCase):
    def test_visual_xml_adds_no_dynamic_degrees_or_contacts(self):
        xml='<mujoco><asset/><worldbody><camera name="agentview"/><body name="robot"><joint name="j"/><geom type="box" size="1 1 1"/></body></worldbody></mujoco>'
        marker={'id':0,'world_position_m':[1,2,3],'radius_m':.002,'emissive_rgb':[1,0,1]}
        result=ET.fromstring(probe.marked_xml(xml,{'agentview':[marker]}))
        self.assertEqual(len(result.findall('.//joint')),1)
        self.assertEqual(len(result.findall('.//body')),1)
        self.assertEqual(len(result.findall('.//camera')),1)
        added=result.find('./worldbody/geom');self.assertEqual(added.get('contype'),'0');self.assertEqual(added.get('conaffinity'),'0')
        self.assertEqual(added.get('group'),'1')
        self.assertEqual(result.find('./asset/material').get('emission'),'1')

    def test_xml_missing_world_is_rejected(self):
        with self.assertRaises(ValueError):probe.marked_xml('<mujoco/>',{})

    def test_selects_only_finite_correct_length_initial_state(self):
        import numpy as np
        states=np.array([[0,1,2],[3,4,5]],dtype=float)
        selected=probe.select_state(states,1,3);np.testing.assert_array_equal(selected,[3,4,5])
        selected[0]=99;self.assertEqual(states[1,0],3)
        for values,index,size in [(states,2,3),(states,-1,3),(states,0,4),([[float('nan'),1,2]],0,3)]:
            with self.assertRaises(ValueError):probe.select_state(values,index,size)

    def test_parent_preserves_venv_path_and_distinguishes_missing_benchmark_init(self):
        for worker_failure in (False,True):
            with self.subTest(worker_failure=worker_failure),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);python=root/'examples/libero/.venv/bin/python';python.parent.mkdir(parents=True);python.write_text('fixture')
                calls=[]
                class Worker:
                    def __init__(self,command,**kwargs):
                        calls.append((command,kwargs));folder=Path(command[command.index('--worker')+1])
                        (folder/'report.json').write_text(json.dumps({'status':'failed' if worker_failure else 'completed',
                            'benchmark_initialization_available':False,'marker_test':{'all_marker_views_passed':True}}))
                    def __enter__(self):return self
                    def __exit__(self,*args):pass
                    def wait(self,timeout=None):return 1 if worker_failure else 0
                args=['probe','--openpi-root',str(root),'--output-root',str(root/'out')]
                with patch.object(sys,'argv',args),patch('run_multipose_geometry.preflight_reference',return_value={'prefix':'fixture-venv'}),patch.object(probe.subprocess,'Popen',Worker),patch.object(Path,'glob',return_value=[]):
                    self.assertEqual(probe.main(),1 if worker_failure else 0)
                manifest=json.loads(next((root/'out').glob('*/run.json')).read_text())
                self.assertEqual(manifest['status'],'failed' if worker_failure else 'completed')
                self.assertEqual(calls[0][0][0],str(python.absolute()))
                self.assertNotIn('VIRTUAL_ENV',calls[0][1]['env'])
                self.assertEqual(calls[0][1]['env']['MUJOCO_GL'],'egl')
                self.assertNotIn('--query',calls[0][0])
                if not worker_failure:self.assertFalse(manifest['benchmark_initialization_available'])


if __name__=='__main__':unittest.main()
