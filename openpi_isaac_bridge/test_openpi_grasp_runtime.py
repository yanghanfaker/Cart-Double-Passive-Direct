import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock,patch
import zipfile
import numpy as np

import openpi_grasp_runtime as runtime_module
from test_online_registered_probe import capture
from kinematic_attribution import rotate


class RuntimeArchiveTests(unittest.TestCase):
    def test_current_inputs_and_invalid_future_output_preserved_in_query_archive(self):
        for invalid in (False,True):
            with self.subTest(invalid=invalid),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);cal=capture(root);geometry=json.loads((root/'geometry_audit.json').read_bytes());geometry['cameras']['external_camera']['frame']=[2];geometry['cameras']['wrist_camera']['frame']=[3]
                base=types.SimpleNamespace(sim=types.SimpleNamespace(get_physics_step_count=lambda:175))
                hand=geometry['sensor_body_world'];tcp=(np.array(hand['position_m'])+rotate(hand['quat_xyzw'],[0,0,.097])).tolist()
                state={'tcp_world_m':tcp,'hand_quat_world_xyzw':hand['quat_xyzw'],'joint_positions_rad':geometry['joint_positions_rad']}
                class Runtime:
                    def __init__(self,env,base,arm,settings,output,tensor,report,save):
                        self.report=report;self.trace=io.StringIO();(output/'queries').mkdir()
                    def state(self):return state
                    def step_model(self,*args):self.report['native_controls_executed']+=1;return state
                    def step_goal(self,*args):raise AssertionError('unexpected verification move')
                    def save_images(self,*args):pass
                def snapshot(base,arm,folder,initial,tensor,index):
                    capture(folder);(folder/'geometry_audit.json').write_text(json.dumps(geometry));return geometry,175
                def driver(read,query,model,goal,**kwargs):
                    req=query(1)
                    for a in req['raw_action_chunk'][:5]:model(a,1.)
                    return {'success':False,'outcome':'approach_query_budget_exhausted'}
                chunk=np.zeros((10,7))
                if invalid:chunk[-1,0]=float('nan')
                client=MagicMock();client.infer.return_value={'actions':chunk};client.get_server_metadata.return_value={}
                module=types.ModuleType('openpi_client.websocket_client_policy');module.WebsocketClientPolicy=MagicMock(return_value=client)
                settings={'calibration':cal,'prompt':'pick up the cube','server_host':'127.0.0.1','server_port':8000,'max_queries':1}
                with (patch.dict(sys.modules,{'openpi_client.websocket_client_policy':module}),patch.object(runtime_module,'check_openpi_health',MagicMock(return_value={'status':'healthy'})),
                        patch.object(runtime_module,'GraspRuntime',Runtime),patch.object(runtime_module,'capture_round',snapshot),patch.object(runtime_module,'run_grasp_experiment',driver)):
                    if invalid:
                        with self.assertRaises(ValueError):runtime_module.run_openpi_grasp(None,base,None,settings,root,None)
                    else:runtime_module.run_openpi_grasp(None,base,None,settings,root,None)
                report=json.loads((root/'online_registered_result.json').read_bytes())
                self.assertEqual(report['policy_queries'],1);self.assertEqual(report['inference_attempts'],1)
                self.assertEqual(report['native_controls_executed'],0 if invalid else 5)
                self.assertEqual(report['status'],'failed' if invalid else 'completed')
                with zipfile.ZipFile(root/report['queries'][0]['archive']) as archive:
                    self.assertIn('geometry_audit.json',archive.namelist());self.assertIn('query_record.json',archive.namelist())
                    self.assertIn('invalid_model_response.txt' if invalid else 'online_model_response.json',archive.namelist())
                    self.assertIn('wrist_camera_policy224_flip_x.png',archive.namelist())
                    if not invalid:
                        request=json.loads(archive.read('online_replay_request.json'))
                        self.assertEqual(request['source_input_sha256'],report['queries'][0]['input']['input_sha256'])


if __name__=='__main__':unittest.main()
