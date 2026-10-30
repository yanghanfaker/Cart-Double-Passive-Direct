import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock,patch
import numpy as np

import online_reference_probe as probe
import run_online_reference_ab as runner
from test_online_registered_probe import capture
from test_clean_scene_probe import fixtures
from run_clean_scene_capture import prepare_cases
from test_target_reference_probe import state
from kinematic_attribution import rotate


def short_source(root):
    config=prepare_cases(*fixtures())[1];cal=fixtures()[2];cal['candidate_geometry_within_thresholds']=True
    initial=state();initial['joint_positions_rad']=config['arm_positions_rad'];gains={};cb=json.dumps(cal).encode();(root/'world_registration.json').write_bytes(cb)
    data={'status':'completed','queries_per_run':20,'repeats':3,'initial_inputs_identical':True,'all_followup_inputs_changed':True,'world_registration_sha256':runner.digest(cb),'cases':[]}
    for repeat in range(1,4):
        for mode in ('current','accumulated'):
            name=f'repeat-{repeat:02d}-{mode}';folder=root/name;folder.mkdir()
            (folder/'clean_scene_request.json').write_text(json.dumps(config));(folder/'joint_gains.json').write_text('{}')
            (folder/'online_settings.json').write_text(json.dumps({'prompt':'cube','calibration':cal,'expected_joint_gains':gains}))
            r={'status':'completed','reference_mode':mode,'policy_queries':20,'inference_attempts':20,'action_controls':100,'hold_controls':10,'native_controls_executed':110,
                'all_followup_inputs_changed':True,'initial':initial,'first_input_sha256':'same'}
            b=json.dumps(r).encode();(folder/'result.json').write_bytes(b)
            data['cases'].append({'name':name,'reference_mode':mode,'result':name+'/result.json','result_sha256':runner.digest(b)})
    (root/'run.json').write_text(json.dumps(data));return data


class DescentTests(unittest.TestCase):
    def test_planned_stop_reasons(self):
        s={'tcp_world_m':[.5,0,.110],'object_pos_world_m':[.5,0,.021]}
        self.assertEqual(probe.descent_stop(s,.089),'near_boundary_reached')
        s['tcp_world_m']=[.65,0,.060];self.assertEqual(probe.descent_stop(s,.155),'descent_height_floor_reached')
        s['tcp_world_m']=[.65,0,.1];self.assertEqual(probe.descent_stop(s,.15),'distance_increased_from_best_over_10mm')
        self.assertEqual(probe.descent_stop(s,.17,.001),'progress_below_2mm_per100_controls')
        self.assertIsNone(probe.descent_stop(s,.17,.005))
        self.assertEqual(probe.comparison_limits('short')['net_m'],.06);self.assertEqual(probe.comparison_limits('descent')['queries'],(80,100))

    def test_early_stop_count_validation_and_checkpoint_summary(self):
        r={'status':'completed','reference_mode':'accumulated','profile':'descent','queries_requested':100,'policy_queries':80,'inference_attempts':80,
            'action_controls':397,'hold_controls':10,'native_controls_executed':407,'stop_reason':'near_boundary_reached','boundary_reached':True}
        runner.validate_run_result(r,'accumulated',100,'descent')
        for change in ({'action_controls':395,'native_controls_executed':405},{'boundary_reached':False},{'stop_reason':'query_budget_exhausted','boundary_reached':False}):
            with self.subTest(change=change),self.assertRaises(RuntimeError):runner.validate_run_result({**r,**change},'accumulated',100,'descent')
        cases=[]
        for i in range(3):
            cases.append({'status':'completed','reference_mode':'current','boundary_reached':i<2,'action_controls':210+i*10,
                'stop_reason':'near_boundary_reached' if i<2 else 'progress_below_2mm_per100_controls',
                'height_checkpoints':{'height_100mm':{'xy_offset_m':.01*(i+1),'object_distance_m':.11,'action_control':200}} if i<2 else {},
                **{k:.001 for k in ('distance_reduction_m','final_xy_offset_m','final_ideal_position_error_m','max_applied_backlog_m','max_pre_update_ideal_gap_m','max_applied_target_error_m')}})
        s=runner.summarize(cases)['current'];self.assertEqual(s['boundary_reached_runs'],2);self.assertEqual(s['action_controls_to_boundary']['mean'],215)
        self.assertEqual(s['height_100mm']['reached_runs'],2);self.assertAlmostEqual(s['height_100mm']['xy_offset_m']['mean'],.015)

    def test_boundary_can_stop_inside_chunk_and_hold_last_actual_prefix(self):
        self.run_boundary_case('descent')

    def test_near_boundary_stops_inside_chunk_records_close_and_holds_open(self):
        self.run_boundary_case('near-descent')

    def run_boundary_case(self,profile):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cal=capture(root);g=json.loads((root/'geometry_audit.json').read_bytes());(root/'joint_gains.json').write_text('{}')
            near=profile=='near-descent';budget=120 if near else 80
            q=[1,0,0,0];initial_tcp=np.array([.5,0,.0952 if near else .1252]);obj=[.5,0,.021];clock={'step':175};holder={}
            def start_state():return {'tcp_world_m':initial_tcp.tolist(),'hand_position_world_m':(initial_tcp-rotate(q,[0,0,.097])).tolist(),
                'hand_quat_world_xyzw':q,'joint_positions_rad':g['joint_positions_rad'],'object_pos_world_m':obj,'finger_positions_m':[.04,.04]}
            class Runtime:
                def __init__(self,env,base,arm,settings,output,tensor,report,save):
                    self.s=start_state();self.controls=0;self.path=0.;self.trace=io.StringIO();self.report=report;holder['r']=self;(output/'queries').mkdir()
                def state(self):return copy.deepcopy(self.s)
                def step(self,target,grip,phase):
                    self_outer.assertEqual(grip,1.0)
                    p=np.array(self.s['tcp_world_m']);new=np.array(target['tcp97_target_world_m']);self.path+=np.linalg.norm(new-p);self.controls+=1;clock['step']+=5
                    self.s['tcp_world_m']=new.tolist();self.s['hand_position_world_m']=(new-rotate(q,[0,0,.097])).tolist()
                    self.report['native_controls_executed']=self.controls;return self.state()
                def save_images(self,*args):pass
            def snapshot(base,arm,folder,initial,tensor,index):
                capture(folder);geo=copy.deepcopy(g);s=holder['r'].state();geo['sensor_body_world']={'position_m':s['hand_position_world_m'],'quat_xyzw':q}
                geo['sensor_tcp_world']['position_m']=(np.array(s['hand_position_world_m'])+rotate(q,[0,0,.1034])).tolist();geo['object_root_world']['position_m']=obj
                for n in ('external_camera','wrist_camera'):geo['cameras'][n]['frame']=[index+1]
                (folder/'geometry_audit.json').write_text(json.dumps(geo));return geo,clock['step']
            self_outer=self
            client=MagicMock();client.infer.return_value={'actions':np.array([[0,0,-.01,0,0,0,1. if near else -1.]]*10)};client.get_server_metadata.return_value={}
            module=types.ModuleType('openpi_client.websocket_client_policy');module.WebsocketClientPolicy=MagicMock(return_value=client)
            base=types.SimpleNamespace(sim=types.SimpleNamespace(get_physics_step_count=lambda:clock['step']))
            settings={'reference_mode':'accumulated','profile':profile,'queries':budget,'calibration':cal,'expected_initial':start_state(),
                'expected_joint_gains':{},'prompt':'cube','server_host':'127.0.0.1','server_port':8000}
            with (patch.dict(sys.modules,{'openpi_client.websocket_client_policy':module}),patch('openpi_grasp_runtime.GraspRuntime',Runtime),
                    patch.object(probe,'check_openpi_health',return_value={'status':'healthy'}),patch.object(probe,'capture_round',snapshot)):
                result=probe.run_online_reference(None,base,None,settings,root,None)
            self.assertEqual(result['stop_reason'],'near_boundary_reached');self.assertTrue(result['boundary_reached'])
            self.assertEqual(result['action_controls'],29);self.assertEqual(result['hold_controls'],10);self.assertEqual(result['native_controls_executed'],39)
            self.assertEqual(client.infer.call_count,6);self.assertEqual(result['queries'][-1]['executed_prefix_controls'],4)
            self.assertEqual(result['height_checkpoints']['height_70mm' if near else 'height_100mm']['action_control'],9)
            runner.validate_run_result(result,'accumulated',budget,profile)
            self.assertEqual(result['requested_close_actions'],29 if near else 0)
            if near:
                self.assertEqual(len(result['close_request_events']),29)
                first=result['first_close_request'];self.assertEqual(first['action_control'],1);self.assertEqual(first['query'],1);self.assertEqual(first['prefix'],1)
                self.assertAlmostEqual(first['height_m'],.0742);self.assertAlmostEqual(first['xy_offset_m'],0);self.assertEqual(first['gripper_applied'],1.0)
                self.assertLess(result['action_end_object_distance_m'],.06);self.assertGreater(result['action_end_object_distance_m'],.05)
            else:self.assertIsNone(result['first_close_request'])
            rows=[json.loads(x) for x in (root/'reference_trace.jsonl').read_text().splitlines()]
            self.assertEqual(rows[-1]['source_query'],6);self.assertEqual(rows[-1]['source_prefix_index'],4)

    def test_near_thresholds_and_profile_budgets(self):
        t=probe.approach_thresholds('near-descent');self.assertEqual(t['minimum_m'],.05);self.assertEqual(t['height_checkpoints_mm'],(200,150,100,80,70,60))
        self.assertEqual(probe.comparison_limits('near-descent')['queries'],(120,140))
        s={'tcp_world_m':[0,0,.075],'object_pos_world_m':[0,0,0]}
        self.assertIsNone(probe.descent_stop(s,.075,profile='near-descent'))
        self.assertEqual(probe.descent_stop(s,.075),'near_boundary_reached')
        s['tcp_world_m']=[0,0,.0599];self.assertEqual(probe.descent_stop(s,.0599,profile='near-descent'),'near_boundary_reached')
        s['tcp_world_m']=[.07,0,.029];self.assertEqual(probe.descent_stop(s,.075,profile='near-descent'),'descent_height_floor_reached')
        with self.assertRaises(ValueError):probe.approach_thresholds('invalid')

    def test_near_source_requires_all_six_descent_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m=short_source(root);m.update(profile='descent',queries_per_run=100)
            for c in m['cases']:
                p=root/c['result'];r=json.loads(p.read_bytes());r.update(profile='descent',queries_requested=100,policy_queries=80,inference_attempts=80,
                    action_controls=397,native_controls_executed=407,boundary_reached=True,stop_reason='near_boundary_reached',action_end_object_distance_m=.0899)
                b=json.dumps(r).encode();p.write_bytes(b);c['result_sha256']=runner.digest(b)
            runner.prepare_online_source(root,m,'descent')
            for modification in ({'boundary_reached':False,'stop_reason':'progress_below_2mm_per100_controls'},{'action_end_object_distance_m':.10},{'first_input_sha256':'different'}):
                p=root/m['cases'][0]['result'];original=p.read_bytes();r={**json.loads(original),**modification};b=json.dumps(r).encode();p.write_bytes(b);m['cases'][0]['result_sha256']=runner.digest(b)
                with self.assertRaises((ValueError,RuntimeError)):runner.prepare_online_source(root,m,'descent')
                p.write_bytes(original);m['cases'][0]['result_sha256']=runner.digest(original)

    def test_close_request_summary_excludes_absent_requests(self):
        cases=[{'status':'completed','reference_mode':'current','requested_close_actions':3,'first_close_request':{'height_m':.08,'xy_offset_m':.01,'object_distance_m':.081,'action_control':400}},
               {'status':'completed','reference_mode':'current','requested_close_actions':0,'first_close_request':None}]
        for c in cases:c.update({k:0. for k in ('distance_reduction_m','final_xy_offset_m','final_ideal_position_error_m','max_applied_backlog_m','max_pre_update_ideal_gap_m','max_applied_target_error_m')})
        s=runner.summarize(cases)['current'];self.assertEqual(s['runs_with_close_request'],1);self.assertEqual(s['requested_close_actions'],3)
        self.assertEqual(s['first_close_request']['height_m']['mean'],.08);self.assertIsNone(s['first_close_request']['height_m']['sample_std'])

    def test_descent_source_and_parent_allow_normal_nonboundary_stops(self):
        self.run_parent_case('descent')

    def test_near_parent_uses_previous_descent_source_and_new_budget(self):
        self.run_parent_case('near-descent')

    def run_parent_case(self,profile):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();m=short_source(source);runner.prepare_online_source(source,m);commands=[]
            budget=120 if profile=='near-descent' else 100
            if profile=='near-descent':
                m.update(profile='descent',queries_per_run=100)
                for c in m['cases']:
                    p=source/c['result'];r=json.loads(p.read_bytes());r.update(profile='descent',queries_requested=100,policy_queries=80,inference_attempts=80,
                        action_controls=397,native_controls_executed=407,boundary_reached=True,stop_reason='near_boundary_reached',action_end_object_distance_m=.0899)
                    b=json.dumps(r).encode();p.write_bytes(b);c['result_sha256']=runner.digest(b)
                (source/'run.json').write_text(json.dumps(m))
            class Process:
                def __init__(self,command,**kwargs):
                    commands.append(command);self.stdout=io.StringIO('');sf=Path(command[command.index('--online-registered-json')+1]);s=json.loads(sf.read_bytes())
                    out=Path(command[command.index('--output-root')+1])/'capture';out.mkdir(parents=True);stalled=len(commands)==2
                    q=80;a=400 if stalled else 397
                    self_outer.assertEqual(s['profile'],profile);self_outer.assertEqual(s['queries'],budget)
                    r={'status':'completed','profile':profile,'reference_mode':s['reference_mode'],'queries_requested':budget,'policy_queries':q,'inference_attempts':q,
                        'action_controls':a,'hold_controls':10,'native_controls_executed':a+10,'boundary_reached':not stalled,
                        'stop_reason':'progress_below_2mm_per100_controls' if stalled else 'near_boundary_reached',
                        'first_input_sha256':'same','all_followup_inputs_changed':True,'requested_close_actions':0,
                        'height_checkpoints':{} if stalled else {'height_100mm':{'xy_offset_m':.01,'object_distance_m':.11,'action_control':350}},
                        **{k:.001 for k in ('distance_reduction_m','final_xy_offset_m','final_ideal_position_error_m','max_applied_backlog_m','max_pre_update_ideal_gap_m','max_applied_target_error_m')}}
                    (out/'online_registered_result.json').write_text(json.dumps(r))
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def wait(self):return 0
            self_outer=self
            args=['descent','--profile',profile,'--source-manifest',str(source/'run.json'),'--output-root',str(root/'out')]
            with patch.object(sys,'argv',args),patch.object(runner,'check_openpi_health',return_value={'status':'healthy'}),patch.object(runner.subprocess,'Popen',Process):
                self.assertEqual(runner.main(),0)
            self.assertEqual(len(commands),6);report=json.loads(next((root/'out').glob('*/run.json')).read_bytes())
            self.assertEqual(report['profile'],profile);self.assertEqual(report['queries_per_run'],budget)
            self.assertEqual(report['summary']['accumulated']['boundary_reached_runs'],2)
            self.assertEqual(report['summary']['accumulated']['height_100mm']['reached_runs'],2)
            self.assertEqual(report['summary']['current']['boundary_reached_runs'],3)


if __name__=='__main__':unittest.main()
