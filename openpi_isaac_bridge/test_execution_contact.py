import copy
import math
import unittest
import numpy as np
from execution_contact import contact_cases,contact_scene,run_contact_experiment,compress_trace
from execution_alignment_probe import validate_case


class ContactTests(unittest.TestCase):
    def test_contact_trace_compression_preserves_exact_bytes(self):
        import tempfile
        import gzip
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'physics_joint_trace.jsonl';payload=b'{"phase":"close"}\n'*500
            path.write_bytes(payload);packed=compress_trace(path)
            self.assertFalse(path.exists())
            with gzip.open(packed,'rb') as source:self.assertEqual(source.read(),payload)

    def test_ten_cases_cross_height_finger_and_repeat(self):
        cases=contact_cases();self.assertEqual(len(cases),10)
        self.assertEqual(len({c['name'] for c in cases}),10)
        for c in cases:
            validate_case(c);self.assertEqual(c['arm_profile'],'k1687p5-d50')
            self.assertEqual(c['cap_m'],.0005);self.assertEqual(c['rotation_cap_deg'],.1)
        positive=[c for c in cases if not c['negative_control']]
        self.assertEqual({(c['grasp_height_m'],c['finger_profile'],c['repeat']) for c in positive},
            {(h,f,r) for h in (0.,.005) for f in ('kp350-kd50','kp1000-kd100') for r in (1,2)})
        self.assertEqual(len([c for c in cases if c['negative_control']]),2)

    def test_scene_moves_only_proxy_xy_using_static_reference(self):
        from test_clean_scene_probe import fixtures
        from run_clean_scene_capture import prepare_cases
        config=prepare_cases(*fixtures())[0];old=copy.deepcopy(config)
        settings={'calibration':{'world_from_isaac_rotation':np.eye(3).tolist(),
            'world_from_isaac_translation_m':[-.66,0,.912]},'native_initial':{'robot0_eef_pos':[-.01,.14,1.03]}}
        result=contact_scene(config,settings)
        np.testing.assert_allclose(result['object_xy_world_m'],[.66,.14])
        self.assertEqual(config,old)
        for key in old:
            if key not in ('name','object_xy_world_m'):self.assertEqual(result[key],old[key])

    def simulator(self,obstruction=True,slip=False,prelift_push=False,transfer_spike=False):
        # Kinematic callback fixture only; does not simulate real contacts or PD dynamics.
        state={'tcp_world_m':[.65,.14,.12],'hand_quat_world_xyzw':[1.,0.,0.,0.],
            'hand_position_world_m':[.65,.14,.217],
            'object_pos_world_m':[.66,.14,.021],'object_quat_world_xyzw':[0.,0.,0.,1.],
            'object_in_tcp_pos_m':[.01,0,.099],'object_in_tcp_quat_xyzw':[1.,0,0,0],
            'finger_positions_m':[.04,.04]}
        calls=[];attached=None;lift_steps=0
        def read():return copy.deepcopy(state)
        def step(p,q,grip,phase):
            nonlocal attached,lift_steps
            calls.append((list(p),list(q),grip,phase))
            d=np.array(p)-state['tcp_world_m'];n=np.linalg.norm(d)
            state['tcp_world_m']=(np.array(state['tcp_world_m'])+d*min(1.,.001/max(n,1e-12))).tolist()
            state['hand_quat_world_xyzw']=q
            state['finger_positions_m']=[.02,.02] if grip<0 and obstruction else ([0.,0.] if grip<0 else [.04,.04])
            if phase=='close' and grip<0 and obstruction:
                attached=state['tcp_world_m'][2]-state['object_pos_world_m'][2]
                if prelift_push:state['object_pos_world_m'][0]+=.021
            if phase in ('lift','hold') and attached is not None and not slip:
                state['object_pos_world_m']=[state['tcp_world_m'][0],state['tcp_world_m'][1],state['tcp_world_m'][2]-attached]
            if phase=='lift':
                lift_steps+=1
                if transfer_spike and lift_steps==10:state['object_pos_world_m'][0]+=.016
            delta=np.array(state['object_pos_world_m'])-state['tcp_world_m']
            state['object_in_tcp_pos_m']=[float(delta[0]),float(-delta[1]),float(-delta[2])]
            return read()
        return read,step,calls

    def test_success_requires_lift_and_two_second_hold(self):
        read,step,calls=self.simulator();result=run_contact_experiment(read,step,contact_cases()[0])
        self.assertTrue(result['success']);self.assertEqual(result['outcome'],'verified_lift_hold')
        self.assertEqual(sum(p=='hold' for _,_,_,p in calls),40)
        self.assertGreaterEqual(result['hold_check']['min_object_lift_m'],.035)

    def test_no_obstruction_prevents_lift(self):
        read,step,calls=self.simulator(obstruction=False)
        result=run_contact_experiment(read,step,contact_cases()[0])
        self.assertFalse(result['success']);self.assertEqual(result['outcome'],'close_candidate_failed')
        self.assertFalse(any(p=='lift' for _,_,_,p in calls))

    def test_open_negative_never_commands_closure_or_lift(self):
        read,step,calls=self.simulator()
        result=run_contact_experiment(read,step,next(c for c in contact_cases() if c['negative_control']))
        self.assertEqual(result['outcome'],'negative_rejected');self.assertTrue(result['negative_check_passed'])
        self.assertTrue(all(grip==1. for _,_,grip,_ in calls));self.assertFalse(result['lift_attempted'])

    def test_closed_fingers_without_object_lift_are_not_success(self):
        read,step,calls=self.simulator(slip=True)
        result=run_contact_experiment(read,step,contact_cases()[0])
        self.assertTrue(result['close_check']['passed']);self.assertFalse(result['success'])
        self.assertEqual(result['outcome'],'lift_hold_failed')

    def test_transfer_slip_cannot_be_hidden_by_final_recovery(self):
        read,step,calls=self.simulator(transfer_spike=True)
        result=run_contact_experiment(read,step,contact_cases()[0])
        self.assertFalse(result['success']);self.assertGreater(result['hold_check']['max_transfer_drift_m'],.015)

    def test_prelift_displacement_stops_experiment(self):
        read,step,calls=self.simulator(prelift_push=True)
        result=run_contact_experiment(read,step,contact_cases()[0])
        self.assertEqual(result['outcome'],'object moved >2cm before lift');self.assertFalse(result['lift_attempted'])


if __name__=='__main__':unittest.main()
