import copy
import math
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from registered_replay_probe import target_from_action,validate_replay,TRANSLATION_CAP,ROTATION_CAP
from run_registered_replay_capture import representative
import run_registered_replay_capture as runner
from geometry_audit import pose_matrix,multiply,inverse
from kinematic_attribution import rotate


def fixture():
    return {'condition':'flip_x','raw_action_chunk':[[0]*7 for _ in range(10)],
        'world_rotation_L_from_I':np.eye(3).tolist(),'world_translation_L_from_I_m':[-.66,0,.912],
        'source_state_float32':[0]*8,'source_arm_positions_rad':[0]*7,
        'registration_quat_xyzw':[0,0,0,1],'source_mapped_hand_quat_xyzw':[1,0,0,0]}


def matrix(q):
    return np.asarray(pose_matrix({'position_m':[0,0,0],'quat_xyzw':q}))[:3,:3]


class RegisteredReplayTests(unittest.TestCase):
    def test_world_reverse_and_translation_cap(self):
        R=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        out=target_from_action([.4,0,.3],[0,0,0,1],R,[1,0,0,0,0,0,-1])
        np.testing.assert_allclose(out['bounded_delta_I_m'],[0,-.0005,0],atol=1e-14)
        np.testing.assert_allclose(R@out['bounded_delta_I_m'],out['bounded_delta_L_m'],atol=1e-14)
        self.assertEqual(out['gripper_applied'],1)

    def test_rotation_keeps_tcp97_fixed_and_moves_ik107(self):
        out=target_from_action([.4,.1,.3],[0,0,0,1],np.eye(3),[0,0,0,1,0,0,1])
        a=ROTATION_CAP
        np.testing.assert_allclose(out['tcp97_target_world_m'],[.4,.1,.397],atol=1e-14)
        np.testing.assert_allclose(out['ik107_target_world_m'],[.4,.1-.01*math.sin(a),.397+.01*math.cos(a)],atol=1e-14)
        self.assertGreater(np.linalg.norm(np.array(out['ik107_target_world_m'])-[.4,.1,.407]),1e-5)

    def test_nonidentity_hand_and_root_roundtrip(self):
        # Hand just beyond pi preserves the model state's negative-w branch.
        q=[math.sin((math.pi+.01)/2),0,0,math.cos((math.pi+.01)/2)]
        reg=[0,0,math.sin(.3),math.cos(.3)];R=matrix(reg)
        out=target_from_action([.45,-.01,.35],q,R,[.4,-.3,.2,.3,.4,-.5,-1])
        target=np.array(out['tcp97_target_world_m']);qi=out['hand_target_quat_world_xyzw']
        np.testing.assert_allclose(np.array(out['ik107_target_world_m'])-matrix(qi)@np.array([0,0,.01]),target,atol=1e-14)
        root_q=[0,math.sin(.2),0,math.cos(.2)];root_p=np.array([.2,.3,.4])
        pr=rotate(inverse(root_q),np.array(out['ik107_target_world_m'])-root_p)
        qr=multiply(inverse(root_q),qi)
        np.testing.assert_allclose(matrix(root_q)@pr+root_p,out['ik107_target_world_m'],atol=1e-14)
        np.testing.assert_allclose(matrix(root_q)@matrix(qr),matrix(qi),atol=1e-14)
        # Independent Rodrigues rotation verifies left/world composition.
        v=np.array(out['bounded_rotation_I_rad']);a=np.linalg.norm(v);u=v/a
        K=np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])
        D=np.eye(3)+math.sin(a)*K+(1-math.cos(a))*(K@K)
        np.testing.assert_allclose(matrix(qi),D@matrix(q),atol=1e-14)
        self.assertAlmostEqual(np.linalg.norm(out['bounded_delta_I_m']),TRANSLATION_CAP)
        self.assertAlmostEqual(np.linalg.norm(v),ROTATION_CAP)

    def test_small_actions_are_not_amplified(self):
        out=target_from_action([0,0,0],[0,0,0,1],np.eye(3),[.001,0,0,0,0,.0001,0])
        np.testing.assert_allclose(out['bounded_delta_I_m'],[.00005,0,0])
        np.testing.assert_allclose(out['bounded_rotation_I_rad'],[0,0,.00005])

    def test_validate_entire_future_chunk_and_registration(self):
        self.assertEqual(validate_replay(fixture())['condition'],'flip_x')
        for edit in ('nan_future','short','reflection','quaternion','arms','condition'):
            item=copy.deepcopy(fixture())
            if edit=='nan_future':item['raw_action_chunk'][-1][-1]=float('nan')
            if edit=='short':item['raw_action_chunk']=item['raw_action_chunk'][:5]
            if edit=='reflection':item['world_rotation_L_from_I'][0][0]=-1
            if edit=='quaternion':item['registration_quat_xyzw']=[1,0,0,0]
            if edit=='arms':item['source_arm_positions_rad']=[0]*6
            if edit=='condition':item['condition']='no_flip'
            with self.subTest(edit=edit),self.assertRaises(ValueError):validate_replay(item)

    def test_representative_is_near_mean_not_best(self):
        queries=[{'condition':'flip_x','repeat':i+1,'first5':{'bounded_local_approach_projection_sum_mm':v}} for i,v in enumerate([0,1,2,3,4,5,6,7,8,100])]
        self.assertEqual(representative({'queries':queries})['repeat'],9)
        queries=[{'condition':'flip_x','repeat':i+1,'first5':{'bounded_local_approach_projection_sum_mm':i%2}} for i in range(10)]
        self.assertEqual(representative({'queries':queries})['repeat'],1)
        with self.assertRaises(ValueError):representative({'queries':queries[:9]})

    def test_runner_preserves_source_and_stops_after_child_failure(self):
        for fail in (False,True):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);clean=root/'clean';clean.mkdir();frozen=root/'frozen';frozen.mkdir()
                original=clean/'run.json';original.write_text('{}')
                report={'status':'completed','actions_executed':0,'source_manifest':str(original),
                    'source_manifest_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'snapshots':[]}
                for name in ('arm-only','arm-and-target-xy'):
                    (clean/name).mkdir();(clean/name/'request.json').write_text(json.dumps({'arm_positions_rad':[9]*7}))
                    geometry={'joint_names':[f'panda_joint{k}' for k in range(1,8)],'joint_positions_rad':[.001*k for k in range(7)]}
                    gb=json.dumps(geometry).encode();(frozen/(name+'_source_geometry.json')).write_bytes(gb)
                    report['snapshots'].append({'name':name,'source_geometry_sha256':hashlib.sha256(gb).hexdigest(),
                        'state_provenance':{'mapped_hand_quat_xyzw':[1,0,0,0],'registration_quat_xyzw':[0,0,0,1],
                            'registration_rotation':np.eye(3).tolist(),'registration_translation_m':[-.66,0,.912]},
                        'queries':[{'condition':'flip_x','repeat':k+1,'input_sha256':'source','state_sent_float32':[0]*8,
                            'raw_action_chunk':[[0]*7 for _ in range(10)],'first5':{'bounded_local_approach_projection_sum_mm':k}} for k in range(10)]})
                source=frozen/'report.json';source.write_text(json.dumps(report));commands=[]
                class Process:
                    def __init__(self,command,**kwargs):
                        commands.append(command);self.stdout=io.StringIO('')
                        self.assertions=kwargs
                        if not fail:
                            out=Path(command[command.index('--output-root')+1])/'capture';out.mkdir(parents=True)
                            (out/'registered_replay_result.json').write_text(json.dumps({'status':'completed',
                                'actual_initial_direction_approach_mm':.4,'command_initial_direction_approach_mm':.5,'final_target_reached':False}))
                    def __enter__(self):return self
                    def __exit__(self,*args):pass
                    def wait(self):return 1 if fail else 0
                argv=['probe','--source-report',str(source),'--output-root',str(root/'out')]
                with patch.object(sys,'argv',argv),patch.object(runner.subprocess,'Popen',Process):
                    self.assertEqual(runner.main(),1 if fail else 0)
                output=next((root/'out').iterdir());result=json.loads((output/'run.json').read_text())
                self.assertEqual(result['status'],'failed' if fail else 'completed')
                self.assertEqual(len(commands),1 if fail else 2)
                self.assertEqual((output/'source_frozen_report.json').read_bytes(),source.read_bytes())
                for command in commands:
                    self.assertIn('--registered-replay-json',command)
                    config=json.loads(Path(command[command.index('--clean-scene-json')+1]).read_text())
                    self.assertEqual(config['arm_positions_rad'],[.001*k for k in range(7)])
                if not fail:self.assertFalse(result['all_final_targets_reached'])


if __name__=='__main__':unittest.main()
