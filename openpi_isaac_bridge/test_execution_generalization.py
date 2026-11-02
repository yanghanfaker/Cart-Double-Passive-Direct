import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from execution_generalization import POSES,generalization_cases,pose_request,validate_reference,prepare_pose
from execution_alignment_probe import validate_case


class GeneralizationTests(unittest.TestCase):
    def settings(self):
        return {'native_initial':{'robot0_joint_pos':[.1,.62,.12,-1.77,-.18,2.43,1.06]},
            'source_native_report_sha256':'native','static_reference_sha256':'static',
            'source_native_bddl_sha256':'task','calibration':{'rotation':'unchanged'}}

    def reference(self,request):
        q=request['joint_positions_rad'][:7];f=[.04,-.04]
        return {'status':'completed','policy_queries':0,'action_steps':0,'bddl_sha256':'task',
            'sim_time_before_forward':0.,'sim_time_after_capture':0.,
            'joint_assignments':[{'joint':n,'target':v,'actual':v} for n,v in
                zip([f'robot0_joint{i}' for i in range(1,8)]+['gripper0_finger_joint1','gripper0_finger_joint2'],q+f)],
            'observation':{'robot0_eef_pos':[.4,0,1.2],'robot0_eef_quat':[0,0,0,1],
                'robot0_joint_pos':q,'robot0_gripper_qpos':f}}

    def test_matrix_and_dependency_order(self):
        cases=generalization_cases();self.assertEqual(len(cases),27);seen=set()
        for c in cases:
            validate_case(c);self.assertTrue(set(c['requires'])<=seen)
            self.assertNotIn(c['name'],seen);seen.add(c['name'])
            if c['kind']=='rotation':self.assertEqual(c['cap_m'],.0005)
        for pose in POSES:
            cross=[c for c in cases if c['pose_variant']==pose and c['arm_profile']!='baseline' and c['kind']=='combined' and c['rotation_cap_deg']==.1]
            self.assertEqual({(c['arm_profile'],c['finger_profile']) for c in cross},
                {(a,f) for a in ('d50','k1687p5-d50') for f in ('kp350-kd50','kp1000-kd100')})
            self.assertEqual(len([c for c in cases if c['pose_variant']==pose]),9)

    def test_declared_joint_offsets_and_no_stale_geometry(self):
        settings=self.settings();original=copy.deepcopy(settings)
        for variant,changes in POSES.items():
            request=pose_request(settings,variant)
            for i,q in enumerate(settings['native_initial']['robot0_joint_pos']):
                self.assertAlmostEqual(request['joint_positions_rad'][i],q+changes.get(i,0))
            self.assertNotIn('robot0_eef_pos',request)
            self.assertNotIn('robot_eef_body',request)
        self.assertEqual(settings,original)
        with self.assertRaises(ValueError):pose_request(settings,'other')

    def test_reference_rejects_tampering_and_motion(self):
        request=pose_request(self.settings(),'center');reference=self.reference(request)
        self.assertEqual(validate_reference(request,reference,'task'),reference['observation'])
        mutations=[lambda r:r.update(bddl_sha256='other'),lambda r:r.update(action_steps=1),
            lambda r:r.update(sim_time_after_capture=.01),
            lambda r:r['joint_assignments'][0].update(actual=.5),
            lambda r:r['joint_assignments'][0].update(actual=float('nan')),
            lambda r:r['joint_assignments'][0].update(joint='robot0_joint2'),
            lambda r:r['observation'].update(robot0_gripper_qpos=[.04,.04]),
            lambda r:r['observation'].update(robot0_eef_quat=[0,0,0,0])]
        for mutate in mutations:
            r=copy.deepcopy(reference);mutate(r)
            with self.assertRaises((ValueError,AssertionError)):validate_reference(request,r,'task')

    def test_cache_reuse_preserves_sources_and_checks_digests(self):
        from test_clean_scene_probe import fixtures
        from run_clean_scene_capture import prepare_cases
        settings=self.settings();original=copy.deepcopy(settings);config=prepare_cases(*fixtures())[0]
        original_config=copy.deepcopy(config)
        request=pose_request(settings,'shoulder-up');reference=self.reference(request)
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'shoulder-up';folder.mkdir()
            (folder/'isaac_source.json').write_text(json.dumps(request))
            payload=json.dumps(reference).encode();(folder/'reference.json').write_bytes(payload)
            manifest={'status':'completed','request_sha256':hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest(),
                'reference_sha256':hashlib.sha256(payload).hexdigest()}
            (folder/'run.json').write_text(json.dumps(manifest))
            with patch('execution_generalization.subprocess.Popen',side_effect=AssertionError('cache must not launch a worker')):
                result,scene,files=prepare_pose(settings,config,'shoulder-up',Path(temp))
            self.assertEqual(result['native_initial']['robot0_joint_pos'],request['joint_positions_rad'][:7])
            self.assertEqual(scene['object_xy_world_m'],[.7,-.25])
            self.assertEqual(set(files),{'run.json','reference.json','isaac_source.json'})
            self.assertEqual(settings,original);self.assertEqual(config,original_config)
            result['native_initial']['robot0_joint_pos'][0]=99
            self.assertEqual(settings,original)
            (folder/'reference.json').write_bytes(payload+b' ')
            with self.assertRaisesRegex(ValueError,'digest'):prepare_pose(settings,config,'shoulder-up',Path(temp))
            (folder/'reference.json').write_bytes(payload)
            altered=copy.deepcopy(settings);altered['source_native_report_sha256']='other'
            with self.assertRaisesRegex(ValueError,'another source'):prepare_pose(altered,config,'shoulder-up',Path(temp))


if __name__=='__main__':unittest.main()
