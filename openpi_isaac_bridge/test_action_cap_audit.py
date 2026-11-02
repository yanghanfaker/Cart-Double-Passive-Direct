import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import action_cap_audit as cap
from low_pose_policy_audit import analyze, sha


def geometry():
    return {'tcp_world_m': [.01, 0, .1], 'object_world_m': [0, 0, 0],
            'world_rotation_L_from_I': np.eye(3).tolist(),
            'hand_position_world_m': [.01, 0, .003], 'hand_quat_world_xyzw': [0, 0, 0, 1]}


def fixture(root):
    report = {'status': 'completed', 'actions_executed': 0, 'simulation_steps': 0,
              'policy_queries': 54, 'repeats_per_snapshot': 3, 'snapshots': []}
    actions = [[.1, 0, -.2, 0, 0, 0, -1]]*7 + [[0, 0, 0, 0, 0, 0, 1]]*3
    for repeat in (1, 2, 3):
        for mode in ('current', 'accumulated'):
            for height in (100, 80, 60):
                name = f'repeat-{repeat:02d}-{mode}-height-{height}mm'
                item = dict(geometry(), name=name, height_requested_mm=height,
                            reference_mode=mode, source_repeat=repeat, input={'input_sha256': 'frozen'}, queries=[])
                folder = root/name; folder.mkdir()
                for i in (1, 2, 3):
                    payload = json.dumps({'actions': actions}).encode(); file = f'{i}.json'
                    (folder/file).write_bytes(payload)
                    item['queries'].append({'repeat': i, 'status': 'completed', 'input_sha256': 'frozen',
                        'response_file': file, 'response_sha256': sha(payload), 'analysis': analyze(actions, item)})
                report['snapshots'].append(item)
    (root/'report.json').write_text(json.dumps(report)); return root/'report.json'


class CapTests(unittest.TestCase):
    def test_known_motion_and_close_bracket(self):
        a = [[0, 0, -.4, 0, 0, 0, -1]]*7 + [[0, 0, -.4, 0, 0, 0, 1]]*3
        small = cap.measure(a, geometry(), .5)
        self.assertAlmostEqual(small['first_close_before_action_height_mm'], 96.5)
        self.assertAlmostEqual(small['first_close_after_action_height_mm'], 96.)
        self.assertEqual(small['first_close_index'], 8)
        full = cap.measure(a, geometry(), None)
        self.assertAlmostEqual(full['first_close_before_action_height_mm'], -40.)
        self.assertFalse(full['first5']['first_close_in_window'])
        self.assertTrue(full['first10']['first_close_in_window'])

    def test_sequence_can_change_direction(self):
        a = [[-1, 0, 0, 0, 0, 0, -1]] + [[.02, 0, 0, 0, 0, 0, -1]]*9
        self.assertLess(cap.measure(a, geometry(), None)['first5']['outward_projection_mm'], 0)
        self.assertGreater(cap.measure(a, geometry(), .5)['first5']['outward_projection_mm'], 0)

    def test_no_cap_still_clips_and_rotates(self):
        g = geometry(); g['world_rotation_L_from_I'] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        a = [[2, 0, 0, 0, 0, 0, -1]]*10
        np.testing.assert_allclose(cap.measure(a, g, None)['first5']['sum_I_mm'], [0, -250, 0], atol=1e-10)

    def test_zero_and_no_close(self):
        g = geometry(); g['tcp_world_m'] = [0, 0, .1]
        d = cap.measure([[0]*7]*10, g, .5)
        self.assertIsNone(d['first5']['outward_projection_mm'])
        self.assertIsNone(d['first5']['path_retained_fraction'])
        self.assertIsNone(d['first_close_index'])
        self.assertEqual(d['first5']['mean_per_action_retained_fraction'], 1)

    def test_reject_invalid_geometry_future_and_caps(self):
        a = [[0]*7]*10
        for c in (-1, 0, float('nan')):
            with self.assertRaises(ValueError): cap.measure(a, geometry(), c)
        bad = copy.deepcopy(a); bad[-1][0] = float('nan')
        with self.assertRaises(ValueError): cap.measure(bad, geometry(), .5)
        g = geometry(); g['world_rotation_L_from_I'][0][0] = -1
        with self.assertRaises(ValueError): cap.measure(a, g, .5)

    def test_complete_paired_report(self):
        with tempfile.TemporaryDirectory() as temp:
            source = fixture(Path(temp)); report, files = cap.evaluate(source)
            self.assertEqual(report['responses'], 54)
            self.assertEqual(len(report['summary']), 18)
            self.assertEqual(len(files), 55)
            self.assertEqual(report['new_policy_queries'], 0)
            self.assertTrue(all(len(r['conditions']) == 6 for r in report['rows']))

    def test_corrupt_response_and_missing_query_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = fixture(root); original = cap.read(source)
            broken = copy.deepcopy(original); broken['snapshots'][0]['queries'].pop()
            source.write_text(json.dumps(broken))
            with self.assertRaises(ValueError): cap.evaluate(source)
            source.write_text(json.dumps(original))
            q = original['snapshots'][0]; (root/q['name']/'1.json').write_text('{}')
            with self.assertRaises(ValueError): cap.evaluate(source)


if __name__ == '__main__': unittest.main()
