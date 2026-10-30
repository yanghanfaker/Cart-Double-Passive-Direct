import copy
import math
import unittest

from frozen_coordinate_probe import CONDITIONS, rotate_vector, state_condition, validate_source, query_snapshot


class CoordinateProbeTests(unittest.TestCase):
    def test_tcp_offset_rotates_with_hand_not_world(self):
        source = [1., 2., 3., 0., math.pi/2, 0., .04, -.04]
        before = list(source)
        b, obj_b = state_condition(source, [2,3,4], CONDITIONS[1])
        c, obj_c = state_condition(source, [2,3,4], CONDITIONS[2])
        self.assertAlmostEqual(c[0]-b[0], -.0064)
        self.assertAlmostEqual(c[2]-b[2], 0)
        self.assertEqual(c[3:], source[3:])
        self.assertEqual(obj_b, obj_c)
        self.assertEqual(source, before)

    def test_world_translation_cancels_in_relative_direction(self):
        source = [.4,0,.3,0,0,0,.04,-.04]
        b, obj = state_condition(source, [.5,0,.1], CONDITIONS[1])
        for i in range(3):
            self.assertAlmostEqual(obj[i]-b[i], [.5,0,.1][i]-source[i])
        a, _ = state_condition(source, [.5,0,.1], CONDITIONS[0])
        self.assertEqual(a, source)
        self.assertIsNot(a, source)

    def test_signed_large_rotvec_and_zero_rotation(self):
        self.assertEqual(rotate_vector([0,0,0], [0,0,-.0064]), [0,0,-.0064])
        first = rotate_vector([0,math.pi/2,0], [0,0,1])
        second = rotate_vector([0,-3*math.pi/2,0], [0,0,1])
        for a,b in zip(first,second):
            self.assertAlmostEqual(a,b)

    def test_rejects_invalid_state(self):
        for state in ([0]*7, [float('nan')]*8, [True]*8):
            with self.assertRaises(ValueError):
                state_condition(state, [0,0,1], CONDITIONS[0])

    def test_real_saved_source_and_wrong_geometry_rejected(self):
        import json
        from pathlib import Path
        path = Path(__file__).resolve().parent.parent/'downloaded_gain_sweep/dc5dcaefdce0/report.json'
        if not path.exists():
            self.skipTest('local prior diagnostic fixture unavailable')
        source = json.loads(path.read_text())
        validate_source(source)
        source['snapshots'][0]['source_setup']['ee_sensor_offset_m'][2] = .107
        with self.assertRaises(ValueError):
            validate_source(source)

    def test_mock_queries_interleave_only_state_changes(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('numpy required for client mock')
        calls = []
        class Client:
            def infer(self, obs):
                calls.append(copy.deepcopy(obs))
                return {'actions': np.zeros((10,7))}
        images = {'external':np.zeros((224,224,3), dtype=np.uint8),
                  'wrist':np.ones((224,224,3), dtype=np.uint8)}
        item = {'name':'fixture', 'state':[.4,0,.3,0,math.pi/2,0,.04,-.04],
                'object_world_m_diagnostic_only':[.5,0,.1], 'prompt':'pick up the cube'}
        saved = []
        rows = query_snapshot(Client(),item,images,3,np,saved.append)
        self.assertEqual(len(rows),9)
        self.assertEqual([r['condition'] for r in rows], list(CONDITIONS)+list(CONDITIONS[1:]+CONDITIONS[:1])+list(CONDITIONS[2:]+CONDITIONS[:2]))
        for obs in calls:
            np.testing.assert_array_equal(obs['observation/image'], images['external'])
            np.testing.assert_array_equal(obs['observation/wrist_image'], images['wrist'])
            np.testing.assert_array_equal(obs['observation/state'][3:], np.asarray(item['state'], dtype=np.float32)[3:])
            self.assertEqual(obs['prompt'], item['prompt'])
        for condition in CONDITIONS:
            self.assertEqual(len({r['input_sha256'] for r in rows if r['condition']==condition}),1)
        self.assertTrue(all(not a['executed'] for r in rows for a in r['actions']))


if __name__ == '__main__':
    unittest.main()
