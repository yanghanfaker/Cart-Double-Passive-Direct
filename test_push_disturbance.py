"""Run without Isaac Lab: python3 -m unittest -v test_push_disturbance.py."""

import ast
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from push_disturbance import RandomPushScheduler


def scheduler(env_ids=(0,), **overrides):
    params = dict(env_ids=env_ids, dt=1/60, seed=42, start_s=5.0,
                  force_min=2.0, force_max=5.0,
                  duration_min_s=0.05, duration_max_s=0.15,
                  wait_min_s=2.0, wait_max_s=4.0)
    params.update(overrides)
    return RandomPushScheduler(**params)


class RandomPushTests(unittest.TestCase):
    def test_reproducibility_and_seed(self):
        a, b, c = scheduler(), scheduler(), scheduler(seed=43)
        trace_a = [a.step() for _ in range(6000)]
        self.assertEqual(trace_a, [b.step() for _ in range(6000)])
        self.assertNotEqual(trace_a, [c.step() for _ in range(6000)])

    def test_pulse_bounds_duration_gap_and_direction(self):
        s = scheduler()
        events = [event for _ in range(20000) for event in s.step()]
        self.assertGreater(len(events), 50)
        self.assertEqual(len(events) % 2, 0)
        signs = set()
        last_stop = None
        for start, stop in zip(events[::2], events[1::2]):
            self.assertEqual(start.env_id, 0)
            self.assertTrue(2 <= abs(start.force) <= 5)
            self.assertEqual(stop.force, 0)
            self.assertEqual(stop.step - start.step, start.duration_steps)
            self.assertTrue(3 <= start.duration_steps <= 9)
            if last_stop is None:
                self.assertTrue(math.ceil(7/s.dt) <= start.step <= math.ceil(9/s.dt))
            else:
                self.assertTrue(120 <= start.step-last_stop <= 240)
            last_stop = stop.step
            signs.add(start.force > 0)
        self.assertEqual(signs, {False, True})

    def test_selected_environment_and_independent_rng(self):
        one = scheduler((3,))
        many = scheduler((0, 3))
        trace_one, trace_three = [], []
        for tick in range(6000):
            if tick % 150 == 0:
                many.reset([0])
            trace_one.extend(one.step())
            trace_three.extend(event for event in many.step() if event.env_id == 3)
        self.assertEqual(trace_one, trace_three)
        self.assertTrue(all(event.env_id == 3 for event in trace_one))

    def test_reset_cancels_old_pulse_and_restarts_grace(self):
        s = scheduler()
        while not s.step():
            pass
        reset_tick = s.tick
        s.reset([0])
        for _ in range(420):
            self.assertEqual(s.step(), [])
        while not (events := s.step()):
            pass
        self.assertGreaterEqual(events[0].step, reset_tick + 420)
        self.assertNotEqual(events[0].force, 0)

    def test_invalid_ranges(self):
        for options in (dict(force_min=0), dict(force_max=1), dict(dt=0),
                        dict(wait_min_s=0), dict(duration_max_s=0.01),
                        dict(start_s=-1), dict(force_max=float('nan'))):
            with self.subTest(options=options), self.assertRaises(ValueError):
                scheduler(**options)

    def test_installer_copies_helper_and_is_idempotent(self):
        root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'source/isaaclab_tasks/isaaclab_tasks/core/cartpole'
            target.mkdir(parents=True)
            (target / 'cartpole_direct_env.py').write_text('# fixture\n')
            (target / '__init__.py').write_text('# original registration\n')
            for _ in range(2):
                subprocess.run([sys.executable, str(root / 'install_passive_double.py'), temp],
                               check=True, capture_output=True, text=True)
            text = (target / '__init__.py').read_text()
            self.assertEqual(text.count('from . import passive_double_cartpole'), 1)
            for name in ('passive_double_cartpole.py', 'push_disturbance.py'):
                self.assertEqual((target / name).read_bytes(), (root / name).read_bytes())
                ast.parse((target / name).read_text(encoding='utf-8'))

    def test_force_target_only_changes_selected_cart(self):
        # Execute the actual environment method with small matrix stand-ins.
        # Checks per-environment wiring without importing Isaac Sim or torch.
        class Matrix:
            def __init__(self, values): self.values = values
            def __add__(self, other):
                return Matrix([[a+b for a,b in zip(ar,br)]
                               for ar,br in zip(self.values,other.values)])
            def __setitem__(self, key, values):
                rows, columns = key
                for index, row in enumerate(range(len(self.values))[rows]):
                    for j, column in enumerate(columns):
                        self.values[row][column] = values.values[index][j]
        root = Path(__file__).resolve().parent
        tree = ast.parse((root / 'passive_double_cartpole.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n,ast.ClassDef)
                   and n.name == 'PassiveDoubleCartpoleEnv')
        method = next(n for n in cls.body if isinstance(n,ast.FunctionDef)
                      and n.name == '_apply_action')
        scope = {'torch': SimpleNamespace(zeros_like=lambda matrix: Matrix(
            [[0.0 for _ in row] for row in matrix.values]))}
        exec(compile(ast.Module(body=[method],type_ignores=[]),'<actual force method>','exec'),scope)
        captured = []
        robot = SimpleNamespace(
            data=SimpleNamespace(joint_pos=SimpleNamespace(torch=Matrix([[0,0,0] for _ in range(4)]))),
            set_joint_effort_target_index=lambda target:captured.append(target.values),
        )
        env = SimpleNamespace(cartpole=robot,_cart_idx=[0],
                              _force=Matrix([[1],[1],[1],[1]]),
                              _push=Matrix([[25],[0],[0],[0]]))
        scope['_apply_action'](env)
        self.assertEqual(captured[0],[[26,0,0],[1,0,0],[1,0,0],[1,0,0]])


if __name__ == '__main__':
    unittest.main()
