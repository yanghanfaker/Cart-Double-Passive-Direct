import hashlib
import unittest
from run_official_libero_baseline import child_command, parse_log, verified_source


def log(flags):
    return '\n'.join(['INFO:root:Success: '+str(x) for x in flags]+[
        'INFO:root:Total success rate: '+str(sum(flags)/len(flags)),
        'INFO:root:Total episodes: '+str(len(flags))])


class OfficialLiberoBaselineTests(unittest.TestCase):
    def test_complete_negative_is_not_grasp_success(self):
        result=parse_log(log([False]*10))
        self.assertTrue(result['complete']);self.assertEqual(result['successes'],0)

    def test_complete_mixed(self):
        result=parse_log(log([True,False]*5))
        self.assertTrue(result['complete']);self.assertEqual(result['success_rate'],.5)

    def test_caught_exception_rejects_successful_exit(self):
        self.assertFalse(parse_log(log([True]*10)+'\nERROR:root:Caught exception: failed')['complete'])

    def test_partial_and_inconsistent_results_rejected(self):
        self.assertFalse(parse_log(log([True]*9))['complete'])
        self.assertFalse(parse_log(log([True]*10).replace('rate: 1.0','rate: 0.5'))['complete'])

    def test_source_hash_rejects_change(self):
        manifest={'sha256':hashlib.sha256(b'upstream').hexdigest()}
        self.assertEqual(verified_source(b'upstream',manifest),b'upstream')
        with self.assertRaises(ValueError):verified_source(b'patched',manifest)

    def test_venv_identity_and_unmodified_action_flags(self):
        args=child_command('/venv/bin/python','/out/main.py','/out/videos')
        self.assertEqual(args[0],'/venv/bin/python')
        self.assertIn('--args.replan-steps',args)
        self.assertEqual(args[args.index('--args.num-trials-per-task')+1],'1')
        self.assertEqual(args[args.index('--args.replan-steps')+1],'5')
        self.assertFalse(any(x in args for x in ('--headless','--viz','--cap-m','--force-open')))


if __name__=='__main__':unittest.main()
