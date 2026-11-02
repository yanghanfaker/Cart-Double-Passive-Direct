import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest
from execution_sync_cases import sync_cases,gain_plan,ARM_PROFILES
from execution_alignment_probe import validate_case,motion_plan


class SyncTests(unittest.TestCase):
    def test_geometry_runner_forwards_requested_gain_file(self):
        import run_geometry_audit
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);request=root/'request.json';output=root/'out'
            gains,_=gain_plan(*self.source(),'d50');request.write_text(json.dumps(gains))
            seen=[]
            class Process:
                def __init__(self,command,**kwargs):
                    seen.append(command);self.stdout=[]
                    folder=Path(command[command.index('--output')+1]);(folder/'geometry_audit.json').write_text('{"status":"completed"}')
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def wait(self):return 0
            with patch('sys.argv',['run_geometry_audit.py','--isaaclab-root',str(root),'--output-root',str(output),'--joint-gains-json',str(request)]),patch.object(run_geometry_audit.subprocess,'Popen',Process):
                self.assertEqual(run_geometry_audit.main(),0)
            forwarded=Path(seen[0][seen[0].index('--joint-gains-json')+1])
            self.assertEqual(json.loads(forwarded.read_bytes()),gains)
            self.assertEqual(json.loads((forwarded.parent/'source_case.json').read_bytes())['joint_overrides'],gains)

    def source(self):
        overrides={f'panda_joint{i}':{'stiffness':k,'damping':d} for i,k,d in ((1,1125,25),(2,5400,120),(3,1125,25),(4,5400,120))}
        audit={'joint_names':[f'panda_joint{i}' for i in range(1,8)],'runtime':{
            'joint_stiffness':{'values':[1125,5400,1125,5400,900,900,900]},
            'joint_damping':{'values':[25,120,25,120,20,20,20]}}}
        return overrides,audit

    def test_twenty_cases_and_dependencies(self):
        seen=set();self.assertEqual(len(sync_cases()),20)
        for case in sync_cases():
            validate_case(case);self.assertTrue(set(case['requires'])<=seen);self.assertNotIn(case['name'],seen);seen.add(case['name'])

    def test_gain_changes_only_named_joints_and_fields(self):
        overrides,audit=self.source();original=copy.deepcopy((overrides,audit))
        for profile,changes in ARM_PROFILES.items():
            result,expected=gain_plan(overrides,audit,profile)
            for i in range(1,8):
                name=f'panda_joint{i}'
                for field in ('stiffness','damping'):
                    baseline=audit['runtime']['joint_'+field]['values'][i-1]
                    self.assertEqual(expected[field][i-1],changes.get(name,{}).get(field,baseline))
                if name in result and name not in changes:self.assertEqual(result[name],overrides[name])
        self.assertEqual((overrides,audit),original)

    def test_single_joint_damping_isolation(self):
        overrides,audit=self.source()
        result,actual=gain_plan(overrides,audit,'j1-d50')
        self.assertEqual(result['panda_joint1'],{'stiffness':1125,'damping':50})
        self.assertEqual(result['panda_joint3'],overrides['panda_joint3'])
        self.assertEqual(actual['stiffness'],audit['runtime']['joint_stiffness']['values'])

    def test_reject_changed_baseline_or_order(self):
        for kind in ('override','runtime','order'):
            overrides,audit=self.source()
            if kind=='override':overrides['panda_joint1']['damping']=30
            elif kind=='runtime':audit['runtime']['joint_stiffness']['values'][2]=900
            else:audit['joint_names'].reverse()
            with self.assertRaises(ValueError):gain_plan(overrides,audit,'d50')
        with self.assertRaises(ValueError):gain_plan(*self.source(),'unknown')

    def test_combined_controls_and_candidate_fingers(self):
        source=[[.1,.2,.3,.4,.5,.6,s] for s in [-1,-1,-1,-1,1]]
        closure=[]
        for case in sync_cases():
            if case['kind']!='combined':continue
            plan=motion_plan(case,source)
            if case.get('force_open'):self.assertTrue(all(a[6]==-1 for _,a in plan[:5]))
            else:
                self.assertEqual([a for _,a in plan[:5]],source);closure.append(case)
        self.assertEqual(len(closure),2)
        self.assertEqual({c['finger_profile'] for c in closure},{'kp350-kd50','kp1000-kd100'})


if __name__=='__main__':unittest.main()
