"""Read native Panda command state and count format_action calls without changing outputs."""
import hashlib
import inspect
import numpy as np
from native_libero_replay import ReplayClient


class NativeGripperClient(ReplayClient):
    def __init__(self,responses):
        super().__init__(responses);self.gripper=None;self.calls=0;self.period_calls=0;self.metadata=None

    def before_step(self,env):
        grip=env.robots[0].gripper
        if grip is not self.gripper:
            self.gripper=grip;original=grip.format_action
            def counted(action):
                self.calls+=1
                return original(action)
            grip.format_action=counted
            source=inspect.getsource(type(grip))
            self.metadata={'class':type(grip).__name__,'speed':float(grip.speed),
                'class_source':source,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
                'note':'wrapper counts calls and returns original output unchanged; no target override'}
        self.period_calls=self.calls

    def observe_step(self,env):
        if env.robots[0].gripper is not self.gripper:raise RuntimeError('gripper changed during control step')
        sim=env.sim;actuators=[]
        for name in self.gripper.actuators:
            aid=sim.model.actuator_name2id(name)
            actuators.append({'name':name,'id':int(aid),'ctrl':float(sim.data.ctrl[aid]),
                'ctrlrange':np.asarray(sim.model.actuator_ctrlrange[aid]).tolist(),
                'transmission_type':int(sim.model.actuator_trntype[aid]),
                'gain_parameters':np.asarray(sim.model.actuator_gainprm[aid]).tolist(),
                'bias_parameters':np.asarray(sim.model.actuator_biasprm[aid]).tolist()})
        if not actuators:raise RuntimeError('no native gripper actuators')
        state=np.asarray(self.gripper.current_action,dtype=float)
        if not np.isfinite(state).all() or not all(np.isfinite(a['ctrl']) for a in actuators):raise RuntimeError('nonfinite native gripper command')
        result={'format_action_calls_this_control':self.calls-self.period_calls,'format_action_calls_total':self.calls,
            'normalized_current_action':state.tolist(),'actuators':actuators,
            'scope':'native command state after one 50ms env.step; ctrl is actuator input, not measured force or necessarily joint position'}
        if self.metadata is not None:result['implementation']=self.metadata;self.metadata=None
        return result
