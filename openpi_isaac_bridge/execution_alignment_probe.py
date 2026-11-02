"""No-contact execution diagnostics; native SDK imports are deferred."""
import math
import numpy as np


def cases():
    result=[{'name':'gripper-'+m,'kind':'gripper','gripper_mode':m,'cap_m':.0005}
            for m in ('binary','period','substep')]
    result.append({'name':'gripper-reverse','kind':'gripper','gripper_mode':'substep','cap_m':.0005,'reverse':True})
    result += [{'name':f'translation-{mm:g}mm','kind':'translation','gripper_mode':'binary','cap_m':mm/1000}
               for mm in (.5,1,2,5)]
    return result


def validate_case(case):
    from execution_coverage_cases import coverage_cases
    if case not in cases()+coverage_cases():raise ValueError('execution case must exactly match a declared no-contact diagnostic')
    return case


def ramp(value,sign,dt):
    if not all(math.isfinite(x) for x in (value,sign,dt)) or not 0<=value<=.04 or dt<=0:
        raise ValueError('invalid gripper ramp')
    return float(np.clip(value-.1*np.sign(sign)*dt,0,.04))


def ramp_schedule(reverse=False):
    # Native sign: positive closes, negative opens. Every entry is 50 ms.
    return [-1.]*4+([1.]*4+[-1.]*4+[1.]*12+[-1.]*12 if reverse else [1.]*16+[-1.]*16)


def motion_plan(case,actions):
    """Return (phase, raw action) without changing archived source actions."""
    if case['kind']=='gripper':return [('gripper',[0.]*6+[s]) for s in ramp_schedule(case.get('reverse',False))]
    plan=[]
    if case.get('prelude'):
        # Synthetic preconditioning at the same start: not native velocity restoration.
        plan += [('prelude',list(a[:3])+[0.,0.,0.,-1.]) for a in actions[:5]]
    for source in actions[:5]:
        a=list(source)
        if case['kind']=='rotation':
            a=[0.]*7;a[3+case['axis']]=float(case['sign']);a[6]=-1.
        elif case['kind']=='translation':a[3:6]=[0.,0.,0.];a[6]=-1.
        plan.append(('command',a))
    return plan+[('hold',None)]*20


def next_target(before,previous,R,raw,case):
    from registered_replay_probe import target_from_action
    from kinematic_attribution import rotate
    hand=before['hand_position_world_m'];quat=before['hand_quat_world_xyzw']
    if previous is not None and case.get('reference_mode')=='accumulated':
        quat=previous['hand_target_quat_world_xyzw']
        hand=(np.array(previous['tcp97_target_world_m'])-rotate(quat,[0,0,.097])).tolist()
    return target_from_action(hand,quat,R,raw,translation_cap=case['cap_m'],rotation_cap=math.radians(case.get('rotation_cap_deg',.1)))


class FingerOverride:
    def __init__(self,runtime,mode):
        self.r=runtime;self.mode=mode;self.value=.04;self.sign=-1.;self.calls=0
        self.term=runtime.base.action_manager.get_term('gripper_action')
        self.had='apply_actions' in self.term.__dict__;self.original=self.term.apply_actions
        self.term.apply_actions=self.apply

    def begin(self,sign):
        self.sign=sign;self.calls=0
        if self.mode=='period':self.value=ramp(self.value,sign,.05)

    def apply(self):
        import torch
        self.original();self.calls+=1
        if self.mode=='binary':return
        if self.mode=='substep':self.value=ramp(self.value,self.sign,float(self.r.base.physics_dt))
        target=torch.full((1,2),self.value,device=self.r.base.device,dtype=torch.float32)
        commands=getattr(self.r.robot.actuators,'target_command',None)
        if commands is not None and callable(getattr(commands,'set_position_index',None)):
            commands.set_position_index(value=target,joint_ids=self.r.fingers)
            commands.set_velocity_index(value=torch.zeros_like(target),joint_ids=self.r.fingers)
        else:
            self.r.robot.set_joint_position_target_index(target=target,joint_ids=self.r.fingers)
            self.r.robot.set_joint_velocity_target_index(target=torch.zeros_like(target),joint_ids=self.r.fingers)

    def close(self):
        if self.had:self.term.apply_actions=self.original
        else:del self.term.apply_actions


def run_alignment(runtime,R,settings,first,report,save):
    from physics_trace import PhysicsJointRecorder
    from registered_replay_probe import target_from_action
    from native_libero_rollout import save_json
    case=validate_case(settings['execution_case']);r=runtime;tensor=r.tensor
    if getattr(r.base,'_physics_handles_decimation',False):raise RuntimeError('substep tracing requires environment-owned decimation')
    if not math.isclose(float(r.base.physics_dt),.01,abs_tol=1e-9):raise RuntimeError('expected 10ms Isaac physics period')
    def optional(field,ids):
        try:
            data=tensor(getattr(r.robot.data,field))[0,ids].detach().cpu().numpy()
            if not np.isfinite(data).all():raise RuntimeError('nonfinite '+field)
            return {'value':data.tolist(),'status':'available'}
        except (AttributeError,NotImplementedError) as error:
            return {'value':None,'status':'unavailable','reason':str(error)}
    def targets(ids):
        commands=getattr(r.robot.actuators,'target_command',None)
        data=commands.position if commands is not None else r.robot.data.joint_pos_target
        values=tensor(data)[0,ids].detach().cpu().numpy()
        if not np.isfinite(values).all():raise RuntimeError('nonfinite target buffer')
        return values.tolist()
    # Read actual buffers, including USD-inherited active gains and zero passive drive.
    finger_audit={key:optional(key,r.fingers) for key in ('joint_stiffness','joint_damping','joint_effort_limits','joint_vel_limits')}
    finger_audit['asset']=str(getattr(getattr(r.robot.cfg,'spawn',None),'usd_path','unavailable'))
    finger_audit['note']='finger1 active; finger2 passive/mimic; authored USD fields may differ from resolved solver properties'
    from execution_coverage_cases import FINGER_PROFILES
    requested=FINGER_PROFILES[case.get('finger_profile','inherited')]
    if requested is not None:
        for key,expected in zip(('joint_stiffness','joint_damping'),requested):
            actual=finger_audit[key]
            if actual['status']!='available' or not np.allclose(actual['value'],[expected,0],atol=1e-4,rtol=0):
                raise RuntimeError('explicit active/passive finger gains not confirmed: '+key)
    try:
        from isaaclab.sim import get_current_stage
        stage=get_current_stage();entries=[]
        for prim in stage.Traverse():
            if 'finger' not in str(prim.GetPath()).lower():continue
            attrs={str(a.GetName()):str(a.Get()) for a in prim.GetAttributes() if str(a.GetName()).startswith(('physics:','physx','drive:'))}
            if attrs:entries.append({'path':str(prim.GetPath()),'attributes':attrs})
        finger_audit['usd_authored_attributes']={'status':'available','prims':entries}
    except (ImportError,AttributeError,RuntimeError) as error:
        finger_audit['usd_authored_attributes']={'status':'unavailable','reason':str(error)}
    save_json(r.output/'finger_dynamics_audit.json',finger_audit)
    dynamic=settings.get('native_dynamic_initial',settings['native_initial'])
    report['initial_velocity_comparison']={'native_joint_velocity_rad_s':dynamic.get('robot0_joint_vel'),
        'isaac_joint_velocity_rad_s':first['joint_velocities_rad_s'],
        'scope':'recorded native dynamic state versus held Isaac start; no native velocity restoration'}
    report['execution_envelopes']={'translation_m':r.period_translation_limit,'rotation_deg':r.period_rotation_limit_deg}
    save()
    def command():
        state=r.state();q=np.array(state['joint_positions_rad']);limits=tensor(r.robot.data.soft_joint_pos_limits)[0,r.joints].detach().cpu().numpy()
        finger_target=targets(r.fingers)
        if override is not None and override.mode!='binary' and not np.allclose(finger_target,[override.value]*2,atol=1e-7,rtol=0):
            raise RuntimeError('incremental gripper target overwritten before physics write')
        jac=tensor(r.arm._compute_frame_jacobian())[0].detach().cpu().numpy()
        if jac.shape!=(6,7) or not np.isfinite(jac).all():raise RuntimeError('expected finite actual 6x7 IK Jacobian')
        sv=np.linalg.svd(jac,compute_uv=False)
        return {'position_target_rad':targets(r.joints),'position_before_rad':state['joint_positions_rad'],
            'finger_target_m':finger_target,'finger_before_m':state['finger_positions_m'],
            'finger_applied_torque':optional('applied_torque',r.fingers),
            'finger_computed_torque':optional('computed_torque',r.fingers),
            'finger_effort_limits':optional('joint_effort_limits',r.fingers),
            'applied_torque':optional('applied_torque',r.joints),'computed_torque':optional('computed_torque',r.joints),
            'velocity_limits':optional('joint_vel_limits',r.joints),'effort_limits':optional('joint_effort_limits',r.joints),
            'joint_limit_margin_rad':np.minimum(q-limits[:,0],limits[:,1]-q).tolist(),
            'jacobian_root':jac.tolist(),'jacobian_singular_values':sv.tolist(),
            'jacobian_condition':float(sv[0]/sv[-1]) if sv[-1]>1e-12 else None,
            'jacobian_note':'mixed translation/rotation units; condition number depends on chosen units'}
    def state():
        s=r.state()
        return {'position_rad':s['joint_positions_rad'],'velocity_rad_s':s['joint_velocities_rad_s'],
            'finger_position_m':s['finger_positions_m'],'finger_velocity':optional('joint_vel',r.fingers),
            'hand_quat_world_xyzw':s['hand_quat_world_xyzw'],'tcp_world_m':s['tcp_world_m']}
    recorder=None;override=None
    try:
        recorder=PhysicsJointRecorder(r.base.scene,float(r.base.physics_dt),.05,command,state,r.output/'physics_joint_trace.jsonl',
            {'joint_names':[f'panda_joint{i}' for i in range(1,8)],'effort_note':'SDK fields; availability explicit; no solver-force measurement claim'})
        override=FingerOverride(r,case['gripper_mode'])
        target=target_from_action(first['hand_position_world_m'],first['hand_quat_world_xyzw'],R,[0]*7)
        plan=motion_plan(case,settings['raw_actions']);sign=-1.
        for i,(phase,planned) in enumerate(plan):
            before=r.state();raw=None
            if planned is not None:
                if planned[6]!=0:sign=float(np.sign(planned[6]))
                if case['kind']!='gripper':
                    raw=list(planned);target=next_target(before,target,R,raw,case)
            if phase=='command' and 'stream_initial' not in report:
                report['stream_initial']=before
                report['preconditioning_note']='synthetic five translation-only actions from the same held start; pose and velocity both change; not matched native dynamic replay' if case.get('prelude') else 'no synthetic preconditioning'
            override.begin(sign);recorder.begin_control({'case':case['name'],'phase':phase,'case_control_step':i+1})
            after=r.step(target,-1. if sign>0 else 1.,'close');recorder.end_control()
            if override.calls!=5:raise RuntimeError('gripper apply_actions not called exactly once per physics substep')
            if math.dist(after['tcp_world_m'],after['object_pos_world_m'])<.10 or math.dist(after['object_pos_world_m'],first['object_pos_world_m'])>.002:
                raise RuntimeError('no-contact condition violated')
            report['trace'].append({'step':i+1,'phase':phase,'before':before,'after':after,'raw_action':raw,'target':target,
                'native_gripper_sign':sign,'gripper_apply_calls':override.calls,'ramp_target_m':None if case['gripper_mode']=='binary' else override.value})
            save()
        report.update(status='completed',case=case,final=r.state(),physics_controls=recorder.control_count,physics_substeps=recorder.substep_count,
            scope='no-contact isolated gripper, rotation or coupled segment diagnostic; no new inference; not full grasp or native dynamics equivalence')
    finally:
        if override is not None:override.close()
        if recorder is not None:
            recorder.close()
            save_json(r.output/'execution_trace_status.json',{'controls':recorder.control_count,'substeps':recorder.substep_count,
                'complete':recorder.context is None,'case':case})
