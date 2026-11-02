"""Declared coverage cases; each starts in an independent, distant-object scene."""

FINGER_PROFILES={'inherited':None,'kp1000-kd10':(1000.,10.),
                 'kp1000-kd50':(1000.,50.),'kp250-kd10':(250.,10.),
                 'kp1000-kd100':(1000.,100.),'kp350-kd50':(350.,50.),'kp350-kd100':(350.,100.)}


def coverage_cases():
    result=[]
    for profile in ('inherited','kp1000-kd10','kp1000-kd50','kp250-kd10'):
        result.append({'name':'finger-'+profile,'kind':'gripper','gripper_mode':'substep','cap_m':.0005,
            'finger_profile':profile,'requires':[] if profile=='inherited' else ['finger-inherited']})
    for deg in (.1,.5):
        for axis in range(3):
            for sign in (-1,1):
                suffix=f"r{'xyz'[axis]}-{'negative' if sign<0 else 'positive'}"
                result.append({'name':f'rotation-{deg:g}deg-'+suffix,'kind':'rotation','axis':axis,'sign':sign,
                    'rotation_cap_deg':deg,'cap_m':.0005,'gripper_mode':'binary',
                    'requires':[] if deg==.1 else ['rotation-0.1deg-'+suffix]})
    for mm,deg in ((2,.1),(5,.5)):
        for mode in ('binary','substep'):
            requires=[c['name'] for c in result if c['kind']=='rotation' and c['rotation_cap_deg']==deg]
            if mm==5:requires.append('combined-2mm-'+mode)
            result.append({'name':f'combined-{mm}mm-'+mode,'kind':'combined','cap_m':mm/1000,
                'rotation_cap_deg':deg,'gripper_mode':mode,'requires':requires})
    base=next(c for c in result if c['name']=='combined-2mm-substep')
    result.append({**base,'name':'combined-2mm-accumulated','reference_mode':'accumulated','requires':[base['name']]})
    result.append({**base,'name':'combined-2mm-preconditioned','prelude':True,'requires':[base['name']]})
    return result


def configure_finger_profile(actuators,profile,*,armature=None):
    """Set only active finger1; never add a drive to passive/mimic finger2."""
    import re
    if profile not in FINGER_PROFILES:raise ValueError('unknown finger profile')
    if armature is not None and (isinstance(armature,bool) or armature!=1.):raise ValueError('only declared armature=1 diagnostic supported')
    names=[f'panda_joint{i}' for i in range(1,8)]+['panda_finger_joint1','panda_finger_joint2']
    active=[];passive=[];audit=[]
    for name,actuator in actuators.items():
        patterns=actuator.joint_names_expr
        if isinstance(patterns,str):patterns=[patterns]
        matched=[j for j in names if any(re.fullmatch(p,j) for p in patterns)]
        if not any(j.startswith('panda_finger') for j in matched):continue
        if len(matched)!=1:raise ValueError('finger coverage requires separate active/passive actuator groups')
        if not getattr(actuator.class_type,'is_implicit_model',False):raise ValueError('expected implicit finger drive')
        audit.append({'group':name,'joint':matched[0],'original_stiffness':actuator.stiffness,
            'original_damping':actuator.damping,'original_armature':getattr(actuator,'armature',None),'model':actuator.class_type.__name__})
        (active if matched[0]=='panda_finger_joint1' else passive).append(actuator)
    if len(active)!=1 or len(passive)!=1:raise ValueError('expected one active finger and one passive finger')
    if passive[0].stiffness!=0 or passive[0].damping!=0:raise ValueError('finger2 is not the expected passive zero-gain drive')
    if armature is not None and not all(hasattr(a,'armature') for a in active+passive):raise ValueError('SDK config does not expose finger armature; no mutation applied')
    requested=FINGER_PROFILES[profile]
    if requested is not None:active[0].stiffness,active[0].damping=requested
    if armature is not None:
        for actuator in active+passive:actuator.armature=armature
    return {'profile':profile,'requested_active_gains':requested,'requested_finger_armature':armature,'groups':audit,
        'scope':'explicit exploratory active-finger gains; passive drive and effort limits unchanged; no equivalence to native kp assumed'}
