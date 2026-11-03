"""Matched online action-cap/prefix matrix with read-only geometry diagnostics."""
import copy


def online_caps_cases():
    from execution_hybrid import hybrid_cases
    reference=copy.deepcopy(hybrid_cases()[0])
    result=[{**reference,'name':'online-caps-reference-start','collision_geometry':True}]
    combinations=[(cap,prefix) for cap in (.0005,.001) for prefix in (1,5)]
    for repeat,order in ((1,combinations),(2,list(reversed(combinations)))):
        for cap,prefix in order:
            base=copy.deepcopy(next(c for c in hybrid_cases() if c.get('handoff_mode')=='geometry' and c['prefix_length']==prefix and c['repeat']==1))
            result.append({**base,'name':f'online-caps-{cap*1000:g}mm-p{prefix}-r{repeat}',
                'cap_m':cap,'repeat':repeat,'collision_geometry':True,
                'requires':['online-caps-reference-start'],'requires_success':['online-caps-reference-start']})
    from execution_limit_audit import limit_cases
    for prefix in (1,5):
        replay=copy.deepcopy(next(c for c in limit_cases() if c.get('limit_audit_role')=='replay' and c['source_prefix']==prefix and c['cap_m']==.002))
        result.append({**replay,'name':f'online-caps-replay-p{prefix}-2mm','collision_geometry':True,
            'requires':['online-caps-reference-start'],'requires_success':['online-caps-reference-start']})
    result.append({**reference,'name':'online-caps-reference-end','collision_geometry':True})
    return result
