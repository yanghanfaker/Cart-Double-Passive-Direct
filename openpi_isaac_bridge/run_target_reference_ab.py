"""Compare target conventions on a validated prefix of the failed grasp stream."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
import numpy as np
from clean_scene_probe import validate_clean_scene
from registered_replay_probe import validate_replay


def digest(payload):return hashlib.sha256(payload).hexdigest()


def prepare_stream(root,data,steps):
    """Take executed first-five prefixes, never unused future model actions."""
    result_bytes=(root/data['result']).read_bytes()
    if digest(result_bytes)!=data['result_sha256']:raise ValueError('source result SHA mismatch')
    result=json.loads(result_bytes);capture=(root/data['result']).parent
    if result['status']!='completed' or result['mode']!='openpi-grasp' or result['native_controls_executed']<steps:raise ValueError('need completed grasp experiment with enough recorded controls')
    events=[json.loads(line) for line in (capture/'motion.jsonl').read_text().splitlines()]
    attempts=[r for r in events if r['event']=='attempt'][:steps];responses=[r for r in events if r['event']=='result'][:steps]
    if len(attempts)!=steps or len(responses)!=steps:raise ValueError('missing source motion trace')
    actions=[];provenance=[];first=None;geometry=None
    for index in range(steps//5):
        query=result['queries'][index]
        if query['query']!=index+1 or query['status']!='validated':raise ValueError('invalid source query order/status')
        archive=capture/query['archive'];payload=archive.read_bytes()
        if digest(payload)!=query['archive_sha256']:raise ValueError('source query archive SHA mismatch')
        with zipfile.ZipFile(archive) as z:
            request=validate_replay(json.loads(z.read('online_replay_request.json')))
            if request['source_input_sha256']!=query['input']['input_sha256']:raise ValueError('source input digest mismatch')
            if json.loads(z.read('online_model_response.json'))['actions']!=request['raw_action_chunk']:raise ValueError('model response and replay request differ')
            if first is None:first=request;geometry=json.loads(z.read('geometry_audit.json'))
            for prefix,action in enumerate(request['raw_action_chunk'][:5],1):
                n=index*5+prefix-1;a=attempts[n];r=responses[n];target=a['target']
                if a['control']!=n+1 or r['control']!=n+1 or a['phase']!='approach' or r['status']!='checked' or r['physics_substeps']!=5 or a['gripper_applied']!=1 or target['source_query']!=index+1 or target['source_prefix_index']!=prefix or target['raw_model_action']!=action:
                    raise ValueError('source prefix was not actually executed open-gripper')
                actions.append(action)
        provenance.append({'query':index+1,'archive':query['archive'],'archive_sha256':query['archive_sha256'],'input_sha256':query['input']['input_sha256']})
    return validate_replay({**first,'raw_action_chunk':actions}),geometry,provenance,result_bytes,capture


def main():
    repo=Path(__file__).resolve().parent.parent;parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest',type=Path)
    parser.add_argument('--output-root',type=Path,default=repo/'diagnostics/target_reference_ab')
    parser.add_argument('--isaaclab-root',type=Path,default=Path('/home/test/Robot/IsaacLab'))
    parser.add_argument('--steps',type=int,choices=(100,200),default=200)
    args=parser.parse_args()
    sources=sorted((repo/'diagnostics/openpi_grasp').glob('*/run.json'))
    source=args.source_manifest or next((p for p in reversed(sources) if json.loads(p.read_bytes()).get('status')=='completed' and json.loads(p.read_bytes()).get('native_controls_executed',0)>=args.steps),None)
    if source is None:parser.error('no completed grasp trace with enough executed actions')
    source=source.expanduser().absolute();data=json.loads(source.read_bytes())
    folder=args.output_root.expanduser().absolute()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');folder.mkdir(parents=True,exist_ok=False)
    (folder/'source_grasp_manifest.json').write_bytes(source.read_bytes())
    report={'status':'preparing','policy_queries':0,'source_manifest':str(source),'source_sha256':digest(source.read_bytes()),'steps':args.steps,'cases':[],
        'scope':'two independent initializations, same recorded action prefixes; current versus accumulated targets; open gripper, no new inference'}
    def save():(folder/'run.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    save();print(f'[Reference AB runner] Output: {folder}',flush=True)
    try:
        request,geometry,provenance,result_bytes,capture=prepare_stream(source.parent,data,args.steps)
        config=validate_clean_scene(json.loads((source.parent/'clean_scene_request.json').read_bytes()))
        config['arm_positions_rad']=request['source_arm_positions_rad']
        validate_clean_scene(config)
        calibration_bytes=(source.parent/'world_registration.json').read_bytes()
        if digest(calibration_bytes)!=data['world_registration_sha256']:raise ValueError('source calibration SHA mismatch')
        calibration=json.loads(calibration_bytes)
        if not calibration.get('candidate_geometry_within_thresholds'):raise ValueError('validated calibration required')
        for key,rkey,ckey in [('reference_world_rotation','world_rotation_L_from_I','world_from_isaac_rotation'),('reference_world_translation_m','world_translation_L_from_I_m','world_from_isaac_translation_m')]:
            if not np.allclose(config[key],calibration[ckey],atol=1e-12,rtol=0) or not np.allclose(request[rkey],calibration[ckey],atol=1e-12,rtol=0):raise ValueError('scene/source registration mismatch')
        gains=json.loads((capture/'joint_gains.json').read_bytes())
        stream_bytes=json.dumps(request['raw_action_chunk'],separators=(',',':'),allow_nan=False).encode();stream_sha=digest(stream_bytes)
        (folder/'fixed_actions.json').write_bytes(stream_bytes);(folder/'source_grasp_result.json').write_bytes(result_bytes)
        (folder/'source_geometry.json').write_text(json.dumps(geometry,indent=2),encoding='utf-8')
        (folder/'world_registration.json').write_bytes(calibration_bytes)
        (folder/'source_queries.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
        report.update(status='running',fixed_stream_sha256=stream_sha,world_registration_sha256=digest(calibration_bytes));save()
        for mode in ('current','accumulated'):
            target=folder/mode;target.mkdir();cf=target/'clean_scene_request.json';cf.write_text(json.dumps(config,indent=2),encoding='utf-8')
            settings={'mode':'target-reference-ab','reference_mode':mode,'steps':args.steps,'replay_request':request,
                'calibration':calibration,'fixed_stream_sha256':stream_sha,'expected_joint_gains':gains,
                'source_object_position_world_m':geometry['object_root_world']['position_m']}
            sf=target/'reference_settings.json';sf.write_text(json.dumps(settings,indent=2),encoding='utf-8')
            command=[sys.executable,str(Path(__file__).with_name('run_geometry_audit.py')),'--isaaclab-root',str(args.isaaclab_root.absolute()),
                '--output-root',str(target/'capture'),'--clean-scene-json',str(cf),'--online-registered-json',str(sf)]
            row={'mode':mode,'status':'running','command':command};report['cases'].append(row);save()
            env=os.environ.copy();env.pop('VIRTUAL_ENV',None);env.pop('PYTHONPATH',None)
            with (target/'runner.log').open('w',encoding='utf-8') as log:
                with subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace') as child:
                    for line in child.stdout:
                        log.write(line);log.flush()
                        if '[Geometry runner]' in line or '[Target reference]' in line:print(line,end='',flush=True)
                    if child.wait()!=0:raise RuntimeError(f'comparison stopped; inspect {target/"runner.log"}')
            outputs=list((target/'capture').glob('*/online_registered_result.json'))
            if len(outputs)!=1:raise RuntimeError('expected one comparison result')
            result=json.loads(outputs[0].read_bytes())
            if result['status']!='completed' or result['action_controls']!=args.steps or result['hold_controls']!=10 or result['reference_mode']!=mode or result['fixed_stream_sha256']!=stream_sha:raise RuntimeError('comparison incomplete or source mismatch')
            row.update(status='completed',result=outputs[0].relative_to(folder).as_posix(),result_sha256=digest(outputs[0].read_bytes()),
                final_ideal_position_error_m=result['final_ideal_position_error_m'],final_applied_target_error_m=result['final_applied_target_error_m']);save()
        report['status']='completed';return 0
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        for row in report['cases']:
            if row['status']=='running':row['status']='failed'
        print(f'[Reference AB runner] STOP: {error}',flush=True);return 1
    finally:
        save();print(f'[Reference AB runner] Manifest: {folder/"run.json"}',flush=True)


if __name__=='__main__':raise SystemExit(main())
