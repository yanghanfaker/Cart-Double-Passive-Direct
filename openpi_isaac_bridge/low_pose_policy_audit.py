"""Repeat frozen low-pose observations; audit decoded/capped directions, no execution."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import zipfile
import numpy as np

from frozen_coordinate_probe import digest_observation
from kinematic_attribution import rotate
from libero_mapping import decode_libero_action
from online_registered_probe import prepare_capture_observation
from openpi_health import check_openpi_health
from openpi_step_probe import read_server_metadata,validate_action_chunk
from registered_replay_probe import target_from_action
from run_online_reference_ab import validate_run_result,validate_initial

HEIGHTS=(100,80,60)
FILES=('geometry_audit.json','online_replay_request.json','online_model_response.json',
       'external_camera_policy224.png','external_camera_policy224_flip_x.png',
       'wrist_camera_policy224.png','wrist_camera_policy224_flip_x.png')


def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())


def local_path(root,name):
    root=Path(root).resolve();p=(root/name).resolve()
    if not p.is_relative_to(root):raise ValueError('source path escapes source root')
    return p


def tcp_from_query(q):
    p=q['input']['state_provenance']
    return np.array(p['source_sensor_position_world_m'])+rotate(p['source_hand_quat_xyzw'],[0,0,-.0064])


def select_query(result,height_mm):
    target=height_mm/1000;obj_z=result['initial']['object_pos_world_m'][2]
    queries=result['queries']
    if not queries:raise ValueError('source has no queries')
    q=min(queries,key=lambda q:(abs(float(tcp_from_query(q)[2])-obj_z-target),q['query']))
    if abs(float(tcp_from_query(q)[2])-obj_z-target)>.002:raise ValueError('no actual query within 2mm of requested height')
    return q


def prepare(source,output):
    source=Path(source);root=source.parent;output=Path(output);m=read(source)
    if m['status']!='completed' or m['profile']!='near-descent' or m['queries_per_run']!=120 or m['repeats']!=3 or not m['initial_inputs_identical'] or not m['all_followup_inputs_changed']:
        raise ValueError('completed six-run 120-query near-descent source required')
    expected={f'repeat-{r:02d}-{mode}' for r in (1,2,3) for mode in ('current','accumulated')}
    if len(m['cases'])!=6 or {c['name'] for c in m['cases']}!=expected:raise ValueError('missing or duplicate source runs')
    cb=(root/'world_registration.json').read_bytes()
    if sha(cb)!=m['world_registration_sha256']:raise ValueError('calibration SHA mismatch')
    cal=json.loads(cb);(output/'world_registration.json').write_bytes(cb);(output/'source_manifest.json').write_bytes(source.read_bytes())
    prepared=[];initial=None;source_gains=None;first_inputs=set()
    for c in m['cases']:
        path=local_path(root,c['result']);payload=path.read_bytes();d=json.loads(payload)
        if sha(payload)!=c['result_sha256']:raise ValueError('source result SHA mismatch')
        validate_run_result(d,c['reference_mode'],120,'near-descent')
        if d['stop_reason']!='near_boundary_reached' or not .05<=d['action_end_object_distance_m']<=.06 or not d['all_followup_inputs_changed']:
            raise ValueError('every source run must reach 60mm with updating inputs')
        if len(d['queries'])!=d['policy_queries'] or sum(q['executed_prefix_controls'] for q in d['queries'])!=d['action_controls']:raise ValueError('incomplete source query prefix counts')
        if initial is None:initial=d['initial']
        validate_initial(d['initial'],initial);first_inputs.add(d['first_input_sha256'])
        settings=read(root/c['name']/'online_settings.json');gains=read(path.parent/'joint_gains.json')
        if settings['calibration']!=cal or settings['expected_joint_gains']!=gains:raise ValueError('source settings disagree')
        if source_gains is None:source_gains=gains
        if gains!=source_gains:raise ValueError('source gains differ')
        (output/(c['name']+'_source_result.json')).write_bytes(payload)
        for height in HEIGHTS:
            q=select_query(d,height);name=f'{c["name"]}-height-{height}mm';folder=output/name;folder.mkdir()
            archive=local_path(path.parent,q['archive']);ab=archive.read_bytes()
            if sha(ab)!=q['archive_sha256']:raise ValueError('source archive SHA mismatch')
            (folder/'source_query.zip').write_bytes(ab)
            with zipfile.ZipFile(archive) as z:
                if len(set(z.namelist()))!=len(z.namelist()):raise ValueError('duplicate ZIP members')
                for file in FILES:(folder/file).write_bytes(z.read(file))
            g=read(folder/'geometry_audit.json');obs,request,meta=prepare_capture_observation(folder,cal,settings['prompt'],g)
            old=read(folder/'online_replay_request.json');response=read(folder/'online_model_response.json')['actions'];validate_action_chunk(response)
            if len(response)<10 or response!=old['raw_action_chunk']:raise ValueError('source response/request mismatch')
            if meta['input_sha256']!=q['input']['input_sha256'] or meta['input_sha256']!=old['source_input_sha256'] or meta['state_sent_float32']!=q['input']['state_sent_float32']:
                raise ValueError('reconstructed observation differs from actual sent source input')
            for key in meta['frames']:
                for digest in ('png_sha256','array_sha256'):
                    if meta['frames'][key][digest]!=q['input']['frames'][key][digest]:raise ValueError('source image digest mismatch')
            hand=g['sensor_body_world'];tcp=np.array(hand['position_m'])+rotate(hand['quat_xyzw'],[0,0,.097]);obj=np.array(g['object_root_world']['position_m'])
            # Native body and sensor transforms are stored through float32 buffers.
            sensor_gap=float(np.linalg.norm(tcp-tcp_from_query(q)))
            if sensor_gap>1e-7 or abs(tcp[2]-obj[2]-height/1000)>.002:raise ValueError('source capture height disagrees')
            item={'name':name,'reference_mode':c['reference_mode'],'source_repeat':c['repeat'],'height_requested_mm':height,
                'height_actual_mm':float(tcp[2]-obj[2])*1000,'sensor_body_tcp_gap_m':sensor_gap,'xy_offset_mm':float(np.linalg.norm(tcp[:2]-obj[:2]))*1000,
                'source_query':q['query'],'capture_physics_step':q['capture_physics_step'],'camera_frames':q['camera_frames'],
                'input':meta,'tcp_world_m':tcp.tolist(),'object_world_m':obj.tolist(),'hand_position_world_m':hand['position_m'],
                'hand_quat_world_xyzw':hand['quat_xyzw'],'world_rotation_L_from_I':cal['world_from_isaac_rotation'],
                'source_result_sha256':c['result_sha256'],'source_archive_sha256':q['archive_sha256'],'source_actions':response,'queries':[]}
            item['source_action_analysis']=analyze(response,item)
            (folder/'snapshot.json').write_text(json.dumps(item,indent=2,allow_nan=False),encoding='utf-8')
            prepared.append((item,obs,folder))
    if len(first_inputs)!=1 or len(prepared)!=18:raise ValueError('source initial inputs differ or snapshots incomplete')
    return prepared


def analyze(chunk,item):
    validate_action_chunk(chunk)
    if len(chunk)<10:raise ValueError('need at least ten finite 7D actions')
    R=np.array(item['world_rotation_L_from_I']);tcp=np.array(item['tcp_world_m']);obj=np.array(item['object_world_m'])
    if R.shape!=(3,3) or not np.isfinite(R).all() or not np.allclose(R.T@R,np.eye(3),atol=1e-5) or abs(np.linalg.det(R)-1)>1e-5:raise ValueError('invalid proper world rotation')
    rel=tcp[:2]-obj[:2];unit=rel/np.linalg.norm(rel) if np.linalg.norm(rel)>1e-9 else None
    decoded=[];bounded=[];retention=[];angles=[];checks=[];closes=[]
    for index,action in enumerate(chunk,1):
        d=decode_libero_action(action);v=np.array(d['delta_world_m']);norm=float(np.linalg.norm(v));gain=min(1,.0005/norm) if norm else 1.
        vI=R.T@v;vB=R.T@(v*gain);rot=np.array(d['delta_world_rad']);rot_norm=float(np.linalg.norm(rot))
        bounded_rot=R.T@(rot*min(1,math.radians(.1)/rot_norm)) if rot_norm else rot
        native=target_from_action(item['hand_position_world_m'],item['hand_quat_world_xyzw'],R,action)
        error=float(np.linalg.norm(vB-native['bounded_delta_I_m']))
        if error>1e-10 or np.linalg.norm(R@vI-v)>1e-10 or np.linalg.norm(bounded_rot-native['bounded_rotation_I_rad'])>1e-10:raise ValueError('offline action decode differs from active bridge or frame round-trip')
        if np.linalg.norm(vI)>1e-12 and np.linalg.norm(vB)>1e-12:angles.append(float(np.dot(vI,vB)/(np.linalg.norm(vI)*np.linalg.norm(vB))))
        decoded.append(vI);bounded.append(vB);retention.append(gain);checks.append(error)
        if action[6]>0:closes.append(index)
    report={'first_close_index':closes[0] if closes else None,'max_bridge_decode_error_m':max(checks),
        'minimum_decoded_capped_direction_cosine':min(angles) if angles else None,
        'scope':'translation-only hypothetical fixed-start integration; no robot/IK/physics execution; pre-cap is after controller input clip and OSC scaling'}
    for count in (5,10):
        windows={}
        for name,vectors in (('decoded_pre_cap',decoded),('bridge_capped',bounded)):
            s=np.sum(vectors[:count],axis=0);end=tcp+s
            windows[name]={'sum_I_mm':(s*1000).tolist(),'radial_outward_sum_mm':float(s[:2]@unit)*1000 if unit is not None else None,
                'endpoint_xy_mm':float(np.linalg.norm(end[:2]-obj[:2]))*1000,
                'endpoint_xy_change_mm':float(np.linalg.norm(end[:2]-obj[:2])-np.linalg.norm(rel))*1000,
                'endpoint_object_distance_mm':float(np.linalg.norm(end-obj))*1000}
        windows.update(close_requests=sum(i<=count for i in closes),translation_capped_actions=sum(x<1-1e-12 for x in retention[:count]),
            mean_translation_retained_fraction=float(np.mean(retention[:count])))
        if closes and closes[0]<=count:
            k=closes[0]-1
            windows['first_close_hypothetical_before_action_height_mm']={name:float(tcp[2]-obj[2]+np.sum(v[:k],axis=0)[2] if k else tcp[2]-obj[2])*1000 for name,v in (('decoded_pre_cap',decoded),('bridge_capped',bounded))}
        report[f'first{count}']=windows
    return report


def stats(values):return {'mean':statistics.mean(values),'sample_std':statistics.stdev(values) if len(values)>1 else None,'min':min(values),'max':max(values)}


def summarize(items):
    result={}
    for mode in ('current','accumulated'):
        result[mode]={}
        for height in HEIGHTS:
            snapshots=[{**i,'queries':[q for q in i['queries'] if q['status']=='completed']} for i in items if i['reference_mode']==mode and i['height_requested_mm']==height];group={'source_snapshots':len(snapshots),'completed_queries':sum(len(i['queries']) for i in snapshots)}
            for count in (5,10):
                window={}
                for condition in ('decoded_pre_cap','bridge_capped'):
                    means=[statistics.mean(q['analysis'][f'first{count}'][condition]['radial_outward_sum_mm'] for q in s['queries']) for s in snapshots if s['queries'] and s['queries'][0]['analysis'][f'first{count}'][condition]['radial_outward_sum_mm'] is not None]
                    values=[q['analysis'][f'first{count}'][condition]['radial_outward_sum_mm'] for s in snapshots for q in s['queries']];values=[v for v in values if v is not None]
                    window[condition]={'source_snapshot_means_mm':stats(means) if means else None,'outward_queries':sum(v>0 for v in values),'direction_defined_queries':len(values)}
                group[f'first{count}']=window
            result[mode][f'height_{height}mm']=group
    return result


def run_queries(client,prepared,repeats,report,save):
    for repeat in range(1,repeats+1):
        order=prepared if repeat%2 else prepared[::-1]
        for item,original,folder in order:
            obs={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in original.items()};fingerprint=digest_observation(obs)
            if fingerprint!=item['input']['input_sha256']:raise ValueError('frozen input changed before query')
            row={'repeat':repeat,'status':'querying','input_sha256':fingerprint,'actions_executed':0};item['queries'].append(row);report['inference_attempts']+=1;save()
            start=time.perf_counter();response=client.infer(obs);row['inference_wall_s']=time.perf_counter()-start;report['policy_queries']+=1
            chunk=response['actions'];chunk=chunk.tolist() if hasattr(chunk,'tolist') else chunk
            file=folder/f'response_{repeat:02d}.json'
            try:payload=json.dumps({'actions':chunk},indent=2,allow_nan=False).encode();file.write_bytes(payload);row.update(response_file=file.name,response_sha256=sha(payload));save()
            except (TypeError,ValueError):
                (folder/f'invalid_response_{repeat:02d}.txt').write_text(repr(chunk),encoding='utf-8');raise
            if digest_observation(obs)!=fingerprint:raise RuntimeError('policy client mutated frozen observation')
            row['analysis']=analyze(chunk,item);row['status']='completed';save()
            print(f'[Low pose audit] {item["name"]} repeat={repeat}/{repeats}; capped first5 outward={row["analysis"]["first5"]["bridge_capped"]["radial_outward_sum_mm"]}; executed=0',flush=True)


def main():
    repo=Path(__file__).resolve().parent.parent;p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-manifest',type=Path);p.add_argument('--output-root',type=Path,default=repo/'diagnostics/low_pose_policy_audit')
    p.add_argument('--server-host',default='127.0.0.1');p.add_argument('--server-port',type=int,default=8000);p.add_argument('--repeats',type=int,choices=(3,5,10),default=5)
    args=p.parse_args();sources=sorted((repo/'diagnostics/near_descent_reference_ab').glob('*/run.json'))
    source=args.source_manifest or next((s for s in reversed(sources) if read(s).get('status')=='completed'),None)
    if source is None:p.error('no completed near-descent source')
    folder=args.output_root.absolute()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');folder.mkdir(parents=True,exist_ok=False)
    report={'status':'preparing','actions_executed':0,'simulation_steps':0,'policy_queries':0,'inference_attempts':0,'repeats_per_snapshot':args.repeats,
        'source_manifest':str(source.absolute()),'source_manifest_sha256':sha(source.read_bytes()),'snapshots':[],
        'scope':'frozen exact source observations; unpaired stochastic inference; no new simulator, robot movement, or native LIBERO comparison',
        'height_selection':'nearest actual source query within 2mm; not exact checkpoint image',
        'summary_unit':'mean per source snapshot, then sample statistics across three source snapshots per mode/height; repeated queries are not independent robot trials'}
    def save():
        report['summary']=summarize(report['snapshots']);(folder/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    save()
    try:
        prepared=prepare(source,folder);report['snapshots']=[i for i,_,_ in prepared];save()
        report['server_health']=check_openpi_health(args.server_host,args.server_port)
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        client=WebsocketClientPolicy(host=args.server_host,port=args.server_port);report['server_metadata']=read_server_metadata(client);report['status']='running';save()
        run_queries(client,prepared,args.repeats,report,save);report['status']='completed';return 0
    except Exception as error:
        for item in report['snapshots']:
            for row in item['queries']:
                if row['status']=='querying':row.update(status='failed',error=str(error))
        report.update(status='failed',error_type=type(error).__name__,error=str(error));print(f'[Low pose audit] STOP: {error}',flush=True);return 1
    finally:save();print(f'[Low pose audit] Report: {folder/"report.json"}',flush=True)


if __name__=='__main__':raise SystemExit(main())
