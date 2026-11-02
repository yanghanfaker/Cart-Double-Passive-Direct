"""Paired translation-cap sensitivity on saved policy responses; zero execution."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from libero_mapping import decode_libero_action
from low_pose_policy_audit import local_path, read, sha, stats
from openpi_step_probe import validate_action_chunk

CAPS_MM = (0.5, 1., 2., 5., 10., None)


def measure(actions, item, cap_mm):
    validate_action_chunk(actions)
    if len(actions) < 10:
        raise ValueError('at least ten finite 7D actions required')
    if cap_mm is not None and (not math.isfinite(cap_mm) or cap_mm <= 0):
        raise ValueError('cap must be positive finite or None')
    R = np.asarray(item['world_rotation_L_from_I'], dtype=float)
    tcp = np.asarray(item['tcp_world_m'], dtype=float)
    obj = np.asarray(item['object_world_m'], dtype=float)
    if (R.shape != (3, 3) or tcp.shape != (3,) or obj.shape != (3,)
            or not all(np.isfinite(x).all() for x in (R, tcp, obj))
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-5)
            or abs(np.linalg.det(R) - 1) > 1e-5):
        raise ValueError('invalid geometry / proper rotation')
    decoded = np.array([R.T @ np.array(decode_libero_action(a)['delta_world_m']) for a in actions[:10]])
    lengths = np.linalg.norm(decoded, axis=1)
    gains = np.ones(10)
    if cap_mm is not None:
        np.divide(cap_mm / 1000, lengths, out=gains, where=lengths > 0)
        gains = np.minimum(gains, 1.)
    vectors = decoded * gains[:, None]
    positions = tcp + np.vstack([np.zeros(3), np.cumsum(vectors, axis=0)])
    rel = tcp[:2] - obj[:2]
    unit = rel / np.linalg.norm(rel) if np.linalg.norm(rel) > 1e-9 else None
    close = next((i for i, a in enumerate(actions[:10]) if a[6] > 0), None)
    result = {'cap_mm': cap_mm, 'first_close_index': None if close is None else close + 1,
              'first_close_before_action_height_mm': None if close is None else float(positions[close, 2] - obj[2]) * 1000,
              'first_close_after_action_height_mm': None if close is None else float(positions[close + 1, 2] - obj[2]) * 1000}
    for n in (5, 10):
        delta = positions[n] - tcp
        result[f'first{n}'] = {
            'sum_I_mm': (delta * 1000).tolist(),
            'outward_projection_mm': None if unit is None else float(delta[:2] @ unit) * 1000,
            'endpoint_xy_mm': float(np.linalg.norm(positions[n, :2] - obj[:2])) * 1000,
            'endpoint_height_mm': float(positions[n, 2] - obj[2]) * 1000,
            'minimum_height_mm': float(np.min(positions[:n+1, 2]) - obj[2]) * 1000,
            'mean_per_action_retained_fraction': float(np.mean(gains[:n])),
            'path_retained_fraction': float(np.sum(lengths[:n] * gains[:n]) / np.sum(lengths[:n])) if np.sum(lengths[:n]) > 0 else None,
            'capped_actions': int(np.sum(gains[:n] < 1 - 1e-12)),
            'first_close_in_window': close is not None and close < n}
    return result


def evaluate(source):
    source = Path(source); root = source.parent; original = read(source)
    snapshots = original['snapshots']; repeats = original['repeats_per_snapshot']
    expected = {f'repeat-{r:02d}-{m}-height-{h}mm' for r in (1, 2, 3)
                for m in ('current', 'accumulated') for h in (100, 80, 60)}
    if (original['status'] != 'completed' or original['actions_executed'] != 0
            or original['simulation_steps'] != 0 or len(snapshots) != 18
            or {s['name'] for s in snapshots} != expected
            or repeats not in (3, 5, 10) or original['policy_queries'] != 18 * repeats):
        raise ValueError('complete frozen low-pose audit required')
    rows = []; files = {'source_report.json': source.read_bytes()}
    for item in snapshots:
        expected_name = f"repeat-{item['source_repeat']:02d}-{item['reference_mode']}-height-{item['height_requested_mm']}mm"
        if item['name'] != expected_name:
            raise ValueError('snapshot labels disagree')
        if len(item['queries']) != repeats or {q['repeat'] for q in item['queries']} != set(range(1, repeats+1)):
            raise ValueError('incomplete / duplicate repetitions')
        for q in item['queries']:
            if q['status'] != 'completed' or q['input_sha256'] != item['input']['input_sha256']:
                raise ValueError('query status or frozen fingerprint mismatch')
            payload = local_path(root, item['name'] + '/' + q['response_file']).read_bytes()
            if sha(payload) != q['response_sha256']:
                raise ValueError('response SHA mismatch')
            relative = f"responses/{item['name']}/{q['repeat']:02d}.json"
            files[relative] = payload
            conditions = [measure(json.loads(payload)['actions'], item, cap) for cap in CAPS_MM]
            # A regression check against the independently saved baseline audit.
            for n in (5, 10):
                old = q['analysis'][f'first{n}']['bridge_capped']['sum_I_mm']
                if not np.allclose(conditions[0][f'first{n}']['sum_I_mm'], old, atol=1e-8, rtol=0):
                    raise ValueError('baseline differs from saved bridge cap analysis')
            rows.append({'snapshot': item['name'], 'height_mm': item['height_requested_mm'],
                         'source_mode': item['reference_mode'], 'repeat': q['repeat'],
                         'response_file': relative, 'response_sha256': sha(payload), 'conditions': conditions})
    summary = []
    for height in (100, 80, 60):
        for index, cap in enumerate(CAPS_MM):
            group = [r for r in rows if r['height_mm'] == height]
            s = {'height_mm': height, 'cap_mm': cap, 'responses': len(group), 'source_snapshots': 6}
            closing = [r for r in group if r['conditions'][index]['first_close_index'] is not None]
            s['responses_with_close_in_first10'] = len(closing)
            for field in ('first_close_before_action_height_mm', 'first_close_after_action_height_mm'):
                means = [float(np.mean([r['conditions'][index][field] for r in closing if r['snapshot'] == name])) for name in sorted({r['snapshot'] for r in closing})]
                s[field + '_snapshot_means'] = stats(means) if means else None
            for n in (5, 10):
                v = [r['conditions'][index][f'first{n}'] for r in group]
                projections = [x['outward_projection_mm'] for x in v if x['outward_projection_mm'] is not None]
                s[f'first{n}'] = {'outward_responses': sum(x > 0 for x in projections), 'direction_defined_responses': len(projections),
                    'endpoint_below_object_center_responses': sum(x['endpoint_height_mm'] < 0 for x in v)}
                s[f'first{n}']['pre_cap_inward_to_outward_responses'] = sum(
                    r['conditions'][-1][f'first{n}']['outward_projection_mm'] is not None
                    and r['conditions'][-1][f'first{n}']['outward_projection_mm'] < 0
                    and r['conditions'][index][f'first{n}']['outward_projection_mm'] > 0 for r in group)
                for key in ('endpoint_height_mm', 'endpoint_xy_mm', 'mean_per_action_retained_fraction'):
                    means = [float(np.mean([r['conditions'][index][f'first{n}'][key] for r in group if r['snapshot'] == name])) for name in sorted({r['snapshot'] for r in group})]
                    s[f'first{n}'][key + '_snapshot_means'] = stats(means)
            summary.append(s)
    return {'status': 'completed', 'source_report_sha256': sha(source.read_bytes()),
        'actions_executed': 0, 'simulation_steps': 0, 'new_policy_queries': 0, 'responses': len(rows),
        'scope': 'paired translation-only fixed-start ideal integration of first 5/10 actions; no dynamics, IK, rotation execution, contact or replanning; no cap still retains OSC input clip and scaling',
        'timing': 'close index unchanged; before/after positions bracket simultaneous arm/gripper action, not actual closure time',
        'statistics': 'repeated responses share 18 inputs; summary means use six source snapshots per height, pooled source-mode labels; these are not independent robot trials',
        'warning': 'below object center is a geometric flag, not proof of table collision; no recommended physical cap is inferred',
        'summary': summary, 'rows': rows}, files


def main():
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-report', type=Path)
    p.add_argument('--output-root', type=Path, default=repo/'diagnostics/action_cap_audit')
    args = p.parse_args()
    source = args.source_report or next((f for f in sorted((repo/'diagnostics/low_pose_policy_audit').glob('*/report.json'), reverse=True) if read(f).get('status') == 'completed'), None)
    if source is None: p.error('no completed low-pose report')
    folder = args.output_root.absolute()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder.mkdir(parents=True, exist_ok=False)
    try:
        report, files = evaluate(source)
        for name, payload in files.items():
            dest = folder/name; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(payload)
        for s in report['summary']:
            v = s['first5']; print(f"[Cap audit] height={s['height_mm']}mm cap={s['cap_mm']}mm first5 outward={v['outward_responses']}/{v['direction_defined_responses']} height={v['endpoint_height_mm_snapshot_means']['mean']:.2f}mm", flush=True)
        code = 0
    except Exception as error:
        report = {'status': 'failed', 'error': str(error), 'error_type': type(error).__name__, 'actions_executed': 0}; code = 1
    (folder/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f"[Cap audit] {report['status']}: {folder/'report.json'}", flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
