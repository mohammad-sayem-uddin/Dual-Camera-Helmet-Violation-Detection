"""TASK 4: forensic breakdown of all 209 decided GT rider tracks (Stage K helmet track-level).

Reproduces Stage K's exact extraction (work/stageK/extract.py + evaluate.py track-level block),
then enriches each track with aggregation diagnostics from Stage J tracks_final.json,
the release validity layer, and per-observation state sequences on both sides.

Outputs: work/postcompletion/helmet_track_breakdown.csv + analysis_summary.json.
Read-only w.r.t. all frozen artifacts.
"""
import json, csv, glob, os, collections, statistics

ROOT = os.path.expanduser('~/workspace/research_helmet')
REL = ROOT + '/repo/data/dual_camera_test_v1/releases/dual_camera_gt_v1.0_ai_assisted_audited'
SJ = ROOT + '/work/stageJ/outputs/stagej_out'
OUT = ROOT + '/work/postcompletion'
PAIRS = [f'pair_{i:03d}' for i in range(1, 26)]
CAMS = ('front', 'back')

# ---------------------------------------------------------------- GT side
gt_tracks = {}   # (pair, cam, sub) -> record
gt_parent_span = {}  # (pair, cam, moto_id) -> (first_frame, last_frame)
for pf in sorted(glob.glob(REL + '/pairs/*/pair_*_gt.json')):
    g = json.load(open(pf)); pr = g['pair']; pair = pr['pair_id']
    for cam in CAMS:
        ca = pr['camera_annotations'][cam]
        for mt in ca['motorcycle_tracks']:
            gt_parent_span[(pair, cam, mt['track_id'])] = (mt['first_frame_index'], mt['last_frame_index'])
            for st in mt['rider_subtracks']:
                gt_tracks[(pair, cam, st['subtrack_id'])] = {
                    'kind': 'moto_subtrack', 'parent': mt['track_id'],
                    'role': st['slot_role'], 'slot_index': st['slot_index'],
                    'gt_state': st['gt_helmet_state'], 'track_state': st['track_helmet_state'],
                    'provenance': st['state_provenance'],
                    'obs': [(o['frame_index'], o['state'], o['confidence'], o['bbox_xyxy'],
                             o['pts_norm_sec']) for o in st['head_observations']],
                    'n_obs_field': st['n_observations']}
        for ot in ca['orphan_rider_tracks']:
            gt_tracks[(pair, cam, ot['track_id'])] = {
                'kind': 'orphan', 'parent': ot['track_id'],
                'role': 'orphan', 'slot_index': None,
                'gt_state': ot['gt_helmet_state'], 'track_state': ot['track_helmet_state'],
                'provenance': ot['state_provenance'],
                'obs': [(o['frame_index'], o['state'], o['confidence'], o['bbox_xyxy'],
                         o['pts_norm_sec']) for o in ot['head_observations']],
                'n_obs_field': len(ot['head_observations'])}

# validity layer: per (pair, cam, subtrack) placement-class counts
validity = collections.defaultdict(collections.Counter)
with open(REL + '/rider_frame_validity.csv') as f:
    for r in csv.DictReader(f):
        validity[(r['pair'], r['cam'], r['subtrack'])][r['placement_class']] += 1
        if r['spatial_use_valid'] == 'true':
            validity[(r['pair'], r['cam'], r['subtrack'])]['SPATIAL_VALID'] += 1

# ---------------------------------------------------------------- pred side (Stage J)
pred_summary = {}   # (pair, cam, sub) -> summary dict from tracks_final
pred_coupled = {}   # (pair, cam, sub) -> {frame: (state, conf, bbox)}
pred_parent_last_stepped = {}  # (pair, cam, moto_id) -> last frame index stepped (from track_frames)
for pair in PAIRS:
    d = os.path.join(SJ, pair)
    for cam in CAMS:
        tf = json.load(open(os.path.join(d, f'{pair}_{cam}_tracks_final.json')))
        for mt in tf['moto_tracks']:
            mid = mt['moto_track_id']
            for slot, hist in mt.get('rider_pts_history', {}).items():
                sub = f"{mid}_rider_s{slot}"
                pred_coupled[(pair, cam, sub)] = {h[0]: (h[3], h[4], h[2]) for h in hist}
            rs = mt.get('rider_subtracks') or {}
            for st in (rs.values() if isinstance(rs, dict) else rs):
                pred_summary[(pair, cam, st['subtrack_id'])] = {
                    'ac': st.get('aggregated_confidence') or {},
                    'temporal_state': st.get('temporal_state'), 'hits': st.get('hits'),
                    'age': st.get('age'), 'tsu': st.get('time_since_update'),
                    'abstained': st.get('abstained_updates_count'),
                    'slot_role': st.get('slot_role'), 'parent': mid,
                    'consecutive_helmet': st.get('consecutive_helmet_frames'),
                    'consecutive_no_helmet': st.get('consecutive_no_helmet_frames')}
        for ot in tf['orphan_tracks']:
            oid = ot.get('orphan_track_id') or ot.get('track_id')
            pred_coupled[(pair, cam, oid)] = {h[0]: (h[3], h[4], h[2]) for h in ot.get('pts_history', [])}
            pred_summary[(pair, cam, oid)] = {
                'ac': ot.get('aggregated_confidence') or {},
                'temporal_state': ot.get('temporal_state'), 'hits': ot.get('hits'),
                'age': ot.get('age'), 'tsu': ot.get('time_since_update'),
                'abstained': ot.get('abstained_updates_count'),
                'slot_role': None, 'parent': oid,
                'consecutive_helmet': ot.get('consecutive_helmet_frames'),
                'consecutive_no_helmet': ot.get('consecutive_no_helmet_frames')}
        # parent last stepped frame from track_frames.jsonl
        last_seen = {}
        with open(os.path.join(d, f'{pair}_{cam}_track_frames.jsonl')) as f:
            for line in f:
                rec = json.loads(line)
                fi = rec.get('frame_index')
                for t in rec.get('active_moto_tracks', []):
                    last_seen[t['moto_track_id']] = fi
                for t in rec.get('terminated_moto_tracks', []) or []:
                    last_seen[t['moto_track_id']] = fi
        for mid, fi in last_seen.items():
            if fi is not None:
                pred_parent_last_stepped[(pair, cam, mid)] = fi

def pred_state_rule(ac):
    """Exact Stage K protocol section 5 rule."""
    mx = max(ac.values()) if ac else 0.0
    if mx >= 0.5:
        return 'HELMET' if ac.get('helmet', 0) >= ac.get('no_helmet', 0) else 'NO_HELMET'
    return 'UNCERTAIN'

def flips(state_seq):
    """Adjacent changes among decided (helmet/no_helmet) observations, in frame order."""
    dec = [s for s in state_seq if s in ('helmet', 'no_helmet')]
    return sum(1 for a, b in zip(dec, dec[1:]) if a != b)

def dist_str(counter):
    return ';'.join(f'{k}:{v}' for k, v in sorted(counter.items()))

# ---------------------------------------------------------------- reproduce Stage K + build rows
decided = {k: v for k, v in gt_tracks.items() if v['gt_state'] in ('helmet', 'no_helmet')}
conf = collections.Counter()
rows = []
for (pair, cam, sub), gt in sorted(decided.items()):
    ps_sum = pred_summary.get((pair, cam, sub))
    key_missing = ps_sum is None
    ac = ps_sum['ac'] if ps_sum else {}
    ac_max = max(ac.values()) if ac else 0.0
    ps = 'UNCERTAIN' if key_missing else pred_state_rule(ac)
    conf[(gt['gt_state'], ps)] += 1

    if ps == 'UNCERTAIN':
        outcome = 'PIPELINE_UNCERTAIN'
    elif ps == gt['gt_state'].upper():
        outcome = 'AGREE'
    elif gt['gt_state'] == 'no_helmet':   # pred HELMET on a no_helmet GT track
        outcome = 'HARD_DISAGREE_FALSE_HELMET'
    else:                                  # pred NO_HELMET on a helmet GT track
        outcome = 'HARD_DISAGREE_FALSE_NO_HELMET'

    # GT observation sequence (frame order)
    gobs = sorted(gt['obs'])
    gt_seq = [s.lower() for _, s, _, _, _ in gobs]
    gt_dist = collections.Counter(gt_seq)
    # pred coupled history
    phist = pred_coupled.get((pair, cam, sub), {})
    pframes = sorted(phist)
    pred_seq = [phist[f][0] for f in pframes]
    pred_dist = collections.Counter(pred_seq)
    frame_sets_equal = set(pframes) == {f for f, *_ in gobs}
    # validity
    vc = validity.get((pair, cam, sub), collections.Counter())
    n_valid_rows = sum(v for k, v in vc.items() if k != 'SPATIAL_VALID')
    n_coupled = vc.get('COUPLED', 0)
    coupled_fraction = round(n_coupled / n_valid_rows, 4) if n_valid_rows else ''
    # tail gap: parent last stepped frame - last coupled frame
    last_coupled = pframes[-1] if pframes else None
    parent_last = pred_parent_last_stepped.get((pair, cam, gt['parent']))
    tail_gap = (parent_last - last_coupled) if (parent_last is not None and last_coupled is not None) else ''
    # head size (mean bbox height over GT obs)
    heights = [b[3] - b[1] for *_, b, _ in [(o[0], o[1], o[2], o[3], o[4]) for o in gobs]] if gobs else []
    heights = [o[3][3] - o[3][1] for o in gobs]
    mean_h = round(statistics.mean(heights), 1) if heights else ''

    rows.append({
        'pair': pair, 'camera': cam, 'track_id': sub, 'parent_track': gt['parent'],
        'kind': gt['kind'], 'role': gt['role'],
        'gt_state': gt['gt_state'].upper(), 'pred_state': ps, 'outcome_category': outcome,
        'gt_state_provenance': gt['provenance'],
        'n_observations': len(gobs), 'n_coupled': n_coupled,
        'validity_rows': n_valid_rows, 'coupled_fraction': coupled_fraction,
        'spatial_valid_rows': vc.get('SPATIAL_VALID', 0),
        'pred_obs_state_distribution': dist_str(pred_dist),
        'gt_obs_state_distribution': dist_str(gt_dist),
        'temporal_flip_count_pred': flips(pred_seq), 'temporal_flip_count_gt': flips(gt_seq),
        'pred_hits': ps_sum['hits'] if ps_sum else '',
        'pred_age': ps_sum['age'] if ps_sum else '',
        'pred_abstained_updates': ps_sum['abstained'] if ps_sum else '',
        'ac_helmet': ac.get('helmet', '') if ac else '', 'ac_no_helmet': ac.get('no_helmet', '') if ac else '',
        'ac_max': ac_max if ps_sum else '',
        'key_missing': key_missing,
        'last_coupled_frame': last_coupled if last_coupled is not None else '',
        'parent_last_stepped_frame': parent_last if parent_last is not None else '',
        'tail_gap_frames': tail_gap,
        'coupled_frames_equal_gt': frame_sets_equal,
        'mean_head_bbox_height_px': mean_h,
        '_gt_obs': gobs, '_pred_hist': phist})

# verify reproduction vs Stage K
sk = json.load(open(ROOT + '/work/stageK/stagek_results.json'))['helmet_track_level']
my_conf = {f'{g}->{p}': v for (g, p), v in sorted(conf.items())}
assert len(rows) == sk['n_decided_gt_tracks'] == 209, (len(rows), sk['n_decided_gt_tracks'])
assert my_conf == sk['confusion'], (my_conf, sk['confusion'])
agree = sum(1 for r in rows if r['outcome_category'] == 'AGREE')
assert agree == sk['agreement'] == 134

with open(OUT + '/reproduction_check.json', 'w') as f:
    json.dump({'stagek_confusion': sk['confusion'], 'reproduced_confusion': my_conf,
               'n_rows': len(rows), 'agreement': agree}, f, indent=1)
print('REPRODUCTION OK: 209 tracks, confusion identical to Stage K, agreement', agree)

# ---------------------------------------------------------------- cause classification
for r in rows:
    notes = []
    if r['outcome_category'] == 'AGREE':
        r['primary_cause'] = 'NONE'
    elif r['outcome_category'] == 'PIPELINE_UNCERTAIN':
        if r['key_missing']:
            r['primary_cause'] = 'PRED_TRACK_MISSING'
            notes.append('GT (sub)track id absent from Stage J tracks_final summaries; '
                         'Stage K default UNCERTAIN applied')
        elif r['ac_max'] == 0:
            r['primary_cause'] = 'AGGREGATION_ABSTENTION_ZERO_MASS'
            tg = r['tail_gap_frames']
            if isinstance(tg, int) and tg >= 10:
                notes.append(f'window flush certain: last coupling {tg} frames before parent last stepped frame (>= EMA window 10)')
            elif isinstance(tg, int):
                notes.append(f'ac zero with tail gap only {tg} frames — flush via abstained/unmatched updates within parent life; see counts')
            else:
                notes.append('ac zero; tail gap not computable')
        elif r['ac_max'] < 0.5:
            r['primary_cause'] = 'AGGREGATION_THRESHOLD'
            notes.append(f"0 < ac_max={r['ac_max']} < 0.5")
        else:
            r['primary_cause'] = 'OTHER'
            notes.append('UNCERTAIN despite ac_max >= 0.5 — investigate')
    else:  # hard disagreements — refined after individual inspection (see disagreements.json)
        r['primary_cause'] = 'TEMPORAL_INCONSISTENCY'
    # secondary flags appended to notes
    if isinstance(r['coupled_fraction'], float) and r['coupled_fraction'] < 0.2 and r['validity_rows']:
        notes.append(f"low coupling: coupled_fraction={r['coupled_fraction']}")
    if isinstance(r['mean_head_bbox_height_px'], float) and r['mean_head_bbox_height_px'] < 40:
        notes.append(f"small head: mean bbox height {r['mean_head_bbox_height_px']} px")
    if r['pred_abstained_updates'] not in ('', None) and isinstance(r['pred_abstained_updates'], int) and r['pred_abstained_updates'] >= 50:
        notes.append(f"high abstained updates: {r['pred_abstained_updates']}")
    r['notes'] = ' | '.join(notes)

# disagreements detail dump for manual inspection
dis = []
for r in rows:
    if r['outcome_category'].startswith('HARD_DISAGREE'):
        dis.append({'pair': r['pair'], 'camera': r['camera'], 'track_id': r['track_id'],
                    'role': r['role'], 'gt_state': r['gt_state'], 'pred_state': r['pred_state'],
                    'provenance': r['gt_state_provenance'], 'ac': [r['ac_helmet'], r['ac_no_helmet']],
                    'gt_obs': [(f, s, c) for f, s, c, _, _ in r['_gt_obs']],
                    'pred_obs_dist': r['pred_obs_state_distribution'],
                    'gt_obs_dist': r['gt_obs_state_distribution'],
                    'flips_gt': r['temporal_flip_count_gt'], 'flips_pred': r['temporal_flip_count_pred'],
                    'coupled_fraction': r['coupled_fraction'], 'n_obs': r['n_observations']})
with open(OUT + '/disagreements.json', 'w') as f:
    json.dump(dis, f, indent=1)

# strip helper fields and write CSV
drop = {'_gt_obs', '_pred_hist'}
fieldnames = [k for k in rows[0].keys() if k not in drop]
with open(OUT + '/helmet_track_breakdown.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow({k: r[k] for k in fieldnames})
print('CSV written:', len(rows), 'rows; disagreements:', len(dis))
print(json.dumps(collections.Counter(r['outcome_category'] for r in rows), indent=1))
print(json.dumps(collections.Counter(r['primary_cause'] for r in rows), indent=1))
