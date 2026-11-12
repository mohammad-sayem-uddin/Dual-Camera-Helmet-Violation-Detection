"""Stage K extraction: normalize GT (frozen release) + Stage J predictions
into plain Python structures. No scoring in this module."""
import json, csv, glob, os, collections
from pathlib import Path

REPO = Path(os.environ.get('RESEARCH_HELMET_ROOT', Path(__file__).resolve().parents[2]))
REL = os.environ.get(
    'GT_RELEASE_DIR',
    str(REPO / 'data/dual_camera_test_v1/releases/dual_camera_gt_v1.0_ai_assisted_audited'),
)
SJ = os.environ.get('STAGEJ_OUT_DIR', str(REPO / 'results/phase12_stagej'))
ROOT = str(REPO)
PAIRS = [f'pair_{i:03d}' for i in range(1, 26)]
CAMS = ('front', 'back')


def load_gt(rel_dir: str | os.PathLike | None = None):
    rel_base = os.fspath(rel_dir) if rel_dir is not None else os.fspath(REL)
    gt = {'head_obs': [],          # dicts: pair, cam, parent, sub, frame, bbox, state, conf
          'moto_tracks': {},       # (pair, cam, track_id) -> record
          'rider_tracks': {},      # (pair, cam, sub_id) -> {parent, gt_state, track_state}
          'excluded': set(),       # (pair, cam, track_id)
          'excluded_kind': {},     # (pair, cam, track_id) -> kind
          'events': [],            # from CSV
          'links': [],             # from CSV
          'validity': {},          # (pair, cam, sub, frame) -> bool spatial_use_valid
          'set_constant_subs': set(),  # (pair, cam, sub) with SET_CONSTANT corrections
          }
    for pf in sorted(glob.glob(rel_base + '/pairs/*/pair_*_gt.json')):
        g = json.load(open(pf)); pr = g['pair']; pair = pr['pair_id']
        for t in pr.get('excluded_tracks', []):
            gt['excluded'].add((pair, t['camera'], t['track_id']))
            gt['excluded_kind'][(pair, t['camera'], t['track_id'])] = t['kind']
        for cam in CAMS:
            ca = pr['camera_annotations'][cam]
            for mt in ca['motorcycle_tracks']:
                gt['moto_tracks'][(pair, cam, mt['track_id'])] = {
                    'first': mt['first_frame_index'], 'last': mt['last_frame_index'],
                    'hits': mt['hits'], 'entry_bbox': mt['entry_bbox_xyxy'],
                    'exit_bbox': mt['exit_bbox_xyxy']}
                for st in mt['rider_subtracks']:
                    gt['rider_tracks'][(pair, cam, st['subtrack_id'])] = {
                        'parent': mt['track_id'], 'gt_state': st['gt_helmet_state'],
                        'track_state': st['track_helmet_state']}
                    for o in st['head_observations']:
                        gt['head_obs'].append({'pair': pair, 'cam': cam,
                            'parent': mt['track_id'], 'sub': st['subtrack_id'],
                            'frame': o['frame_index'], 'bbox': o['bbox_xyxy'],
                            'state': o['state'], 'conf': o['confidence']})
            for ot in ca['orphan_rider_tracks']:
                gt['rider_tracks'][(pair, cam, ot['track_id'])] = {
                    'parent': ot['track_id'], 'gt_state': ot['gt_helmet_state'],
                    'track_state': ot['track_helmet_state']}
                for o in ot['head_observations']:
                    gt['head_obs'].append({'pair': pair, 'cam': cam,
                        'parent': ot['track_id'], 'sub': ot['track_id'],
                        'frame': o['frame_index'], 'bbox': o['bbox_xyxy'],
                        'state': o['state'], 'conf': o['confidence']})
    with open(rel_base + '/dual_camera_gt_events.csv') as f:
        for r in csv.DictReader(f):
            gt['events'].append({'pair': r['pair_id'], 'gid': r['global_moto_id'],
                'type': r['event_type'],
                'front': r['front_track_id'] or None, 'back': r['back_track_id'] or None,
                'w0': float(r['violation_window_start_ref_sec']) if r['violation_window_start_ref_sec'] else None,
                'w1': float(r['violation_window_end_ref_sec']) if r['violation_window_end_ref_sec'] else None})
    with open(rel_base + '/cross_camera_identity_links.csv') as f:
        for r in csv.DictReader(f):
            gt['links'].append({'pair': r['pair'], 'kind': r['kind'],
                'front': r['front_track_id'] or None, 'back': r['back_track_id'] or None,
                'outcome': r['outcome'], 'link_status': r['link_status']})
    with open(rel_base + '/rider_frame_validity.csv') as f:
        for r in csv.DictReader(f):
            gt['validity'][(r['pair'], r['cam'], r['subtrack'], int(r['frame']))] = (r['spatial_use_valid'] == 'true')
    with open(rel_base + '/annotation_corrections.csv') as f:
        for r in csv.DictReader(f):
            if r['correction_type'] == 'SET_CONSTANT_HELMET_STATE':
                # ref names the (sub)track the constant was applied to
                gt['set_constant_subs'].add((r['pair'], r['cam'], r['ref']))
    return gt


def _redesign(rec, hist):
    """TASK-04 states: exported fields if present, else pts_history fallback (legacy Stage J)."""
    import sys
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from track_state_redesign import _entry
    return _entry(rec, hist, set())


def load_pred(sj_dir: str | os.PathLike | None = None, pairs: list[str] | None = None):
    sj_base = os.fspath(sj_dir) if sj_dir is not None else os.fspath(SJ)
    pair_list = list(pairs) if pairs is not None else PAIRS
    pred = {'detections': collections.defaultdict(list),  # (pair,cam,frame) -> {'motorcycles':[], 'persons':[], 'heads':[]}
            'coupled': {},        # (pair, cam, sub) -> {frame: (bbox, state, conf)}
            'coupled_parent': {}, # (pair, cam, sub) -> parent track id
            'moto_tracks': {},    # (pair, cam, track_id) -> {first,last,observed}
            'orphan_tracks': set(),
            'track_state': {},    # (pair, cam, sub) -> derived state per protocol §5
            'track_state_redesign': {},
            'events': [], 'assoc_moto': [], 'assoc_orphan': [], 'assoc_rider': [],
            'rider_outcome_counts': collections.Counter(),
            'sync': {}, 'coverage': {}}
    for pair in pair_list:
        d = os.path.join(sj_base, pair)
        for cam in CAMS:
            with open(os.path.join(d, f'{pair}_{cam}_detections.jsonl')) as f:
                for line in f:
                    rec = json.loads(line)
                    pred['detections'][(pair, cam, rec['frame_index'])] = {
                        'motorcycles': rec['motorcycles'], 'persons': rec['persons'], 'heads': rec['heads']}
            tf = json.load(open(os.path.join(d, f'{pair}_{cam}_tracks_final.json')))
            for mt in tf['moto_tracks']:
                for slot, hist in mt.get('rider_pts_history', {}).items():
                    sub = f"{mt['moto_track_id']}_rider_s{slot}"
                    pred['coupled'][(pair, cam, sub)] = {
                        h[0]: (h[2], h[3], h[4]) for h in hist}
                    pred['coupled_parent'][(pair, cam, sub)] = mt['moto_track_id']
            for ot in tf['orphan_tracks']:
                oid = ot.get('orphan_track_id') or ot.get('track_id')
                pred['coupled'][(pair, cam, oid)] = {
                    h[0]: (h[2], h[3], h[4]) for h in ot.get('pts_history', [])}
                pred['coupled_parent'][(pair, cam, oid)] = oid
                pred['orphan_tracks'].add((pair, cam, oid))
            # predicted track-level rider states (protocol §5 rule) from rider_subtracks summaries
            for mt in tf['moto_tracks']:
                rph = mt.get('rider_pts_history', {})
                _rs = mt.get('rider_subtracks') or []
                for st in (_rs.values() if isinstance(_rs, dict) else _rs):
                    ac = st.get('aggregated_confidence') or {}
                    mx = max(ac.values()) if ac else 0.0
                    state = 'UNCERTAIN'
                    if mx >= 0.5:
                        state = 'HELMET' if ac.get('helmet', 0) >= ac.get('no_helmet', 0) else 'NO_HELMET'
                    pred['track_state'][(pair, cam, st['subtrack_id'])] = state
                    pred['track_state_redesign'][(pair, cam, st['subtrack_id'])] = _redesign(
                        st, rph.get(str(st.get('slot_index')), []))
            for ot in tf['orphan_tracks']:
                oid = ot.get('orphan_track_id') or ot.get('track_id')
                ac = ot.get('aggregated_confidence') or {}
                mx = max(ac.values()) if ac else 0.0
                state = 'UNCERTAIN'
                if mx >= 0.5:
                    state = 'HELMET' if ac.get('helmet', 0) >= ac.get('no_helmet', 0) else 'NO_HELMET'
                pred['track_state'][(pair, cam, oid)] = state
                pred['track_state_redesign'][(pair, cam, oid)] = _redesign(ot, ot.get('pts_history', []))
            cov = os.path.join(d, f'{pair}_{cam}_coverage.csv')
            with open(cov) as f:
                pred['coverage'][(pair, cam)] = sum(1 for _ in f) - 1  # minus header
        a = json.load(open(os.path.join(d, f'{pair}_association.json')))
        for cam in CAMS:
            for tl in a[f'{cam}_motorcycle_tracklets']:
                pred['moto_tracks'][(pair, cam, tl['moto_track_id'])] = {
                    'first': tl['first_frame_index'], 'last': tl['last_frame_index'],
                    'observed': tl['observed_frame_count']}
        for r in a['motorcycle_results']:
            pred['assoc_moto'].append({'pair': pair, 'front': r.get('front_moto_track_id'),
                'back': r.get('back_moto_track_id'), 'outcome': r['outcome']})
            for ra in r.get('rider_associations') or []:
                pred['assoc_rider'].append({
                    'pair': pair,
                    'front': ra.get('front_subtrack_id') or ra.get('front_rider_subtrack_id'),
                    'back': ra.get('back_subtrack_id') or ra.get('back_rider_subtrack_id'),
                    'outcome': ra['outcome'],
                    'association_score': ra.get('association_score'),
                    'parent_front': (
                        ra.get('front_moto_track_id')
                        or ra.get('parent_front_moto_track_id')
                        or r.get('front_moto_track_id')
                    ),
                    'parent_back': (
                        ra.get('back_moto_track_id')
                        or ra.get('parent_back_moto_track_id')
                        or r.get('back_moto_track_id')
                    ),
                })
        for r in a['orphan_results']:
            pred['assoc_orphan'].append({'pair': pair, 'front': r.get('front_track_id'),
                'back': r.get('back_track_id'), 'outcome': r['outcome']})
        for k, v in (a.get('rider_outcome_counts') or {}).items():
            pred['rider_outcome_counts'][k] += v
        for e in json.load(open(os.path.join(d, f'{pair}_events.json'))):
            w = e.get('violation_window_ref_sec')
            pred['events'].append({'pair': pair, 'gid': e['global_moto_id'],
                'type': e['violation_event_type'], 'front': e.get('front_track_id'),
                'back': e.get('back_track_id'),
                'w0': w[0] if w else None, 'w1': w[1] if w else None})
        pred['sync'][pair] = json.load(open(os.path.join(d, f'{pair}_sync_diagnostics.json')))
    return pred


def iou(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1]); x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0
