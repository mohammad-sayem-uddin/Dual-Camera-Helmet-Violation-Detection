"""Stage K scoring — implements EVAL_PROTOCOL.md exactly (frozen 2026-11-09).
Writes stagek_results.json, per_pair_stats.json and detail CSVs to work/stageK/."""
import json, csv, collections, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract import load_gt, load_pred, iou, PAIRS, CAMS

OUT = os.path.dirname(os.path.abspath(__file__))
gt = load_gt(); pred = load_pred()
R = {}            # master results
PP = {p: {} for p in PAIRS}   # per-pair bootstrap sufficient statistics

def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {'precision': round(p, 4), 'recall': round(r, 4), 'f1': round(f, 4), 'tp': tp, 'fp': fp, 'fn': fn}

# ---------------------------------------------------------------- detection: head
gt_by_frame = collections.defaultdict(list)
for o in gt['head_obs']:
    gt_by_frame[(o['pair'], o['cam'], o['frame'])].append(o)
pred_heads = []   # (pair, cam, frame, bbox, conf)
for (pair, cam, frame), cats in pred['detections'].items():
    for h in cats['heads']:
        pred_heads.append((pair, cam, frame, h['bbox'], h['confidence']))

# coupling lookup for FP decomposition: (pair,cam,frame,rounded bbox) -> (sub, parent)
couple_map = {}
for (pair, cam, sub), hist in pred['coupled'].items():
    for fr, (bbox, state, conf) in hist.items():
        couple_map[(pair, cam, fr, tuple(round(v, 2) for v in bbox))] = (sub, pred['coupled_parent'][(pair, cam, sub)])

def match_frame(gt_list, pr_list, thr):
    """greedy by pred conf desc; returns matched gt idx set, per-pred tp flag list aligned with pr_list order sorted by conf"""
    order = sorted(range(len(pr_list)), key=lambda i: -pr_list[i][4])
    used = set(); flags = {}
    for i in order:
        best, bj = thr, -1
        for j, g in enumerate(gt_list):
            if j in used: continue
            v = iou(pr_list[i][3], g['bbox'])
            if v >= best: best, bj = v, j
        if bj >= 0:
            used.add(bj); flags[i] = True
        else:
            flags[i] = False
    return used, flags

# operating point (IoU 0.50): per-pair counts + FP/FN decomposition
det_pp = {}
fp_decomp = collections.Counter(); fp_coupled_retained_nogt = []
fn_by_pair = collections.Counter()
scopes = {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in ('front', 'back', 'pooled')}
pred_by_frame = collections.defaultdict(list)
for ph in pred_heads: pred_by_frame[(ph[0], ph[1], ph[2])].append(ph)
for pair in PAIRS:
    tp = fp = fn = 0
    for cam in CAMS:
        frames = {f for (p, c, f) in gt_by_frame if p == pair and c == cam} | \
                 {f for (p, c, f) in pred_by_frame if p == pair and c == cam}
        for fr in frames:
            gl = gt_by_frame.get((pair, cam, fr), [])
            pl = pred_by_frame.get((pair, cam, fr), [])
            used, flags = match_frame(gl, pl, 0.5)
            for i, ph in enumerate(pl):
                if flags[i]:
                    tp += 1; scopes[cam]['tp'] += 1; scopes['pooled']['tp'] += 1
                else:
                    fp += 1; scopes[cam]['fp'] += 1; scopes['pooled']['fp'] += 1
                    key = (pair, cam, fr, tuple(round(v, 2) for v in ph[3]))
                    if key in couple_map:
                        sub, parent = couple_map[key]
                        if (pair, cam, parent) in gt['excluded']:
                            fp_decomp['correction_delta_coupled_to_excluded_track'] += 1
                        else:
                            fp_decomp['coupled_to_retained_track_no_gt_obs'] += 1
                            fp_coupled_retained_nogt.append({'pair': pair, 'cam': cam, 'frame': fr, 'sub': sub})
                    else:
                        fp_decomp['uncoupled_gt_construction'] += 1
            fn += len(gl) - len(used); scopes[cam]['fn'] += len(gl) - len(used); scopes['pooled']['fn'] += len(gl) - len(used)
            fn_by_pair[pair] += len(gl) - len(used)
    det_pp[pair] = {'tp': tp, 'fp': fp, 'fn': fn}
    PP[pair]['head_det'] = {'tp': tp, 'fp': fp, 'fn': fn}
R['detection_head_operating_point'] = {s: prf(v['tp'], v['fp'], v['fn']) for s, v in scopes.items()}
R['detection_head_fp_decomposition'] = dict(fp_decomp)
R['detection_head_fp_coupled_retained_list'] = fp_coupled_retained_nogt

# AP50 / AP50:95 per scope
def ap_for_scope(cam_filter, thr):
    preds = [ph for ph in pred_heads if cam_filter is None or ph[1] == cam_filter]
    preds_sorted = sorted(preds, key=lambda x: -x[4])
    matched = collections.defaultdict(set)  # (pair,cam,frame) -> gt idx used
    tp_cum, fp_cum = [], []
    t = f = 0
    gt_total = sum(1 for o in gt['head_obs'] if cam_filter is None or o['cam'] == cam_filter)
    for ph in preds_sorted:
        gl = gt_by_frame.get((ph[0], ph[1], ph[2]), [])
        best, bj = thr, -1
        for j, g in enumerate(gl):
            if j in matched[(ph[0], ph[1], ph[2])]: continue
            v = iou(ph[3], g['bbox'])
            if v >= best: best, bj = v, j
        if bj >= 0:
            matched[(ph[0], ph[1], ph[2])].add(bj); t += 1
        else:
            f += 1
        tp_cum.append(t); fp_cum.append(f)
    if gt_total == 0 or not tp_cum: return 0.0
    rec = [x / gt_total for x in tp_cum]
    prec = [tp_cum[i] / (tp_cum[i] + fp_cum[i]) for i in range(len(tp_cum))]
    ap = 0.0
    for k in range(101):
        r = k / 100.0
        p = max([prec[i] for i in range(len(rec)) if rec[i] >= r], default=0.0)
        ap += p / 101.0
    return ap

R['detection_head_ap'] = {}
for scope_name, cf in (('front', 'front'), ('back', 'back'), ('pooled', None)):
    aps = [ap_for_scope(cf, 0.5 + 0.05 * i) for i in range(10)]
    R['detection_head_ap'][scope_name] = {'AP50': round(aps[0], 4), 'AP50_95': round(sum(aps) / 10, 4)}

# ---------------------------------------------------------------- detection: motorcycle (track level)
moto_fp_decomp = collections.Counter(); matched_agree = []
gt_moto = set(gt['moto_tracks'].keys()); pred_moto = set(pred['moto_tracks'].keys())
tp_m = len(gt_moto & pred_moto)
for k in sorted(pred_moto - gt_moto):
    moto_fp_decomp['excluded_fp_track_correction_delta' if k in gt['excluded'] else 'other_extra_track'] += 1
for k in sorted(gt_moto & pred_moto):
    g, p = gt['moto_tracks'][k], pred['moto_tracks'][k]
    matched_agree.append({'pair': k[0], 'cam': k[1], 'track': k[2],
        'd_first': p['first'] - g['first'], 'd_last': p['last'] - g['last'], 'd_hits': p['observed'] - g['hits']})
R['detection_motorcycle_tracklevel'] = {
    **prf(tp_m, len(pred_moto - gt_moto), len(gt_moto - pred_moto)),
    'gt_tracks': len(gt_moto), 'pred_tracks': len(pred_moto),
    'fp_decomposition': dict(moto_fp_decomp),
    'fn_tracks': [list(k) for k in sorted(gt_moto - pred_moto)]}
if matched_agree:
    import statistics as st
    R['detection_motorcycle_tracklevel']['temporal_agreement'] = {
        'mean_abs_d_first': round(st.mean(abs(x['d_first']) for x in matched_agree), 3),
        'mean_abs_d_last': round(st.mean(abs(x['d_last']) for x in matched_agree), 3),
        'mean_abs_d_hits': round(st.mean(abs(x['d_hits']) for x in matched_agree), 3),
        'exact_first_last_hits': sum(1 for x in matched_agree if x['d_first'] == 0 and x['d_last'] == 0 and x['d_hits'] == 0)}
# anchored-box identity verification (not scored)
anch = {'checked': 0, 'verbatim': 0}
pred_det_keys = collections.defaultdict(set)
for (pair, cam, frame), cats in pred['detections'].items():
    for m in cats['motorcycles']:
        pred_det_keys[(pair, cam, frame)].add(tuple(round(v, 2) for v in m['bbox']))
for k in sorted(gt_moto & pred_moto):
    g = gt['moto_tracks'][k]
    for fr, bb in ((g['first'], g['entry_bbox']), (g['last'], g['exit_bbox'])):
        anch['checked'] += 1
        if tuple(round(v, 2) for v in bb) in pred_det_keys.get((k[0], k[1], fr), set()): anch['verbatim'] += 1
R['detection_motorcycle_anchored_box_check'] = anch

# ---------------------------------------------------------------- rider family
gt_obs_map = {}   # (pair,cam,sub) -> {frame: bbox}
for o in gt['head_obs']:
    gt_obs_map.setdefault((o['pair'], o['cam'], o['sub']), {})[o['frame']] = o['bbox']

def rider_view(use_validity):
    tp = fp = fn = 0; ious = []; per_pair = {}
    ctx = collections.Counter()
    for pair in PAIRS:
        ptp = pfp = pfn = 0; piou = []
        for cam in CAMS:
            subs = {s for (p, c, s) in gt_obs_map if p == pair and c == cam}
            for sub in subs:
                gmap = gt_obs_map[(pair, cam, sub)]
                pmap = pred['coupled'].get((pair, cam, sub), {})
                for fr, gb in gmap.items():
                    if use_validity:
                        v = gt['validity'].get((pair, cam, sub, fr))
                        if v is None: ctx['gt_obs_validity_row_missing'] += 1; continue
                        if not v: ctx['gt_obs_excluded_by_validity_filter'] += 1; continue
                    if fr in pmap:
                        v = iou(gb, pmap[fr][0]); ious.append(v); piou.append(v)
                        if v >= 0.5: tp += 1; ptp += 1
                        else: tp += 0; fp += 1; fn += 1; pfp += 1; pfn += 1
                    else:
                        fn += 1; pfn += 1
                for fr in pmap:
                    if fr not in gmap:
                        if use_validity and gt['validity'].get((pair, cam, sub, fr)):
                            fp += 1; pfp += 1
                        elif not use_validity:
                            fp += 1; pfp += 1
            # predicted coupled obs outside GT inventory (excluded tracks etc.)
            for (p, c, sub), hist in pred['coupled'].items():
                if p == pair and c == cam and (p, c, sub) not in gt_obs_map:
                    parent = pred['coupled_parent'][(p, c, sub)]
                    if (p, c, parent) in gt['excluded'] or (p, c, sub) in gt['excluded']:
                        ctx['pred_coupled_on_excluded_tracks'] += len(hist)
                    else:
                        ctx['pred_coupled_on_noninventory_tracks'] += len(hist)
        per_pair[pair] = {'tp': ptp, 'fp': pfp, 'fn': pfn,
                          'iou_sum': sum(piou), 'iou_n': len(piou)}
    import statistics as st
    res = {**prf(tp, fp, fn), 'mean_iou_matched': round(st.mean(ious), 4) if ious else 0.0,
           'n_iou_pairs': len(ious), 'context': dict(ctx)}
    return res, per_pair

R['rider_primary_spatially_valid'], rp = rider_view(True)
R['rider_sensitivity_all_coupled'], rs = rider_view(False)
for pair in PAIRS:
    PP[pair]['rider_primary'] = rp[pair]; PP[pair]['rider_sensitivity'] = rs[pair]

# ---------------------------------------------------------------- helmet family
conf = collections.Counter(); unmatched_gt = collections.Counter(); disagg = collections.Counter()
gt_decided = [o for o in gt['head_obs'] if o['state'] in ('helmet', 'no_helmet')]
excl_occluded = sum(1 for o in gt['head_obs'] if o['state'] == 'occluded_unknown')
hel_pp = {}
for o in gt_decided:
    hist = pred['coupled'].get((o['pair'], o['cam'], o['sub']), {})
    if o['frame'] in hist:
        ps = hist[o['frame']][1]
        conf[(o['state'], ps)] += 1
        if ps != o['state']:
            disagg['on_set_constant_subtrack' if (o['pair'], o['cam'], o['sub']) in gt['set_constant_subs'] else 'on_ai_direct_subtrack'] += 1
    else:
        unmatched_gt[o['pair']] += 1
def class_prf(pos):
    tp = conf[(pos, pos)]
    fp = sum(v for (g, p), v in conf.items() if p == pos and g != pos)
    fn = sum(v for (g, p), v in conf.items() if g == pos and p != pos)
    return prf(tp, fp, fn)
R['helmet_observation_level'] = {
    'confusion': {f'{g}->{p}': v for (g, p), v in sorted(conf.items())},
    'helmet': class_prf('helmet'), 'no_helmet': class_prf('no_helmet'),
    'gt_decided_total': len(gt_decided), 'matched': sum(conf.values()),
    'gt_unmatched_no_pred_obs': dict(unmatched_gt), 'unmatched_total': sum(unmatched_gt.values()),
    'excluded_occluded_unknown': excl_occluded,
    'disagreement_decomposition': dict(disagg)}
for pair in PAIRS:
    sub_conf = collections.Counter()
    for o in gt_decided:
        if o['pair'] != pair: continue
        hist = pred['coupled'].get((o['pair'], o['cam'], o['sub']), {})
        if o['frame'] in hist: sub_conf[(o['state'], hist[o['frame']][1])] += 1
    PP[pair]['helmet'] = {
        'helmet': {'tp': sub_conf[('helmet', 'helmet')],
                   'fp': sub_conf[('no_helmet', 'helmet')], 'fn': sub_conf[('helmet', 'no_helmet')]},
        'no_helmet': {'tp': sub_conf[('no_helmet', 'no_helmet')],
                      'fp': sub_conf[('helmet', 'no_helmet')], 'fn': sub_conf[('no_helmet', 'helmet')]}}
# track level
tl = collections.Counter(); tl_total = 0
for (pair, cam, sub), rt in gt['rider_tracks'].items():
    if rt['gt_state'] not in ('helmet', 'no_helmet'): continue
    tl_total += 1
    ps = pred['track_state'].get((pair, cam, sub), 'UNCERTAIN')
    tl[(rt['gt_state'], ps)] += 1
agree = sum(v for (g, p), v in tl.items() if g.upper() == p)
R['helmet_track_level'] = {'n_decided_gt_tracks': tl_total,
    'agreement': agree, 'agreement_rate': round(agree / tl_total, 4) if tl_total else 0,
    'confusion': {f'{g}->{p}': v for (g, p), v in sorted(tl.items())}}

# TASK-04 amended block (additive; the legacy 'helmet_track_level' above is untouched)
if pred.get('track_state_redesign'):
    _rd = collections.defaultdict(collections.Counter)
    for (pair, cam, sub), rt in gt['rider_tracks'].items():
        if rt['gt_state'] not in ('helmet', 'no_helmet'): continue
        _p = pred['track_state_redesign'].get((pair, cam, sub))
        for _m in ('ema_final', 'last_coupled', 'majority', 'conf_weighted', 'primary'):
            _rd[_m][(rt['gt_state'], _p[_m] if _p else 'UNCERTAIN')] += 1
    R['helmet_track_level_redesign'] = {_m: {
        'n_decided_gt_tracks': sum(c.values()),
        'agreement': sum(v for (g, p), v in c.items() if g.upper() == p),
        'abstentions': sum(v for (g, p), v in c.items() if p == 'UNCERTAIN'),
        'confusion': {f'{g}->{p}': v for (g, p), v in sorted(c.items())}} for _m, c in _rd.items()}

# ---------------------------------------------------------------- tracking family (head-observation basis)
import numpy as np
from scipy.optimize import linear_sum_assignment
pred_coupled_by_frame = collections.defaultdict(list)
for (pair, cam, sub), hist in pred['coupled'].items():
    for fr, (bbox, state, conf) in hist.items():
        pred_coupled_by_frame[(pair, cam, fr)].append((sub, bbox))
gt_by_frame_sub = collections.defaultdict(list)
for o in gt['head_obs']:
    gt_by_frame_sub[(o['pair'], o['cam'], o['frame'])].append((o['sub'], o['bbox']))

trk = {'idtp': 0, 'idfp': 0, 'idfn': 0, 'switches': 0}
for pair in PAIRS:
    p_idtp = p_pred = p_gt = p_sw = 0
    for cam in CAMS:
        frames = {f for (p, c, f) in gt_by_frame_sub if p == pair and c == cam} | \
                 {f for (p, c, f) in pred_coupled_by_frame if p == pair and c == cam}
        counts = collections.Counter(); matched_units = collections.defaultdict(list)
        n_gt = n_pr = 0
        for fr in sorted(frames):
            gl = gt_by_frame_sub.get((pair, cam, fr), [])
            pl = pred_coupled_by_frame.get((pair, cam, fr), [])
            n_gt += len(gl); n_pr += len(pl)
            cand = []
            for gi, (gs, gb) in enumerate(gl):
                for pi, (ps, pb) in enumerate(pl):
                    v = iou(gb, pb)
                    if v >= 0.5: cand.append((v, gi, pi))
            cand.sort(reverse=True); ug, up = set(), set()
            for v, gi, pi in cand:
                if gi in ug or pi in up: continue
                ug.add(gi); up.add(pi)
                counts[(gl[gi][0], pl[pi][0])] += 1
                matched_units[gl[gi][0]].append((fr, pl[pi][0]))
        gids = sorted({g for g, _ in counts}); pids = sorted({p for _, p in counts})
        idtp_cam = 0
        if gids and pids:
            M = np.zeros((len(gids), len(pids)), dtype=int)
            for (g, p), v in counts.items(): M[gids.index(g), pids.index(p)] = v
            ri, ci = linear_sum_assignment(-M)
            idtp_cam = int(M[ri, ci].sum())
        for gsub, units in matched_units.items():
            units.sort()
            p_sw += sum(1 for i in range(1, len(units)) if units[i][1] != units[i - 1][1])
        p_idtp += idtp_cam; p_gt += n_gt; p_pred += n_pr
    trk['idtp'] += p_idtp; trk['idfn'] += p_gt - p_idtp; trk['idfp'] += p_pred - p_idtp
    trk['switches'] += p_sw
    PP[pair]['tracking'] = {'idtp': p_idtp, 'idfp': p_pred - p_idtp, 'idfn': p_gt - p_idtp, 'switches': p_sw}
idp = trk['idtp'] / (trk['idtp'] + trk['idfp']) if trk['idtp'] + trk['idfp'] else 0
idr = trk['idtp'] / (trk['idtp'] + trk['idfn']) if trk['idtp'] + trk['idfn'] else 0
R['tracking_head_basis'] = {'idtp': trk['idtp'], 'idfp': trk['idfp'], 'idfn': trk['idfn'],
    'id_precision': round(idp, 4), 'id_recall': round(idr, 4),
    'idf1': round(2 * idp * idr / (idp + idr), 4) if idp + idr else 0,
    'id_switches': trk['switches'],
    'hota': 'NOT COMPUTED — protocol §6: release lacks per-frame box trajectories for motorcycle/person; head-only HOTA would not be the standard metric.'}

# ---------------------------------------------------------------- association family
def assoc_eval(gt_rows, pred_rows, kind):
    gt_units = {(r['pair'], r['front'] or '', r['back'] or ''): r['outcome'] for r in gt_rows}
    pr_units = {(r['pair'], r['front'] or '', r['back'] or ''): r['outcome'] for r in pred_rows}
    cross = collections.Counter()
    for k in set(gt_units) | set(pr_units):
        cross[(gt_units.get(k, 'ABSENT'), pr_units.get(k, 'ABSENT'))] += 1
    gt_linked = {k for k, v in gt_units.items() if v == 'LINKED'}
    pr_linked = {k for k, v in pr_units.items() if v == 'LINKED'}
    tp = len(gt_linked & pr_linked)
    res = {**prf(tp, len(pr_linked - gt_linked), len(gt_linked - pr_linked)),
           'gt_units': len(gt_units), 'pred_units': len(pr_units),
           'gt_linked': len(gt_linked), 'pred_linked': len(pr_linked),
           'pred_abstain': sum(1 for v in pr_units.values() if v in ('AMBIGUOUS_ABSTAIN', 'INSUFFICIENT_EVIDENCE')),
           'pred_abstain_rate': round(sum(1 for v in pr_units.values() if v in ('AMBIGUOUS_ABSTAIN', 'INSUFFICIENT_EVIDENCE')) / len(pr_units), 4) if pr_units else 0,
           'crosstab': {f'{g}|{p}': v for (g, p), v in sorted(cross.items())},
           'fp_linked_pairs': [list(k) for k in sorted(pr_linked - gt_linked)],
           'fn_linked_pairs': [list(k) for k in sorted(gt_linked - pr_linked)]}
    pp = {}
    for pair in PAIRS:
        g = {k for k in gt_linked if k[0] == pair}; p = {k for k in pr_linked if k[0] == pair}
        pp[pair] = {'tp': len(g & p), 'fp': len(p - g), 'fn': len(g - p)}
    return res, pp

R['association_motorcycle'], am = assoc_eval(
    [r for r in gt['links'] if r['kind'] == 'motorcycle'], pred['assoc_moto'], 'motorcycle')
R['association_orphan'], ao = assoc_eval(
    [r for r in gt['links'] if r['kind'] == 'orphan_rider'], pred['assoc_orphan'], 'orphan_rider')
for pair in PAIRS:
    PP[pair]['assoc_moto'] = am[pair]
gt_rider_counts = collections.Counter(r['outcome'] for r in gt['links'] if r['kind'] == 'rider')
R['association_rider'] = {'status': 'item-level NOT COMPUTABLE — Stage J association.json has no per-item rider results (protocol §7)',
    'gt_outcome_counts': dict(gt_rider_counts), 'pred_outcome_counts_aggregate': dict(pred['rider_outcome_counts'])}

# ---------------------------------------------------------------- event family
def tiou(a, b):
    if a[0] is None or b[0] is None: return None
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0

gt_ev = {(e['pair'], e['front'] or '', e['back'] or ''): e for e in gt['events']}
pr_ev = {(e['pair'], e['front'] or '', e['back'] or ''): e for e in pred['events']}
matches, mismatches = [], []
used_g, used_p = set(), set()
for k in sorted(set(gt_ev) & set(pr_ev)):
    g, p = gt_ev[k], pr_ev[k]
    t = tiou((g['w0'], g['w1']), (p['w0'], p['w1']))
    if g['w0'] is not None and (t is None or t < 0.5): continue
    used_g.add(k); used_p.add(k)
    rec = {'pair': k[0], 'front': k[1], 'back': k[2], 'gt_type': g['type'], 'pred_type': p['type'],
           'tiou': round(t, 4) if t is not None else None,
           'd_onset': round(p['w0'] - g['w0'], 4) if g['w0'] is not None and p['w0'] is not None else None,
           'd_end': round(p['w1'] - g['w1'], 4) if g['w1'] is not None and p['w1'] is not None else None}
    (matches if g['type'] == p['type'] else mismatches).append(rec)

def involves_excluded(e, side_tracks):
    pair = e['pair']
    for tid, cam in side_tracks(e):
        if tid and (pair, cam, tid) in gt['excluded']: return True
    return False

def tracks_of(e):
    return [(e['front'], 'front'), (e['back'], 'back')]

unmatched_pred, unmatched_gt = [], []
gt_by_pair = collections.defaultdict(list)
for k, e in gt_ev.items():
    if k not in used_g: gt_by_pair[e['pair']].append((k, e))
for k, e in pr_ev.items():
    if k in used_p: continue
    cls = 'residual'
    if involves_excluded(e, tracks_of): cls = 'correction_delta_excluded_track'
    else:
        for gk, ge in gt_by_pair[e['pair']]:
            ps = {(e['front'] or ''), (e['back'] or '')}; gs = {(ge['front'] or ''), (ge['back'] or '')}
            diff = ps ^ gs
            if ps & gs and all((e['pair'], 'front' if d == (e['front'] or ge['front']) else 'back', d) in gt['excluded'] or
                               any((e['pair'], c, d) in gt['excluded'] for c in CAMS) for d in diff if d):
                cls = 'would_match_differs_only_by_excluded_track'; break
    unmatched_pred.append({'pair': e['pair'], 'front': e['front'], 'back': e['back'], 'type': e['type'], 'class': cls})
for k, e in gt_ev.items():
    if k in used_g: continue
    cls = 'residual'
    for pk, pe in [(kk, ee) for kk, ee in pr_ev.items() if kk not in used_p and ee['pair'] == e['pair']]:
        ps = {(pe['front'] or ''), (pe['back'] or '')}; gs = {(e['front'] or ''), (e['back'] or '')}
        diff = ps ^ gs
        if ps & gs and all(any((e['pair'], c, d) in gt['excluded'] for c in CAMS) for d in diff if d):
            cls = 'would_match_differs_only_by_excluded_track'; break
    unmatched_gt.append({'pair': e['pair'], 'front': e['front'], 'back': e['back'], 'type': e['type'], 'class': cls})

tp_ev = len(matches)
per_type = {}
for t in sorted({e['type'] for e in gt['events']} | {e['type'] for e in pred['events']}):
    tp_t = sum(1 for m in matches if m['gt_type'] == t)
    fp_t = sum(1 for e in pred['events'] if e['type'] == t) - tp_t
    fn_t = sum(1 for e in gt['events'] if e['type'] == t) - tp_t
    per_type[t] = prf(tp_t, fp_t, fn_t)
import statistics as st
timed = [m for m in matches if m['d_onset'] is not None]
R['events'] = {**prf(tp_ev, len(pred['events']) - tp_ev, len(gt['events']) - tp_ev),
    'strict_identity_matches': len(matches) + len(mismatches), 'type_agreeing': tp_ev,
    'type_mismatches': mismatches, 'per_type': per_type,
    'timing_errors_sec': {
        'n': len(timed),
        'd_onset_mean': round(st.mean(m['d_onset'] for m in timed), 4) if timed else None,
        'd_onset_median': round(st.median(m['d_onset'] for m in timed), 4) if timed else None,
        'd_onset_mean_abs': round(st.mean(abs(m['d_onset']) for m in timed), 4) if timed else None,
        'd_end_mean': round(st.mean(m['d_end'] for m in timed), 4) if timed else None,
        'd_end_median': round(st.median(m['d_end'] for m in timed), 4) if timed else None,
        'd_end_mean_abs': round(st.mean(abs(m['d_end']) for m in timed), 4) if timed else None},
    'unmatched_pred_classes': dict(collections.Counter(u['class'] for u in unmatched_pred)),
    'unmatched_gt_classes': dict(collections.Counter(u['class'] for u in unmatched_gt)),
    'unmatched_pred': unmatched_pred, 'unmatched_gt': unmatched_gt, 'matches': matches}
for pair in PAIRS:
    tp_p = sum(1 for m in matches if m['pair'] == pair)
    PP[pair]['events'] = {'tp': tp_p,
        'fp': sum(1 for e in pred['events'] if e['pair'] == pair) - tp_p,
        'fn': sum(1 for e in gt['events'] if e['pair'] == pair) - tp_p}

# ---------------------------------------------------------------- synchronization summary
BOUND = {'pair_004', 'pair_005', 'pair_008', 'pair_015', 'pair_016', 'pair_019'}
NEAR = {'pair_004', 'pair_005', 'pair_008', 'pair_015'}
MEDIUM = {'pair_008', 'pair_019', 'pair_022'}
sync_rows = []
for pair in PAIRS:
    s = pred['sync'][pair]; q = s.get('q5_overlap_lookup_stats') or {}
    row = {'pair': pair, 'offset_sec': s.get('offset_sec'), 'epsilon_sync_ms': s.get('epsilon_sync_ms'),
           'drift_ms': s.get('drift_ms'), 'sync_quality': s.get('sync_quality'),
           'q5_front_mean_ms': (q.get('delta_pts_front_ms') or {}).get('mean'),
           'q5_front_max_ms': (q.get('delta_pts_front_ms') or {}).get('max'),
           'q5_back_mean_ms': (q.get('delta_pts_back_ms') or {}).get('mean'),
           'q5_back_max_ms': (q.get('delta_pts_back_ms') or {}).get('max'),
           'q5_skew_mean_ms': (q.get('inter_camera_pts_skew_ms') or {}).get('mean'),
           'q5_skew_max_ms': (q.get('inter_camera_pts_skew_ms') or {}).get('max'),
           'q5_n_samples': q.get('n_samples'),
           'decoded_front': pred['coverage'].get((pair, 'front')), 'decoded_back': pred['coverage'].get((pair, 'back'))}
    if pair in BOUND and row['epsilon_sync_ms'] is not None and row['drift_ms'] is not None:
        row['labelled_bound_eps_plus_abs_drift_ms'] = round(row['epsilon_sync_ms'] + abs(row['drift_ms']), 3)
    sync_rows.append(row)
def strat(rows):
    return {'pairs': len(rows),
            'median_epsilon_ms': round(st.median(r['epsilon_sync_ms'] for r in rows), 3),
            'median_abs_drift_ms': round(st.median(abs(r['drift_ms']) for r in rows), 3),
            'max_abs_drift_ms': max(abs(r['drift_ms']) for r in rows),
            'median_q5_skew_max_ms': round(st.median(r['q5_skew_max_ms'] for r in rows if r['q5_skew_max_ms'] is not None), 3)}
R['synchronization'] = {'by_pair': sync_rows,
    'strata': {'pooled': strat(sync_rows),
               'near_frame': strat([r for r in sync_rows if r['pair'] in NEAR]),
               'medium_quality': strat([r for r in sync_rows if r['pair'] in MEDIUM])},
    'gap1_caveat': 'Drift is reported, never applied (GAP-1): where drift != 0, epsilon_sync_ms is a LOWER BOUND on end-of-window misalignment. Labelled bound epsilon+|drift| given for pairs 004/005/008/015/016/019 per design OQ-2.',
    'coverage': 'Decoded frames per camera in by_pair (decoded_*); Stage J gates verified decoded == Phase 12C forensic counts for all 50 videos (corpus front 6,557 / back 8,684).'}
R['not_computable'] = {
    'person_detection': 'No GT person entities exist in the frozen release (protocol §3). Substitute: Stage J count reproduction delta 0 + head/rider families.',
    'motorcycle_frame_level_detection_ap': 'Release stores no per-frame GT motorcycle boxes (protocol §3); track-level metrics + anchored-box verification reported instead.',
    'hota': R['tracking_head_basis']['hota'],
    'rider_association_item_level': 'Stage J output has no per-item rider association results (protocol §7).'}

with open(OUT + '/stagek_results.json', 'w') as f: json.dump(R, f, indent=1)
with open(OUT + '/per_pair_stats.json', 'w') as f: json.dump(PP, f, indent=1)
with open(OUT + '/sync_by_pair.csv', 'w', newline='') as f:
    _fn = list(sync_rows[0].keys())
    if 'labelled_bound_eps_plus_abs_drift_ms' not in _fn: _fn.append('labelled_bound_eps_plus_abs_drift_ms')
    w = csv.DictWriter(f, fieldnames=_fn); w.writeheader(); w.writerows(sync_rows)
with open(OUT + '/events_detail.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['side', 'pair', 'front', 'back', 'gt_type', 'pred_type', 'tiou', 'd_onset', 'd_end', 'class'])
    for m in matches: w.writerow(['match', m['pair'], m['front'], m['back'], m['gt_type'], m['pred_type'], m['tiou'], m['d_onset'], m['d_end'], 'type_agree'])
    for m in mismatches: w.writerow(['match', m['pair'], m['front'], m['back'], m['gt_type'], m['pred_type'], m['tiou'], m['d_onset'], m['d_end'], 'type_mismatch'])
    for u in unmatched_pred: w.writerow(['pred_only', u['pair'], u['front'], u['back'], '', u['type'], '', '', '', u['class']])
    for u in unmatched_gt: w.writerow(['gt_only', u['pair'], u['front'], u['back'], u['type'], '', '', '', '', u['class']])
print('EVALUATE DONE')
print(json.dumps({k: R[k] for k in ('detection_head_operating_point', 'events') if k in R}, indent=1)[:800])
