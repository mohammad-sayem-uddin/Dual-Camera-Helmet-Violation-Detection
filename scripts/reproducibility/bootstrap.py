"""Stage K bootstrap — protocol §10: pair-level resampling, B=2000, seed 20261109,
percentile 95% CIs from per-pair sufficient statistics recorded during scoring."""
import json, os
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
PP = json.load(open(OUT + '/per_pair_stats.json'))
PAIRS = sorted(PP.keys())
rng = np.random.default_rng(20261109)
B = 2000

def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)

def sumkey(sample, fam, keys):
    return {k: sum(PP[p][fam][k] for p in sample) for k in keys}

metrics = {k: [] for k in ('head_det_p', 'head_det_r', 'head_det_f1', 'rider_f1', 'rider_mean_iou',
                           'helmet_f1', 'no_helmet_f1', 'assoc_moto_f1', 'event_f1', 'idf1')}
for b in range(B):
    sample = [PAIRS[i] for i in rng.integers(0, len(PAIRS), len(PAIRS))]
    s = sumkey(sample, 'head_det', ('tp', 'fp', 'fn'))
    p, r, f = prf(s['tp'], s['fp'], s['fn'])
    metrics['head_det_p'].append(p); metrics['head_det_r'].append(r); metrics['head_det_f1'].append(f)
    s = sumkey(sample, 'rider_primary', ('tp', 'fp', 'fn', 'iou_sum', 'iou_n'))
    _, _, f = prf(s['tp'], s['fp'], s['fn']); metrics['rider_f1'].append(f)
    metrics['rider_mean_iou'].append(s['iou_sum'] / s['iou_n'] if s['iou_n'] else 0.0)
    for cls, mk in (('helmet', 'helmet_f1'), ('no_helmet', 'no_helmet_f1')):
        tp = sum(PP[p]['helmet'][cls]['tp'] for p in sample)
        fp = sum(PP[p]['helmet'][cls]['fp'] for p in sample)
        fn = sum(PP[p]['helmet'][cls]['fn'] for p in sample)
        metrics[mk].append(prf(tp, fp, fn)[2])
    s = sumkey(sample, 'assoc_moto', ('tp', 'fp', 'fn'))
    metrics['assoc_moto_f1'].append(prf(s['tp'], s['fp'], s['fn'])[2])
    s = sumkey(sample, 'events', ('tp', 'fp', 'fn'))
    metrics['event_f1'].append(prf(s['tp'], s['fp'], s['fn'])[2])
    s = sumkey(sample, 'tracking', ('idtp', 'idfp', 'idfn'))
    idp = s['idtp'] / (s['idtp'] + s['idfp']) if s['idtp'] + s['idfp'] else 0.0
    idr = s['idtp'] / (s['idtp'] + s['idfn']) if s['idtp'] + s['idfn'] else 0.0
    metrics['idf1'].append(2 * idp * idr / (idp + idr) if idp + idr else 0.0)

cis = {}
for k, v in metrics.items():
    lo, hi = np.percentile(v, [2.5, 97.5])
    cis[k] = {'ci95_lo': round(float(lo), 4), 'ci95_hi': round(float(hi), 4),
              'bootstrap_mean': round(float(np.mean(v)), 4)}
json.dump({'B': B, 'seed': 20261109, 'unit': 'pair (25 pairs, resampled with replacement)',
           'cis': cis}, open(OUT + '/bootstrap_cis.json', 'w'), indent=1)
print('BOOTSTRAP DONE', json.dumps(cis, indent=1))
