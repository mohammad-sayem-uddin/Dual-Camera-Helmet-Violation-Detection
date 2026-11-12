"""Stage K bootstrap — protocol §10: pair-level resampling, B=2000, seed 20261109,
percentile 95% CIs from per-pair sufficient statistics recorded during scoring."""
import argparse, json, os
from pathlib import Path
import numpy as np

REPO = Path(os.environ.get('RESEARCH_HELMET_ROOT', Path(__file__).resolve().parents[2]))
DEFAULT_OUT = os.environ.get('STAGEK_OUT_DIR', str(REPO / 'results' / 'stagek_reproduced'))


def run_bootstrap(out_dir: str | None = None) -> dict:
    out_base = out_dir or DEFAULT_OUT
    PP = json.load(open(os.path.join(out_base, 'per_pair_stats.json')))
    PAIRS = sorted(PP.keys())
    rng = np.random.default_rng(20261109)
    B = 2000

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    def sumkey(sample, fam, keys):
        return {k: sum(PP[p][fam][k] for p in sample) for k in keys}

    has_assoc_rider = all('assoc_rider' in PP[p] for p in PAIRS)
    metric_keys = [
        'head_det_p', 'head_det_r', 'head_det_f1', 'rider_f1', 'rider_mean_iou',
        'helmet_f1', 'no_helmet_f1', 'assoc_moto_f1',
    ]
    if has_assoc_rider:
        metric_keys.append('assoc_rider_f1')
    metric_keys.extend(['event_f1', 'idf1'])
    metrics = {k: [] for k in metric_keys}

    for _ in range(B):
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
        if has_assoc_rider:
            s = sumkey(sample, 'assoc_rider', ('tp', 'fp', 'fn'))
            metrics['assoc_rider_f1'].append(prf(s['tp'], s['fp'], s['fn'])[2])
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
    payload = {'B': B, 'seed': 20261109, 'unit': 'pair (25 pairs, resampled with replacement)', 'cis': cis}
    with open(os.path.join(out_base, 'bootstrap_cis.json'), 'w') as f:
        json.dump(payload, f, indent=1)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=DEFAULT_OUT)
    args = ap.parse_args()
    payload = run_bootstrap(out_dir=args.out)
    print('BOOTSTRAP DONE', json.dumps(payload['cis'], indent=1))


if __name__ == '__main__':
    main()
