"""TASK 4 visual evidence: sequentially decode source videos (no seeking) and save frames
for 8 example tracks. Full frame (GT head bbox outlined in red, downscaled to 1280w) +
native-resolution crop around the head bbox. Outputs PNGs to
repo/docs/phase12/track_failure_examples/ and a manifest to work/postcompletion/."""
import json, os, glob
import cv2

ROOT = os.path.expanduser('~/workspace/research_helmet')
REL = ROOT + '/repo/data/dual_camera_test_v1/releases/dual_camera_gt_v1.0_ai_assisted_audited'
VID = ROOT + '/repo/data/dual_camera_test_v1'
OUTDIR = ROOT + '/repo/docs/phase12/track_failure_examples'
os.makedirs(OUTDIR, exist_ok=True)

# load GT observations per (pair, cam, track)
gt_obs = {}
for pf in sorted(glob.glob(REL + '/pairs/*/pair_*_gt.json')):
    g = json.load(open(pf)); pr = g['pair']; pair = pr['pair_id']
    for cam in ('front', 'back'):
        ca = pr['camera_annotations'][cam]
        for mt in ca['motorcycle_tracks']:
            for st in mt['rider_subtracks']:
                gt_obs[(pair, cam, st['subtrack_id'])] = {
                    o['frame_index']: o for o in st['head_observations']}
        for ot in ca['orphan_rider_tracks']:
            gt_obs[(pair, cam, ot['track_id'])] = {
                o['frame_index']: o for o in ot['head_observations']}

def largest_frame(key, want_state=None):
    obs = gt_obs[key]
    cand = [(f, o) for f, o in obs.items() if want_state is None or o['state'] == want_state]
    f, o = max(cand, key=lambda fo: (fo[1]['bbox_xyxy'][2]-fo[1]['bbox_xyxy'][0]) *
                                    (fo[1]['bbox_xyxy'][3]-fo[1]['bbox_xyxy'][1]))
    return f

EXAMPLES = [
    # (pair, cam, track, frame or 'largest', want_state, label)
    ('pair_002', 'front', 'front_moto_0001_rider_s0', 'largest', None, 'E1_pair002_uncertain'),
    ('pair_024', 'front', 'front_moto_0001_rider_s1', 'largest', None, 'E2_evidence_rich_flushed'),
    ('pair_019', 'front', 'front_moto_0002_rider_s0', 'largest', None, 'E3_sparse_abstention'),
    ('pair_012', 'back',  'back_moto_0002_rider_s0',  'largest', None, 'E4_back_sparse_abstention'),
    ('pair_004', 'front', 'front_orphan_0003', 35, None, 'E5_donning_before'),
    ('pair_004', 'front', 'front_orphan_0003', 42, None, 'E5_donning_after'),
    ('pair_004', 'front', 'front_orphan_0005', 80, None, 'E6_false_no_helmet'),
    ('pair_006', 'front', 'front_moto_0001_rider_s0', 'largest', None, 'E7_agree_subtrack_control'),
    ('pair_004', 'front', 'front_orphan_0007', 'largest', 'helmet', 'E8_agree_orphan_control'),
]

targets = {}   # (pair, cam) -> {frame: [(track, label)]}
manifest = []
for pair, cam, track, fr, want, label in EXAMPLES:
    key = (pair, cam, track)
    frame = largest_frame(key, want) if fr == 'largest' else fr
    o = gt_obs[key][frame]
    targets.setdefault((pair, cam), {})[frame] = (track, label, o)
    manifest.append({'label': label, 'pair': pair, 'camera': cam, 'track_id': track,
                     'frame_index': frame, 'pts_norm_sec': o['pts_norm_sec'],
                     'gt_obs_state_at_frame': o['state'], 'gt_obs_confidence': o['confidence'],
                     'head_bbox_xyxy': o['bbox_xyxy'],
                     'full_png': f'{pair}_{track}_f{frame:04d}.png',
                     'crop_png': f'{pair}_{track}_f{frame:04d}_crop.png'})

for (pair, cam), fmap in sorted(targets.items()):
    vpath = os.path.join(VID, cam, f'{pair}_{cam}.mp4')
    cap = cv2.VideoCapture(vpath)
    assert cap.isOpened(), vpath
    want = sorted(fmap)
    idx = -1
    got = {}
    while want:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx == want[0]:
            got[idx] = frame.copy()
            want.pop(0)
    cap.release()
    assert len(got) == len(fmap), (pair, cam, len(got), len(fmap))
    for fi, img in got.items():
        track, label, o = fmap[fi]
        x1, y1, x2, y2 = [int(round(v)) for v in o['bbox_xyxy']]
        full = img.copy()
        cv2.rectangle(full, (x1, y1), (x2, y2), (0, 0, 255), 6)
        h, w = full.shape[:2]
        scale = 1280 / w
        full_small = cv2.resize(full, (1280, int(h * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(OUTDIR, f'{pair}_{track}_f{fi:04d}.png'), full_small)
        # native-res crop: bbox expanded 3x, min side 320 px, clipped to frame
        bw, bh = x2 - x1, y2 - y1
        cw, ch = max(320, 3 * bw), max(320, 3 * bh)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cx1, cy1 = max(0, cx - cw // 2), max(0, cy - ch // 2)
        cx2, cy2 = min(w, cx1 + cw), min(h, cy1 + ch)
        crop = img[cy1:cy2, cx1:cx2]
        cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(OUTDIR, f'{pair}_{track}_f{fi:04d}_crop.png'), crop)
        print('saved', pair, cam, track, 'frame', fi, 'state', o['state'])

with open(ROOT + '/work/postcompletion/examples_manifest.json', 'w') as f:
    json.dump(manifest, f, indent=1)
print('manifest entries:', len(manifest))
