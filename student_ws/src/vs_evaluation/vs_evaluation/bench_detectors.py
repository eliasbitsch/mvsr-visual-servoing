"""Offline localisation benchmark: saved camera frames x detectors x prompt formulations.

Ground truth boxes come from projecting the object geometry (targets.yaml poses, camera pose and
K saved by run_eval), so every object in a frame is a test case, not only the servo target.
    $VS_PYTHON -m vs_evaluation.bench_detectors --campaign /root/vs_data/eval/final
Writes <campaign>/bench/results.csv (one row per query) and prints a summary.
"""
import argparse
import csv
import gc
import glob
import json
import os
import time

import cv2
import numpy as np
import yaml

# Prompt levels: plain category, the main-run attribute prompt, synonym, indirect description
PROMPTS = {
    'red_can': {'category': 'can', 'attribute': 'red can', 'synonym': 'red soda tin',
                'description': 'the tallest object on the table'},
    'blue_box': {'category': 'box', 'attribute': 'blue box', 'synonym': 'blue carton',
                 'description': 'the flat rectangular object'},
    'green_cup': {'category': 'cup', 'attribute': 'green cup', 'synonym': 'green mug',
                  'description': 'the drinking vessel'},
    'yellow_ball': {'category': 'ball', 'attribute': 'yellow ball', 'synonym': 'yellow sphere',
                    'description': 'the round object that can roll'},
}
# Not in the scene: any detection is a false positive
NEGATIVES = ['purple bottle', 'black phone']
# Radius for round objects; the box uses its footprint
SHAPES = {'red_can': ('cylinder', 0.033), 'green_cup': ('cylinder', 0.04),
          'yellow_ball': ('sphere', 0.035), 'blue_box': ('box', None)}
CONDITIONS = {'main_dino_sam2': 'nominal', 'rob_occl40': 'occlusion 40%',
              'rob_light30': 'light 30%', 'rob_light150': 'light 150%'}
DETECTORS = {
    'grounding_dino': {}, 'owlv2': {}, 'yolo_world': {}, 'florence2': {}, 'sam3': {},
    'grounding_dino_base': {'name': 'grounding_dino', 'model_id': 'IDEA-Research/grounding-dino-base'},
    'yoloe': {},
    # ~8 s per call (it repeats its boxes up to the token limit): same subset as Opus
    'locate_anything': {'conditions': 1, 'starts': ['S1', 'S4', 'S8']},
    'qwen3vl_8b': {'name': 'vlm', 'model': 'qwen3-vl:8b-instruct', 'conditions': 2},
    'gemma4_e4b': {'name': 'vlm', 'model': 'gemma4:e4b', 'conditions': 2},
    # Each call is a CLI round trip on the subscription: a small, fixed subset
    'opus55': {'name': 'vlm', 'model': 'claude-opus-5-5', 'url': 'http://localhost:11436/v1',
               'conditions': 1, 'starts': ['S1', 'S4', 'S8']},
    'gpt6_astra': {'name': 'vlm', 'model': 'gpt-6-astra', 'url': 'http://localhost:11436/v1',
                   'conditions': 1, 'starts': ['S1', 'S4', 'S8']},
}
FIELDS = ['detector', 'condition', 'image', 'object', 'level', 'prompt', 'visible_frac',
          'n_dets', 'iou', 'correct', 'false_positive', 'latency_s']
IOU_OK = 0.5


def quat_matrix(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def object_points(cfg, name):
    x, y, _, yaw = cfg['pose']
    top = cfg['top']
    kind, r = SHAPES[name]
    a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    if kind == 'cylinder':
        p = np.concatenate([np.stack([r * np.cos(a), r * np.sin(a), np.full_like(a, z)], 1)
                            for z in (0.0, top)])
    elif kind == 'sphere':
        u, v = np.meshgrid(a, np.linspace(0, np.pi, 24))
        p = np.stack([r * np.cos(u) * np.sin(v), r * np.sin(u) * np.sin(v),
                      top / 2 + r * np.cos(v)], -1).reshape(-1, 3)
    else:
        fx, fy = cfg['footprint']
        p = np.array([[sx * fx / 2, sy * fy / 2, z]
                      for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, top)])
    c, s = np.cos(yaw), np.sin(yaw)
    return p @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T + [x, y, 0]


def gt_box(view, points, w, h):
    """Projected box clipped to the image, and the fraction of it inside the image."""
    K = np.array(view['K']).reshape(3, 3)
    cam = view['camera_pose_world']
    pc = (points - cam[:3]) @ quat_matrix(*cam[3:])  # world -> optical frame
    if (pc[:, 2] <= 0.01).any():
        return None, 0.0
    uv = pc @ K.T
    uv = uv[:, :2] / uv[:, 2:]
    full = np.r_[uv.min(0), uv.max(0)]
    clip = np.clip(full, 0, [w, h, w, h])
    area = lambda b: max(b[2] - b[0], 0) * max(b[3] - b[1], 0)
    return clip, area(clip) / max(area(full), 1e-9)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def load_frames(campaign, targets_cfg, smoke):
    frames = []
    for block, condition in CONDITIONS.items():
        for png in sorted(glob.glob(os.path.join(campaign, block, 'images', '*.png'))):
            meta = png[:-4] + '.json'
            name = os.path.basename(png)[:-4]
            # Reference frames are taken before the occluder or light change: nominal only
            if not os.path.exists(meta) or (condition != 'nominal' and name.endswith('_reference')):
                continue
            view = json.load(open(meta))
            img = cv2.cvtColor(cv2.imread(png), cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            objects = {}
            # Under occlusion only the occluded target is a test case; the others are not covered
            candidates = [view['target']] if condition.startswith('occlusion') else list(PROMPTS)
            for obj in candidates:
                box, frac = gt_box(view, object_points(targets_cfg[obj], obj), w, h)
                if box is not None and frac >= 0.5:
                    objects[obj] = (box, frac)
            frames.append({'block': block, 'condition': condition,
                           'image': name, 'rgb': img, 'objects': objects})
    if smoke:
        frames = [f for f in frames if f['image'].endswith('_S1')][:2]
    return frames


def queries(frame):
    for name, (box, frac) in frame['objects'].items():
        for level, prompt in PROMPTS[name].items():
            yield name, level, prompt, box, frac
    for prompt in NEGATIVES:
        yield '', 'negative', prompt, None, 0.0


def unload_ollama(url='http://localhost:11434'):
    """Free the VRAM a VLM of the previous detector may still hold (keep-alive 30 min)."""
    import urllib.request
    try:
        with urllib.request.urlopen(f'{url}/api/ps', timeout=3) as r:
            names = [m['name'] for m in json.load(r).get('models', [])]
        for name in names:
            body = json.dumps({'model': name, 'keep_alive': 0}).encode()
            urllib.request.urlopen(urllib.request.Request(
                f'{url}/api/generate', body, {'Content-Type': 'application/json'}), timeout=60)
    except OSError:
        pass


def make_detector(key, spec):
    from vs_perception.detectors import DETECTORS as REGISTRY
    opts = {k: v for k, v in spec.items() if k in ('model', 'url', 'model_id')}
    if spec.get('name') == 'vlm':
        opts['timeout'] = 300  # the first call loads the model into the server
    det = REGISTRY.create(spec.get('name', key), device='cuda', score_threshold=0.3, **opts)
    det.load()
    return det


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--campaign', required=True)
    ap.add_argument('--detectors', nargs='+', default=list(DETECTORS), choices=list(DETECTORS))
    ap.add_argument('--targets-yaml', help='default: vs_sim share directory')
    ap.add_argument('--smoke', action='store_true', help='two frames per detector')
    args = ap.parse_args()

    path = args.targets_yaml
    if path is None:
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(get_package_share_directory('vs_sim'), 'config', 'targets.yaml')
    targets_cfg = yaml.safe_load(open(path))['targets']
    frames = load_frames(args.campaign, targets_cfg, args.smoke)
    out_dir = os.path.join(args.campaign, 'bench')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'results.csv')
    done = set()
    if os.path.exists(out_path):
        done = {r['detector'] for r in csv.DictReader(open(out_path))}
    print(f'{len(frames)} frames, {sum(len(f["objects"]) for f in frames)} objects', flush=True)

    import torch
    failed = []
    with open(out_path, 'a', newline='') as fh:
        writer = csv.DictWriter(fh, FIELDS)
        if fh.tell() == 0:  # also after a crash that left only the header behind
            writer.writeheader()
        for key in args.detectors:
            if key in done:
                print(f'{key}: done, skipped', flush=True)
                continue
            spec = DETECTORS[key]
            subset = [f for f in frames
                      if f['condition'] in list(CONDITIONS.values())[:spec.get('conditions', 4)]
                      and (not spec.get('starts') or f['image'].split('_')[-1] in spec['starts'])]
            if spec.get('name') != 'vlm':
                unload_ollama()
            try:
                det = make_detector(key, spec)
                det.detect(subset[0]['rgb'], 'red can')  # warm-up, not timed
            except Exception as exc:  # e.g. a VLM server that is not up: keep the others
                print(f'{key}: load failed: {exc}', flush=True)
                failed.append(key)
                continue
            rows = []
            # Same prompt back to back: models that encode the text once per prompt change
            # (YOLO-World) are timed as in the closed loop, where the prompt is fixed
            jobs = sorted(((q[2], i, frame, q) for i, frame in enumerate(subset)
                           for q in queries(frame)), key=lambda j: (j[0], j[1]))
            for _, _, frame, (name, level, prompt, box, frac) in jobs:
                t0 = time.monotonic()
                try:
                    dets = det.detect(frame['rgb'], prompt)
                except Exception as exc:
                    print(f'{key} {frame["image"]} "{prompt}": {exc}', flush=True)
                    dets = []
                latency = time.monotonic() - t0
                best = max(dets, key=lambda d: d.score) if dets else None
                score = iou(best.box, box) if best is not None and box is not None else 0.0
                rows.append({'detector': key, 'condition': frame['condition'],
                             'image': frame['image'], 'object': name, 'level': level,
                             'prompt': prompt, 'visible_frac': round(frac, 3),
                             'n_dets': len(dets), 'iou': round(score, 4),
                             'correct': int(box is not None and score >= IOU_OK),
                             'false_positive': int(box is None and bool(dets)),
                             'latency_s': round(latency, 4)})
            writer.writerows(rows)
            fh.flush()
            pos = [r for r in rows if r['level'] != 'negative']
            neg = [r for r in rows if r['level'] == 'negative']
            print(f'{key}: {len(rows)} queries, correct {np.mean([r["correct"] for r in pos]):.2f}, '
                  f'FP on negatives {np.mean([r["false_positive"] for r in neg]):.2f}, '
                  f'median latency {np.median([r["latency_s"] for r in rows]):.3f} s', flush=True)
            del det
            gc.collect()
            torch.cuda.empty_cache()
    if failed:  # visible to run_campaign.py instead of a silently shorter table
        print(f'models that did not load: {failed}', flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
