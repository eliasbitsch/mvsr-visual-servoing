"""Calibrate a box verifier (vs_perception.verifiers) on the saved campaign frames.

    $VS_PYTHON -m vs_evaluation.bench_verifier --campaign /root/vs_data/eval/final

Every visible object's ground-truth box is scored against its own prompts (positives, per prompt
level) and against the prompts of the other objects and of absent objects (negatives: what a
detector box on the wrong object would be checked against). Prints how many positives and
negatives each threshold keeps, and writes the scores to <campaign>/bench/verifier_scores.csv.
The verifier's default threshold should keep nearly all attribute prompts (the closed-loop prompt)
while dropping most negatives.
"""
import argparse
import csv
import os

import numpy as np
import yaml

from vs_evaluation.bench_detectors import NEGATIVES, PROMPTS, load_frames

THRESHOLDS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--campaign', required=True)
    ap.add_argument('--verifier', default='siglip')
    ap.add_argument('--model', help='verifier weights, default: the verifier\'s own')
    ap.add_argument('--targets-yaml', help='default: vs_sim share directory')
    args = ap.parse_args()

    path = args.targets_yaml
    if path is None:
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(get_package_share_directory('vs_sim'), 'config', 'targets.yaml')
    frames = load_frames(args.campaign, yaml.safe_load(open(path))['targets'], smoke=False)

    from vs_perception.verifiers import VERIFIERS
    opts = {'model': args.model} if args.model else {}
    verifier = VERIFIERS.create(args.verifier, device='auto', **opts)
    verifier.load()

    rows = []
    for frame in frames:
        for obj, (box, _) in frame['objects'].items():
            others = [PROMPTS[o]['attribute'] for o in PROMPTS if o != obj]
            cases = [(lvl, p) for lvl, p in PROMPTS[obj].items()] \
                + [('other object', p) for p in others] + [('absent object', p) for p in NEGATIVES]
            for kind, prompt in cases:
                score = verifier.scores(frame['rgb'], prompt, [box])[0]
                rows.append({'image': frame['image'], 'condition': frame['condition'],
                             'object': obj, 'kind': kind, 'prompt': prompt,
                             'score': round(score, 6)})

    out = os.path.join(args.campaign, 'bench', f'verifier_scores_{args.verifier}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    kinds = ['attribute', 'category', 'synonym', 'description', 'other object', 'absent object']
    print(f'{len(frames)} frames, {len(rows)} crop/prompt pairs; share kept per threshold')
    print('threshold ' + ' '.join(f'{k:>14s}' for k in kinds))
    for t in THRESHOLDS:
        kept = [np.mean([r['score'] >= t for r in rows if r['kind'] == k]) for k in kinds]
        print(f'{t:9.3f} ' + ' '.join(f'{v:14.2f}' for v in kept))
    for obj in PROMPTS:
        s = sorted(r['score'] for r in rows if r['object'] == obj and r['kind'] == 'attribute')
        if s:
            print(f'{obj:12s} attribute prompt: min {s[0]:.4f}, median {s[len(s) // 2]:.4f}')
    print(f'written to {out}')


if __name__ == '__main__':
    main()
