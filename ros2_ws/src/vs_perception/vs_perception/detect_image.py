"""Run one detector on image files, without ROS. For the FM comparison study.

  ros2 run vs_perception detect_image --detector owlv2 --prompt "red box" /data/frames/*.png \
      --out /data/results/owlv2_red_box
Writes annotated images and a detections.csv (file, prompt, rank, box, score, latency_ms).
"""
import argparse
import csv
import time
from pathlib import Path

import cv2

from vs_perception.detectors import DETECTORS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('images', nargs='+', type=Path)
    ap.add_argument('--detector', required=True, choices=DETECTORS.names())
    ap.add_argument('--prompt', action='append', required=True,
                    help='repeat to compare several prompts in one run')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--threshold', type=float, default=0.2)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    detector = DETECTORS.create(args.detector, device=args.device, score_threshold=args.threshold)
    detector.load()
    args.out.mkdir(parents=True, exist_ok=True)

    with open(args.out / 'detections.csv', 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['file', 'prompt', 'rank', 'x_min', 'y_min', 'x_max', 'y_max', 'score',
                         'latency_ms'])
        for path in args.images:
            bgr = cv2.imread(str(path))
            if bgr is None:
                print(f'skip unreadable {path}')
                continue
            rgb = bgr[:, :, ::-1].copy()
            for prompt in args.prompt:
                t0 = time.perf_counter()
                dets = sorted(detector.detect(rgb, prompt), key=lambda d: -d.score)
                ms = (time.perf_counter() - t0) * 1000
                vis = bgr.copy()
                for rank, d in enumerate(dets):
                    writer.writerow([path.name, prompt, rank, *[round(x, 1) for x in d.box],
                                     round(d.score, 4), round(ms, 1)])
                    x0, y0, x1, y1 = map(int, d.box)
                    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0) if rank == 0 else (0, 160, 255), 2)
                    cv2.putText(vis, f'{d.score:.2f}', (x0, max(y0 - 4, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                if not dets:
                    writer.writerow([path.name, prompt, -1, '', '', '', '', '', round(ms, 1)])
                tag = prompt.replace(' ', '_')
                cv2.imwrite(str(args.out / f'{path.stem}__{tag}.jpg'), vis)
                print(f'{path.name} "{prompt}": {len(dets)} detections, {ms:.0f} ms')
