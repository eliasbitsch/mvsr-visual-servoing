#!/usr/bin/env python3
"""Generates the textures of the target models.

    python3 tools/generate_textures.py            # from the vs_sim package folder

  models/aruco_target/materials/textures/aruco_4x4_50_id0.png
      ArUco DICT_4X4_50, ID 0. The marker covers 10/14 of the texture,
      the rest is a white border (board 0.14 m -> marker 0.10 m).
  models/feature_target/materials/textures/feature_board.png
      Texture-rich board for keypoint detection / feature matching.

Set --mirror if a check in the simulation shows that Gazebo maps the texture
mirrored onto the box (ArUco markers are then not detected).
"""

import argparse
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "..", "models")


def aruco_texture(marker_id=0, px=1400, marker_frac=10.0 / 14.0):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    m = int(round(px * marker_frac))
    if hasattr(cv2.aruco, "generateImageMarker"):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, m)
    else:  # OpenCV < 4.7
        marker = cv2.aruco.drawMarker(dictionary, marker_id, m)
    img = np.full((px, px), 255, np.uint8)
    o = (px - m) // 2
    img[o:o + m, o:o + m] = marker
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def feature_texture(w=1600, h=1200, seed=7):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 235, np.uint8)
    # smooth colour gradient background
    gx = np.linspace(0, 1, w)[None, :, None]
    gy = np.linspace(0, 1, h)[:, None, None]
    img = (img * (0.75 + 0.25 * gx) * (0.8 + 0.2 * gy)).astype(np.uint8)

    def colour():
        return tuple(int(c) for c in rng.integers(0, 230, 3))

    for _ in range(70):
        kind = rng.integers(0, 4)
        c = colour()
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        s = int(rng.integers(20, 140))
        if kind == 0:
            cv2.circle(img, (x, y), s // 2, c, -1, cv2.LINE_AA)
        elif kind == 1:
            cv2.rectangle(img, (x, y), (x + s, y + int(s * rng.uniform(0.4, 1.5))), c, -1)
        elif kind == 2:
            pts = rng.integers(-s, s, (5, 2)) + np.array([x, y])
            cv2.fillPoly(img, [pts.astype(np.int32)], c, cv2.LINE_AA)
        else:
            cv2.line(img, (x, y), (x + int(rng.integers(-300, 300)), y + int(rng.integers(-300, 300))),
                     c, int(rng.integers(3, 15)), cv2.LINE_AA)
    # checkerboard patch and text give strong corners
    for i in range(6):
        for j in range(4):
            if (i + j) % 2 == 0:
                cv2.rectangle(img, (60 + i * 40, h - 220 + j * 40), (100 + i * 40, h - 180 + j * 40), (20, 20, 20), -1)
    for k, word in enumerate(["FH TECHNIKUM", "VISUAL", "SERVOING", "UR5e"]):
        cv2.putText(img, word, (int(w * 0.45), 160 + k * 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0,
                    (30, 30, 120), 8, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 0), 12)
    return img


def save(img, *parts):
    path = os.path.join(MODELS, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)
    print("wrote", os.path.normpath(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", action="store_true", help="mirror textures horizontally")
    args = ap.parse_args()
    a = aruco_texture()
    f = feature_texture()
    if args.mirror:
        a, f = a[:, ::-1], f[:, ::-1]
    save(a, "aruco_target", "materials", "textures", "aruco_4x4_50_id0.png")
    save(f, "feature_target", "materials", "textures", "feature_board.png")


if __name__ == "__main__":
    main()
