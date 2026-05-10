#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""video_extractor.py

byebye / comehere タスクで共通して使う MediaPipe Pose+Hand 座標抽出。
推論フロー・初期化手順はオリジナルスクリプトから変更していない。
"""

from __future__ import annotations

import os
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "16")

from pathlib import Path
import threading
from typing import Any

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options


def _pose_side_indices(side: str) -> dict[str, int]:
    """Return BlazePose landmark indices for the requested body side."""
    if side == "Left":
        return {"HIP": 23, "SHOULDER": 11, "ELBOW": 13, "WRIST": 15, "INDEX": 19}
    return {"HIP": 24, "SHOULDER": 12, "ELBOW": 14, "WRIST": 16, "INDEX": 20}


def _pick_pose_px(lms: Any, idx: int, width: int, height: int) -> tuple[float, float, float]:
    lm = lms[idx]
    return float(lm.x * width), float(lm.y * height), float(getattr(lm, "presence", 1.0))


def _pick_hand_px(hand_lms: Any, idx: int, width: int, height: int) -> tuple[float, float]:
    lm = hand_lms[idx]
    return float(lm.x * width), float(lm.y * height)


def extract_pose_hand_px_from_video(
    video_path: Path,
    pose_model_path: Path,
    hand_model_path: Path,
    side: str,
) -> tuple[pd.DataFrame, int, int, float, float]:
    """Extract per-frame Pose+Hand landmark coordinates (pixel units).

    byebye / comehere タスク共通の抽出関数。
    推論フロー・ランドマーク選択ロジックはオリジナルスクリプトから変更していない。

    Returns (raw_df, width, height, src_fps, fps).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"動画を開けません: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    if src_fps <= 0:
        src_fps = 30.0
    fps = src_fps

    pose_idx = _pose_side_indices(side)

    # macOS 新バージョンで GPU 初期化がハングする問題を回避するため CPU を明示指定する。
    # VIDEO モードを維持することで、フレーム間の時系列情報を使った安定したトラッキングを保証する。
    pose_opt = vision.PoseLandmarkerOptions(
        base_options=base_options.BaseOptions(
            model_asset_path=str(pose_model_path),
            delegate=base_options.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False,
    )
    hand_opt = vision.HandLandmarkerOptions(
        base_options=base_options.BaseOptions(
            model_asset_path=str(hand_model_path),
            delegate=base_options.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )

    mp_image_format = mp.ImageFormat.SRGB
    rows: list[dict[str, Any]] = []
    frame = 0

    pose_lm = vision.PoseLandmarker.create_from_options(pose_opt)
    hand_lm = vision.HandLandmarker.create_from_options(hand_opt)
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break

            if frame % 50 == 0:
                print(f"  extracting frame {frame}...", flush=True)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp_image_format, data=rgb)

            ts_ms = int(frame * 1000.0 / fps)
            pose_res = pose_lm.detect_for_video(mp_img, ts_ms)
            hand_res = hand_lm.detect_for_video(mp_img, ts_ms)

            # Pose 側の主要上肢点を取得する。
            hip_x = hip_y = sh_x = sh_y = el_x = el_y = wr_x = wr_y = pose_ix_x = pose_ix_y = np.nan
            if pose_res.pose_landmarks:
                lms = pose_res.pose_landmarks[0]
                hip_x, hip_y, _ = _pick_pose_px(lms, pose_idx["HIP"], width, height)
                sh_x, sh_y, _ = _pick_pose_px(lms, pose_idx["SHOULDER"], width, height)
                el_x, el_y, _ = _pick_pose_px(lms, pose_idx["ELBOW"], width, height)
                wr_x, wr_y, _ = _pick_pose_px(lms, pose_idx["WRIST"], width, height)
                pose_ix_x, pose_ix_y, _ = _pick_pose_px(lms, pose_idx["INDEX"], width, height)

            hand_wrist_x = hand_wrist_y = np.nan
            idx_mcp_x = idx_mcp_y = np.nan
            idx_pip_x = idx_pip_y = np.nan
            hand_ix_x = hand_ix_y = np.nan

            # 複数 hand 候補がある場合は Pose wrist に最も近い手を採用する。
            chosen = None
            if hand_res.hand_landmarks:
                if np.isfinite(wr_x) and np.isfinite(wr_y):
                    best_d = 1e18
                    for hl in hand_res.hand_landmarks:
                        hwx, hwy = _pick_hand_px(hl, 0, width, height)
                        d = (hwx - wr_x) ** 2 + (hwy - wr_y) ** 2
                        if d < best_d:
                            best_d = d
                            chosen = hl
                else:
                    chosen = hand_res.hand_landmarks[0]

            if chosen is not None:
                hand_wrist_x, hand_wrist_y = _pick_hand_px(chosen, 0, width, height)
                idx_mcp_x, idx_mcp_y = _pick_hand_px(chosen, 5, width, height)
                idx_pip_x, idx_pip_y = _pick_hand_px(chosen, 6, width, height)
                hand_ix_x, hand_ix_y = _pick_hand_px(chosen, 8, width, height)  # INDEX_TIP

            # 代表 index は hand INDEX_TIP を優先し、欠損時は Pose INDEX にフォールバック。
            if np.isfinite(hand_ix_x) and np.isfinite(hand_ix_y):
                index_x = hand_ix_x
                index_y = hand_ix_y
                index_source = "hand"
            elif np.isfinite(pose_ix_x) and np.isfinite(pose_ix_y):
                index_x = pose_ix_x
                index_y = pose_ix_y
                index_source = "pose"
            else:
                index_x = np.nan
                index_y = np.nan
                index_source = "none"

            rows.append({
                "frame": frame,
                "hip_x_px": hip_x, "hip_y_px": hip_y,
                "shoulder_x_px": sh_x, "shoulder_y_px": sh_y,
                "elbow_x_px": el_x, "elbow_y_px": el_y,
                "wrist_x_px": wr_x, "wrist_y_px": wr_y,
                "pose_index_x_px": pose_ix_x, "pose_index_y_px": pose_ix_y,
                "hand_index_tip_x_px": hand_ix_x, "hand_index_tip_y_px": hand_ix_y,
                "index_x_px": index_x, "index_y_px": index_y,
                "index_source": index_source,
                "hand_wrist_x_px": hand_wrist_x, "hand_wrist_y_px": hand_wrist_y,
                "index_mcp_x_px": idx_mcp_x, "index_mcp_y_px": idx_mcp_y,
                "index_pip_x_px": idx_pip_x, "index_pip_y_px": idx_pip_y,
            })

            frame += 1
    finally:
        cap.release()
        # macOS の MediaPipe VIDEO モードでは close() がハングすることがあるため
        # タイムアウト付きのデーモンスレッドで呼び出す。
        for _lm in (pose_lm, hand_lm):
            _t = threading.Thread(target=_lm.close, daemon=True)
            _t.start()
            _t.join(timeout=10.0)

    raw_df = pd.DataFrame(rows)
    return raw_df, width, height, src_fps, fps
