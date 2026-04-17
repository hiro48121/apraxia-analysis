#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""video_to_byebye_metrics.py

目的
- 1本の動画を解析して、同じ out_dir に以下を保存する（後処理スクリプト不要）:
  - frames.csv
  - cycles.csv
  - summary.csv
  - waveform_<動画名>.png

備考
- byebye では PoseLandmarker と HandLandmarker を両方使用します。
- FFmpeg 由来の warning 表示を抑えるため、OpenCV import 前に
  OPENCV_FFMPEG_LOGLEVEL=16 を既定値として設定します。
  （外部で別値を export 済みの場合はその値を優先します。）
"""

from __future__ import annotations

import os
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "16")

import argparse
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options


# 詳細コメント版について
# - この版では、解析ロジック・閾値・分岐・出力列は変えず、処理理解のための説明コメントだけを追加しています。
# - byebye 系は、(1) Pose/Hand の両方から上肢・手指座標を抽出し、(2) 代表 index を構成し、
#   (3) 外れ値補正・平滑化・動作区間推定を行い、(4) 周期検出後に frames/cycles/summary を保存します。
# - comehere と似ていますが、byebye 固有の抽出関数や手・Pose の統合方法が分かるよう補足しています。


# =========================
# math helpers (same as byebye v2)
# =========================
# ------------------------------------------------------------------------------
# angle_deg
# 役割: 3点から角度 ABC を計算する基本関数。
# 入力: ax..cy。
# 出力: 角度 degree。
# 注意: 肩・肘・手関節・示指 MCP 角度の算出に再利用する。
# ------------------------------------------------------------------------------
def angle_deg(ax, ay, bx, by, cx, cy):
    """Angle ABC in degrees. Returns NaN if any point is missing."""
    pts = [ax, ay, bx, by, cx, cy]
    if any(pd.isna(v) for v in pts):
        return np.nan
    v1x, v1y = ax - bx, ay - by
    v2x, v2y = cx - bx, cy - by
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 == 0 or n2 == 0:
        return np.nan
    cosv = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
    return math.degrees(math.acos(cosv))


# ------------------------------------------------------------------------------
# pca_plane_deg
# 役割: 2次元軌跡の主方向角を求める。
# 入力: x, y。
# 出力: degree。
# 注意: サイクル運動方向の要約指標。
# ------------------------------------------------------------------------------
def pca_plane_deg(x, y):
    """Principal direction angle (deg) of trajectory in 2D."""
    xy = np.column_stack([x, y]).astype(float)
    xy = xy[~np.isnan(xy).any(axis=1)]
    if len(xy) < 3:
        return np.nan
    xy = xy - xy.mean(axis=0, keepdims=True)
    cov = np.cov(xy.T)
    w, v = np.linalg.eig(cov)
    pc = v[:, int(np.argmax(w))]
    return float(math.degrees(math.atan2(pc[1], pc[0])))


# ------------------------------------------------------------------------------
# traj_len_px
# 役割: 軌跡長を pixel 単位で計算する。
# 入力: x, y。
# 出力: 総移動距離。
# 注意: NaN を除外して隣接差分を足す。
# ------------------------------------------------------------------------------
def traj_len_px(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~np.isnan(x) & ~np.isnan(y)
    x = x[m]; y = y[m]
    if len(x) < 2:
        return np.nan
    dx = np.diff(x); dy = np.diff(y)
    return float(np.nansum(np.hypot(dx, dy)))


# ------------------------------------------------------------------------------
# max_speed_px_s
# 役割: 最大速度を計算する。
# 入力: x, y, fps。
# 出力: px/s。
# 注意: 周期ごとの速度指標。
# ------------------------------------------------------------------------------
def max_speed_px_s(x, y, fps):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~np.isnan(x) & ~np.isnan(y)
    x = x[m]; y = y[m]
    if len(x) < 2:
        return np.nan
    dx = np.diff(x); dy = np.diff(y)
    v = np.hypot(dx, dy) * float(fps)
    return float(np.nanmax(v)) if len(v) else np.nan


# ------------------------------------------------------------------------------
# rolling_mean
# 役割: 中心化移動平均を返す。
# 入力: x, win。
# 出力: 平滑化配列。
# 注意: ピーク検出と QC 用。
# ------------------------------------------------------------------------------
def rolling_mean(x, win):
    s = pd.Series(x, dtype="float64")
    return s.rolling(int(win), center=True, min_periods=1).mean().to_numpy()


# ------------------------------------------------------------------------------
# speed_series_px_s
# 役割: 位置系列からフレーム単位速度を作る。
# 入力: x, y, fps。
# 出力: 速度系列。
# 注意: onset・trim に使用。
# ------------------------------------------------------------------------------
def speed_series_px_s(x, y, fps):
    """Per-frame speed (px/s) aligned to frame indices. speed[0] is NaN."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    v = np.full(n, np.nan, dtype=float)
    if n < 2:
        return v
    for i in range(1, n):
        if np.isfinite(x[i]) and np.isfinite(y[i]) and np.isfinite(x[i-1]) and np.isfinite(y[i-1]):
            v[i] = math.hypot(x[i] - x[i-1], y[i] - y[i-1]) * float(fps)
    return v


# ------------------------------------------------------------------------------
# detect_onset_frame
# 役割: cue 後の動作開始フレームを robust 閾値で検出する。
# 入力: 速度系列など。
# 出力: (onset_frame, threshold)。
# 注意: baseline の median + k*MAD を基準とする。
# ------------------------------------------------------------------------------
def detect_onset_frame(speed_px_s_arr, cue_frame, baseline_frames, k_mad=3.0, hold_frames=5):
    """
    Detect movement onset after cue_frame using baseline median + k*MAD threshold.

    Baseline: prefer immediately BEFORE cue_frame if possible, else AFTER cue_frame.
    """
    speed = np.asarray(speed_px_s_arr, dtype=float)
    n = len(speed)
    cue_frame = int(cue_frame)
    if n == 0 or cue_frame >= n:
        return None, np.nan

    bf = max(1, int(baseline_frames))

    if cue_frame >= bf:
        b0 = cue_frame - bf
        b1 = cue_frame
    else:
        b0 = cue_frame
        b1 = min(n, cue_frame + bf)

    baseline = speed[b0:b1]
    baseline = baseline[np.isfinite(baseline)]
    if len(baseline) == 0:
        baseline = speed[:min(n, bf)]
        baseline = baseline[np.isfinite(baseline)]
    if len(baseline) == 0:
        return None, np.nan

    med = float(np.nanmedian(baseline))
    mad = float(np.nanmedian(np.abs(baseline - med)))
    thr = med + float(k_mad) * mad

    hold = max(1, int(hold_frames))
    for i in range(cue_frame, n - hold + 1):
        w = speed[i:i+hold]
        if np.all(np.isfinite(w)) and np.all(w > thr):
            return int(i), float(thr)

    return None, float(thr)

# =========================
# outlier handling helpers (A: add columns to frames.csv; no extra files)
# =========================
# ------------------------------------------------------------------------------
# _as_str_array
# 役割: 配列を文字列配列化する。
# 入力: 配列。
# 出力: 文字列 ndarray。
# 注意: reason 列などの整形用。
# ------------------------------------------------------------------------------
def _as_str_array(a):
    if a is None:
        return np.array([], dtype=object)
    s = pd.Series(a).astype(str).to_numpy(dtype=object)
    return s

# ------------------------------------------------------------------------------
# detect_index_outliers
# 役割: 代表 index 軌跡の外れ値を検出する。
# 入力: raw index と hand/pose 参照情報。
# 出力: flag, reason, 補助数値。
# 注意: jump と hand/pose 不一致の2系統を扱う。
# ------------------------------------------------------------------------------
def detect_index_outliers(
    ix_raw: np.ndarray,
    iy_raw: np.ndarray,
    index_source: np.ndarray,
    pose_ix_x: np.ndarray,
    pose_ix_y: np.ndarray,
    hand_ix_x: np.ndarray,
    hand_ix_y: np.ndarray,
    jump_px: float = 200.0,
    hand_pose_dist_px: float = 150.0,
):
    """Detect outliers in representative index (x,y) series.

    Outlier rules:
      1) jump: per-frame step distance hypot(dx,dy) > jump_px
      2) hand_pose: if index_source=='hand' and both pose/hand index exist,
                    hypot(hand_index - pose_index) > hand_pose_dist_px

    Returns:
      outlier_flag (int 0/1),
      outlier_reason (str),
      step_px (float),
      hand_pose_dist_px_arr (float)
    """
    ix = np.asarray(ix_raw, dtype=float)
    iy = np.asarray(iy_raw, dtype=float)
    n = int(len(ix))
    src = _as_str_array(index_source)
    if src.size != n:
        src = np.array(["none"] * n, dtype=object)

    step = np.full(n, np.nan, dtype=float)
    if n >= 2:
        dx = np.diff(ix)
        dy = np.diff(iy)
        step[1:] = np.hypot(dx, dy)

    jump = np.isfinite(step) & (step > float(jump_px))

    # hand vs pose distance (only meaningful when both exist)
    pose_x = np.asarray(pose_ix_x, dtype=float)
    pose_y = np.asarray(pose_ix_y, dtype=float)
    hand_x = np.asarray(hand_ix_x, dtype=float)
    hand_y = np.asarray(hand_ix_y, dtype=float)

    hp_dist = np.hypot(hand_x - pose_x, hand_y - pose_y)
    hand_pose = (src == "hand") & np.isfinite(hp_dist) & (hp_dist > float(hand_pose_dist_px))

    flag = (jump | hand_pose).astype(int)

    reason = np.array([""] * n, dtype=object)
    reason[jump] = "jump"
    # append if both apply
    both = jump & hand_pose
    reason[hand_pose & ~jump] = "hand_pose"
    reason[both] = "jump|hand_pose"

    return flag, reason, step, hp_dist

# ------------------------------------------------------------------------------
# _interpolate_with_gap_limit
# 役割: 短い欠損だけ線形補間する。
# 入力: arr, max_gap_frames。
# 出力: 補間後配列。
# 注意: 過補間を防ぐ。
# ------------------------------------------------------------------------------
def _interpolate_with_gap_limit(arr: np.ndarray, max_gap_frames: int) -> np.ndarray:
    """Linear interpolation with a maximum consecutive-gap limit."""
    s = pd.Series(arr, dtype="float64")
    lim = int(max(1, max_gap_frames))
    # interpolate only small gaps
    s = s.interpolate(method="linear", limit=lim, limit_direction="both")
    # fill edges only for small gaps
    s = s.ffill(limit=lim).bfill(limit=lim)
    return s.to_numpy(dtype=float)

# ------------------------------------------------------------------------------
# apply_outlier_cleaning_2d
# 役割: 2次元座標に対する outlier 除去＋短欠損補間。
# 入力: raw x/y, flag, max_gap_frames。
# 出力: clean x/y。
# 注意: raw を残しつつ解析用 clean を作る。
# ------------------------------------------------------------------------------
def apply_outlier_cleaning_2d(ix_raw: np.ndarray, iy_raw: np.ndarray, outlier_flag: np.ndarray, max_gap_frames: int):
    """Replace outlier frames with NaN, then interpolate (limited gap) for x and y."""
    ix = np.asarray(ix_raw, dtype=float).copy()
    iy = np.asarray(iy_raw, dtype=float).copy()
    m = np.asarray(outlier_flag, dtype=int) == 1
    ix[m] = np.nan
    iy[m] = np.nan
    ix_c = _interpolate_with_gap_limit(ix, int(max_gap_frames))
    iy_c = _interpolate_with_gap_limit(iy, int(max_gap_frames))
    return ix_c, iy_c



# =========================
# movement segment + cycle selection helpers
# =========================
# ------------------------------------------------------------------------------
# detect_movement_segment
# 役割: 手関節速度から主要動作区間を推定する。
# 入力: 速度系列、cue、各種閾値。
# 出力: (start_frame, end_frame, thr, method)。
# 注意: cycle search window を trim に合わせる場合に使う。
# ------------------------------------------------------------------------------
def detect_movement_segment(speed_px_s_arr, cue_frame, baseline_frames,
                           k_mad=3.0, hold_frames=5,
                           quiet_frames=15, min_movement_frames=15):
    """
    Detect movement segment [start_frame, end_frame] using per-frame speed (px/s).

    - start_frame: movement onset after cue_frame (baseline median + k*MAD, sustained hold_frames)
    - end_frame: last frame before a sustained quiet period (<= threshold) of quiet_frames,
                 falling back to the last frame above threshold if no quiet period is found.

    Returns (start_frame, end_frame, thr, method)
      method: 'quiet' (end found by quiet period) or 'last_above_thr'
    """
    speed = np.asarray(speed_px_s_arr, dtype=float)
    n = len(speed)
    cue_frame = int(cue_frame)
    if n == 0 or cue_frame >= n:
        return None, None, np.nan, "no_data"

    # threshold (same baseline logic as detect_onset_frame)
    bf = max(1, int(baseline_frames))
    if cue_frame >= bf:
        b0, b1 = cue_frame - bf, cue_frame
    else:
        b0, b1 = cue_frame, min(n, cue_frame + bf)

    baseline = speed[b0:b1]
    baseline = baseline[np.isfinite(baseline)]
    if len(baseline) == 0:
        baseline = speed[:min(n, bf)]
        baseline = baseline[np.isfinite(baseline)]
    if len(baseline) == 0:
        return None, None, np.nan, "no_baseline"

    med = float(np.nanmedian(baseline))
    mad = float(np.nanmedian(np.abs(baseline - med)))
    thr = med + float(k_mad) * mad

    start_frame, _ = detect_onset_frame(
        speed_px_s_arr=speed,
        cue_frame=cue_frame,
        baseline_frames=baseline_frames,
        k_mad=float(k_mad),
        hold_frames=int(hold_frames),
    )
    if start_frame is None:
        return None, None, float(thr), "no_onset"

    start_frame = int(start_frame)
    quiet = max(1, int(quiet_frames))
    min_mv = max(0, int(min_movement_frames))

    # Find the first sustained quiet window after at least min_movement_frames.
    end_by_quiet = None
    search0 = min(n - quiet, start_frame + min_mv)
    for i in range(search0, n - quiet + 1):
        w = speed[i:i+quiet]
        # require finite values for stability
        if np.all(np.isfinite(w)) and np.all(w <= thr):
            end_by_quiet = int(i - 1)
            break

    if end_by_quiet is not None and end_by_quiet >= start_frame:
        return start_frame, end_by_quiet, float(thr), "quiet"

    # fallback: last frame above threshold
    last = None
    for i in range(n - 1, start_frame - 1, -1):
        if np.isfinite(speed[i]) and speed[i] > thr:
            last = int(i)
            break
    if last is None:
        last = n - 1
    return start_frame, last, float(thr), "last_above_thr"


# ------------------------------------------------------------------------------
# select_best_contiguous_cycles_by_cv
# 役割: 周期 CV 最小の連続 target_n サイクル窓を選ぶ。
# 入力: cycles, fps, target_n。
# 出力: selected_cycles, selected_cv, start_idx。
# 注意: 安定した連続区間を機械的に選ぶ。
# ------------------------------------------------------------------------------
def select_best_contiguous_cycles_by_cv(cycles, fps, target_n=10):
    """
    Select a contiguous window of target_n cycles whose cycle_time_s CV is minimal.
    Returns (selected_cycles, best_cv, window_start_index).

    If len(cycles) < target_n, returns (cycles, NaN, 0).
    """
    cycles = list(cycles) if cycles is not None else []
    if len(cycles) < int(target_n) or int(target_n) <= 0:
        return cycles, np.nan, 0

    durs = np.array([(c["end_frame"] - c["start_frame"]) / float(fps) for c in cycles], dtype=float)
    best_cv = None
    best_i = 0

    for i in range(0, len(cycles) - int(target_n) + 1):
        w = durs[i:i+int(target_n)]
        if not np.all(np.isfinite(w)):
            continue
        m = float(np.mean(w))
        if m == 0:
            continue
        sd = float(np.std(w, ddof=1)) if len(w) >= 2 else 0.0
        cv = sd / m
        if (best_cv is None) or (cv < best_cv):
            best_cv = cv
            best_i = i

    if best_cv is None:
        # if all windows invalid, return first target_n
        best_i = 0
        best_cv = np.nan

    selected = cycles[best_i:best_i+int(target_n)]
    return selected, float(best_cv) if best_cv is not None else np.nan, int(best_i)

# =========================
# cycle detection (same core as byebye v2)
# =========================
# ------------------------------------------------------------------------------
# find_local_extrema_prom
# 役割: 平滑化信号から極値候補を抽出する。
# 入力: 信号と検出条件。
# 出力: maxima, minima。
# 注意: prominence と間隔条件で雑音極値を抑える。
# ------------------------------------------------------------------------------
def find_local_extrema_prom(x, min_sep_frames=6, min_amp_px=20.0,
                            min_prom_ratio=0.15, min_prom_px=0.0, prom_win_frames=15):
    """
    Return (maxima_idx, minima_idx) for a smoothed series x, with:
      - local extremum rule
      - minimum separation (frames)
      - global amplitude floor (min_amp_px)
      - prominence threshold:
          prom_px >= max(min_prom_px, min_prom_ratio * (global_range))
        where prom_px is estimated within a +/- prom_win_frames window.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    maxima, minima = [], []
    last_max = -10**9
    last_min = -10**9

    x_valid = x[~np.isnan(x)]
    if len(x_valid) == 0:
        return maxima, minima

    x_min = float(np.nanmin(x_valid))
    x_max = float(np.nanmax(x_valid))
    global_range = x_max - x_min
    prom_thr = max(float(min_prom_px), float(min_prom_ratio) * float(global_range))

    w = int(max(3, prom_win_frames))

    for i in range(1, n - 1):
        if np.isnan(x[i-1]) or np.isnan(x[i]) or np.isnan(x[i+1]):
            continue

        # local max
        if (x[i] >= x[i-1] and x[i] >= x[i+1]) and (x[i] > x[i-1] or x[i] > x[i+1]):
            if i - last_max >= int(min_sep_frames):
                if (x[i] - x_min) >= float(min_amp_px):
                    l0 = max(0, i - w)
                    r1 = min(n, i + w + 1)
                    left = x[l0:i+1]
                    right = x[i:r1]
                    left_min = float(np.nanmin(left)) if np.any(~np.isnan(left)) else np.nan
                    right_min = float(np.nanmin(right)) if np.any(~np.isnan(right)) else np.nan
                    base = max(left_min, right_min)
                    prom = float(x[i] - base) if np.isfinite(base) else np.nan
                    if np.isfinite(prom) and prom >= prom_thr:
                        maxima.append(i)
                        last_max = i

        # local min
        if (x[i] <= x[i-1] and x[i] <= x[i+1]) and (x[i] < x[i-1] or x[i] < x[i+1]):
            if i - last_min >= int(min_sep_frames):
                if (x_max - x[i]) >= float(min_amp_px):
                    l0 = max(0, i - w)
                    r1 = min(n, i + w + 1)
                    left = x[l0:i+1]
                    right = x[i:r1]
                    left_max = float(np.nanmax(left)) if np.any(~np.isnan(left)) else np.nan
                    right_max = float(np.nanmax(right)) if np.any(~np.isnan(right)) else np.nan
                    base = min(left_max, right_max)
                    prom = float(base - x[i]) if np.isfinite(base) else np.nan
                    if np.isfinite(prom) and prom >= prom_thr:
                        minima.append(i)
                        last_min = i

    return maxima, minima


# ------------------------------------------------------------------------------
# build_cycles_from_extrema
# 役割: 極値配列から1サイクル辞書を組み立てる。
# 入力: x_smooth, maxima, minima, fps など。
# 出力: cycle list。
# 注意: start-opp-end を規格化して下流集計へ渡す。
# ------------------------------------------------------------------------------
def build_cycles_from_extrema(x_smooth, maxima, minima,
                              start_search_frame, fps,
                              min_cycle_s=0.3, max_cycle_s=3.0):
    """
    Cycle definition:
      If first extremum after start_search_frame is MAX:  max -> min -> next max
      If first extremum after start_search_frame is MIN:  min -> max -> next min
    Returns list of dict: {cycle_id, start_frame, opp_frame, end_frame}
    """
    x_smooth = np.asarray(x_smooth, dtype=float)
    start_search = int(start_search_frame)

    ext = [(i, "max") for i in maxima] + [(i, "min") for i in minima]
    ext = [(i, t) for (i, t) in ext if i >= start_search]
    ext.sort(key=lambda z: z[0])
    if not ext:
        return []

    start_kind = ext[0][1]  # "max" or "min"
    cycles = []
    cid = 1

    if start_kind == "max":
        starts = [i for i in maxima if i >= start_search]
        opps = minima
        for k in range(len(starts) - 1):
            s = starts[k]
            e = starts[k + 1]
            mids = [j for j in opps if s < j < e]
            if not mids:
                continue
            mid = mids[int(np.nanargmin(x_smooth[mids]))]  # deepest minimum
            dur = (e - s) / float(fps)
            if dur < float(min_cycle_s) or dur > float(max_cycle_s):
                continue
            cycles.append({"cycle_id": cid, "start_frame": s, "opp_frame": mid, "end_frame": e})
            cid += 1
    else:
        starts = [i for i in minima if i >= start_search]
        opps = maxima
        for k in range(len(starts) - 1):
            s = starts[k]
            e = starts[k + 1]
            mids = [j for j in opps if s < j < e]
            if not mids:
                continue
            mid = mids[int(np.nanargmax(x_smooth[mids]))]  # highest maximum
            dur = (e - s) / float(fps)
            if dur < float(min_cycle_s) or dur > float(max_cycle_s):
                continue
            cycles.append({"cycle_id": cid, "start_frame": s, "opp_frame": mid, "end_frame": e})
            cid += 1

    return cycles




# =========================
# waveform + similarity helpers (aligned conceptually to hammer)
# =========================
# ------------------------------------------------------------------------------
# _odd
# 役割: 窓長を奇数へそろえる。
# 入力: n。
# 出力: 奇数窓長。
# 注意: 中心化 rolling 用。
# ------------------------------------------------------------------------------
def _odd(n: int) -> int:
    n = int(max(1, n))
    return n if (n % 2 == 1) else (n + 1)


# ------------------------------------------------------------------------------
# save_waveform_png
# 役割: 代表波形を PNG として保存する。
# 入力: frames_df, out_dir。
# 出力: png path。
# 注意: clean 系を優先描画。
# ------------------------------------------------------------------------------
def save_waveform_png(frames_df: pd.DataFrame, out_dir: Path) -> Path:
    """Save waveform png for byebye. Prefer cleaned cycle-signal / index-x so the exported waveform matches lateral byebye motion and reflects outlier processing."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    stem = out_dir.name

    x = frames_df["time_s"] if "time_s" in frames_df.columns else (
        frames_df["t_s"] if "t_s" in frames_df.columns else frames_df.get("frame_idx", frames_df.index)
    )

    # For byebye, the lateral component is the main cycle signal.
    # Prefer the cleaned/smoothed cycle-signal first, then cleaned index-x, then raw x.
    if "cycle_signal_smooth_px" in frames_df.columns:
        y = frames_df["cycle_signal_smooth_px"]
        ylab = "cycle_signal_smooth_px"
    elif "cycle_signal_px" in frames_df.columns:
        y = frames_df["cycle_signal_px"]
        ylab = "cycle_signal_px"
    elif "index_x_px_sm" in frames_df.columns:
        y = frames_df["index_x_px_sm"]
        ylab = "index_x_px_sm"
    elif "index_x_px_clean" in frames_df.columns:
        y = frames_df["index_x_px_clean"]
        ylab = "index_x_px_clean"
    elif "index_x_px_raw" in frames_df.columns:
        y = frames_df["index_x_px_raw"]
        ylab = "index_x_px_raw"
    elif "index_x_px_sm_raw" in frames_df.columns:
        y = frames_df["index_x_px_sm_raw"]
        ylab = "index_x_px_sm_raw"
    elif "wrist_x_px_sm" in frames_df.columns:
        y = frames_df["wrist_x_px_sm"]
        ylab = "wrist_x_px_sm"
    elif "wrist_x_px_raw" in frames_df.columns:
        y = frames_df["wrist_x_px_raw"]
        ylab = "wrist_x_px_raw"
    else:
        y = frames_df.iloc[:, 0]
        ylab = frames_df.columns[0]

    plt.figure()
    plt.plot(x, y)
    plt.xlabel("time_s" if "time_s" in frames_df.columns else "frame_idx")
    plt.ylabel(ylab)
    plt.title(stem)

    png = out_dir / f"waveform_{stem}.png"
    plt.savefig(png, dpi=200, bbox_inches="tight")
    plt.close()
    return png


# ------------------------------------------------------------------------------
# _resample_1d_nan
# 役割: 1周期波形を比較用に共通長へ再標本化する。
# 入力: arr, n。
# 出力: 長さ n の配列。
# 注意: 波形 QC 用。
# ------------------------------------------------------------------------------
def _resample_1d_nan(arr: np.ndarray, n: int) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    n = int(n)
    if n <= 1:
        return np.array([float(np.nanmean(a))]) if np.isfinite(a).any() else np.array([np.nan])
    if a.size < 2 or (np.isfinite(a).sum() < 2):
        return np.full(n, np.nan, dtype=float)
    x = np.arange(a.size, dtype=float)
    m = np.isfinite(a)
    xp = x[m]
    fp = a[m]
    x_new = np.linspace(0.0, float(a.size - 1), n)
    y_new = np.interp(x_new, xp, fp)
    return y_new.astype(float)


# ------------------------------------------------------------------------------
# _cycle_waveforms_from_y
# 役割: 各サイクルの y 波形を抽出して共通長にそろえる。
# 入力: y_sm, cycles_df, resample_n。
# 出力: 波形行列。
# 注意: selected10 の QC に使う。
# ------------------------------------------------------------------------------
def _cycle_waveforms_from_y(y_sm: np.ndarray, cycles_df: pd.DataFrame, resample_n: int) -> np.ndarray:
    """Return (n_cycles, resample_n) matrix of per-cycle waveforms from y_sm."""
    y_sm = np.asarray(y_sm, dtype=float)
    resample_n = int(resample_n)
    if cycles_df is None or len(cycles_df) == 0:
        return np.empty((0, resample_n), dtype=float)

    mats = []
    for _, r in cycles_df.iterrows():
        s = r.get("start_frame", np.nan)
        e = r.get("end_frame", np.nan)
        if not (np.isfinite(s) and np.isfinite(e)):
            mats.append(np.full(resample_n, np.nan, dtype=float))
            continue
        s_i = int(max(0, int(s)))
        e_i = int(min(len(y_sm) - 1, int(e)))
        if e_i <= s_i:
            mats.append(np.full(resample_n, np.nan, dtype=float))
            continue
        seg = y_sm[s_i : e_i + 1]
        mats.append(_resample_1d_nan(seg, resample_n))

    return np.vstack(mats).astype(float) if len(mats) else np.empty((0, resample_n), dtype=float)


# ------------------------------------------------------------------------------
# _corr_to_mean_wave
# 役割: 平均波形に対する各波形の相関を返す。
# 入力: waves。
# 出力: 相関配列。
# 注意: 波形の一貫性評価。
# ------------------------------------------------------------------------------
def _corr_to_mean_wave(waves: np.ndarray) -> np.ndarray:
    """Correlation of each waveform to the mean waveform (NaN-safe, requires enough valid points)."""
    w = np.asarray(waves, dtype=float)
    if w.ndim != 2 or w.size == 0:
        return np.array([], dtype=float)

    mean_w = np.nanmean(w, axis=0)
    corrs = []
    min_pts = max(10, int(0.20 * w.shape[1]))
    for i in range(w.shape[0]):
        a = w[i, :]
        m = np.isfinite(a) & np.isfinite(mean_w)
        if int(m.sum()) < min_pts:
            corrs.append(np.nan)
            continue
        aa = a[m]
        bb = mean_w[m]
        if np.nanstd(aa) == 0 or np.nanstd(bb) == 0:
            corrs.append(np.nan)
            continue
        c = float(np.corrcoef(aa, bb)[0, 1])
        corrs.append(c)
    return np.asarray(corrs, dtype=float)

# ------------------------------------------------------------------------------
# _block_wave_stats
# 役割: ブロック内波形の平均/最小相関を返す。
# 入力: waves_block。
# 出力: (mean_corr, min_corr)。
# 注意: 候補ブロック比較用。
# ------------------------------------------------------------------------------
def _block_wave_stats(waves_block: np.ndarray) -> tuple[float, float]:
    corrs = _corr_to_mean_wave(waves_block)
    if np.isfinite(corrs).any():
        return float(np.nanmean(corrs)), float(np.nanmin(corrs))
    return float("nan"), float("nan")


# ------------------------------------------------------------------------------
# _best_contiguous_block_by_waveform_then_cv
# 役割: 波形一貫性優先、CV 次点で連続ブロックを選ぶ。
# 入力: cycle_times, waveforms, target_n。
# 出力: best index と QC 指標。
# 注意: selected10 選択ロジック。
# ------------------------------------------------------------------------------
def _best_contiguous_block_by_waveform_then_cv(
    cycle_times: np.ndarray,
    waveforms: np.ndarray,
    target: int,
) -> tuple[int, float, float, float]:
    """Hammerと同様: mean_corr最大を優先し、同等ならCV最小。waveformsが無効ならCV最小。"""
    t = np.asarray(cycle_times, dtype=float)
    n = int(len(t))
    target = int(target)
    if target <= 0 or n < target:
        return 0, float("inf"), float("nan"), float("nan")

    best_i = 0
    best_cv = float("inf")
    best_mean_corr = float("nan")
    best_min_corr = float("nan")

    for i in range(0, n - target + 1):
        seg_t = t[i : i + target]
        seg_t_f = seg_t[np.isfinite(seg_t)]
        mean = float(np.nanmean(seg_t_f)) if len(seg_t_f) > 0 else float("nan")
        sd = float(np.nanstd(seg_t_f, ddof=1)) if len(seg_t_f) >= 2 else float("nan")
        cv = float(sd / mean) if (np.isfinite(sd) and np.isfinite(mean) and mean != 0.0) else float("inf")

        if waveforms is not None and np.asarray(waveforms).ndim == 2 and int(waveforms.shape[0]) >= (i + target):
            seg_w = np.asarray(waveforms[i : i + target, :], dtype=float)
            mean_corr, min_corr = _block_wave_stats(seg_w)
        else:
            mean_corr, min_corr = float("nan"), float("nan")

        if np.isfinite(mean_corr):
            if (not np.isfinite(best_mean_corr)) or (mean_corr > best_mean_corr + 1e-6) or (
                abs(mean_corr - best_mean_corr) <= 1e-6 and cv < best_cv
            ):
                best_i = i
                best_cv = cv
                best_mean_corr = mean_corr
                best_min_corr = min_corr
        else:
            if (not np.isfinite(best_mean_corr)) and (cv < best_cv):
                best_i = i
                best_cv = cv
                best_mean_corr = mean_corr
                best_min_corr = min_corr

    return int(best_i), float(best_cv), float(best_mean_corr), float(best_min_corr)



# =========================
# video extraction helpers
# =========================
# ------------------------------------------------------------------------------
# _pose_side_indices
# 役割: 左右指定から BlazePose の対応 index を返す。
# 入力: side。
# 出力: ランドマーク index の辞書。
# 注意: 抽出関数で左右差を一元化する。
# ------------------------------------------------------------------------------
def _pose_side_indices(side: str) -> dict[str, int]:
    """Return BlazePose landmark indices for the requested body side."""
    if side == "Left":
        return {"HIP": 23, "SHOULDER": 11, "ELBOW": 13, "WRIST": 15, "INDEX": 19}
    return {"HIP": 24, "SHOULDER": 12, "ELBOW": 14, "WRIST": 16, "INDEX": 20}


# ------------------------------------------------------------------------------
# _pick_pose_px
# 役割: Pose ランドマーク1点を pixel 座標へ変換する。
# 入力: lms, idx, width, height。
# 出力: (x_px, y_px, score)。
# 注意: presence も合わせて返し、後段の有効判定に使えるようにする。
# ------------------------------------------------------------------------------
def _pick_pose_px(lms: Any, idx: int, width: int, height: int) -> tuple[float, float, float]:
    lm = lms[idx]
    return float(lm.x * width), float(lm.y * height), float(getattr(lm, "presence", 1.0))


# ------------------------------------------------------------------------------
# _pick_hand_px
# 役割: Hand ランドマーク1点を pixel 座標へ変換する。
# 入力: hand_lms, idx, width, height。
# 出力: (x_px, y_px)。
# 注意: hand 側は座標のみで十分なため score は返さない。
# ------------------------------------------------------------------------------
def _pick_hand_px(hand_lms: Any, idx: int, width: int, height: int) -> tuple[float, float]:
    lm = hand_lms[idx]
    return float(lm.x * width), float(lm.y * height)


# ------------------------------------------------------------------------------
# extract_byebye_px_from_video
# 役割: byebye 用に動画から Pose/Hand 座標を抽出し、代表 index を構成した raw_df を返す。
# 入力: 動画、モデル、左右など。
# 出力: (raw_df, fps, video_file)。
# 注意: main から抽出処理を切り出しているのが comehere との大きな違い。
# ------------------------------------------------------------------------------
def extract_byebye_px_from_video(
    video_path: Path,
    pose_model_path: Path,
    hand_model_path: Path,
    side: str,
) -> tuple[pd.DataFrame, int, int, float, float]:
    """Read one video and return per-frame landmark rows plus basic video metadata."""
    # 動画を開き、解像度と fps を取得する。
    # fps が取得できない場合は 30fps を代替値にする。
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
    # Pose / Hand の landmarker を VIDEO モードで作成する。
    # byebye では hand と pose の両方を使って代表点を構成する。
    pose_opt = vision.PoseLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=str(pose_model_path)),
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False,
    )
    hand_opt = vision.HandLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=str(hand_model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )

    mp_image_format = mp.ImageFormat.SRGB
    # フレームごとの抽出結果を rows に蓄積する。
    rows: list[dict[str, Any]] = []
    frame = 0

    try:
        with vision.PoseLandmarker.create_from_options(pose_opt) as pose_lm,              vision.HandLandmarker.create_from_options(hand_opt) as hand_lm:

            # フレーム走査。
            # 各フレームの Pose / Hand 座標を pixel 化し、必要列だけ辞書へ保存する。
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp_image_format, data=rgb)
                ts_ms = int(frame * 1000.0 / fps)

                pose_res = pose_lm.detect_for_video(mp_img, ts_ms)
                hand_res = hand_lm.detect_for_video(mp_img, ts_ms)

                # Pose 側の主要上肢点を取得する。
                # 取得不能な点は NaN のまま残し、後段で clean 化する。
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

                # 複数 hand 候補がある場合は、Pose wrist に最も近い手を採用する。
                # 同じ画角内の不要 hand を拾いにくくするための手順。
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
                    hand_ix_x, hand_ix_y = _pick_hand_px(chosen, 8, width, height)

                # 代表 index は hand INDEX_TIP を優先し、
                # hand が欠損したフレームだけ Pose INDEX にフォールバックする。
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

    raw_df = pd.DataFrame(rows)
    return raw_df, width, height, src_fps, fps


# =========================
# main
# =========================
# ------------------------------------------------------------------------------
# main
# 役割: byebye 解析の入口。抽出結果を前処理し、周期検出・集計・保存まで行う。
# 入力: CLI 引数。
# 出力: そのまま処理終了。
# 注意: 抽出は extract_byebye_px_from_video に委譲し、その後の流れは comehere と近い。
# ------------------------------------------------------------------------------
def main():
    # main 全体の流れ:
    # 1) CLI 引数を定義・読込
    # 2) extract_byebye_px_from_video で raw_df を作成
    # 3) raw index を clean 系列へ整形し、速度・角度・周期信号を作成
    # 4) onset / trim / cycle detection を実施
    # 5) frames.csv, cycles.csv, summary.csv, waveform png を保存
    # 解析条件を CLI から再現できるよう、引数をここで一括定義する。
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--pose_model", required=True)
    ap.add_argument("--hand_model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--participant_id", required=True)
    ap.add_argument("--condition", default="byebye")
    ap.add_argument("--set_id", type=int, required=True)
    ap.add_argument("--trial_id", type=int, required=True)
    ap.add_argument("--cue_frame", type=int, default=0)
    ap.add_argument("--side", choices=["Left", "Right"], default="Left")

    # cycle params (aligned to byebye v2)
    ap.add_argument("--cycle_signal", choices=["dist", "dx", "dy"], default="dx",
                    help="Cycle detection signal. dist: hypot(index-shoulder), dx: index_x-shoulder_x, dy: index_y-shoulder_y (byebye default: dx)")
    ap.add_argument("--smooth_sec", type=float, default=0.10)
    ap.add_argument("--start_search_sec", type=float, default=0.20)
    ap.add_argument("--cycle_search_use_trim", action="store_true", default=True,
                    help="Restrict cycle search to detected movement segment (trim_start_frame..trim_end_frame). "
                         "Default: enabled.")

    ap.add_argument("--min_sep_s", type=float, default=0.10)
    ap.add_argument("--min_amp_px", type=float, default=80.0)
    ap.add_argument("--min_prom_ratio", type=float, default=0.10)
    ap.add_argument("--min_prom_px", type=float, default=0.0)
    ap.add_argument("--prom_win_s", type=float, default=0.35)
    ap.add_argument("--min_cycle_s", type=float, default=0.30)
    ap.add_argument("--max_cycle_s", type=float, default=3.00)

    # onset params (aligned to byebye v2)
    ap.add_argument("--onset_baseline_s", type=float, default=0.5)
    ap.add_argument("--onset_k_mad", type=float, default=3.0)
    ap.add_argument("--onset_hold_frames", type=int, default=5)
    ap.add_argument("--onset_speed_smooth_win", type=int, default=5)


    # outlier handling (A: no extra output files; adds columns to frames.csv)
    ap.add_argument("--outlier_disable", action="store_true",
                    help="Disable outlier handling (default: enabled).")
    ap.add_argument("--outlier_jump_px", type=float, default=200.0,
                    help="Outlier rule 1: per-frame step distance threshold in px (default: 200).")
    ap.add_argument("--outlier_hand_pose_dist_px", type=float, default=150.0,
                    help="Outlier rule 2: when index_source is hand, if hand-vs-pose index distance exceeds this (px), treat as outlier (default: 150).")
    ap.add_argument("--outlier_max_gap_s", type=float, default=0.30,
                    help="Maximum gap (seconds) to interpolate when filling outliers (default: 0.30s).")



    # movement segment trimming + cycle selection
    ap.add_argument("--trim_k_mad", type=float, default=3.0,
                    help="Wrist-speed threshold factor for trimming segment (median + k*MAD).")
    ap.add_argument("--trim_hold_frames", type=int, default=5,
                    help="Consecutive frames above threshold to accept movement onset (trimming).")
    ap.add_argument("--trim_quiet_s", type=float, default=0.60,
                    help="Seconds of sustained quiet (<= threshold) to mark movement end (trimming).")
    ap.add_argument("--trim_min_movement_s", type=float, default=0.50,
                    help="Minimum seconds after onset before looking for the quiet window.")
    ap.add_argument("--target_cycles", type=int, default=10,
                    help="Number of consecutive cycles to select as final analysis target.")
    ap.add_argument("--save_all_cycles", action="store_true",
                    help="Option kept for compatibility; no additional CSV is written.")


    # waveform similarity check (aligned conceptually to hammer)
    ap.add_argument("--waveform_resample_n", type=int, default=100,
                help="Waveform similarity check: resample each cycle waveform to this length (default: 100).")
    ap.add_argument("--waveform_min_corr", type=float, default=0.75,
                help="Waveform similarity check: minimum correlation to mean waveform for the selected block (default: 0.75).")


    # 以降は指定された条件で raw_df 作成 → 解析へ進む。
    args = ap.parse_args()

    video_path = Path(args.video).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # まず動画から raw_df を作る。
    # 抽出処理は専用関数に分離されているため、main 側では前処理以降に集中できる。
    raw_df, width, height, src_fps, fps = extract_byebye_px_from_video(
        video_path=video_path,
        pose_model_path=Path(args.pose_model).expanduser(),
        hand_model_path=Path(args.hand_model).expanduser(),
        side=args.side,
    )

    cue = int(args.cue_frame)


    # 肩を基準点として代表 index の相対位置系列を作る準備をする。
    base_x = raw_df["shoulder_x_px"].to_numpy(dtype=float)
    base_y = raw_df["shoulder_y_px"].to_numpy(dtype=float)

    # Representative index (raw): Hand INDEX_TIP preferred, else Pose INDEX
    ix_raw = raw_df["index_x_px"].to_numpy(dtype=float)
    iy_raw = raw_df["index_y_px"].to_numpy(dtype=float)

    # For outlier checks
    index_source_arr = raw_df["index_source"].to_numpy()
    pose_ix_x = raw_df["pose_index_x_px"].to_numpy(dtype=float)
    pose_ix_y = raw_df["pose_index_y_px"].to_numpy(dtype=float)
    hand_ix_x = raw_df["hand_index_tip_x_px"].to_numpy(dtype=float)
    hand_ix_y = raw_df["hand_index_tip_y_px"].to_numpy(dtype=float)

    # Outlier handling (default: enabled). This does NOT delete raw; it creates clean series for analysis.
    # outlier handling。
    # raw を消さず、clean 系列のみ別に作ることで QC 可能性を残す。
    if bool(args.outlier_disable):
        outlier_flag = np.zeros(len(raw_df), dtype=int)
        outlier_reason = np.array([""] * len(raw_df), dtype=object)
        outlier_step_px = np.full(len(raw_df), np.nan, dtype=float)
        outlier_hand_pose_dist_px = np.full(len(raw_df), np.nan, dtype=float)
        ix_clean = ix_raw.copy()
        iy_clean = iy_raw.copy()
        max_gap_frames = int(round(float(args.outlier_max_gap_s) * float(fps)))
    else:
        outlier_flag, outlier_reason, outlier_step_px, outlier_hand_pose_dist_px = detect_index_outliers(
            ix_raw=ix_raw,
            iy_raw=iy_raw,
            index_source=index_source_arr,
            pose_ix_x=pose_ix_x,
            pose_ix_y=pose_ix_y,
            hand_ix_x=hand_ix_x,
            hand_ix_y=hand_ix_y,
            jump_px=float(args.outlier_jump_px),
            hand_pose_dist_px=float(args.outlier_hand_pose_dist_px),
        )
        max_gap_frames = int(round(float(args.outlier_max_gap_s) * float(fps)))
        max_gap_frames = max(1, max_gap_frames)
        ix_clean, iy_clean = apply_outlier_cleaning_2d(ix_raw, iy_raw, outlier_flag, max_gap_frames)

    # raw / clean 双方で相対位置・距離を計算する。
    # raw は確認用、clean は解析用。
    # Raw & clean kinematics
    dx_raw = ix_raw - base_x
    dy_raw = iy_raw - base_y
    dist_raw = np.hypot(dx_raw, dy_raw)
    dist_smooth_raw = rolling_mean(dist_raw, 5)

    dx_clean = ix_clean - base_x
    dy_clean = iy_clean - base_y
    dist_clean = np.hypot(dx_clean, dy_clean)
    dist_smooth_clean = rolling_mean(dist_clean, 5)


    # 周期検出に使う代表信号を選ぶ。
    # 実際の極値探索は clean 系列で行い、raw は比較参照用として残す。
    # cycle signal (raw and clean). Cycle detection uses the *clean* signal.
    if args.cycle_signal == "dx":
        sig_raw = dx_raw
        sig_used = dx_clean
    elif args.cycle_signal == "dy":
        sig_raw = dy_raw
        sig_used = dy_clean
    else:
        sig_raw = dist_raw
        sig_used = dist_clean

    win_sig = int(max(3, round(float(args.smooth_sec) * float(fps))))
    if win_sig % 2 == 0:
        win_sig += 1

    sig_smooth_raw = rolling_mean(sig_raw, win_sig)
    sig_smooth = rolling_mean(sig_used, win_sig)


    
    # index tip 速度を計算する。
    # raw は spikes 確認用、clean は解析/QC 用。
    # speed (index tip)
    # - raw: may contain spikes (useful to confirm outlier existence)
    # - clean: used for analysis/QC
    speed_index_raw = speed_series_px_s(ix_raw, iy_raw, fps)
    speed_index_clean_raw = speed_series_px_s(ix_clean, iy_clean, fps)
    speed_index_smooth_raw = rolling_mean(speed_index_raw, int(max(1, args.onset_speed_smooth_win)))
    speed_index_smooth = rolling_mean(speed_index_clean_raw, int(max(1, args.onset_speed_smooth_win)))

    # onset と trim は wrist 速度ベースで行う。
    # hand wrist が取れたフレームはそちらを優先し、無ければ pose wrist を使う。
    # speed (wrist) - used for onset + movement trimming (hand wrist preferred)
    pose_wrx = raw_df["wrist_x_px"].to_numpy(dtype=float)
    pose_wry = raw_df["wrist_y_px"].to_numpy(dtype=float)
    hand_wrx = raw_df["hand_wrist_x_px"].to_numpy(dtype=float)
    hand_wry = raw_df["hand_wrist_y_px"].to_numpy(dtype=float)
    wrist_x = np.where(np.isfinite(hand_wrx), hand_wrx, pose_wrx)
    wrist_y = np.where(np.isfinite(hand_wry), hand_wry, pose_wry)

    speed_wrist_raw = speed_series_px_s(wrist_x, wrist_y, fps)
    speed_wrist_smooth = rolling_mean(speed_wrist_raw, int(max(1, args.onset_speed_smooth_win)))

    baseline_frames = int(round(float(args.onset_baseline_s) * float(fps)))

    # cue 後の動作開始フレームを検出する。
    # start_to_onset_s は summary に記録される時間指標。
    # onset after cue based on wrist speed
    onset_frame, onset_thr = detect_onset_frame(
    speed_wrist_smooth,
    cue_frame=cue,
    baseline_frames=baseline_frames,
    k_mad=float(args.onset_k_mad),
    hold_frames=int(args.onset_hold_frames),
    )
    start_to_onset_s = float((onset_frame - cue) / float(fps)) if onset_frame is not None else np.nan

    # 手関節速度から主要動作区間を推定する。
    # cycle_search_use_trim を有効にした場合、この区間で周期探索を絞り込む。
    # movement segment trimming (start/end) based on wrist speed
    quiet_frames = int(round(float(args.trim_quiet_s) * float(fps)))
    min_movement_frames = int(round(float(args.trim_min_movement_s) * float(fps)))
    trim_start_frame, trim_end_frame, trim_thr, trim_method = detect_movement_segment(
    speed_wrist_smooth,
    cue_frame=cue,
    baseline_frames=baseline_frames,
    k_mad=float(args.trim_k_mad),
    hold_frames=int(args.trim_hold_frames),
    quiet_frames=quiet_frames,
    min_movement_frames=min_movement_frames,
    )

    # 各フレームの関節角度を計算する。
    # cycles.csv では各サイクルの range/mean として要約される。
    # angles per frame
    shoulder_deg = np.array([angle_deg(raw_df.at[i, "hip_x_px"], raw_df.at[i, "hip_y_px"],
                                       raw_df.at[i, "shoulder_x_px"], raw_df.at[i, "shoulder_y_px"],
                                       raw_df.at[i, "elbow_x_px"], raw_df.at[i, "elbow_y_px"])
                             for i in range(len(raw_df))], dtype=float)
    elbow_deg = np.array([angle_deg(raw_df.at[i, "shoulder_x_px"], raw_df.at[i, "shoulder_y_px"],
                                    raw_df.at[i, "elbow_x_px"], raw_df.at[i, "elbow_y_px"],
                                    raw_df.at[i, "wrist_x_px"], raw_df.at[i, "wrist_y_px"])
                          for i in range(len(raw_df))], dtype=float)
    wrist_deg = np.array([angle_deg(raw_df.at[i, "elbow_x_px"], raw_df.at[i, "elbow_y_px"],
                                    raw_df.at[i, "wrist_x_px"], raw_df.at[i, "wrist_y_px"],
                                    raw_df.at[i, "index_x_px"], raw_df.at[i, "index_y_px"])
                          for i in range(len(raw_df))], dtype=float)
    index_mcp_deg = np.array([angle_deg(raw_df.at[i, "hand_wrist_x_px"], raw_df.at[i, "hand_wrist_y_px"],
                                        raw_df.at[i, "index_mcp_x_px"], raw_df.at[i, "index_mcp_y_px"],
                                        raw_df.at[i, "index_pip_x_px"], raw_df.at[i, "index_pip_y_px"])
                              for i in range(len(raw_df))], dtype=float)

    # 周期検出の準備。
    # 平滑化した代表信号から極値を取り、周期候補を組み立てる。
    # cycle detection
    min_sep_frames = int(round(float(args.min_sep_s) * float(fps)))
    prom_win_frames = int(max(3, round(float(args.prom_win_s) * float(fps))))
    sig_smooth_for_peaks = _interpolate_with_gap_limit(sig_smooth, max_gap_frames=max_gap_frames)

    start_search_frame0 = int(cue + float(args.start_search_sec) * float(fps))
    search_end0 = int(len(sig_smooth_for_peaks) - 1)

    # 実際に周期探索するフレーム範囲を決める。
    # byebye では設定により trim 区間優先で search window を狭められる。
    # Cycle search window: defaultはtrim segmentを優先
    search_start = int(start_search_frame0)
    search_end = int(search_end0)
    cycle_search_used_trim = 0
    if bool(getattr(args, 'cycle_search_use_trim', False)):
        if trim_start_frame is not None and np.isfinite(trim_start_frame):
            search_start = max(search_start, int(trim_start_frame))
        if trim_end_frame is not None and np.isfinite(trim_end_frame):
            search_end = min(search_end, int(trim_end_frame))
        cycle_search_used_trim = 1

    # 段階的閾値緩和の考え方。
    # まず標準条件で探し、足りなければ prominence / amplitude を少しずつ緩める。
    # Teacher-advice alignment:
    # 1) detect movement segment by wrist speed and restrict search to that segment
    # 2) relax thresholds stepwise to avoid missing small cycles
    # 3) from all detected cycles, select contiguous target_n cycles with minimum CV of cycle time
    detect_attempt_prom_ratio = [
        float(args.min_prom_ratio),
        max(0.01, float(args.min_prom_ratio) * 0.85),
        max(0.01, float(args.min_prom_ratio) * 0.70),
    ]
    detect_attempt_prom_px = [
        float(args.min_prom_px),
        max(0.0, float(args.min_prom_px) * 0.75),
        max(0.0, float(args.min_prom_px) * 0.50),
    ]
    detect_attempt_amp_px = [
        float(args.min_amp_px),
        max(1.0, float(args.min_amp_px) * 0.85),
        max(1.0, float(args.min_amp_px) * 0.70),
    ]

    maxima_f, minima_f = [], []
    cycles_all = []
    selected_cycles, selected_cv, selected_window_start_idx = [], np.nan, 0
    best_mean_corr, best_min_corr = np.nan, np.nan
    detect_attempt_used = 0
    detect_prom_ratio_used = np.nan
    detect_prom_px_used = np.nan
    detect_amp_px_used = np.nan

    # search window が妥当なら、ここで周期候補を構築する。
    if search_end >= search_start:
        target_n = int(args.target_cycles)
        best_pack = None

        for att_i, (prom_ratio_i, prom_px_i, amp_px_i) in enumerate(
            zip(detect_attempt_prom_ratio, detect_attempt_prom_px, detect_attempt_amp_px)
        ):
            maxima_i, minima_i = find_local_extrema_prom(
                sig_smooth_for_peaks,
                min_sep_frames=min_sep_frames,
                min_amp_px=float(amp_px_i),
                min_prom_ratio=float(prom_ratio_i),
                min_prom_px=float(prom_px_i),
                prom_win_frames=prom_win_frames,
            )
            maxima_f_i = [i for i in maxima_i if search_start <= i <= search_end]
            minima_f_i = [i for i in minima_i if search_start <= i <= search_end]

            cycles_i = build_cycles_from_extrema(
                sig_smooth_for_peaks,
                maxima_f_i,
                minima_f_i,
                start_search_frame=search_start,
                fps=fps,
                min_cycle_s=float(args.min_cycle_s),
                max_cycle_s=float(args.max_cycle_s),
            )

            pack_i = {
                "maxima_f": maxima_f_i,
                "minima_f": minima_f_i,
                "cycles_all": cycles_i,
                "attempt_used": int(att_i),
                "prom_ratio_used": float(prom_ratio_i),
                "prom_px_used": float(prom_px_i),
                "amp_px_used": float(amp_px_i),
            }

            if (best_pack is None) or (len(cycles_i) > len(best_pack["cycles_all"])):
                best_pack = pack_i
            if target_n > 0 and len(cycles_i) >= target_n:
                best_pack = pack_i
                break

        if best_pack is not None:
            maxima_f = best_pack["maxima_f"]
            minima_f = best_pack["minima_f"]
            cycles_all = best_pack["cycles_all"]
            detect_attempt_used = int(best_pack["attempt_used"])
            detect_prom_ratio_used = float(best_pack["prom_ratio_used"])
            detect_prom_px_used = float(best_pack["prom_px_used"])
            detect_amp_px_used = float(best_pack["amp_px_used"])

        # 候補が十分あれば、連続 target_n サイクル窓を selected10 として選ぶ。
        # 選択は波形一貫性優先、その次に周期 CV を参照する設計にしている。
        # Select contiguous target_n cycles with minimum CV (teacher advice)
        if len(cycles_all) >= target_n and target_n > 0:
            selected_cycles, selected_cv, selected_window_start_idx = select_best_contiguous_cycles_by_cv(
                cycles_all, fps=fps, target_n=target_n
            )
        else:
            selected_cycles = cycles_all
            selected_cv = np.nan
            selected_window_start_idx = 0

    # -------------------------
    # frames.csv (always saved)
    # -------------------------
    # frames.csv を組み立てる。
    # raw / clean / smoothed / signal / angle / motion flag をフレーム単位で保存する。
    frames = raw_df.copy()

    # Align column names with hammer while keeping backward compatibility
    if "frame_idx" not in frames.columns:
        frames.insert(0, "frame_idx", frames["frame"].astype(int) if "frame" in frames.columns else np.arange(len(frames), dtype=int))
    if "time_s" not in frames.columns:
        frames.insert(1, "time_s", frames["frame_idx"].astype(float) / float(fps))

    # combined wrist (hand preferred) for waveform export & QC
    frames["wrist_x_px_raw"] = wrist_x
    frames["wrist_y_px_raw"] = wrist_y
    # Representative index (INDEX_TIP preferred, else pose INDEX)
        # outlier info
    frames["outlier_flag"] = outlier_flag
    frames["outlier_reason"] = outlier_reason
    frames["outlier_step_px"] = outlier_step_px
    frames["outlier_hand_pose_dist_px"] = outlier_hand_pose_dist_px

    # representative index (raw & clean)
    frames["index_x_px_raw"] = ix_raw
    frames["index_y_px_raw"] = iy_raw
    frames["index_x_px_clean"] = ix_clean
    frames["index_y_px_clean"] = iy_clean

    win_wrist = _odd(int(round(float(args.smooth_sec) * float(fps))))
    frames["wrist_x_px_sm"] = rolling_mean(wrist_x, win_wrist)
    frames["wrist_y_px_sm"] = rolling_mean(wrist_y, win_wrist)

    # smoothed index (raw & clean)
    frames["index_x_px_sm_raw"] = rolling_mean(ix_raw, win_wrist)
    frames["index_y_px_sm_raw"] = rolling_mean(iy_raw, win_wrist)
    frames["index_x_px_sm"] = rolling_mean(ix_clean, win_wrist)
    frames["index_y_px_sm"] = rolling_mean(iy_clean, win_wrist)

    # kinematics / signals (raw & clean)
    frames["dx_px_raw"] = dx_raw
    frames["dy_px_raw"] = dy_raw
    frames["dist_index_to_shoulder_px_raw"] = dist_raw
    frames["dist_smooth_px_raw"] = dist_smooth_raw

    frames["dx_px"] = dx_clean
    frames["dy_px"] = dy_clean
    frames["dist_index_to_shoulder_px"] = dist_clean
    frames["dist_smooth_px"] = dist_smooth_clean

    frames["speed_index_px_s_raw"] = speed_index_smooth_raw
    frames["speed_index_px_s"] = speed_index_smooth

    frames["cycle_signal"] = args.cycle_signal
    frames["cycle_signal_px_raw"] = sig_raw
    frames["cycle_signal_smooth_px_raw"] = sig_smooth_raw
    frames["cycle_signal_px"] = sig_used
    frames["cycle_signal_smooth_px"] = sig_smooth

    frames["speed_px_s"] = speed_wrist_smooth
    frames["speed_wrist_px_s"] = speed_wrist_smooth
    frames["speed_index_px_s"] = speed_index_smooth

    # movement window flag (trim segment)
    in_motion = np.zeros(len(frames), dtype=int)
    if (trim_start_frame is not None) and (trim_end_frame is not None) and np.isfinite(trim_start_frame) and np.isfinite(trim_end_frame):
        ts = int(trim_start_frame)
        te = int(trim_end_frame)
        if 0 <= ts < len(frames) and 0 <= te < len(frames) and te >= ts:
            in_motion[ts : te + 1] = 1
    frames["in_motion"] = in_motion
    frames["in_trim_segment"] = in_motion

    frames["shoulder_deg"] = shoulder_deg
    frames["elbow_deg"] = elbow_deg
    frames["wrist_deg"] = wrist_deg
    frames["index_mcp_deg"] = index_mcp_deg

    # -------------------------
    # cycles.csv (ALWAYS output all detected cycles)
    #   - selected10: the chosen contiguous window (usually 10) used for QC/summary
    #   - if detected < 10: selected10 will just mark all detected cycles
    # -------------------------
    # cycles.csv を組み立てる。
    # すべての検出サイクルを残し、その中で selected10 に入ったものへ印を付ける。
    selected_keys = {(int(c["start_frame"]), int(c["end_frame"])) for c in (selected_cycles or [])}

    cyc_rows = []
    for c in (cycles_all or []):
        s0 = int(c["start_frame"])
        e0 = int(c["end_frame"])
        mid = int(c["opp_frame"])

        seg_x = ix_clean[s0:e0+1]
        seg_y = iy_clean[s0:e0+1]

        x_range = float(np.nanmax(seg_x) - np.nanmin(seg_x)) if np.any(~np.isnan(seg_x)) else np.nan
        y_range = float(np.nanmax(seg_y) - np.nanmin(seg_y)) if np.any(~np.isnan(seg_y)) else np.nan
        area = float(x_range * y_range) if (np.isfinite(x_range) and np.isfinite(y_range)) else np.nan

        seg_dist_s = dist_smooth_clean[s0:e0+1]
        dist_range = float(np.nanmax(seg_dist_s) - np.nanmin(seg_dist_s)) if np.any(~np.isnan(seg_dist_s)) else np.nan

        seg_sig = np.asarray(sig_smooth[s0:e0+1], dtype=float)
        amp_px = float(np.nanmax(seg_sig) - np.nanmin(seg_sig)) if np.any(~np.isnan(seg_sig)) else np.nan

        row = {
            "cycle_id_detected": int(c.get("cycle_id", 0)),
            "start_frame": s0,
            "opp_frame": mid,
            "end_frame": e0,
            "start_time_s": float(s0 / float(fps)),
            "opp_time_s": float(mid / float(fps)),
            "end_time_s": float(e0 / float(fps)),
            "cycle_time_s": float((e0 - s0) / float(fps)),
            "amp_px": amp_px,
            "selected10": int((s0, e0) in selected_keys),
            "area_px2": area,
            "x_range_px": x_range,
            "y_range_px": y_range,
            "traj_len_px": traj_len_px(seg_x, seg_y),
            "max_speed_px_s": max_speed_px_s(seg_x, seg_y, fps),
            "dist_range_px": dist_range,
            "plane_deg": pca_plane_deg(seg_x, seg_y),
            "wave_corr_to_mean10": np.nan,
        }

        for name, arr in [
            ("shoulder_deg", shoulder_deg),
            ("elbow_deg", elbow_deg),
            ("wrist_deg", wrist_deg),
            ("index_mcp_deg", index_mcp_deg),
        ]:
            seg = np.asarray(arr[s0:e0+1], dtype=float)
            if np.any(~np.isnan(seg)):
                row[f"{name}_range"] = float(np.nanmax(seg) - np.nanmin(seg))
                row[f"{name}_mean"] = float(np.nanmean(seg))
            else:
                row[f"{name}_range"] = np.nan
                row[f"{name}_mean"] = np.nan

        cyc_rows.append(row)

    cycles_df = pd.DataFrame(cyc_rows)

    # Sort + add sequential id for convenience
    if len(cycles_df) > 0:
        cycles_df = cycles_df.sort_values("start_frame").reset_index(drop=True)
        cycles_df.insert(0, "cycle_id", np.arange(1, len(cycles_df) + 1, dtype=int))
    else:
        # keep stable header even when no cycles
        cycles_df = pd.DataFrame(columns=[
            "cycle_id","cycle_id_detected","start_frame","opp_frame","end_frame",
            "start_time_s","opp_time_s","end_time_s","cycle_time_s","amp_px",
            "selected10","area_px2","x_range_px","y_range_px","traj_len_px","max_speed_px_s",
            "dist_range_px","plane_deg",
            "shoulder_deg_range","elbow_deg_range","wrist_deg_range","index_mcp_deg_range",
            "shoulder_deg_mean","elbow_deg_mean","wrist_deg_mean","index_mcp_deg_mean",
            "wave_corr_to_mean10",
        ])

    # -------------------------
    # Waveform similarity check for the selected block (typically 10 cycles)
    #   - uses smoothed cycle-signal per cycle, resampled, compared to mean waveform
    # -------------------------
    n_sel = int((cycles_df.get("selected10", 0) == 1).sum()) if len(cycles_df) > 0 else 0
    waveform_pass_10 = 0
    wave_mean_corr_10 = np.nan
    wave_min_corr_10 = np.nan

    # selected10 ブロックについて波形相関を計算し、cycles_df に書き戻す。
    if n_sel > 0 and len(cycles_df) > 0:
        sel_df = cycles_df[cycles_df["selected10"] == 1].copy().sort_values("start_frame").reset_index(drop=True)
        waves = _cycle_waveforms_from_y(frames["cycle_signal_smooth_px"].to_numpy(dtype=float), sel_df, int(args.waveform_resample_n))
        corrs = _corr_to_mean_wave(waves)

        if len(corrs) == len(sel_df):
            sel_df["wave_corr_to_mean10"] = corrs

            # write back
            for _, r in sel_df.iterrows():
                s0 = int(r["start_frame"]); e0 = int(r["end_frame"])
                m = (cycles_df["start_frame"] == s0) & (cycles_df["end_frame"] == e0)
                cycles_df.loc[m, "wave_corr_to_mean10"] = float(r["wave_corr_to_mean10"]) if np.isfinite(r["wave_corr_to_mean10"]) else np.nan

            if np.any(np.isfinite(corrs)):
                wave_mean_corr_10 = float(np.nanmean(corrs))
                wave_min_corr_10 = float(np.nanmin(corrs))

        # pass definition: (exactly target_cycles selected) AND (all corrs >= threshold)
        if (n_sel == int(args.target_cycles)) and (len(corrs) == n_sel) and np.all(np.isfinite(corrs)) and np.all(corrs >= float(args.waveform_min_corr)):
            waveform_pass_10 = 1
        else:
            waveform_pass_10 = 0

    # -------------------------
    # summary.csv (always saved)
    # -------------------------
    # cycle time stats for selected block (preferred), else all cycles
    # summary 集計用の補助関数。
    def _cycle_time_stats(df_: pd.DataFrame) -> tuple[float, float, float]:
        if df_ is None or len(df_) == 0 or "cycle_time_s" not in df_.columns:
            return np.nan, np.nan, np.nan
        t = df_["cycle_time_s"].to_numpy(dtype=float)
        if not np.any(np.isfinite(t)):
            return np.nan, np.nan, np.nan
        t_mean = float(np.nanmean(t))
        t_sd = float(np.nanstd(t, ddof=1)) if np.sum(np.isfinite(t)) >= 2 else np.nan
        cv = float(t_sd / t_mean) if (np.isfinite(t_sd) and t_mean != 0) else np.nan
        return t_mean, t_sd, cv

    sel_block_df = cycles_df[cycles_df["selected10"] == 1].copy() if ("selected10" in cycles_df.columns) else pd.DataFrame()
    t_mean_sel, t_sd_sel, rhythm_cv_sel = _cycle_time_stats(sel_block_df)
    t_mean_all, t_sd_all, rhythm_cv_all = _cycle_time_stats(cycles_df)
    # outlier summary
    _out_n = int(np.sum(outlier_flag == 1))
    _reasons = pd.Series(outlier_reason).astype(str)
    _out_n_jump = int(_reasons.str.contains("jump", regex=False).sum())
    _out_n_hand_pose = int(_reasons.str.contains("hand_pose", regex=False).sum())



    # summary.csv 1行分のメタ情報・QC・統計をここでまとめる。
    meta = {
        "participant_id": args.participant_id,
        "condition": args.condition,
        "set_id": int(args.set_id),
        "trial_id": int(args.trial_id),
        "side": args.side,
        "cue_frame": int(cue),
        "src_fps": float(src_fps),
        "fps_used": float(fps),
        "width_px": int(width),
        "height_px": int(height),
        "n_frames": int(len(frames)),

        "outlier_enabled": int(not bool(args.outlier_disable)),
        "outlier_jump_px": float(args.outlier_jump_px),
        "outlier_hand_pose_dist_px": float(args.outlier_hand_pose_dist_px),
        "outlier_max_gap_s": float(args.outlier_max_gap_s),
        "outlier_max_gap_frames": int(max_gap_frames),
        "n_outliers_index": _out_n,
        "n_outliers_jump": _out_n_jump,
        "n_outliers_hand_pose": _out_n_hand_pose,
        "n_cycles_detected": int(len(cycles_df)),
        "target_cycles": int(args.target_cycles),
        "n_cycles_selected10": int(n_sel),
        "selected_cycles_cv": float(selected_cv) if np.isfinite(selected_cv) else np.nan,
        "selected_window_wave_mean_corr": float(best_mean_corr) if np.isfinite(best_mean_corr) else np.nan,
        "selected_window_wave_min_corr": float(best_min_corr) if np.isfinite(best_min_corr) else np.nan,
        "selected_window_start_index": int(selected_window_start_idx),
        "cycle_search_start_frame": int(search_start) if np.isfinite(search_start) else np.nan,
        "cycle_search_end_frame": int(search_end) if np.isfinite(search_end) else np.nan,
        "cycle_search_used_trim": int(cycle_search_used_trim),
        "detect_attempt_used": int(detect_attempt_used),
        "detect_prom_ratio_used": float(detect_prom_ratio_used) if np.isfinite(detect_prom_ratio_used) else np.nan,
        "detect_prom_px_used": float(detect_prom_px_used) if np.isfinite(detect_prom_px_used) else np.nan,
        "detect_amp_px_used": float(detect_amp_px_used) if np.isfinite(detect_amp_px_used) else np.nan,
        "trim_start_frame": int(trim_start_frame) if (trim_start_frame is not None and np.isfinite(trim_start_frame)) else np.nan,
        "trim_end_frame": int(trim_end_frame) if (trim_end_frame is not None and np.isfinite(trim_end_frame)) else np.nan,
        "trim_thr_px_s": float(trim_thr) if np.isfinite(trim_thr) else np.nan,
        "trim_method": trim_method,
        "start_to_onset_s": start_to_onset_s,
        "onset_frame": int(onset_frame) if onset_frame is not None else np.nan,
        "onset_thr_px_s": float(onset_thr) if np.isfinite(onset_thr) else np.nan,

        # cycle time summary
        "cycle_time_mean_s_selected10": t_mean_sel,
        "cycle_time_sd_s_selected10": t_sd_sel,
        "rhythm_cv_selected10": rhythm_cv_sel,
        "cycle_time_mean_s_all": t_mean_all,
        "cycle_time_sd_s_all": t_sd_all,
        "rhythm_cv_all": rhythm_cv_all,

        # waveform similarity QC for selected block
        "waveform_resample_n": int(args.waveform_resample_n),
        "waveform_min_corr_th": float(args.waveform_min_corr),
        "waveform_mean_corr_10": wave_mean_corr_10,
        "waveform_min_corr_10": wave_min_corr_10,
        "waveform_pass_10": int(waveform_pass_10),
    }

    # add means over selected block if available, else over all cycles
    def _add_means(prefix: str, df_: pd.DataFrame):
        if df_ is None or len(df_) == 0:
            return
        for k in [
            "area_px2", "x_range_px", "y_range_px", "traj_len_px", "max_speed_px_s",
            "dist_range_px", "plane_deg",
            "shoulder_deg_range", "elbow_deg_range", "wrist_deg_range", "index_mcp_deg_range",
            "shoulder_deg_mean", "elbow_deg_mean", "wrist_deg_mean", "index_mcp_deg_mean",
        ]:
            if k in df_.columns:
                meta[f"{prefix}{k}_mean_over_cycles"] = float(np.nanmean(df_[k].to_numpy(dtype=float)))

    if n_sel > 0:
        _add_means("selected10_", sel_block_df)
    else:
        _add_means("all_", cycles_df)

    # 1試行1行の summary.csv を作成する。
    summary_df = pd.DataFrame([meta])

    # -------------------------
    # write outputs (aligned to hammer: frames/cycles/summary + waveform png)
    # -------------------------
    # 既定ファイル名で保存する。出力名は既存運用との互換性維持のため固定。
    frames.to_csv(out_dir / "frames.csv", index=False)
    cycles_df.to_csv(out_dir / "cycles.csv", index=False)
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    png = save_waveform_png(frames, out_dir)

    # QC print (similar to hammer)
    n_det = int(len(cycles_df))
    target = int(args.target_cycles)
    pass10 = int(waveform_pass_10)
    print(f"QC: detected_cycles={n_det}, selected_block={n_sel}, target={target}, waveform_pass_10={pass10}")

    print("Saved:")
    print(" ", out_dir / "frames.csv")
    print(" ", out_dir / "cycles.csv")
    print(" ", out_dir / "summary.csv")
    print(" ", png)



if __name__ == "__main__":
    import sys, traceback, faulthandler
    faulthandler.enable()
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)