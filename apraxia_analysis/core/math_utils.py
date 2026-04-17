#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""math_utils.py

byebye / comehere タスクで共通して使う数学・信号処理ユーティリティ。
計算ロジック・型変換・NaN 判定順序はオリジナルスクリプトから変更していない。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


# =========================
# 基本数学ヘルパー
# =========================

def angle_deg(
    ax: float, ay: float,
    bx: float, by: float,
    cx: float, cy: float,
) -> float:
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


def pca_plane_deg(x: np.ndarray, y: np.ndarray) -> float:
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


def traj_len_px(x: np.ndarray, y: np.ndarray) -> float:
    """Total trajectory length in pixels (NaN-safe)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~np.isnan(x) & ~np.isnan(y)
    x = x[m]; y = y[m]
    if len(x) < 2:
        return np.nan
    dx = np.diff(x); dy = np.diff(y)
    return float(np.nansum(np.hypot(dx, dy)))


def max_speed_px_s(x: np.ndarray, y: np.ndarray, fps: float) -> float:
    """Maximum frame-to-frame speed in px/s (NaN-safe)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~np.isnan(x) & ~np.isnan(y)
    x = x[m]; y = y[m]
    if len(x) < 2:
        return np.nan
    dx = np.diff(x); dy = np.diff(y)
    v = np.hypot(dx, dy) * float(fps)
    return float(np.nanmax(v)) if len(v) else np.nan


def rolling_mean(x: np.ndarray, win: int) -> np.ndarray:
    """Centred rolling mean with min_periods=1 (NaN-safe)."""
    s = pd.Series(x, dtype="float64")
    return s.rolling(int(win), center=True, min_periods=1).mean().to_numpy()


def speed_series_px_s(x: np.ndarray, y: np.ndarray, fps: float) -> np.ndarray:
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


def _odd(n: int) -> int:
    """Return n if odd, n+1 if even. Minimum value is 1."""
    n = int(max(1, n))
    return n if (n % 2 == 1) else (n + 1)


# =========================
# 外れ値処理ヘルパー
# =========================

def _as_str_array(a: Any) -> np.ndarray:
    """Convert any array-like to a numpy object array of strings."""
    if a is None:
        return np.array([], dtype=object)
    s = pd.Series(a).astype(str).to_numpy(dtype=object)
    return s


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def _interpolate_with_gap_limit(arr: np.ndarray, max_gap_frames: int) -> np.ndarray:
    """Linear interpolation with a maximum consecutive-gap limit."""
    s = pd.Series(arr, dtype="float64")
    lim = int(max(1, max_gap_frames))
    # interpolate only small gaps
    s = s.interpolate(method="linear", limit=lim, limit_direction="both")
    # fill edges only for small gaps
    s = s.ffill(limit=lim).bfill(limit=lim)
    return s.to_numpy(dtype=float)


def apply_outlier_cleaning_2d(
    ix_raw: np.ndarray,
    iy_raw: np.ndarray,
    outlier_flag: np.ndarray,
    max_gap_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
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
# 動作開始・区間検出
# =========================

def detect_onset_frame(
    speed_px_s_arr: np.ndarray,
    cue_frame: int,
    baseline_frames: int,
    k_mad: float = 3.0,
    hold_frames: int = 5,
) -> tuple[int | None, float]:
    """Detect movement onset after cue_frame using baseline median + k*MAD threshold.

    Baseline: prefer immediately BEFORE cue_frame if possible, else AFTER cue_frame.
    Returns (onset_frame_index_or_None, threshold_px_s).
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


def detect_movement_segment(
    speed_px_s_arr: np.ndarray,
    cue_frame: int,
    baseline_frames: int,
    k_mad: float = 3.0,
    hold_frames: int = 5,
    quiet_frames: int = 15,
    min_movement_frames: int = 15,
) -> tuple[int | None, int | None, float, str]:
    """Detect movement segment [start_frame, end_frame] using per-frame speed (px/s).

    Returns (start_frame, end_frame, threshold_px_s, method).
    method: 'quiet' (end found by quiet period) or 'last_above_thr'.
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


def select_best_contiguous_cycles_by_cv(
    cycles: list[dict],
    fps: float,
    target_n: int = 10,
) -> tuple[list[dict], float, int]:
    """Select a contiguous window of target_n cycles whose cycle_time_s CV is minimal.

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
# サイクル検出
# =========================

def find_local_extrema_prom(
    x: np.ndarray,
    min_sep_frames: int = 6,
    min_amp_px: float = 20.0,
    min_prom_ratio: float = 0.15,
    min_prom_px: float = 0.0,
    prom_win_frames: int = 15,
) -> tuple[list[int], list[int]]:
    """Return (maxima_idx, minima_idx) for a smoothed series x, with:
      - local extremum rule
      - minimum separation (frames)
      - global amplitude floor (min_amp_px)
      - prominence threshold:
          prom_px >= max(min_prom_px, min_prom_ratio * (global_range))
        where prom_px is estimated within a +/- prom_win_frames window.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    maxima: list[int] = []
    minima: list[int] = []
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


def build_cycles_from_extrema(
    x_smooth: np.ndarray,
    maxima: list[int],
    minima: list[int],
    start_search_frame: int,
    fps: float,
    min_cycle_s: float = 0.3,
    max_cycle_s: float = 3.0,
) -> list[dict]:
    """Build cycle list from detected extrema.

    Cycle definition:
      If first extremum after start_search_frame is MAX:  max -> min -> next max
      If first extremum after start_search_frame is MIN:  min -> max -> next min
    Returns list of dict: {cycle_id, start_frame, opp_frame, end_frame}.
    """
    x_smooth = np.asarray(x_smooth, dtype=float)
    start_search = int(start_search_frame)

    ext = [(i, "max") for i in maxima] + [(i, "min") for i in minima]
    ext = [(i, t) for (i, t) in ext if i >= start_search]
    ext.sort(key=lambda z: z[0])
    if not ext:
        return []

    start_kind = ext[0][1]  # "max" or "min"
    cycles: list[dict] = []
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
# 波形類似度
# =========================

def _resample_1d_nan(arr: np.ndarray, n: int) -> np.ndarray:
    """Resample a 1D array to n points using linear interpolation (NaN-safe)."""
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


def _cycle_waveforms_from_y(
    y_sm: np.ndarray,
    cycles_df: pd.DataFrame,
    resample_n: int,
) -> np.ndarray:
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


def _block_wave_stats(waves_block: np.ndarray) -> tuple[float, float]:
    """Mean and min correlation of waveforms to their mean waveform."""
    corrs = _corr_to_mean_wave(waves_block)
    if np.isfinite(corrs).any():
        return float(np.nanmean(corrs)), float(np.nanmin(corrs))
    return float("nan"), float("nan")


def _best_contiguous_block_by_waveform_then_cv(
    cycle_times: np.ndarray,
    waveforms: np.ndarray,
    target: int,
) -> tuple[int, float, float, float]:
    """Select best contiguous block prioritising mean waveform correlation, then CV.

    Returns (best_start_index, best_cv, best_mean_corr, best_min_corr).
    """
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
