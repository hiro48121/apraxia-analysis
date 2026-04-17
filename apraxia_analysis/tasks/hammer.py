#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tasks/hammer.py

hammer タスク固有の解析ロジック。
計算ロジック・型変換・NaN 判定順序はオリジナルスクリプトから変更していない。
Hammer は Pose のみ使用するため、抽出・数学ユーティリティは自己完結している。

【設計上の注意】
  下記ユーティリティ関数は core/math_utils.py に類似関数が存在するが、
  シグネチャや実装の細部が hammer 固有のため、出力の再現性を保証するため
  意図的にここで独立定義している。共通化を行う際は必ず出力値を照合すること。

  関数                    | core/ との相違点
  -----------------------|------------------------------------------------
  _odd()                 | max(3, n) を使用。core は max(1, n)
  angle_deg()            | 引数が Tuple[float,float] x3。core は float x6
  _interpolate_small_gaps| ループベースの補間。core は pandas.interpolate
  _rolling_mean()        | win<=1 のガード付き。core に同等品あり
  _corr_to_mean_wave()   | ドット積で相関計算。core は np.corrcoef
  _cycle_waveforms_from_y| start_i/end_i 列を使用。core は start_frame/end_frame
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  Hammer 固有ユーティリティ
#  （core/math_utils.py との違いは上記モジュール docstring を参照）
# ─────────────────────────────────────────────────────────────────────────────

def _odd(n: int) -> int:
    """Return n if odd, n+1 if even. Minimum value is 3 (hammer-specific)."""
    n = int(max(3, n))
    return n if (n % 2 == 1) else n + 1


def angle_deg(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float],
) -> float:
    """Angle ABC in degrees. Takes three (x, y) tuples. Returns NaN if any point is missing."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    if not (
        np.isfinite(ax) and np.isfinite(ay)
        and np.isfinite(bx) and np.isfinite(by)
        and np.isfinite(cx) and np.isfinite(cy)
    ):
        return float("nan")
    v1x, v1y = ax - bx, ay - by
    v2x, v2y = cx - bx, cy - by
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 == 0 or n2 == 0:
        return float("nan")
    cosang = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
    return math.degrees(math.acos(cosang))


def _interpolate_small_gaps(
    s: pd.Series,
    max_gap: int,
) -> tuple[pd.Series, pd.Series]:
    """Interpolate NaN runs up to max_gap frames. Returns (filled, interpolated_flag).

    Loop-based implementation to preserve exact interpolation behaviour.
    """
    x = s.astype(float).copy()
    interp_flag = pd.Series(np.zeros(len(x), dtype=int), index=x.index)

    isn = x.isna().to_numpy()
    if not isn.any():
        return x, interp_flag

    n = len(x)
    i = 0
    while i < n:
        if not isn[i]:
            i += 1
            continue
        j = i
        while j < n and isn[j]:
            j += 1
        run_len = j - i
        if run_len <= max_gap:
            left = i - 1
            right = j
            if left >= 0 and right < n and np.isfinite(x.iloc[left]) and np.isfinite(x.iloc[right]):
                x.iloc[i:j] = np.linspace(x.iloc[left], x.iloc[right], run_len + 2)[1:-1]
                interp_flag.iloc[i:j] = 1
        i = j
    return x, interp_flag


def _rolling_mean(x: np.ndarray, win: int) -> np.ndarray:
    """Centred rolling mean. Short-circuits when win <= 1."""
    if win <= 1:
        return x.astype(float)
    s = pd.Series(x.astype(float))
    return s.rolling(window=win, center=True, min_periods=1).mean().to_numpy(dtype=float)


def _find_peaks_safely(
    y: np.ndarray,
    distance: int,
    prominence: float,
) -> np.ndarray:
    """Try scipy.signal.find_peaks; fall back to a lightweight local-extrema detector."""
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return np.array([], dtype=int)

    try:
        from scipy.signal import find_peaks  # type: ignore

        peaks, _props = find_peaks(
            y, distance=max(1, int(distance)), prominence=max(0.0, float(prominence))
        )
        return peaks.astype(int)
    except Exception:
        peaks_list: List[int] = []
        last = -10**9
        for i in range(1, len(y) - 1):
            if i - last < max(1, int(distance)):
                continue
            if not (np.isfinite(y[i - 1]) and np.isfinite(y[i]) and np.isfinite(y[i + 1])):
                continue
            if y[i] >= y[i - 1] and y[i] >= y[i + 1]:
                lo = np.nanmin(y[max(0, i - distance) : i + 1])
                hi = np.nanmin(y[i : min(len(y), i + distance + 1)])
                prom_est = y[i] - max(lo, hi)
                if np.isfinite(prom_est) and prom_est >= prominence:
                    peaks_list.append(i)
                    last = i
        return np.array(peaks_list, dtype=int)


def _detect_motion_window(
    speed: np.ndarray,
    wrist_valid: np.ndarray,
    fps: float,
    cfg: "HammerConfig",
) -> tuple[int, int, float]:
    """Detect the active motion window [start_i, end_i] using wrist speed."""
    n = int(len(speed))
    if n <= 0:
        return 0, -1, float("nan")

    sp = np.asarray(speed, dtype=float)
    valid = (np.asarray(wrist_valid, dtype=int) == 1) & np.isfinite(sp)
    if not valid.any():
        return 0, n - 1, float("nan")

    p95 = float(np.nanpercentile(sp[valid], 95))
    th = float(cfg.motion_speed_th_frac) * p95

    active = (sp >= th) & valid

    win = max(1, int(round(float(cfg.motion_min_active_s) * fps)))
    ker = np.ones(win, dtype=float)
    a = active.astype(float)
    mov = np.convolve(a, ker, mode="same")
    active2 = mov >= max(1.0, 0.5 * win)

    if not active2.any():
        return 0, n - 1, th

    start = int(np.argmax(active2))
    end = int(n - 1 - np.argmax(active2[::-1]))

    pad = int(round(float(cfg.motion_pad_s) * fps))
    start = max(0, start - pad)
    end = min(n - 1, end + pad)
    if end <= start:
        return 0, n - 1, th
    return start, end, th


def _resample_1d_nan(arr: np.ndarray, n: int) -> np.ndarray:
    """Resample 1D array to n points via linear interpolation (NaN-safe)."""
    a = np.asarray(arr, dtype=float)
    if n <= 1:
        return np.array([float(np.nanmean(a))]) if np.isfinite(a).any() else np.array([np.nan])
    if a.size < 2 or (np.isfinite(a).sum() < 2):
        return np.full(int(n), np.nan, dtype=float)
    x = np.arange(a.size, dtype=float)
    m = np.isfinite(a)
    xp = x[m]
    fp = a[m]
    x_new = np.linspace(0.0, float(a.size - 1), int(n))
    y_new = np.interp(x_new, xp, fp)
    return y_new.astype(float)


def _cycle_waveforms_from_y(
    y_sm: np.ndarray,
    cycles_df: pd.DataFrame,
    resample_n: int,
) -> np.ndarray:
    """Return (n_cycles, resample_n) waveform matrix.

    Uses start_i / end_i columns (hammer-specific, unlike core which uses start_frame/end_frame).
    """
    mats: List[np.ndarray] = []
    if cycles_df is None or len(cycles_df) == 0:
        return np.empty((0, int(resample_n)), dtype=float)

    for _, r in cycles_df.iterrows():
        s = r.get("start_i", np.nan)
        e = r.get("end_i", np.nan)
        if not (np.isfinite(s) and np.isfinite(e)):
            mats.append(np.full(int(resample_n), np.nan, dtype=float))
            continue
        s_i = int(max(0, int(s)))
        e_i = int(min(len(y_sm) - 1, int(e)))
        if e_i <= s_i:
            mats.append(np.full(int(resample_n), np.nan, dtype=float))
            continue
        seg = y_sm[s_i : e_i + 1]
        mats.append(_resample_1d_nan(seg, int(resample_n)))

    return np.vstack(mats).astype(float) if len(mats) > 0 else np.empty((0, int(resample_n)), dtype=float)


def _corr_to_mean_wave(waves: np.ndarray) -> np.ndarray:
    """Correlation of each waveform to the mean waveform.

    Uses dot-product formula (hammer-specific; core uses np.corrcoef).
    Both are mathematically equivalent Pearson r.
    """
    w = np.asarray(waves, dtype=float)
    if w.ndim != 2 or w.size == 0:
        return np.array([], dtype=float)

    mean_w = np.nanmean(w, axis=0)
    corrs: List[float] = []
    min_pts = max(10, int(0.20 * w.shape[1]))

    for i in range(w.shape[0]):
        a = w[i]
        mask = np.isfinite(a) & np.isfinite(mean_w)
        if int(mask.sum()) < int(min_pts):
            corrs.append(float("nan"))
            continue
        aa = a[mask]
        bb = mean_w[mask]
        aa = aa - float(np.mean(aa))
        bb = bb - float(np.mean(bb))
        sa = float(np.std(aa))
        sb = float(np.std(bb))
        if sa == 0.0 or sb == 0.0:
            corrs.append(float("nan"))
            continue
        corr = float(np.dot(aa, bb) / (float(len(aa)) * sa * sb))
        corrs.append(corr)

    return np.asarray(corrs, dtype=float)


def _block_wave_stats(waves_block: np.ndarray) -> tuple[float, float]:
    """Mean and min correlation of waveforms to their mean waveform."""
    corrs = _corr_to_mean_wave(waves_block)
    if np.isfinite(corrs).any():
        return float(np.nanmean(corrs)), float(np.nanmin(corrs))
    return float("nan"), float("nan")


def select_best_contiguous_cycles_by_cv(
    cycle_times: np.ndarray,
    target: int,
) -> tuple[int, float]:
    """Select a contiguous block whose cycle-time CV is minimal.

    Returns (best_start_index, best_cv).
    If len(cycle_times) < target, returns (0, NaN).

    Note: takes a flat np.ndarray of cycle durations (different from core's version
    which takes a list[dict] and fps).
    """
    t = np.asarray(cycle_times, dtype=float)
    n = int(len(t))
    target = int(target)
    if target <= 0 or n < target:
        return 0, float("nan")

    best_i = 0
    best_cv = None
    for i in range(0, n - target + 1):
        seg = t[i : i + target]
        seg = seg[np.isfinite(seg)]
        if len(seg) == 0:
            continue
        mean = float(np.nanmean(seg))
        if not np.isfinite(mean) or mean == 0.0:
            continue
        sd = float(np.nanstd(seg, ddof=1)) if len(seg) >= 2 else 0.0
        cv = float(sd / mean)
        if (best_cv is None) or (cv < best_cv):
            best_i = i
            best_cv = cv

    if best_cv is None:
        return 0, float("nan")
    return int(best_i), float(best_cv)


def _best_contiguous_block_by_waveform_then_cv(
    cycle_times: np.ndarray,
    waveforms: np.ndarray,
    target: int,
) -> tuple[int, float, float, float]:
    """Select best block: prioritise mean waveform correlation, then CV.

    Returns (best_start_index, best_cv, best_mean_corr, best_min_corr).
    """
    t = np.asarray(cycle_times, dtype=float)
    n = int(len(t))
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


# ─────────────────────────────────────────────────────────────────────────────
#  設定クラス
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HammerConfig:
    fps: float

    interp_max_gap_s: float = 0.10
    smooth_s: float = 0.15

    # Outlier removal (Approach A: step-jump based)
    outlier_enabled: bool = True
    outlier_jump_px: float = 200.0

    motion_speed_th_frac: float = 0.15
    motion_min_active_s: float = 0.30
    motion_pad_s: float = 0.40

    min_sep_s: float = 0.25
    peak_prom_frac: float = 0.12
    peak_prom_min_abs: float = 5.0

    min_cycle_s: float = 0.20
    max_cycle_s: float = 3.00
    cycle_amp_frac: float = 0.15

    target_cycles: int = 10

    waveform_resample_n: int = 100
    waveform_min_corr: float = 0.75

    use_cycles_from: int = 4
    use_cycles_to: int = 8


# ─────────────────────────────────────────────────────────────────────────────
#  解析コア
# ─────────────────────────────────────────────────────────────────────────────

def process_hammer_trial(
    raw_df: pd.DataFrame,
    cfg: HammerConfig,
    meta: Dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run full hammer analysis on a single trial.

    Returns (frames_df, cycles_df, summary_df).
    """
    fps = float(cfg.fps) if cfg.fps and cfg.fps > 0 else 30.0

    df = raw_df.copy()
    if "frame_idx" not in df.columns:
        raise ValueError("raw_df must have frame_idx column")

    max_gap_frames = int(round(cfg.interp_max_gap_s * fps))
    x_raw = df.get("wrist_x_px_raw", pd.Series([np.nan] * len(df))).astype(float)
    y_raw = df.get("wrist_y_px_raw", pd.Series([np.nan] * len(df))).astype(float)
    wrist_valid = df.get("wrist_valid", pd.Series(np.zeros(len(df), dtype=int))).to_numpy(dtype=int)

    outlier_enabled = bool(getattr(cfg, "outlier_enabled", True))
    outlier_jump_th = float(getattr(cfg, "outlier_jump_px", 200.0))
    outlier_step_px = np.full(len(df), np.nan, dtype=float)
    outlier_flag = np.zeros(len(df), dtype=int)

    if outlier_enabled and len(df) >= 2 and np.isfinite(outlier_jump_th) and outlier_jump_th > 0:
        xr = x_raw.to_numpy(dtype=float)
        yr = y_raw.to_numpy(dtype=float)
        for i in range(1, len(df)):
            if wrist_valid[i] != 1 or wrist_valid[i - 1] != 1:
                continue
            if not (np.isfinite(xr[i]) and np.isfinite(yr[i]) and np.isfinite(xr[i - 1]) and np.isfinite(yr[i - 1])):
                continue
            step = float(math.hypot(float(xr[i] - xr[i - 1]), float(yr[i] - yr[i - 1])))
            outlier_step_px[i] = step
            if step > outlier_jump_th:
                outlier_flag[i] = 1

    # 外れ値フレームを NaN にしてから補間・平滑化に使う（生値は残す）
    x_for_interp = x_raw.copy()
    y_for_interp = y_raw.copy()
    if outlier_enabled and int(outlier_flag.sum()) > 0:
        x_for_interp[outlier_flag == 1] = np.nan
        y_for_interp[outlier_flag == 1] = np.nan

    x_filled, x_interp = _interpolate_small_gaps(x_for_interp, max_gap=max_gap_frames)
    y_filled, y_interp = _interpolate_small_gaps(y_for_interp, max_gap=max_gap_frames)
    interpolated = ((x_interp.to_numpy() == 1) | (y_interp.to_numpy() == 1)).astype(int)

    win = _odd(int(round(cfg.smooth_s * fps)))
    x_sm = _rolling_mean(x_filled.to_numpy(dtype=float), win=win)
    y_sm = _rolling_mean(y_filled.to_numpy(dtype=float), win=win)

    frame_idx = df["frame_idx"].to_numpy(dtype=float)
    time_s = frame_idx / fps

    vx = np.gradient(x_sm, 1.0 / fps)
    vy = np.gradient(y_sm, 1.0 / fps)
    speed = np.hypot(vx, vy)

    motion_start_i, motion_end_i, motion_speed_th = _detect_motion_window(speed, wrist_valid, fps=fps, cfg=cfg)
    in_motion = np.zeros(len(df), dtype=int)
    if 0 <= motion_start_i < len(df) and 0 <= motion_end_i < len(df) and motion_end_i >= motion_start_i:
        in_motion[motion_start_i : motion_end_i + 1] = 1

    def _colpair(xc: str, yc: str) -> np.ndarray:
        if {xc, yc}.issubset(df.columns):
            return df[[xc, yc]].to_numpy(dtype=float)
        return np.full((len(df), 2), np.nan, dtype=float)

    hip = _colpair("hip_x_px", "hip_y_px")
    shoulder = _colpair("shoulder_x_px", "shoulder_y_px")
    elbow = _colpair("elbow_x_px", "elbow_y_px")
    wrist = np.column_stack([x_sm, y_sm])
    indexp = _colpair("index_x_px", "index_y_px")

    shoulder_deg = np.array(
        [angle_deg(tuple(hip[i]), tuple(shoulder[i]), tuple(elbow[i])) for i in range(len(df))], dtype=float
    )
    elbow_deg = np.array(
        [angle_deg(tuple(shoulder[i]), tuple(elbow[i]), tuple(wrist[i])) for i in range(len(df))], dtype=float
    )
    wrist_deg = np.array(
        [angle_deg(tuple(elbow[i]), tuple(wrist[i]), tuple(indexp[i])) for i in range(len(df))], dtype=float
    )

    w0 = int(motion_start_i)
    w1 = int(motion_end_i)
    if not (0 <= w0 < len(df) and 0 <= w1 < len(df) and w1 > w0):
        w0, w1 = 0, len(df) - 1

    # MediaPipe の画像座標は Y 軸が下向き正のため、ハンマーの打鍵点（最も手が下がる瞬間）は
    # Y が最大値になる。-y_sm にすることで打鍵点が局所極大になり、find_peaks で正しく検出できる。
    sig = -y_sm[w0 : w1 + 1]
    if np.isfinite(sig).any():
        lo, hi = np.nanpercentile(sig, [5, 95])
    else:
        lo, hi = 0.0, 0.0

    prom = float(cfg.peak_prom_frac) * float(hi - lo)
    prom = max(float(cfg.peak_prom_min_abs), float(prom))
    dist = int(round(cfg.min_sep_s * fps))

    y_win = y_sm[w0 : w1 + 1]
    if np.isfinite(y_win).any():
        y_lo, y_hi = np.nanpercentile(y_win, [5, 95])
        min_amp_y = float(cfg.cycle_amp_frac) * float(y_hi - y_lo)
    else:
        min_amp_y = 0.0

    target = int(getattr(cfg, "target_cycles", 10))

    def _build_cycles_from_peaks(
        up_peaks_abs: np.ndarray,
        min_amp_y_local: float,
    ) -> tuple[List[Dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cycles_local: List[Dict] = []
        is_upper_local = np.zeros(len(df), dtype=int)
        is_lower_local = np.zeros(len(df), dtype=int)
        cycle_id_per_frame_local = np.full(len(df), fill_value=np.nan)
        is_cycle_start_local = np.zeros(len(df), dtype=int)

        for _k in range(len(up_peaks_abs) - 1):
            s = int(up_peaks_abs[_k])
            e = int(up_peaks_abs[_k + 1])
            if e <= s:
                continue

            seg = y_sm[s : e + 1]
            if not np.isfinite(seg).any():
                continue
            amp_y = float(np.nanmax(seg) - np.nanmin(seg))
            if np.isfinite(min_amp_y_local) and amp_y < float(min_amp_y_local):
                continue
            # seg 内の Y 最大点 = 手が最も下に下がった打鍵点（hit_i）
            low = s + int(np.nanargmax(seg))

            cycle_time = (e - s) / fps
            if cycle_time < float(cfg.min_cycle_s) or cycle_time > float(cfg.max_cycle_s):
                continue

            dx = x_sm[low] - x_sm[s]
            dy = y_sm[low] - y_sm[s]
            direction_deg_abs = (
                float(abs(math.degrees(math.atan2(dy, dx))))
                if (np.isfinite(dx) and np.isfinite(dy))
                else float("nan")
            )

            xs = x_sm[s : e + 1]
            ys = y_sm[s : e + 1]
            traj_len = float(np.nansum(np.hypot(np.diff(xs), np.diff(ys))))
            vmax = float(np.nanmax(speed[s : e + 1])) if np.isfinite(speed[s : e + 1]).any() else float("nan")

            def _range(arr: np.ndarray) -> float:
                if not np.isfinite(arr).any():
                    return float("nan")
                return float(np.nanmax(arr) - np.nanmin(arr))

            def _mean(arr: np.ndarray) -> float:
                if not np.isfinite(arr).any():
                    return float("nan")
                return float(np.nanmean(arr))

            rom_sh = _range(shoulder_deg[s : e + 1])
            rom_el = _range(elbow_deg[s : e + 1])
            rom_wr = _range(wrist_deg[s : e + 1])

            mean_sh = _mean(shoulder_deg[s : e + 1])
            mean_el = _mean(elbow_deg[s : e + 1])
            mean_wr = _mean(wrist_deg[s : e + 1])

            hit_time = (low - s) / fps
            lift_time = (e - low) / fps

            miss = (
                (wrist_valid[s : e + 1] == 0)
                | (outlier_flag[s : e + 1] == 1)
                | (~np.isfinite(x_raw.to_numpy(dtype=float)[s : e + 1]))
                | (~np.isfinite(y_raw.to_numpy(dtype=float)[s : e + 1]))
            )
            n_missing = int(np.sum(miss))
            max_run = 0
            run = 0
            for m in miss:
                if m:
                    run += 1
                    max_run = max(max_run, run)
                else:
                    run = 0

            cid = int(len(cycles_local))
            cycles_local.append(
                {
                    "cycle_id": cid,
                    "cycle_seq": int(cid) + 1,
                    "start_i": int(s),
                    "end_i": int(e),
                    "hit_i": int(low),
                    "start_frame": int(df["frame_idx"].iloc[s]),
                    "end_frame": int(df["frame_idx"].iloc[e]),
                    "hit_frame": int(df["frame_idx"].iloc[low]),
                    "cycle_time_s": float(cycle_time),
                    "hit_time_s": float(hit_time),
                    "lift_time_s": float(lift_time),
                    "direction_deg_abs": float(direction_deg_abs),
                    "traj_len_px": float(traj_len),
                    "vmax_px_s": float(vmax),
                    "amp_y_px": float(amp_y),
                    "rom_shoulder_deg": rom_sh,
                    "rom_elbow_deg": rom_el,
                    "rom_wrist_deg": rom_wr,
                    "mean_shoulder_deg": mean_sh,
                    "mean_elbow_deg": mean_el,
                    "mean_wrist_deg": mean_wr,
                    "cycle_valid": 1,
                    "n_frames_in_cycle": int(e - s + 1),
                    "n_missing_frames": n_missing,
                    "max_missing_run": int(max_run),
                }
            )

            cycle_id_per_frame_local[s : e + 1] = cid
            is_cycle_start_local[s] = 1
            is_upper_local[s] = 1
            is_lower_local[low] = 1

        return cycles_local, is_upper_local, is_lower_local, cycle_id_per_frame_local, is_cycle_start_local

    def _detect_once(
        prom_local: float,
        min_amp_y_local: float,
        extra_pad_frames: int = 0,
    ) -> tuple:
        w0_local = max(0, int(w0) - int(extra_pad_frames))
        w1_local = min(len(df) - 1, int(w1) + int(extra_pad_frames))
        sig_local = -y_sm[w0_local : w1_local + 1]
        up = _find_peaks_safely(sig_local, distance=dist, prominence=float(prom_local))
        up = (up + w0_local).astype(int)
        up = up[(up >= w0_local) & (up <= w1_local)]
        cycles_local, is_upper_local, is_lower_local, cycle_pf_local, is_cs_local = _build_cycles_from_peaks(
            up, float(min_amp_y_local)
        )
        return cycles_local, is_upper_local, is_lower_local, cycle_pf_local, is_cs_local, up, w0_local, w1_local

    # 3段階の閾値セット（prominence・振れ幅・探索パッドをそれぞれ段階的に緩和）。
    # target 数に達した時点で打ち切り（過剰な緩和を防ぐ）。
    prom_list = [
        float(prom),
        max(float(cfg.peak_prom_min_abs) * 0.60, float(prom) * 0.75),
        max(float(cfg.peak_prom_min_abs) * 0.40, float(prom) * 0.60),
    ]
    amp_list = [float(min_amp_y), float(min_amp_y) * 0.85, float(min_amp_y) * 0.70]
    pad_list = [0, int(round(0.20 * fps)), int(round(0.40 * fps))]

    best_pack = None
    attempt_used = 0

    for i, (p_i, a_i, pad_i) in enumerate(zip(prom_list, amp_list, pad_list)):
        pack = _detect_once(p_i, a_i, extra_pad_frames=pad_i)
        cycles_i = pack[0]
        # より多くのサイクルが取れた場合に best_pack を更新する
        if best_pack is None or len(cycles_i) > len(best_pack[0]):
            best_pack = (*pack, p_i, a_i, i)
        # target 個以上得られたので、これ以上閾値を緩和しない
        if target > 0 and len(cycles_i) >= target:
            best_pack = (*pack, p_i, a_i, i)
            break

    cycles, is_upper, is_lower, cycle_id_per_frame, is_cycle_start, up_peaks, w0_used, w1_used, prom_used, min_amp_used, attempt_used = best_pack
    n_up_peaks = int(len(up_peaks))

    w0, w1 = int(w0_used), int(w1_used)
    prom = float(prom_used)
    min_amp_y = float(min_amp_used)

    cycles_df = pd.DataFrame(cycles)

    cycles_df["selected10"] = 0
    cycles_df["cycle_id10"] = np.nan
    cycles_df["selected_block"] = 0
    cycles_df["cycle_in_block"] = np.nan
    cycles_df["wave_corr_selblock"] = np.nan

    best_start = 0
    best_cv = float("nan")
    best_mean_corr = float("nan")
    best_min_corr = float("nan")

    resample_n = int(getattr(cfg, "waveform_resample_n", 100))
    waves_all = _cycle_waveforms_from_y(y_sm, cycles_df, resample_n=resample_n) if len(cycles_df) > 0 else np.empty((0, resample_n), dtype=float)

    if len(cycles_df) > 0 and target > 0:
        if len(cycles_df) >= target:
            # hammer はサイクル時間 CV 最小の窓を選ぶ（comehere の波形優先とは異なる）
            best_start, best_cv = select_best_contiguous_cycles_by_cv(
                cycles_df["cycle_time_s"].to_numpy(dtype=float),
                target=target,
            )
            sel_idx = list(range(best_start, best_start + target))
        else:
            sel_idx = list(range(len(cycles_df)))

        cycles_df.loc[sel_idx, "selected10"] = 1
        cycles_df.loc[sel_idx, "selected_block"] = 1
        cycles_df.loc[sel_idx, "cycle_id10"] = np.arange(1, len(sel_idx) + 1, dtype=float)
        cycles_df.loc[sel_idx, "cycle_in_block"] = np.arange(1, len(sel_idx) + 1, dtype=float)

        if waves_all.size > 0 and len(sel_idx) >= 2:
            waves_block = waves_all[sel_idx, :]
            best_mean_corr, best_min_corr = _block_wave_stats(waves_block)
            corrs_block = _corr_to_mean_wave(waves_block)
            for j, c in zip(sel_idx, corrs_block):
                cycles_df.loc[j, "wave_corr_selblock"] = float(c)

    cycle_id10_per_frame = np.full(len(df), fill_value=np.nan)
    if len(cycles_df) > 0 and "cycle_id" in cycles_df.columns:
        max_cid = (
            int(np.nanmax(cycles_df["cycle_id"].to_numpy(dtype=float)))
            if np.isfinite(cycles_df["cycle_id"].to_numpy(dtype=float)).any()
            else 0
        )
        map_arr = np.full(max_cid + 1, np.nan, dtype=float)
        for _cid, _cid10 in zip(cycles_df["cycle_id"].to_numpy(dtype=float), cycles_df["cycle_id10"].to_numpy(dtype=float)):
            if np.isfinite(_cid) and int(_cid) >= 0 and int(_cid) < len(map_arr):
                map_arr[int(_cid)] = _cid10
        valid_cycle_mask = np.isfinite(cycle_id_per_frame)
        cid_int = np.zeros(len(cycle_id_per_frame), dtype=int)
        cid_int[valid_cycle_mask] = cycle_id_per_frame[valid_cycle_mask].astype(int)
        cid_int = np.clip(cid_int, 0, len(map_arr) - 1)
        cycle_id10_per_frame = map_arr[cid_int]
        cycle_id10_per_frame[~valid_cycle_mask] = np.nan

    is_cycle_start10 = ((is_cycle_start == 1) & np.isfinite(cycle_id10_per_frame)).astype(int)

    frames_df = pd.DataFrame(
        {
            "frame_idx": df["frame_idx"].astype(int),
            "wrist_x_px_raw": x_raw.to_numpy(dtype=float),
            "wrist_y_px_raw": y_raw.to_numpy(dtype=float),
            "wrist_valid": wrist_valid,
            "wrist_x_px_clean": x_filled.to_numpy(dtype=float),
            "wrist_y_px_clean": y_filled.to_numpy(dtype=float),
            "outlier_step_px": outlier_step_px,
            "outlier_flag": outlier_flag,
            "outlier_jump_th_px": float(outlier_jump_th) if (outlier_enabled and np.isfinite(outlier_jump_th)) else np.nan,
            "hip_x_px": df.get("hip_x_px", np.nan),
            "hip_y_px": df.get("hip_y_px", np.nan),
            "shoulder_x_px": df.get("shoulder_x_px", np.nan),
            "shoulder_y_px": df.get("shoulder_y_px", np.nan),
            "elbow_x_px": df.get("elbow_x_px", np.nan),
            "elbow_y_px": df.get("elbow_y_px", np.nan),
            "index_x_px": df.get("index_x_px", np.nan),
            "index_y_px": df.get("index_y_px", np.nan),
            "fps": float(fps),
            "time_s": time_s,
            "interpolated": interpolated,
            "wrist_x_px_sm": x_sm,
            "wrist_y_px_sm": y_sm,
            "vx_px_s": vx,
            "vy_px_s": vy,
            "speed_px_s": speed,
            "is_upper": is_upper,
            "is_lower": is_lower,
            "shoulder_deg": shoulder_deg,
            "elbow_deg": elbow_deg,
            "wrist_deg": wrist_deg,
            "cycle_id": cycle_id_per_frame,
            "cycle_seq": cycle_id_per_frame + 1,
            "cycle_id10": cycle_id10_per_frame,
            "is_cycle_start": is_cycle_start,
            "is_cycle_start10": is_cycle_start10,
            "in_motion": in_motion,
        }
    )

    for k, v in meta.items():
        frames_df[k] = v

    n_all = int(len(cycles_df))
    sel = cycles_df[cycles_df.get("selected10", 0) == 1].copy() if n_all > 0 else cycles_df.copy()
    n_sel = int(len(sel))

    if n_sel > 0 and n_all > 0 and target > 0 and n_all >= target:
        sel_label = f"best{target}(raw_cycle {best_start + 1}-{best_start + target})"
    elif n_sel > 0:
        sel_label = f"all_detected({n_sel})"
    else:
        sel_label = "none"

    # 選択ブロック内のウォームアップ（1-3番）と疲労（9-10番）を除いた
    # 中間サイクル（use_cycles_from〜use_cycles_to）を集計指標の算出に使う
    if n_sel >= int(cfg.use_cycles_to) and "cycle_id10" in sel.columns:
        use = sel[(sel["cycle_id10"] >= int(cfg.use_cycles_from)) & (sel["cycle_id10"] <= int(cfg.use_cycles_to))].copy()
        selected = f"{sel_label}; use {cfg.use_cycles_from}-{cfg.use_cycles_to}"
    else:
        use = sel.copy()
        selected = f"{sel_label}; use all"

    def _col_mean(col: str) -> float:
        if col not in use.columns or len(use) == 0:
            return float("nan")
        x = use[col].to_numpy(dtype=float)
        return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")

    def _col_sd(col: str) -> float:
        if col not in use.columns or len(use) == 0:
            return float("nan")
        x = use[col].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        return float(np.std(x, ddof=1)) if len(x) >= 2 else float("nan")

    ct_mean = _col_mean("cycle_time_s")
    ct_sd = _col_sd("cycle_time_s")
    rhythm_cv = float(ct_sd / ct_mean) if (np.isfinite(ct_sd) and np.isfinite(ct_mean) and ct_mean != 0) else float("nan")

    summary = {
        "participant_id": meta.get("participant_id", ""),
        "task": "hammer",
        "condition": meta.get("condition", "hammer"),
        "set_id": meta.get("set_id", ""),
        "trial_id": meta.get("trial_id", ""),
        "video_file": meta.get("video_file", meta.get("video", "")),
        "cue_frame": meta.get("cue_frame", np.nan),
        "src_fps": float(fps),
        "n_frames": int(len(frames_df)),
        "outlier_enabled": int(outlier_enabled),
        "outlier_jump_th_px": float(outlier_jump_th) if (outlier_enabled and np.isfinite(outlier_jump_th)) else np.nan,
        "n_outlier_frames": int(np.sum(outlier_flag)) if outlier_enabled else 0,
        "n_cycles": int(len(use)),
        "n_cycles_detected": int(n_all),
        "n_cycles_selected10": int(n_sel),
        "best10_cv": float(best_cv) if np.isfinite(best_cv) else float("nan"),
        "best10_mean_corr": float(best_mean_corr) if np.isfinite(best_mean_corr) else float("nan"),
        "best10_min_corr": float(best_min_corr) if np.isfinite(best_min_corr) else float("nan"),
        "waveform_resample_n": int(getattr(cfg, "waveform_resample_n", 100)),
        "waveform_min_corr_th": float(getattr(cfg, "waveform_min_corr", 0.75)),
        # 合格条件: ① target 個以上検出 ② 選択ブロックの最小相関値が閾値以上
        # （byebye/comehere は n_sel == target を要求するが、hammer は n_all >= target でよい）
        "waveform_pass_10": int(
            (int(n_all) >= int(target))
            and np.isfinite(float(best_min_corr))
            and (float(best_min_corr) >= float(getattr(cfg, "waveform_min_corr", 0.75)))
        )
        if ("best_min_corr" in locals())
        else 0,
        "motion_start_frame": int(frames_df["frame_idx"].iloc[motion_start_i]) if (0 <= int(motion_start_i) < len(frames_df)) else np.nan,
        "motion_end_frame": int(frames_df["frame_idx"].iloc[motion_end_i]) if (0 <= int(motion_end_i) < len(frames_df)) else np.nan,
        "motion_speed_th": float(motion_speed_th) if np.isfinite(motion_speed_th) else np.nan,
        "peak_prominence_used": float(prom),
        "n_up_peaks": int(n_up_peaks),
        "cycle_detect_attempt": int(attempt_used),
        "analysis_start_frame": int(frames_df["frame_idx"].iloc[w0]) if (0 <= int(w0) < len(frames_df)) else np.nan,
        "analysis_end_frame": int(frames_df["frame_idx"].iloc[w1]) if (0 <= int(w1) < len(frames_df)) else np.nan,
        "min_amp_y_used": float(min_amp_y),
        "cycle_time_mean_s": ct_mean,
        "cycle_time_sd_s": ct_sd,
        "rhythm_cv": rhythm_cv,
        "hit_time_mean_s": _col_mean("hit_time_s"),
        "lift_time_mean_s": _col_mean("lift_time_s"),
        "direction_deg_abs_mean": _col_mean("direction_deg_abs"),
        "traj_len_px_mean": _col_mean("traj_len_px"),
        "vmax_px_s_mean": _col_mean("vmax_px_s"),
        "rom_shoulder_deg_mean": _col_mean("rom_shoulder_deg"),
        "rom_elbow_deg_mean": _col_mean("rom_elbow_deg"),
        "rom_wrist_deg_mean": _col_mean("rom_wrist_deg"),
        "mean_shoulder_deg_mean": _col_mean("mean_shoulder_deg"),
        "mean_elbow_deg_mean": _col_mean("mean_elbow_deg"),
        "mean_wrist_deg_mean": _col_mean("mean_wrist_deg"),
        "selected_cycles": selected,
    }

    summary_df = pd.DataFrame([summary])
    return frames_df, cycles_df, summary_df


# ─────────────────────────────────────────────────────────────────────────────
#  MediaPipe 動画からの座標抽出
# ─────────────────────────────────────────────────────────────────────────────

POSE_LM: Dict[str, Dict[str, int]] = {
    "Left": {
        "hip": 23, "shoulder": 11, "elbow": 13, "wrist": 15, "index": 19,
    },
    "Right": {
        "hip": 24, "shoulder": 12, "elbow": 14, "wrist": 16, "index": 20,
    },
}


def _lm_score(lm: object) -> float:
    """Extract presence/visibility score from a MediaPipe landmark."""
    if lm is None:
        return 0.0
    if hasattr(lm, "presence") and lm.presence is not None:
        try:
            return float(lm.presence)
        except Exception:
            pass
    if hasattr(lm, "visibility") and lm.visibility is not None:
        try:
            return float(lm.visibility)
        except Exception:
            pass
    return 1.0


def extract_pose_px_from_video(
    video_path: Path,
    pose_model: Path,
    side: str,
    presence_th: float,
) -> tuple[pd.DataFrame, float, str]:
    """Extract minimal landmark pixel coords per frame using MediaPipe Pose.

    Returns (raw_df, src_fps, video_file).
    """
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not (fps > 0):
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    base_options = python.BaseOptions(model_asset_path=str(pose_model))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    idx_map = POSE_LM[side]

    rows: List[Dict] = []
    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        ts_ms = int(round((frame_idx / fps) * 1000.0))

        result = landmarker.detect_for_video(mp_img, ts_ms)
        pose_lms = None
        if result is not None and hasattr(result, "pose_landmarks") and result.pose_landmarks:
            pose_lms = result.pose_landmarks[0]

        def _get_xy(name: str) -> tuple[float, float]:
            if pose_lms is None:
                return float("nan"), float("nan")
            lm = pose_lms[idx_map[name]]
            if lm is None:
                return float("nan"), float("nan")
            x = float(lm.x) * float(width) if np.isfinite(width) and width > 0 else float("nan")
            y = float(lm.y) * float(height) if np.isfinite(height) and height > 0 else float("nan")
            return x, y

        wx, wy = _get_xy("wrist")
        hx, hy = _get_xy("hip")
        sx, sy = _get_xy("shoulder")
        ex, ey = _get_xy("elbow")
        ix, iy = _get_xy("index")

        w_valid = 0
        if pose_lms is not None:
            wscore = _lm_score(pose_lms[idx_map["wrist"]])
            w_valid = int(float(wscore) >= float(presence_th) and np.isfinite(wx) and np.isfinite(wy))

        # if invalid, set wrist coords to NaN (missing)
        if not w_valid:
            wx, wy = float("nan"), float("nan")

        rows.append(
            {
                "frame_idx": int(frame_idx),
                "wrist_x_px_raw": float(wx),
                "wrist_y_px_raw": float(wy),
                "wrist_valid": int(w_valid),
                "hip_x_px": float(hx),
                "hip_y_px": float(hy),
                "shoulder_x_px": float(sx),
                "shoulder_y_px": float(sy),
                "elbow_x_px": float(ex),
                "elbow_y_px": float(ey),
                "index_x_px": float(ix),
                "index_y_px": float(iy),
            }
        )

        frame_idx += 1

    cap.release()
    try:
        landmarker.close()
    except Exception:
        pass

    raw_df = pd.DataFrame(rows)
    return raw_df, fps, str(video_path)


# ─────────────────────────────────────────────────────────────────────────────
#  波形 PNG 出力
# ─────────────────────────────────────────────────────────────────────────────

def save_waveform_png(frames_df: pd.DataFrame, out_dir: Path) -> Path:
    """Save wrist-y waveform as a PNG file."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    stem = out_dir.name

    x = frames_df["time_s"] if "time_s" in frames_df.columns else frames_df["frame_idx"]
    if "wrist_y_px_clean" in frames_df.columns:
        y = frames_df["wrist_y_px_clean"]
        ylab = "wrist_y_px_clean"
    elif "wrist_y_px_sm" in frames_df.columns:
        y = frames_df["wrist_y_px_sm"]
        ylab = "wrist_y_px_sm"
    else:
        y = frames_df["wrist_y_px_raw"]
        ylab = "wrist_y_px_raw"

    plt.figure()
    plt.plot(x, y)
    plt.xlabel("time_s" if "time_s" in frames_df.columns else "frame_idx")
    plt.ylabel(ylab)
    plt.title(stem)

    png = out_dir / f"waveform_{stem}.png"
    plt.savefig(png, dpi=200, bbox_inches="tight")
    plt.close()
    return png


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    """Build the argument parser for the hammer task CLI."""
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="入力MOV/MP4")
    p.add_argument("--pose_model", required=True, help="pose_landmarker_full.task などの .task へのパス")
    p.add_argument("--hand_model", default=None, help="(unused) hand_landmarker.task")
    p.add_argument("--side", choices=["Left", "Right"], default="Left")
    p.add_argument("--out_dir", required=True, help="出力フォルダ")

    p.add_argument("--participant_id", default="")
    p.add_argument("--condition", default="hammer")
    p.add_argument("--set_id", default="")
    p.add_argument("--trial_id", default="")

    p.add_argument("--cue_frame", type=int, default=0)

    p.add_argument("--presence_th", type=float, default=0.50, help="wrist_valid判定のpresence閾値")

    p.add_argument("--outlier_disable", action="store_true", help="飛び値処理（方式A）を無効化")
    p.add_argument("--outlier_jump_px", type=float, default=200.0, help="飛び値判定: 連続フレーム間の位置ジャンプがこのpxを超えたら除外")

    p.add_argument("--target_cycles", type=int, default=10, help="最終的に採用する連続サイクル数 (default: 10)")

    p.add_argument("--motion_speed_th_frac", type=float, default=0.15,
                   help="動作区間検出: speed閾値 = frac * P95(speed)")
    p.add_argument("--motion_min_active_s", type=float, default=0.30,
                   help="動作区間検出: 最小連続活動時間 (秒)")
    p.add_argument("--motion_pad_s", type=float, default=0.40,
                   help="動作区間検出: 前後に足す余白 (秒)")

    p.add_argument("--peak_prom_frac", type=float, default=0.12,
                   help="ピーク検出: prominence = frac * (P95-P5) を使用")
    p.add_argument("--peak_prom_min_abs", type=float, default=5.0,
                   help="ピーク検出: prominenceの最小値 (px)")
    p.add_argument("--cycle_amp_frac", type=float, default=0.15,
                   help="偽サイクル除外: min_amp_y = frac * (P95(y)-P5(y))")

    p.add_argument("--waveform_resample_n", type=int, default=100,
                   help="同一波形チェック: 1サイクル波形をこの点数に正規化して相関を計算 (default: 100)")
    p.add_argument("--waveform_min_corr", type=float, default=0.75,
                   help="同一波形チェック: 相関の最低許容値 (default: 0.75)")

    return p


def run_hammer(argv: list[str] | None = None) -> int:
    """CLI entry point for the hammer task."""
    args = build_argparser().parse_args(argv)

    video = Path(args.video)
    pose_model = Path(args.pose_model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_df, src_fps, video_file = extract_pose_px_from_video(
        video_path=video,
        pose_model=pose_model,
        side=args.side,
        presence_th=float(args.presence_th),
    )

    cfg = HammerConfig(
        fps=float(src_fps),
        motion_speed_th_frac=float(args.motion_speed_th_frac),
        motion_min_active_s=float(args.motion_min_active_s),
        motion_pad_s=float(args.motion_pad_s),
        peak_prom_frac=float(args.peak_prom_frac),
        peak_prom_min_abs=float(args.peak_prom_min_abs),
        cycle_amp_frac=float(args.cycle_amp_frac),
        target_cycles=int(args.target_cycles),
        waveform_resample_n=int(args.waveform_resample_n),
        waveform_min_corr=float(args.waveform_min_corr),
        outlier_enabled=(not bool(args.outlier_disable)),
        outlier_jump_px=float(args.outlier_jump_px),
    )

    meta = {
        "participant_id": args.participant_id,
        "task": "hammer",
        "condition": args.condition,
        "set_id": args.set_id,
        "trial_id": args.trial_id,
        "video_file": video_file,
        "src_fps": float(src_fps),
        "side": args.side,
        "cue_frame": int(args.cue_frame),
    }

    frames_df, cycles_df, summary_df = process_hammer_trial(raw_df=raw_df, cfg=cfg, meta=meta)

    frames_path = out_dir / "frames.csv"
    cycles_path = out_dir / "cycles.csv"
    summary_path = out_dir / "summary.csv"

    frames_df.to_csv(frames_path, index=False)
    cycles_df.to_csv(cycles_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    png = save_waveform_png(frames_df, out_dir)

    n_det = int(summary_df.loc[0, "n_cycles_detected"]) if ("n_cycles_detected" in summary_df.columns) else len(cycles_df)
    n_sel = int(summary_df.loc[0, "n_cycles_selected10"]) if ("n_cycles_selected10" in summary_df.columns) else int((cycles_df.get("selected10", 0) == 1).sum())
    target = int(args.target_cycles)
    pass10 = int(summary_df.loc[0, "waveform_pass_10"]) if ("waveform_pass_10" in summary_df.columns) else 0
    print(f"QC: detected_cycles={n_det}, selected_block={n_sel}, target={target}, waveform_pass_10={pass10}")

    print("Saved:")
    print(f"  {frames_path}")
    print(f"  {cycles_path}")
    print(f"  {summary_path}")
    print(f"  {png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run_hammer())
