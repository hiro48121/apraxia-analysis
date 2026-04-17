#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""video_to_hammer_metrics.py

目的
- 1本の動画を解析して、同じ out_dir に以下を保存する（後処理スクリプト不要）:
  - frames.csv
  - cycles.csv
  - summary.csv
  - waveform_<動画名>.png 例: waveform_IMG_1858.png

備考
- hand_model 引数は互換性のために受け取りますが、現状未使用です。
- 解析ロジック（サイクル検出・指標算出）は hammer_metrics.py の設計を踏襲した実装を内蔵しています。

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd


# 詳細コメント版について
# - この版では、解析ロジック・閾値・分岐・出力形式は変更せず、処理理解のためのコメントのみを追記しています。
# - hammer 系は、(1) MediaPipe で手関節周辺の座標を抽出し、(2) 前処理で欠損・飛び値を扱い、
#   (3) y 方向の往復運動からサイクルを検出し、(4) 各サイクル指標と summary を出力する流れです。
# - コメントは『何をしているか』だけでなく、『何のためにその処理があるか』が追えるように付与しています。

# -------------------------
#  hammer_metrics (embedded)
# -------------------------

# ------------------------------------------------------------------------------
# _odd
# 役割: 移動平均などで使う窓長を奇数にそろえる補助関数。
# 入力: n: 候補となる窓長。
# 出力: 3以上の奇数。
# 注意: 中心化 rolling を使う処理では奇数窓の方が対称に扱いやすいため、最小3・偶数なら+1に補正する。
# ------------------------------------------------------------------------------
def _odd(n: int) -> int:
    n = int(max(3, n))
    return n if (n % 2 == 1) else n + 1


# ------------------------------------------------------------------------------
# angle_deg
# 役割: 3点 A-B-C から関節角度 ABC を度数で計算する。
# 入力: a, b, c: 2次元座標。b が頂点。
# 出力: 角度（degree）。計算不能なら NaN。
# 注意: 関節角度の時系列を作る基礎関数として使う。
# ------------------------------------------------------------------------------
def angle_deg(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """Angle ABC in degrees. Returns NaN if any point is missing."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    if not (
        np.isfinite(ax)
        and np.isfinite(ay)
        and np.isfinite(bx)
        and np.isfinite(by)
        and np.isfinite(cx)
        and np.isfinite(cy)
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


# ------------------------------------------------------------------------------
# _interpolate_small_gaps
# 役割: 短い欠損区間だけを線形補間し、どこを補間したかも返す。
# 入力: s: 欠損を含む系列, max_gap: 補間してよい最大フレーム長。
# 出力: (filled, interpolated_flag)。
# 注意: 長い欠損まで埋めると波形を作り込みすぎるため、『短い欠損のみ補間』としている。
# ------------------------------------------------------------------------------
def _interpolate_small_gaps(s: pd.Series, max_gap: int) -> tuple[pd.Series, pd.Series]:
    """Interpolate NaN runs up to max_gap frames. Returns (filled, interpolated_flag)."""
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


# ------------------------------------------------------------------------------
# _rolling_mean
# 役割: 中心化移動平均で系列を平滑化する。
# 入力: x: 数値配列, win: 窓長。
# 出力: 平滑化後の配列。
# 注意: win<=1 の場合は元系列をそのまま float 化して返す。
# ------------------------------------------------------------------------------
def _rolling_mean(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.astype(float)
    s = pd.Series(x.astype(float))
    return s.rolling(window=win, center=True, min_periods=1).mean().to_numpy(dtype=float)


# ------------------------------------------------------------------------------
# _find_peaks_safely
# 役割: ピーク検出を行う。SciPy があれば find_peaks を使い、無ければ簡易実装にフォールバックする。
# 入力: y: 信号, distance: 最小ピーク間隔, prominence: 顕著さ閾値。
# 出力: ピーク位置の整数 index 配列。
# 注意: 環境差で SciPy が無い場合でも解析を継続できるようにしている。
# ------------------------------------------------------------------------------
def _find_peaks_safely(y: np.ndarray, distance: int, prominence: float) -> np.ndarray:
    """Try scipy.signal.find_peaks; otherwise use a lightweight local-extrema detector."""
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
        peaks: List[int] = []
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
                    peaks.append(i)
                    last = i
        return np.array(peaks, dtype=int)


# ------------------------------------------------------------------------------
# _detect_motion_window
# 役割: 手関節速度から『動いている区間』の開始・終了を推定する。
# 入力: speed: 速度系列, wrist_valid: 有効フラグ, fps/cfg: 閾値設定。
# 出力: (start_i, end_i, speed_threshold)。
# 注意: サイクル検出を動画全体でなく主要動作区間に絞るための前処理。
# ------------------------------------------------------------------------------
def _detect_motion_window(
    speed: np.ndarray, wrist_valid: np.ndarray, fps: float, cfg: "HammerConfig"
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


# ------------------------------------------------------------------------------
# _resample_1d_nan
# 役割: 欠損を含む1次元波形を、NaN を保ちながら指定点数へ再サンプリングする。
# 入力: arr: 元波形, n: 目標点数。
# 出力: 長さ n の配列。
# 注意: サイクルごとの波形長をそろえて相関比較するために使う。
# ------------------------------------------------------------------------------
def _resample_1d_nan(arr: np.ndarray, n: int) -> np.ndarray:
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


# ------------------------------------------------------------------------------
# _cycle_waveforms_from_y
# 役割: 各サイクル区間の y 波形を切り出し、共通長に正規化して行列化する。
# 入力: y_sm: 平滑化済み y, cycles_df: start/end を含む表, resample_n: 再標本化点数。
# 出力: shape=(cycle数, resample_n) の配列。
# 注意: 波形類似度チェック用。
# ------------------------------------------------------------------------------
def _cycle_waveforms_from_y(y_sm: np.ndarray, cycles_df: pd.DataFrame, resample_n: int) -> np.ndarray:
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


# ------------------------------------------------------------------------------
# _corr_to_mean_wave
# 役割: 各サイクル波形と平均波形の相関を計算する。
# 入力: waves: サイクル波形行列。
# 出力: 各サイクルの相関係数。
# 注意: 波形の揃い具合を QC 指標として使う。
# ------------------------------------------------------------------------------
def _corr_to_mean_wave(waves: np.ndarray) -> np.ndarray:
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


# ------------------------------------------------------------------------------
# _block_wave_stats
# 役割: 連続サイクルブロックの平均相関・最小相関をまとめて返す。
# 入力: waves_block: 連続ブロックの波形行列。
# 出力: (mean_corr, min_corr)。
# 注意: 『どの10サイクルを採用するか』を決める補助情報。
# ------------------------------------------------------------------------------
def _block_wave_stats(waves_block: np.ndarray) -> tuple[float, float]:
    corrs = _corr_to_mean_wave(waves_block)
    if np.isfinite(corrs).any():
        return float(np.nanmean(corrs)), float(np.nanmin(corrs))
    return float("nan"), float("nan")


# ------------------------------------------------------------------------------
# select_best_contiguous_cycles_by_cv
# 役割: 連続 target サイクルの候補のうち、周期時間 CV が最小の開始位置を選ぶ。
# 入力: cycle_times: 各サイクル時間, target: 採用サイクル数。
# 出力: (best_start_index, best_cv)。
# 注意: 教師助言に沿った『リズムが最も安定した連続区間』の選定。
# ------------------------------------------------------------------------------
def select_best_contiguous_cycles_by_cv(cycle_times: np.ndarray, target: int) -> tuple[int, float]:
    """Select a contiguous block whose cycle-time CV is minimal.

    Returns (best_start_index, best_cv). If len(cycle_times) < target, returns (0, NaN).
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


# ------------------------------------------------------------------------------
# _best_contiguous_block_by_waveform_then_cv
# 役割: 連続サイクル群を、まず波形一貫性、同点なら周期 CV で順位付けして選ぶ。
# 入力: cycle_times, waveforms, target。
# 出力: (best_i, best_cv, best_mean_corr, best_min_corr)。
# 注意: hammer では CV だけでなく波形の揃い具合も QC に使う。
# ------------------------------------------------------------------------------
def _best_contiguous_block_by_waveform_then_cv(
    cycle_times: np.ndarray,
    waveforms: np.ndarray,
    target: int,
) -> tuple[int, float, float, float]:
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


# ------------------------------------------------------------------------------
# HammerConfig
# 役割: hammer 解析で使う閾値・窓長・QC 条件を一括保持する設定クラス。
# 入力: CLI から渡す各種引数。
# 出力: 設定オブジェクト。
# 注意: 解析本体から設定値を切り離し、意味を追いやすくするためのコンテナ。
# ------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------
# process_hammer_trial
# 役割: hammer 解析の中心処理。生データから frames/cycles/summary を構成する。
# 入力: raw_df: 抽出済み座標表, cfg: 設定, meta: メタ情報。
# 出力: (frames_df, cycles_df, summary_df)。
# 注意: この関数内では『前処理→動作区間推定→サイクル検出→集計』を順に行う。
# ------------------------------------------------------------------------------
def process_hammer_trial(raw_df: pd.DataFrame, cfg: HammerConfig, meta: Dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 解析で使う fps。メタに 0 や欠損が入っても破綻しないよう、最低限 30fps を代替値とする。
    fps = float(cfg.fps) if cfg.fps and cfg.fps > 0 else 30.0

    df = raw_df.copy()
    if "frame_idx" not in df.columns:
        raise ValueError("raw_df must have frame_idx column")

    # ここから前処理。
    # 生の手関節座標を取り出し、飛び値除去と補間の準備をする。
    # 補間許容長は秒指定をフレーム数へ直して扱う。
    max_gap_frames = int(round(cfg.interp_max_gap_s * fps))
    x_raw = df.get("wrist_x_px_raw", pd.Series([np.nan] * len(df))).astype(float)
    y_raw = df.get("wrist_y_px_raw", pd.Series([np.nan] * len(df))).astype(float)
    wrist_valid = df.get("wrist_valid", pd.Series(np.zeros(len(df), dtype=int))).to_numpy(dtype=int)

    # -------------------------
    # Outlier removal (Approach A)
    # - detect abrupt step-jumps in wrist (x,y) between consecutive frames
    # - mark those frames as NaN and then interpolate only small gaps
    # -------------------------
    outlier_enabled = bool(getattr(cfg, "outlier_enabled", True))
    outlier_jump_th = float(getattr(cfg, "outlier_jump_px", 200.0))
    outlier_step_px = np.full(len(df), np.nan, dtype=float)
    outlier_flag = np.zeros(len(df), dtype=int)

    # 連続フレーム間で不自然に大きく跳んだ手関節位置を outlier とみなす。
    # ここで raw を直接消さず、後で補間用系列だけ NaN 化するのがポイント。
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

    # clean series used for interpolation/smoothing
    x_for_interp = x_raw.copy()
    y_for_interp = y_raw.copy()
    if outlier_enabled and int(outlier_flag.sum()) > 0:
        x_for_interp[outlier_flag == 1] = np.nan
        y_for_interp[outlier_flag == 1] = np.nan

    # 短い欠損だけを補間して、平滑化や速度計算に使う連続系列を作る。
    # 補間フラグは frames.csv に残し、どのフレームが人工的に埋められたか追えるようにする。
    x_filled, x_interp = _interpolate_small_gaps(x_for_interp, max_gap=max_gap_frames)
    y_filled, y_interp = _interpolate_small_gaps(y_for_interp, max_gap=max_gap_frames)
    interpolated = ((x_interp.to_numpy() == 1) | (y_interp.to_numpy() == 1)).astype(int)

    # 手関節軌跡を平滑化して高周波ノイズを抑える。
    # 速度・ピーク検出・ROM など下流計算は、この平滑化系列を基準に進める。
    win = _odd(int(round(cfg.smooth_s * fps)))
    x_sm = _rolling_mean(x_filled.to_numpy(dtype=float), win=win)
    y_sm = _rolling_mean(y_filled.to_numpy(dtype=float), win=win)

    # 時間軸と速度系列を作成する。
    # speed は x/y の勾配から求めるため、欠損が処理された後の smoothed series を使う。
    frame_idx = df["frame_idx"].to_numpy(dtype=float)
    time_s = frame_idx / fps

    vx = np.gradient(x_sm, 1.0 / fps)
    vy = np.gradient(y_sm, 1.0 / fps)
    speed = np.hypot(vx, vy)

    # 動作区間推定。
    # hammer の往復動作が実際に行われている中心区間を見つけ、後段のピーク探索範囲を絞る。
    motion_start_i, motion_end_i, motion_speed_th = _detect_motion_window(speed, wrist_valid, fps=fps, cfg=cfg)
    in_motion = np.zeros(len(df), dtype=int)
    if 0 <= motion_start_i < len(df) and 0 <= motion_end_i < len(df) and motion_end_i >= motion_start_i:
        in_motion[motion_start_i : motion_end_i + 1] = 1

    # 関節角度計算用に、df から 2列1組の座標配列を安全に取り出す補助関数。
    # 対応列が無い場合は NaN 配列を返し、後続処理を止めないようにしている。
    def _colpair(xc: str, yc: str) -> np.ndarray:
        if {xc, yc}.issubset(df.columns):
            return df[[xc, yc]].to_numpy(dtype=float)
        return np.full((len(df), 2), np.nan, dtype=float)

    hip = _colpair("hip_x_px", "hip_y_px")
    shoulder = _colpair("shoulder_x_px", "shoulder_y_px")
    elbow = _colpair("elbow_x_px", "elbow_y_px")
    wrist = np.column_stack([x_sm, y_sm])
    indexp = _colpair("index_x_px", "index_y_px")

    # 各フレームの肩・肘・手関節角度を算出する。
    # 手関節は clean/smoothed wrist を使うため、角度系列も飛び値の影響を受けにくい。
    shoulder_deg = np.array(
        [angle_deg(tuple(hip[i]), tuple(shoulder[i]), tuple(elbow[i])) for i in range(len(df))], dtype=float
    )
    elbow_deg = np.array(
        [angle_deg(tuple(shoulder[i]), tuple(elbow[i]), tuple(wrist[i])) for i in range(len(df))], dtype=float
    )
    wrist_deg = np.array(
        [angle_deg(tuple(elbow[i]), tuple(wrist[i]), tuple(indexp[i])) for i in range(len(df))], dtype=float
    )

    # ピーク探索を行う解析窓を決める。
    # 動作区間推定に失敗した場合は、保険として動画全体を対象にする。
    w0 = int(motion_start_i)
    w1 = int(motion_end_i)
    if not (0 <= w0 < len(df) and 0 <= w1 < len(df) and w1 > w0):
        w0, w1 = 0, len(df) - 1

    sig = -y_sm[w0 : w1 + 1]
    if np.isfinite(sig).any():
        lo, hi = np.nanpercentile(sig, [5, 95])
    else:
        lo, hi = 0.0, 0.0

    # ピーク検出条件を、窓内波形のレンジから自動設定する。
    # prominence は比率ベースと絶対下限の大きい方を使い、振幅の小さい試行でも極端に緩くしすぎない。
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

    # 上側ピーク列（hammer の持ち上げ側）から cycle 辞書列を作る内部関数。
    # start, hit, end を決め、時間・軌跡長・ROM・欠損量などをここで一括計算する。
    def _build_cycles_from_peaks(up_peaks_abs: np.ndarray, min_amp_y_local: float):
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

    # 1回分のピーク検出を実行する内部関数。
    # 閾値や探索窓の余白を変えながら複数回試し、『10サイクル届かない』状況を救済するために使う。
    def _detect_once(prom_local: float, min_amp_y_local: float, extra_pad_frames: int = 0):
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

    # 閾値緩和の試行列。
    # まず標準条件で試し、足りなければ prominence や振幅条件を段階的に緩める。
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
        if best_pack is None or len(cycles_i) > len(best_pack[0]):
            best_pack = (*pack, p_i, a_i, i)
        if target > 0 and len(cycles_i) >= target:
            best_pack = (*pack, p_i, a_i, i)
            break

    cycles, is_upper, is_lower, cycle_id_per_frame, is_cycle_start, up_peaks, w0_used, w1_used, prom_used, min_amp_used, attempt_used = best_pack
    n_up_peaks = int(len(up_peaks))

    w0, w1 = int(w0_used), int(w1_used)
    prom = float(prom_used)
    min_amp_y = float(min_amp_used)

    # 検出された cycle 辞書列を DataFrame 化する。
    # 以後の selected10 付与や waveform QC はこの表に対して行う。
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

    # frames.csv 用のフレーム単位表を組み立てる。
    # raw / clean / smoothed / motion flag を並べ、後で QC を追跡しやすくする。
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

    # 最後に summary.csv 1行分を構成する。
    # ここには trial メタ情報、QC 指標、周期統計、ROM 統計をまとめて保存する。
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


# -------------------------
#  MediaPipe video ingest
# -------------------------

POSE_LM = {
    "Left": {
        "hip": 23,
        "shoulder": 11,
        "elbow": 13,
        "wrist": 15,
        "index": 19,
    },
    "Right": {
        "hip": 24,
        "shoulder": 12,
        "elbow": 14,
        "wrist": 16,
        "index": 20,
    },
}


# ------------------------------------------------------------------------------
# _lm_score
# 役割: MediaPipe ランドマークの presence / visibility を共通的に読み出す。
# 入力: lm: NormalizedLandmark 互換オブジェクト。
# 出力: 信頼度スコア。
# 注意: MediaPipe バージョン差を吸収するため、presence が無ければ visibility を試す。
# ------------------------------------------------------------------------------
def _lm_score(lm) -> float:
    # mediapipe NormalizedLandmark has presence/visibility depending on version
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


# ------------------------------------------------------------------------------
# extract_pose_px_from_video
# 役割: 動画を1フレームずつ読み、PoseLandmarker から解析に必要な座標を pixel 単位で抽出する。
# 入力: 動画パス、pose モデル、左右、presence 閾値。
# 出力: (raw_df, fps, video_file)。
# 注意: hammer では hand モデルは使わず、Pose 由来の最小限の座標だけを取り出す。
# ------------------------------------------------------------------------------
def extract_pose_px_from_video(video_path: Path, pose_model: Path, side: str, presence_th: float) -> tuple[pd.DataFrame, float, str]:
    """Extract minimal landmark pixel coords per frame.

    Returns (raw_df, src_fps, video_file).

    raw_df columns:
      frame_idx, wrist_x_px_raw, wrist_y_px_raw, wrist_valid,
      hip_x_px, hip_y_px, shoulder_x_px, shoulder_y_px, elbow_x_px, elbow_y_px, index_x_px, index_y_px
    """
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    # ここから動画読み込み処理。
    # 1フレームずつ PoseLandmarker に渡し、解析に最低限必要な座標を pixel 単位で回収する。
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

    # 抽出結果をフレーム単位辞書として蓄積する。
    # 後で DataFrame 化しやすいよう、1フレーム = 1辞書の形式で保存する。
    rows: List[Dict] = []
    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        # BGR->RGB
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


# -------------------------
#  waveform export
# -------------------------

# ------------------------------------------------------------------------------
# save_waveform_png
# 役割: frames_df から代表 y 波形を描画し、waveform_*.png として保存する。
# 入力: frames_df, out_dir。
# 出力: 保存した png パス。
# 注意: 列優先順位は clean → smoothed → raw。
# ------------------------------------------------------------------------------
def save_waveform_png(frames_df: pd.DataFrame, out_dir: Path) -> Path:
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


# -------------------------
#  CLI
# -------------------------

# ------------------------------------------------------------------------------
# build_argparser
# 役割: CLI 引数定義をまとめて作成する。
# 入力: なし。
# 出力: ArgumentParser。
# 注意: 解析条件をコマンドラインから再現可能にする入口。
# ------------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
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

    p.add_argument(
        "--motion_speed_th_frac",
        type=float,
        default=0.15,
        help="動作区間検出: speed閾値 = frac * P95(speed)",
    )
    p.add_argument(
        "--motion_min_active_s",
        type=float,
        default=0.30,
        help="動作区間検出: 最小連続活動時間 (秒)",
    )
    p.add_argument(
        "--motion_pad_s",
        type=float,
        default=0.40,
        help="動作区間検出: 前後に足す余白 (秒)",
    )

    p.add_argument(
        "--peak_prom_frac",
        type=float,
        default=0.12,
        help="ピーク検出: prominence = frac * (P95-P5) を使用",
    )
    p.add_argument(
        "--peak_prom_min_abs",
        type=float,
        default=5.0,
        help="ピーク検出: prominenceの最小値 (px)",
    )
    p.add_argument(
        "--cycle_amp_frac",
        type=float,
        default=0.15,
        help="偽サイクル除外: min_amp_y = frac * (P95(y)-P5(y))",
    )

    p.add_argument(
        "--waveform_resample_n",
        type=int,
        default=100,
        help="同一波形チェック: 1サイクル波形をこの点数に正規化して相関を計算 (default: 100)",
    )
    p.add_argument(
        "--waveform_min_corr",
        type=float,
        default=0.75,
        help="同一波形チェック: 相関の最低許容値 (default: 0.75)",
    )

    return p


# ------------------------------------------------------------------------------
# main
# 役割: CLI 実行時の入口。引数解釈、抽出、解析、保存、QC 表示までを担う。
# 入力: コマンドライン引数。
# 出力: 終了コード 0。
# 注意: 実際の解析ロジックは extract_pose_px_from_video と process_hammer_trial に委譲している。
# ------------------------------------------------------------------------------
def main() -> int:
    # CLI 引数を読み込む。
    # main は『引数解釈 → 動画から座標抽出 → hammer 本体解析 → 保存 → QC 表示』の順で進む。
    args = build_argparser().parse_args()

    video = Path(args.video)
    pose_model = Path(args.pose_model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # まず動画から Pose 座標を抽出する。
    # ここではまだサイクル検出はせず、raw_df を作る段階に留める。
    raw_df, src_fps, video_file = extract_pose_px_from_video(
        video_path=video,
        pose_model=pose_model,
        side=args.side,
        presence_th=float(args.presence_th),
    )

    # CLI 引数を HammerConfig に束ね、解析本体へ渡す。
    # これにより process_hammer_trial 側では cfg 経由で閾値を参照できる。
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

    # summary に埋め込む trial メタ情報を整理する。
    # participant/task/condition などの識別情報はここでまとめておく。
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

    # 解析結果を既定ファイル名で保存する。
    # 出力名は既存運用と互換性を保つため変更しない。
    frames_path = out_dir / "frames.csv"
    cycles_path = out_dir / "cycles.csv"
    summary_path = out_dir / "summary.csv"

    frames_df.to_csv(frames_path, index=False)
    cycles_df.to_csv(cycles_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    png = save_waveform_png(frames_df, out_dir)

    # 端末上の簡易 QC 表示。
    # detected / selected / waveform_pass_10 を見れば、最低限の成功可否をその場で確認できる。
    # QC print (similar to your previous output)
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
    raise SystemExit(main())
