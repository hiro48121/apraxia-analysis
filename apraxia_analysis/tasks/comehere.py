#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""apraxia_analysis/tasks/comehere.py

comehere タスク（おいでおいで動作）の解析エントリポイント。
共通処理は core モジュールから import し、comehere 固有ロジックのみをここに置く。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.math_utils import (
    angle_deg,
    pca_plane_deg,
    traj_len_px,
    max_speed_px_s,
    rolling_mean,
    speed_series_px_s,
    _odd,
    detect_index_outliers,
    _interpolate_with_gap_limit,
    apply_outlier_cleaning_2d,
    detect_onset_frame,
    detect_movement_segment,
    find_local_extrema_prom,
    build_cycles_from_extrema,
    _cycle_waveforms_from_y,
    _corr_to_mean_wave,
    _best_contiguous_block_by_waveform_then_cv,
)
from ..core.video_extractor import extract_pose_hand_px_from_video


# ─── comehere固有ユーティリティ ───────────────────────────────────────────────

def _cycle_time_stats(df: pd.DataFrame) -> tuple[float, float, float]:
    """Compute mean, SD, CV of cycle_time_s. Returns (mean, sd, cv) — NaN if insufficient data."""
    if df is None or len(df) == 0 or "cycle_time_s" not in df.columns:
        return np.nan, np.nan, np.nan
    t = df["cycle_time_s"].to_numpy(dtype=float)
    if not np.any(np.isfinite(t)):
        return np.nan, np.nan, np.nan
    t_mean = float(np.nanmean(t))
    t_sd = float(np.nanstd(t, ddof=1)) if np.sum(np.isfinite(t)) >= 2 else np.nan
    cv = float(t_sd / t_mean) if (np.isfinite(t_sd) and t_mean != 0) else np.nan
    return t_mean, t_sd, cv


def _add_cycle_means(meta: dict[str, Any], prefix: str, df: pd.DataFrame) -> None:
    """Append nanmean of each cycle metric to *meta* under ``{prefix}{key}_mean_over_cycles``."""
    if df is None or len(df) == 0:
        return
    for k in [
        "area_px2", "x_range_px", "y_range_px", "traj_len_px", "max_speed_px_s",
        "dist_range_px", "plane_deg",
        "shoulder_deg_range", "elbow_deg_range", "wrist_deg_range", "index_mcp_deg_range",
        "shoulder_deg_mean", "elbow_deg_mean", "wrist_deg_mean", "index_mcp_deg_mean",
    ]:
        if k in df.columns:
            meta[f"{prefix}{k}_mean_over_cycles"] = float(np.nanmean(df[k].to_numpy(dtype=float)))


def _detect_cycles_comehere(
    sig_smooth_for_peaks: np.ndarray,
    fps: float,
    search_start: int,
    search_end: int,
    target_n: int,
    min_sep_frames: int,
    prom_win_frames: int,
    min_prom_ratio: float,
    min_prom_px: float,
    min_amp_px: float,
    min_cycle_s: float,
    max_cycle_s: float,
    index_y_sm: np.ndarray,
    waveform_resample_n: int,
) -> dict[str, Any]:
    """Single-pass cycle detection + waveform-priority block selection for comehere.

    Selects the best contiguous block by waveform consistency first, then CV
    (``_best_contiguous_block_by_waveform_then_cv``). Returns a dict with keys:
    maxima_f, minima_f, cycles_all, selected_cycles, selected_cv,
    selected_window_start_idx, best_mean_corr, best_min_corr.
    """
    _empty: dict[str, Any] = {
        "maxima_f": [], "minima_f": [], "cycles_all": [],
        "selected_cycles": [], "selected_cv": np.nan, "selected_window_start_idx": 0,
        "best_mean_corr": np.nan, "best_min_corr": np.nan,
    }
    if search_end < search_start:
        return _empty

    maxima, minima = find_local_extrema_prom(
        sig_smooth_for_peaks,
        min_sep_frames=min_sep_frames,
        min_amp_px=float(min_amp_px),
        min_prom_ratio=float(min_prom_ratio),
        min_prom_px=float(min_prom_px),
        prom_win_frames=prom_win_frames,
    )

    maxima_f = [i for i in maxima if search_start <= i <= search_end]
    minima_f = [i for i in minima if search_start <= i <= search_end]

    cycles_all = build_cycles_from_extrema(
        sig_smooth_for_peaks,
        maxima_f,
        minima_f,
        start_search_frame=search_start,
        fps=fps,
        min_cycle_s=float(min_cycle_s),
        max_cycle_s=float(max_cycle_s),
    )

    # Select the best contiguous block of N cycles (波形の一貫性を優先し、同等ならCV最小)
    if len(cycles_all) >= target_n and target_n > 0:
        tmp_cycles = pd.DataFrame([
            {"start_frame": int(c["start_frame"]), "end_frame": int(c["end_frame"])}
            for c in cycles_all
        ])
        cycle_times = np.array(
            [(int(c["end_frame"]) - int(c["start_frame"])) / float(fps) for c in cycles_all],
            dtype=float,
        )
        waveforms_all = _cycle_waveforms_from_y(index_y_sm, tmp_cycles, waveform_resample_n)

        best_i, best_cv, best_mean_corr, best_min_corr = _best_contiguous_block_by_waveform_then_cv(
            cycle_times, waveforms_all, target_n
        )
        selected_window_start_idx = int(best_i)
        selected_cv = float(best_cv) if np.isfinite(best_cv) else np.nan
        selected_cycles = cycles_all[selected_window_start_idx : selected_window_start_idx + target_n]
    else:
        selected_cycles = cycles_all
        selected_cv = np.nan
        selected_window_start_idx = 0
        best_mean_corr, best_min_corr = np.nan, np.nan

    return {
        "maxima_f": maxima_f, "minima_f": minima_f, "cycles_all": cycles_all,
        "selected_cycles": selected_cycles, "selected_cv": selected_cv,
        "selected_window_start_idx": selected_window_start_idx,
        "best_mean_corr": best_mean_corr, "best_min_corr": best_min_corr,
    }


def _compute_waveform_qc(
    cycles_df: pd.DataFrame,
    signal_array: np.ndarray,
    waveform_resample_n: int,
    waveform_min_corr: float,
    target_cycles: int,
) -> tuple[int, float, float]:
    """Waveform similarity QC for the selected (selected10==1) cycle block.

    Modifies *cycles_df* in-place to write back ``wave_corr_to_mean10``.
    Returns (waveform_pass_10, wave_mean_corr_10, wave_min_corr_10).
    """
    n_sel = int((cycles_df.get("selected10", 0) == 1).sum()) if len(cycles_df) > 0 else 0
    waveform_pass_10 = 0
    wave_mean_corr_10 = np.nan
    wave_min_corr_10 = np.nan

    if n_sel > 0 and len(cycles_df) > 0:
        sel_df = cycles_df[cycles_df["selected10"] == 1].copy().sort_values("start_frame").reset_index(drop=True)
        waves = _cycle_waveforms_from_y(signal_array, sel_df, waveform_resample_n)
        corrs = _corr_to_mean_wave(waves)

        if len(corrs) == len(sel_df):
            sel_df["wave_corr_to_mean10"] = corrs
            for _, r in sel_df.iterrows():
                s0 = int(r["start_frame"]); e0 = int(r["end_frame"])
                m = (cycles_df["start_frame"] == s0) & (cycles_df["end_frame"] == e0)
                cycles_df.loc[m, "wave_corr_to_mean10"] = float(r["wave_corr_to_mean10"]) if np.isfinite(r["wave_corr_to_mean10"]) else np.nan

            if np.any(np.isfinite(corrs)):
                wave_mean_corr_10 = float(np.nanmean(corrs))
                wave_min_corr_10 = float(np.nanmin(corrs))

        if (n_sel == target_cycles) and (len(corrs) == n_sel) and np.all(np.isfinite(corrs)) and np.all(corrs >= float(waveform_min_corr)):
            waveform_pass_10 = 1
        else:
            waveform_pass_10 = 0

    return waveform_pass_10, wave_mean_corr_10, wave_min_corr_10


def _build_argparser_comehere() -> argparse.ArgumentParser:
    """Return the argument parser for the comehere task CLI."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--pose_model", required=True)
    ap.add_argument("--hand_model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--participant_id", required=True)
    ap.add_argument("--condition", default="comehere")
    ap.add_argument("--set_id", type=int, required=True)
    ap.add_argument("--trial_id", type=int, required=True)
    ap.add_argument("--cue_frame", type=int, default=0)
    ap.add_argument("--side", choices=["Left", "Right"], default="Left")

    # cycle params (aligned to byebye v2)
    ap.add_argument("--cycle_signal", choices=["dist", "dx", "dy"], default="dy",
                    help="Cycle detection signal. dist: hypot(index-shoulder), dx: index_x-shoulder_x, dy: index_y-shoulder_y")
    ap.add_argument("--smooth_sec", type=float, default=0.10)
    ap.add_argument("--start_search_sec", type=float, default=0.20)
    ap.add_argument("--cycle_search_use_trim", action="store_true",
                    help="Restrict cycle search to detected movement segment (trim_start_frame..trim_end_frame). "
                         "Default: search full video (less dependent on trimming).")

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

    return ap


# ─── 波形PNG出力 ─────────────────────────────────────────────────────────────

def save_waveform_png(frames_df: pd.DataFrame, out_dir: Path) -> Path:
    """Save representative index y waveform png (y axis in px). Uses cleaned series if available."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    stem = out_dir.name

    x = frames_df["time_s"] if "time_s" in frames_df.columns else (
        frames_df["t_s"] if "t_s" in frames_df.columns else frames_df.get("frame_idx", frames_df.index)
    )

    # Prefer representative index Y in px for comehere waveform.
    # With outlier handling (方式A), we prioritize the cleaned series so the saved waveform reflects outlier processing.
    if "index_y_px_clean" in frames_df.columns:
        y = frames_df["index_y_px_clean"]
        ylab = "index_y_px_clean"
    elif "index_y_px_sm" in frames_df.columns:
        y = frames_df["index_y_px_sm"]
        ylab = "index_y_px_sm"
    elif "index_y_px_raw" in frames_df.columns:
        y = frames_df["index_y_px_raw"]
        ylab = "index_y_px_raw"
    elif "index_y_px_sm_raw" in frames_df.columns:
        y = frames_df["index_y_px_sm_raw"]
        ylab = "index_y_px_sm_raw"
    elif "wrist_y_px_raw" in frames_df.columns:
        y = frames_df["wrist_y_px_raw"]
        ylab = "wrist_y_px_raw"
    elif "wrist_y_px_sm" in frames_df.columns:
        y = frames_df["wrist_y_px_sm"]
        ylab = "wrist_y_px_sm"
    else:
        y = frames_df.get("cycle_signal_smooth_px", frames_df.iloc[:, 0])
        ylab = "cycle_signal_smooth_px"

    plt.figure()
    plt.plot(x, y)
    plt.xlabel("time_s" if "time_s" in frames_df.columns else "frame_idx")
    plt.ylabel(ylab)
    plt.title(stem)

    png = out_dir / f"waveform_{stem}.png"
    plt.savefig(png, dpi=200, bbox_inches="tight")
    plt.close()
    return png


# ─── メインエントリポイント ───────────────────────────────────────────────────

def run_comehere(argv: list[str] | None = None) -> None:
    args = _build_argparser_comehere().parse_args(argv)

    video_path = Path(args.video).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_df, width, height, src_fps, fps = extract_pose_hand_px_from_video(
        video_path=video_path,
        pose_model_path=Path(args.pose_model).expanduser(),
        hand_model_path=Path(args.hand_model).expanduser(),
        side=args.side,
    )

    cue = int(args.cue_frame)

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

    # Raw & clean kinematics
    dx_raw = ix_raw - base_x
    dy_raw = iy_raw - base_y
    dist_raw = np.hypot(dx_raw, dy_raw)
    dist_smooth_raw = rolling_mean(dist_raw, 5)

    dx_clean = ix_clean - base_x
    dy_clean = iy_clean - base_y
    dist_clean = np.hypot(dx_clean, dy_clean)
    dist_smooth_clean = rolling_mean(dist_clean, 5)

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

    # speed (index tip)
    speed_index_raw = speed_series_px_s(ix_raw, iy_raw, fps)
    speed_index_clean_raw = speed_series_px_s(ix_clean, iy_clean, fps)
    speed_index_smooth_raw = rolling_mean(speed_index_raw, int(max(1, args.onset_speed_smooth_win)))
    speed_index_smooth = rolling_mean(speed_index_clean_raw, int(max(1, args.onset_speed_smooth_win)))

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

    # onset after cue based on wrist speed
    onset_frame, onset_thr = detect_onset_frame(
        speed_wrist_smooth,
        cue_frame=cue,
        baseline_frames=baseline_frames,
        k_mad=float(args.onset_k_mad),
        hold_frames=int(args.onset_hold_frames),
    )
    start_to_onset_s = float((onset_frame - cue) / float(fps)) if onset_frame is not None else np.nan

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

    # cycle detection
    min_sep_frames = int(round(float(args.min_sep_s) * float(fps)))
    prom_win_frames = int(max(3, round(float(args.prom_win_s) * float(fps))))
    sig_smooth_for_peaks = _interpolate_with_gap_limit(sig_smooth, max_gap_frames=max_gap_frames)

    start_search_frame0 = int(cue + float(args.start_search_sec) * float(fps))
    search_end0 = int(len(sig_smooth_for_peaks) - 1)

    # Cycle search window (less dependent on trimming by default)
    search_start = int(start_search_frame0)
    search_end = int(search_end0)
    cycle_search_used_trim = 0
    if bool(getattr(args, 'cycle_search_use_trim', False)):
        if trim_start_frame is not None and np.isfinite(trim_start_frame):
            search_start = max(search_start, int(trim_start_frame))
        if trim_end_frame is not None and np.isfinite(trim_end_frame):
            search_end = min(search_end, int(trim_end_frame))
        cycle_search_used_trim = 1

    # index_y_sm is needed for waveform-based block selection inside _detect_cycles_comehere
    win_wrist = _odd(int(round(float(args.smooth_sec) * float(fps))))
    index_y_sm = rolling_mean(iy_clean, win_wrist)

    det = _detect_cycles_comehere(
        sig_smooth_for_peaks=sig_smooth_for_peaks,
        fps=fps,
        search_start=search_start,
        search_end=search_end,
        target_n=int(args.target_cycles),
        min_sep_frames=min_sep_frames,
        prom_win_frames=prom_win_frames,
        min_prom_ratio=float(args.min_prom_ratio),
        min_prom_px=float(args.min_prom_px),
        min_amp_px=float(args.min_amp_px),
        min_cycle_s=float(args.min_cycle_s),
        max_cycle_s=float(args.max_cycle_s),
        index_y_sm=index_y_sm,
        waveform_resample_n=int(args.waveform_resample_n),
    )
    cycles_all = det["cycles_all"]
    selected_cycles = det["selected_cycles"]
    selected_cv = det["selected_cv"]
    selected_window_start_idx = det["selected_window_start_idx"]
    best_mean_corr = det["best_mean_corr"]
    best_min_corr = det["best_min_corr"]

    # -------------------------
    # frames.csv (always saved)
    # -------------------------
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

    frames["wrist_x_px_sm"] = rolling_mean(wrist_x, win_wrist)
    frames["wrist_y_px_sm"] = rolling_mean(wrist_y, win_wrist)

    # smoothed index (raw & clean)
    frames["index_x_px_sm_raw"] = rolling_mean(ix_raw, win_wrist)
    frames["index_y_px_sm_raw"] = rolling_mean(iy_raw, win_wrist)
    frames["index_x_px_sm"] = rolling_mean(ix_clean, win_wrist)
    frames["index_y_px_sm"] = index_y_sm

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
    # -------------------------
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

        row: dict[str, Any] = {
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
    # -------------------------
    n_sel = int((cycles_df.get("selected10", 0) == 1).sum()) if len(cycles_df) > 0 else 0
    waveform_pass_10, wave_mean_corr_10, wave_min_corr_10 = _compute_waveform_qc(
        cycles_df=cycles_df,
        signal_array=frames["index_y_px_sm"].to_numpy(dtype=float),
        waveform_resample_n=int(args.waveform_resample_n),
        waveform_min_corr=float(args.waveform_min_corr),
        target_cycles=int(args.target_cycles),
    )

    # -------------------------
    # summary.csv (always saved)
    # -------------------------
    sel_block_df = cycles_df[cycles_df["selected10"] == 1].copy() if ("selected10" in cycles_df.columns) else pd.DataFrame()
    t_mean_sel, t_sd_sel, rhythm_cv_sel = _cycle_time_stats(sel_block_df)
    t_mean_all, t_sd_all, rhythm_cv_all = _cycle_time_stats(cycles_df)
    _out_n = int(np.sum(outlier_flag == 1))
    _reasons = pd.Series(outlier_reason).astype(str)
    _out_n_jump = int(_reasons.str.contains("jump", regex=False).sum())
    _out_n_hand_pose = int(_reasons.str.contains("hand_pose", regex=False).sum())

    meta: dict[str, Any] = {
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
        # QC values for the selected 10-cycle window (not selection criteria).
        "selected_window_wave_mean_corr": float(best_mean_corr) if np.isfinite(best_mean_corr) else np.nan,
        "selected_window_wave_min_corr": float(best_min_corr) if np.isfinite(best_min_corr) else np.nan,
        "selected_window_start_index": int(selected_window_start_idx),
        "cycle_search_start_frame": int(search_start) if np.isfinite(search_start) else np.nan,
        "cycle_search_end_frame": int(search_end) if np.isfinite(search_end) else np.nan,
        "cycle_search_used_trim": int(cycle_search_used_trim),
        "trim_start_frame": int(trim_start_frame) if (trim_start_frame is not None and np.isfinite(trim_start_frame)) else np.nan,
        "trim_end_frame": int(trim_end_frame) if (trim_end_frame is not None and np.isfinite(trim_end_frame)) else np.nan,
        "trim_thr_px_s": float(trim_thr) if np.isfinite(trim_thr) else np.nan,
        "trim_method": trim_method,
        "start_to_onset_s": start_to_onset_s,
        "onset_frame": int(onset_frame) if onset_frame is not None else np.nan,
        "onset_thr_px_s": float(onset_thr) if np.isfinite(onset_thr) else np.nan,

        "cycle_time_mean_s_selected10": t_mean_sel,
        "cycle_time_sd_s_selected10": t_sd_sel,
        "rhythm_cv_selected10": rhythm_cv_sel,
        "cycle_time_mean_s_all": t_mean_all,
        "cycle_time_sd_s_all": t_sd_all,
        "rhythm_cv_all": rhythm_cv_all,

        "waveform_resample_n": int(args.waveform_resample_n),
        "waveform_min_corr_th": float(args.waveform_min_corr),
        # QC values for waveform similarity in the selected 10-cycle window (not selection criteria).
        "waveform_mean_corr_10": wave_mean_corr_10,
        "waveform_min_corr_10": wave_min_corr_10,
        "waveform_pass_10": int(waveform_pass_10),
    }

    if n_sel > 0:
        _add_cycle_means(meta, "selected10_", sel_block_df)
    else:
        _add_cycle_means(meta, "all_", cycles_df)

    summary_df = pd.DataFrame([meta])

    # -------------------------
    # write outputs
    # -------------------------
    frames.to_csv(out_dir / "frames.csv", index=False)
    cycles_df.to_csv(out_dir / "cycles.csv", index=False)
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    png = save_waveform_png(frames, out_dir)

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
    import sys
    import traceback
    import faulthandler
    faulthandler.enable()
    try:
        run_comehere()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
