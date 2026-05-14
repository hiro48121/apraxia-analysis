#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apraxia_app.py  —  Apraxia Analysis Desktop App

起動方法 (ターミナル):
  python apraxia_app.py

起動方法 (ダブルクリック):
  macOS:   apraxia_app.command をダブルクリック
  Windows: apraxia_app.bat     をダブルクリック

注意:
  - このファイルは apraxia_analysis/ フォルダと同じ場所に置いてください
  - 解析数値はタスクモジュール側で計算します（このファイルは変更不要）
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

# ─────────────────────────────────────────────────────────────
#  パス設定（環境に合わせて変更してください）
# ─────────────────────────────────────────────────────────────

# .venv の python インタープリタ
PYTHON = sys.executable

# このファイルがある場所 = apraxia_analysis/ の親ディレクトリ
APP_DIR = Path(__file__).parent

# 設定の保存先（ホームディレクトリに隠しファイル）
CONFIG_PATH = Path.home() / ".apraxia_app_config.json"

# ─────────────────────────────────────────────────────────────
#  定数
# ─────────────────────────────────────────────────────────────

TASKS = ["hammer", "byebye", "comehere"]
SIDES = ["Left", "Right"]

# 波形連動表示で使用する列（タスクごとに既存 waveform PNG と同じ列を使用）
WAVEFORM_COL = {
    "hammer":   "wrist_y_px_sm",
    "comehere": "index_y_px_sm",
    "byebye":   "index_x_px_sm",
}

# フォント（OS別：Windows は日本語対応フォントを明示指定）
if sys.platform == "win32":
    FONT_UI   = "Yu Gothic UI"   # Windows 8.1+
    FONT_MONO = "MS Gothic"      # 日本語対応等幅フォント
else:
    FONT_UI   = "Helvetica Neue"
    FONT_MONO = "Courier"

# summary.csv のうち画面に表示する列（タスク共通 + タスク別）
SUMMARY_KEYS_COMMON = [
    "participant_id", "task", "set_id", "trial_id", "side",
    "n_frames", "src_fps",
    "n_cycles_detected", "n_cycles_selected10",
    "waveform_pass_10",
]
SUMMARY_KEYS_HAMMER = [
    "start_to_onset_s",
    "cycle_time_mean_s_selected10", "cycle_time_sd_s_selected10",
    # rhythm_cv_selected10 は hammer では使用しないため表示対象外
    "hit_time_mean_s", "lift_time_mean_s",
    "direction_deg_abs_mean",
    "traj_len_px_mean",
    "vmax_px_s_mean",
    "shoulder_deg_range_mean", "elbow_deg_range_mean", "wrist_deg_range_mean",
    "shoulder_deg_mean_mean", "elbow_deg_mean_mean", "wrist_deg_mean_mean",
    "waveform_mean_corr_10", "waveform_min_corr_10",
]
SUMMARY_KEYS_BYEBYE_COMEHERE = [
    "start_to_onset_s",
    "cycle_time_mean_s_selected10", "cycle_time_sd_s_selected10",
    "rhythm_cv_selected10",
    "selected10_area_px2_mean_over_cycles",
    "selected10_traj_len_px_mean_over_cycles",
    "selected10_max_speed_px_s_mean_over_cycles",
    "shoulder_deg_range_mean", "elbow_deg_range_mean", "wrist_deg_range_mean",
    "index_mcp_deg_range_mean",
    "shoulder_deg_mean_mean", "elbow_deg_mean_mean", "wrist_deg_mean_mean",
    "index_mcp_deg_mean_mean",
    "waveform_mean_corr_10", "waveform_min_corr_10",
]
SUMMARY_KEYS_CENTRAL5 = [
    "central5_available",
    "n_cycles_central5",
    "qc_cycle_count_warning",
    "cycle_time_mean_s_central5", "cycle_time_sd_s_central5",
    "rhythm_cv_central5",
    "amp_mean_px_central5",      "amp_sd_px_central5",
    "traj_len_mean_px_central5", "traj_len_sd_px_central5",
    "max_speed_mean_px_s_central5", "max_speed_sd_px_s_central5",
]

# ─────────────────────────────────────────────────────────────
#  解析結果サマリ 日本語ラベル辞書
# ─────────────────────────────────────────────────────────────

LABEL_COMMON = {
    "participant_id":      "参加者ID",
    "task":                "タスク",
    "set_id":              "セットID",
    "trial_id":            "試行ID",
    "side":                "側",
    "n_frames":            "フレーム数",
    "src_fps":             "フレームレート",
    "n_cycles_detected":   "検出サイクル数",
    "n_cycles_selected10": "選択サイクル数",
    "waveform_pass_10":    "波形一致判定",
}

LABEL_HAMMER = {
    "start_to_onset_s":             "開始までの時間",
    "cycle_time_mean_s_selected10": "サイクル時間平均",
    "cycle_time_sd_s_selected10":   "サイクル時間標準偏差",
    "hit_time_mean_s":              "打撃時間平均",
    "lift_time_mean_s":             "振り上げ時間平均",
    "direction_deg_abs_mean":       "方向角平均",
    "traj_len_px_mean":             "軌道長平均",
    "vmax_px_s_mean":               "最大速度平均",
    "shoulder_deg_range_mean":      "肩関節可動域平均",
    "elbow_deg_range_mean":         "肘関節可動域平均",
    "wrist_deg_range_mean":         "手関節可動域平均",
    "shoulder_deg_mean_mean":       "肩関節平均角",
    "elbow_deg_mean_mean":          "肘関節平均角",
    "wrist_deg_mean_mean":          "手関節平均角",
    "waveform_mean_corr_10":        "波形相関平均",
    "waveform_min_corr_10":         "波形相関最小値",
}

LABEL_BYEBYE_COMEHERE = {
    "start_to_onset_s":                          "開始までの時間",
    "cycle_time_mean_s_selected10":              "サイクル時間平均",
    "cycle_time_sd_s_selected10":                "サイクル時間標準偏差",
    "rhythm_cv_selected10":                      "リズム変動係数",
    "selected10_area_px2_mean_over_cycles":      "運動面積平均",
    "selected10_traj_len_px_mean_over_cycles":   "軌道長平均",
    "selected10_max_speed_px_s_mean_over_cycles":"最大速度平均",
    "shoulder_deg_range_mean":                   "肩関節可動域平均",
    "elbow_deg_range_mean":                      "肘関節可動域平均",
    "wrist_deg_range_mean":                      "手関節可動域平均",
    "index_mcp_deg_range_mean":                  "手指MP関節可動域平均",
    "shoulder_deg_mean_mean":                    "肩関節平均角",
    "elbow_deg_mean_mean":                       "肘関節平均角",
    "wrist_deg_mean_mean":                       "手関節平均角",
    "index_mcp_deg_mean_mean":                   "手指MP関節平均角",
    "waveform_mean_corr_10":                     "波形相関平均",
    "waveform_min_corr_10":                      "波形相関最小値",
}
LABEL_CENTRAL5 = {
    "central5_available":          "中央5サイクル利用可",
    "n_cycles_central5":           "中央5サイクル数",
    "qc_cycle_count_warning":      "サイクル数確認フラグ",
    "cycle_time_mean_s_central5":  "中央5サイクル時間平均",
    "cycle_time_sd_s_central5":    "中央5サイクル時間標準偏差",
    "rhythm_cv_central5":          "中央5リズム変動係数",
    "amp_mean_px_central5":        "中央5振幅平均",
    "amp_sd_px_central5":          "中央5振幅標準偏差",
    "traj_len_mean_px_central5":   "中央5軌道長平均",
    "traj_len_sd_px_central5":     "中央5軌道長標準偏差",
    "max_speed_mean_px_s_central5":"中央5最大速度平均",
    "max_speed_sd_px_s_central5":  "中央5最大速度標準偏差",
}

# ─────────────────────────────────────────────────────────────
#  設定の保存・読み込み
# ─────────────────────────────────────────────────────────────

def _load_cfg() -> dict:
    try:
        if CONFIG_PATH.exists():
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_cfg(cfg: dict) -> None:
    try:
        CONFIG_PATH.write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
#  メインウィンドウ
# ─────────────────────────────────────────────────────────────

class ApraxiaApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Apraxia Analysis")
        self.geometry("1150x740")
        self.minsize(900, 620)

        self._cfg = _load_cfg()
        self._video_path: str = ""
        self._running = False

        # ── オーバーレイ動画生成状態 ──
        self._last_trial_out: str = ""
        self._last_task: str = ""
        self._overlay_running: bool = False

        # ── スクリーンショット保存 ──
        self._screenshot_btn = None

        # ── 解析前チェック ──
        self._precheck_summary_label = None
        self._precheck_detail_btn = None
        self._precheck_results: list = []
        self._video_is_hevc = None   # True=HEVC確定/疑い, False=非HEVC, None=未判定

        # ── 動画プレーヤー状態 ──
        self._player_cap = None          # cv2.VideoCapture（再生中は開いたまま）
        self._player_total_frames: int = 0
        self._player_fps: float = 30.0
        self._player_current_frame: int = 0
        self._player_playing: bool = False
        self._player_after_id = None
        self._player_slider_busy: bool = False

        # ── 波形連動表示状態 ──
        self._waveform_fig = None          # matplotlib Figure
        self._waveform_ax = None           # matplotlib Axes
        self._waveform_canvas = None       # FigureCanvasTkAgg
        self._waveform_cursor_line = None  # 縦線カーソル（Line2D）
        self._waveform_time_arr: list = [] # time_s の値リスト
        self._waveform_loaded: bool = False
        self._waveform_col: str = ""       # 使用中の列名

        self._build_ui()
        self._restore_values()
        self._apply_default_models()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ──────────────────────────────────────────────────────────
    #  UI 構築
    # ──────────────────────────────────────────────────────────

    def _build_ui(self):
        BG       = "#f5f2ee"
        SURFACE  = "#ffffff"
        SURFACE2 = "#e8e4de"
        ACCENT   = "#2d5a8e"
        HEADER   = "#1a1612"

        self.configure(bg=BG)

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame",         background=BG)
        style.configure("Surface.TFrame", background=SURFACE)
        style.configure("TLabel",         background=BG, font=(FONT_UI, 13))
        style.configure("TButton",        font=(FONT_UI, 13), padding=5)
        style.configure("TRadiobutton",   background=BG, font=(FONT_UI, 13))
        style.configure("TEntry",         font=(FONT_UI, 13), padding=4)
        style.configure("TLabelframe",    background=BG)
        style.configure("TLabelframe.Label",
                        background=BG, font=(FONT_UI, 12, "bold"))

        # アクセントボタン
        style.configure("Accent.TButton",
                        background=ACCENT, foreground="white",
                        font=(FONT_UI, 14, "bold"), padding=10)
        style.map("Accent.TButton",
                  background=[("active", "#1e4070"), ("disabled", "#9ab0cc")],
                  foreground=[("disabled", "#e0e8f4")])

        # ── ヘッダー ──
        header = tk.Frame(self, bg=HEADER, height=50)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text="APRAXIA ANALYSIS", bg=HEADER, fg="#f5f2ee",
                 font=(FONT_MONO, 15, "bold")).pack(side="left", padx=22, pady=12)
        tk.Label(header, text="v1.0", bg="#2a2a2a", fg="#888",
                 font=(FONT_MONO, 10)).pack(side="left", pady=12)

        # ── メインレイアウト ──
        main = ttk.Frame(self, padding=12)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        # 左パネル
        left = ttk.Frame(main, style="Surface.TFrame", padding=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self._build_left(left, SURFACE, SURFACE2, ACCENT)

        # 右パネル（スクロール可能）
        right_outer = ttk.Frame(main)
        right_outer.grid(row=0, column=1, sticky="nsew")
        right_outer.rowconfigure(0, weight=1)
        right_outer.columnconfigure(0, weight=1)

        canvas = tk.Canvas(right_outer, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(right_outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

        right = ttk.Frame(canvas, padding=4)
        win_id = canvas.create_window((0, 0), window=right, anchor="nw")

        right.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win_id, width=e.width))
        # マウスホイールでスクロール（Windows/Mac: MouseWheel、Linux: Button-4/5）
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-1 * e.delta / 120), "units"))
        canvas.bind_all("<Button-4>",
                        lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>",
                        lambda e: canvas.yview_scroll(1, "units"))

        self._build_right(right, BG, ACCENT)

    # ── 左パネル ──────────────────────────────────────────────

    def _build_left(self, parent, surface, surface2, accent):
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        # 動画選択エリア
        drop_frame = tk.Frame(parent, bg=surface2, relief="flat", bd=0)
        drop_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self._video_label = tk.Label(
            drop_frame,
            text="動画ファイルを選択してください\n（MOV / MP4 / AVI）",
            bg=surface2, fg="#6b5f52",
            font=(FONT_UI, 13),
            pady=16, wraplength=400,
        )
        self._video_label.pack(side="left", padx=14, expand=True)
        ttk.Button(drop_frame, text="ファイルを選択",
                   command=self._select_video).pack(side="right", padx=10, pady=10)

        # 波形プレビューエリア
        self._preview_frame = tk.Frame(parent, bg=surface, bd=0)
        self._preview_frame.grid(row=1, column=0, sticky="nsew")

        self._preview_label = tk.Label(
            self._preview_frame, bg=surface, fg="#9e9088",
            text="解析後にここに波形グラフが表示されます",
            font=(FONT_UI, 12),
        )
        self._preview_label.pack(expand=True, fill="both")

        self._video_info_label = tk.Label(
            self._preview_frame, bg=surface, fg="#555555",
            text="", font=(FONT_UI, 11),
        )
        self._video_info_label.pack(side="bottom", pady=(0, 4))

        # プレーヤーコントロール
        self._build_player_controls(parent, surface, surface2)

        # 波形連動表示エリア
        self._build_waveform_area(parent, surface, surface2)

        # ログ
        log_lf = ttk.LabelFrame(parent, text="ログ", padding=4)
        log_lf.grid(row=4, column=0, sticky="ew", pady=(8, 0))

        self._log = scrolledtext.ScrolledText(
            log_lf, height=8,
            font=(FONT_MONO, 11),
            bg="#1a1612", fg="#d0ccc6",
            insertbackground="white",
            relief="flat", state="disabled",
        )
        self._log.pack(fill="both", expand=True)

    # ── 右パネル ──────────────────────────────────────────────

    def _build_right(self, parent, bg, accent):

        # ── タスク選択 ──
        task_lf = ttk.LabelFrame(parent, text="タスク", padding=8)
        task_lf.pack(fill="x", pady=(0, 6))
        self._task_var = tk.StringVar(value="hammer")
        for t in TASKS:
            ttk.Radiobutton(task_lf, text=t, value=t,
                            variable=self._task_var).pack(side="left", padx=10)

        # ── 側（Side）──
        side_lf = ttk.LabelFrame(parent, text="側（Side）", padding=8)
        side_lf.pack(fill="x", pady=(0, 6))
        self._side_var = tk.StringVar(value="Left")
        for s in SIDES:
            ttk.Radiobutton(side_lf, text=s, value=s,
                            variable=self._side_var).pack(side="left", padx=10)

        # ── 患者情報 ──
        info_lf = ttk.LabelFrame(parent, text="患者情報", padding=8)
        info_lf.pack(fill="x", pady=(0, 6))
        self._pid_var   = tk.StringVar()
        self._set_var   = tk.StringVar(value="1")
        self._trial_var = tk.StringVar(value="1")
        self._cue_var   = tk.StringVar(value="0")

        for label, var in [
            ("参加者ID",   self._pid_var),
            ("セットID",   self._set_var),
            ("試行ID",     self._trial_var),
            ("Cueフレーム", self._cue_var),
        ]:
            row = ttk.Frame(info_lf)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=13).pack(side="left")
            ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)

        ttk.Label(
            info_lf,
            text="※ 本研究では動画開始時点をCueフレームとするため、通常は0のまま使用してください。",
            font=(FONT_UI, 10),
            foreground="#6b5f52",
            wraplength=220,
            justify="left",
        ).pack(anchor="w", pady=(4, 0))

        # ── モデルファイル ──
        model_lf = ttk.LabelFrame(parent, text="モデルファイル", padding=8)
        model_lf.pack(fill="x", pady=(0, 6))
        self._pose_model_var = tk.StringVar()
        self._hand_model_var = tk.StringVar()

        for label, var in [
            ("Pose (.task)",            self._pose_model_var),
            ("Hand (.task)\nbyebye/comehere用", self._hand_model_var),
        ]:
            row = ttk.Frame(model_lf)
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=label, width=20, wraplength=130,
                      justify="left").pack(side="left")
            ttk.Entry(row, textvariable=var).pack(
                side="left", fill="x", expand=True, padx=(4, 0))
            ttk.Button(
                row, text="…", width=3,
                command=lambda v=var: self._pick_file(
                    v, [("Task files", "*.task"), ("All", "*.*")])
            ).pack(side="left", padx=(2, 0))

        # ── 出力先フォルダ ──
        out_lf = ttk.LabelFrame(parent, text="出力先フォルダ", padding=8)
        out_lf.pack(fill="x", pady=(0, 6))
        self._out_dir_var = tk.StringVar(
            value=str(APP_DIR / "results"))
        row = ttk.Frame(out_lf)
        row.pack(fill="x")
        ttk.Entry(row, textvariable=self._out_dir_var).pack(
            side="left", fill="x", expand=True)
        ttk.Button(row, text="…", width=3,
                   command=lambda: self._pick_dir(self._out_dir_var)).pack(
                       side="left", padx=(2, 0))

        # ── 解析開始ボタン ──
        self._analyze_btn = ttk.Button(
            parent, text="解析開始 ▶", style="Accent.TButton",
            command=self._start_analysis,
        )
        self._analyze_btn.pack(fill="x", pady=(6, 8))

        # ── 解析前チェック 要約 ──
        precheck_row = ttk.Frame(parent)
        precheck_row.pack(fill="x", pady=(0, 4))
        self._precheck_summary_label = ttk.Label(
            precheck_row,
            text="解析前チェック：未実行",
            font=(FONT_UI, 11),
            foreground="#888888",
        )
        self._precheck_summary_label.pack(side="left", padx=(2, 6))
        self._precheck_detail_btn = ttk.Button(
            precheck_row, text="詳細", width=5,
            command=lambda: self._show_pre_check_detail(self._precheck_results),
            state="disabled",
        )
        self._precheck_detail_btn.pack(side="left")

        # ── オーバーレイ動画作成ボタン ──
        self._overlay_btn = ttk.Button(
            parent, text="オーバーレイ動画を作成",
            command=self._start_overlay,
        )
        self._overlay_btn.pack(fill="x", pady=(0, 8))

        # ── スクリーンショット保存ボタン ──
        self._screenshot_btn = ttk.Button(
            parent, text="スクリーンショット保存",
            command=self._save_screenshot,
            state="disabled",
        )
        self._screenshot_btn.pack(fill="x", pady=(0, 8))

        # ── 解析結果 ──
        res_lf = ttk.LabelFrame(parent, text="解析結果サマリ", padding=8)
        res_lf.pack(fill="x", pady=(0, 4))
        self._result_text = tk.Text(
            res_lf, height=16,
            font=(FONT_MONO, 11),
            bg="#f0ece6", fg="#1a1612",
            relief="flat", state="disabled", wrap="none",
        )
        self._result_text.pack(fill="both", expand=True)

    # ──────────────────────────────────────────────────────────
    #  ファイル・ディレクトリ選択
    # ──────────────────────────────────────────────────────────

    def _select_video(self):
        path = filedialog.askopenfilename(
            title="動画ファイルを選択",
            filetypes=[
                ("動画ファイル", "*.mov *.mp4 *.avi *.MOV *.MP4 *.AVI"),
                ("すべて", "*.*"),
            ],
        )
        if path:
            self._player_reset()
            self._video_path = path
            self._video_label.config(
                text=f"選択済: {Path(path).name}",
                fg="#1a1612",
            )
            self._preview_label.config(image="", text="動画情報を読み込み中...", fg="#9e9088")
            self._preview_label.image = None
            self._video_info_label.config(text="", fg="#555555")
            threading.Thread(
                target=self._load_video_preview,
                args=(path,),
                daemon=True,
            ).start()

    def _pick_file(self, var: tk.StringVar, filetypes=None):
        path = filedialog.askopenfilename(
            filetypes=filetypes or [("All", "*.*")])
        if path:
            var.set(path)

    def _pick_dir(self, var: tk.StringVar):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    # ──────────────────────────────────────────────────────────
    #  ログ・結果表示
    # ──────────────────────────────────────────────────────────

    def _log_write(self, text: str):
        self._log.configure(state="normal")
        self._log.insert("end", text + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _show_result(self, text: str):
        self._result_text.configure(state="normal")
        self._result_text.delete("1.0", "end")
        self._result_text.insert("1.0", text)
        self._result_text.configure(state="disabled")

    def _show_waveform(self, png_path: str):
        """解析後の波形PNGをプレビューに表示する。"""
        try:
            from PIL import Image, ImageTk  # type: ignore

            img = Image.open(png_path)
            # プレビューエリアのサイズに合わせてリサイズ
            self._preview_frame.update_idletasks()
            w = max(200, self._preview_frame.winfo_width() - 20)
            h = max(150, self._preview_frame.winfo_height() - 20)
            img.thumbnail((w, h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._preview_label.config(image=photo, text="")
            self._preview_label.image = photo  # GC防止
        except ImportError:
            self._preview_label.config(
                text=f"波形PNG保存済み:\n{png_path}\n\n(Pillow未インストールのためプレビュー不可)")
        except Exception as e:
            self._preview_label.config(text=f"プレビューエラー:\n{e}")

    # ──────────────────────────────────────────────────────────
    #  解析実行
    # ──────────────────────────────────────────────────────────

    def _confirm_cue_frame(self, cue_val: int) -> str:
        """Cueフレームが0以外のとき確認ダイアログを表示し、結果を返す。

        Returns: "proceed" / "reset" / "cancel"
        """
        result = {"value": "cancel"}

        dlg = tk.Toplevel(self)
        dlg.title("Cueフレームの確認")
        dlg.resizable(False, False)
        dlg.grab_set()

        msg = (
            f"Cueフレームが0以外に設定されています。\n\n"
            f"本研究では通常 cue_frame = 0 を使用します。\n"
            f"このまま解析を実行しますか？\n\n"
            f"現在のCueフレーム：{cue_val}"
        )
        ttk.Label(dlg, text=msg, padding=16, wraplength=320, justify="left").pack()

        btn_frame = ttk.Frame(dlg, padding=(12, 0, 12, 12))
        btn_frame.pack()

        def on_proceed():
            result["value"] = "proceed"
            dlg.destroy()

        def on_reset():
            result["value"] = "reset"
            dlg.destroy()

        def on_cancel():
            result["value"] = "cancel"
            dlg.destroy()

        ttk.Button(btn_frame, text="このまま解析する", command=on_proceed).pack(
            side="left", padx=4)
        ttk.Button(btn_frame, text="0に戻す", command=on_reset).pack(
            side="left", padx=4)
        ttk.Button(btn_frame, text="キャンセル", command=on_cancel).pack(
            side="left", padx=4)

        dlg.update_idletasks()
        w = dlg.winfo_reqwidth()
        h = dlg.winfo_reqheight()
        x = self.winfo_rootx() + (self.winfo_width() - w) // 2
        y = self.winfo_rooty() + (self.winfo_height() - h) // 2
        dlg.geometry(f"+{x}+{y}")

        self.wait_window(dlg)
        return result["value"]

    def _start_analysis(self):
        if self._running:
            messagebox.showinfo("解析中", "現在解析中です。完了をお待ちください。")
            return

        self._player_stop()

        # ── 解析前チェック ──
        results = self._run_pre_check()
        self._precheck_results = results
        self._update_pre_check_label(results)
        if any(r["status"] == "stop" for r in results):
            self._show_pre_check_detail(results)
            return

        # ── バリデーション ──
        if not self._video_path:
            messagebox.showerror("エラー", "動画ファイルを選択してください。")
            return
        if not Path(self._video_path).exists():
            messagebox.showerror("エラー", f"動画ファイルが見つかりません:\n{self._video_path}")
            return
        if not self._pose_model_var.get():
            messagebox.showerror("エラー", "Poseモデル (.task) のパスを設定してください。")
            return
        if not Path(self._pose_model_var.get()).exists():
            messagebox.showerror("エラー",
                                 f"Poseモデルが見つかりません:\n{self._pose_model_var.get()}")
            return

        task = self._task_var.get()
        if task in ("byebye", "comehere"):
            if not self._hand_model_var.get():
                messagebox.showerror("エラー",
                                     f"{task} タスクには Hand モデル (.task) が必要です。")
                return
            if not Path(self._hand_model_var.get()).exists():
                messagebox.showerror("エラー",
                                     f"Handモデルが見つかりません:\n{self._hand_model_var.get()}")
                return

        # ── Windows：非ASCIIパスの事前チェック ──
        if sys.platform == "win32":
            checks = [
                ("動画ファイル",   self._video_path),
                ("Poseモデル",     self._pose_model_var.get()),
                ("出力先フォルダ", self._out_dir_var.get()),
                ("アプリフォルダ", str(APP_DIR)),
            ]
            if task in ("byebye", "comehere"):
                checks.append(("Handモデル", self._hand_model_var.get()))
            problem = [f"・{label}:\n  {path}"
                       for label, path in checks if path and not path.isascii()]
            if problem:
                messagebox.showerror(
                    "パスエラー",
                    "以下のパスに日本語や全角文字が含まれています。\n"
                    "Windows環境ではMediaPipeが正しく動作しない場合があります。\n\n"
                    + "\n".join(problem) +
                    "\n\n英数字のみのパスに変更してから再度お試しください。"
                )
                return

        # ── Cueフレーム確認ダイアログ ──
        cue_raw = self._cue_var.get().strip()
        try:
            cue_val = int(cue_raw)
        except ValueError:
            cue_val = 0
        if cue_val != 0:
            action = self._confirm_cue_frame(cue_val)
            if action == "cancel":
                self._log_write("Cueフレーム確認により解析をキャンセルしました。")
                return
            if action == "reset":
                self._cue_var.set("0")
                self._log_write("Cueフレームを0に戻して解析を開始しました。")

        pid = self._pid_var.get().strip() or "noname"

        # ── 出力フォルダを決定 ──
        base_out = Path(self._out_dir_var.get())
        set_id   = self._set_var.get().strip()   or "1"
        trial_id = self._trial_var.get().strip() or "1"
        trial_out = base_out / pid / task / f"set{set_id}_trial{trial_id}"
        try:
            trial_out.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            messagebox.showerror("エラー", f"出力フォルダを作成できません:\n{e}")
            return

        self._save_config()
        self._running = True
        self._analyze_btn.configure(state="disabled", text="解析中…")
        self._show_result("")
        self._log_write(f"=== 解析開始 ===")
        self._log_write(f"タスク    : {task}")
        self._log_write(f"動画      : {Path(self._video_path).name}")
        self._log_write(f"出力先    : {trial_out}")
        self._log_write("")

        threading.Thread(
            target=self._run_analysis,
            args=(task, pid, set_id, trial_id, str(trial_out)),
            daemon=True,
        ).start()

    # ──────────────────────────────────────────────────────────
    #  動画プレビュー（選択直後）
    # ──────────────────────────────────────────────────────────

    def _open_cap_with_timeout(self, video_path: str, timeout: float = 4.0):
        """cv2.VideoCapture をタイムアウト付きで開く。
        戻り値: (cap, opened, timed_out)"""
        result = {"cap": None, "opened": False}

        def _open():
            import cv2
            cap = cv2.VideoCapture(video_path)
            opened = cap.isOpened()
            if opened:
                ret, _ = cap.read()
                opened = ret
                if ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            result["cap"] = cap
            result["opened"] = opened

        t = threading.Thread(target=_open, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            return None, False, True
        return result["cap"], result["opened"], False

    def _load_video_preview(self, video_path: str):
        """動画選択直後にプレビューフレームと動画情報を表示する（バックグラウンドスレッド）。"""
        self._video_is_hevc = None   # 判定開始前にリセット
        # PyAV でコーデックを直接確認（インストール済みの場合）
        is_hevc = False
        try:
            import av as _av_check
            with _av_check.open(video_path) as _f:
                is_hevc = _f.streams.video[0].codec_context.name.lower() in (
                    "hevc", "h265", "hvc1")
        except Exception:
            pass

        if is_hevc:
            self._video_is_hevc = True
            msg = "⚠ HEVC（H.265）形式です\n解析時に自動的にH.264へ変換されます"
            self.after(0, lambda: self._video_info_label.config(text=msg, fg="#b85c00"))
            self.after(0, lambda: self._preview_label.config(
                text="プレビュー不可（HEVC形式）", fg="#9e9088"))
            return

        cap, opened, timed_out = self._open_cap_with_timeout(video_path, timeout=8.0)

        if timed_out or not opened:
            if cap:
                try:
                    cap.release()
                except Exception:
                    pass
            self._video_is_hevc = True
            msg = "⚠ HEVC（H.265）形式の可能性があります\n解析時に自動的にH.264へ変換されます"
            self.after(0, lambda: self._video_info_label.config(text=msg, fg="#b85c00"))
            self.after(0, lambda: self._preview_label.config(
                text="プレビュー不可（HEVC形式）", fg="#9e9088"))
            return

        self._video_is_hevc = False
        import cv2
        width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps      = cap.get(cv2.CAP_PROP_FPS)
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = n_frames / fps if fps > 0 else 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        cap.release()

        info = f"{width}×{height}  /  {fps:.1f} fps  /  {duration:.1f} 秒"
        self.after(0, lambda: self._video_info_label.config(text=info, fg="#444444"))

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.after(0, lambda f=frame_rgb: self._display_frame_preview(f))
            # プレーヤーを初期化（既にOpenCVで読めることが確認済みのためここで実施）
            self.after(0, lambda p=video_path, n=int(n_frames), r=fps:
                       self._player_init(p, n, r))
        else:
            self.after(0, lambda: self._preview_label.config(
                text="フレーム取得失敗", fg="#9e9088"))

    def _display_frame_preview(self, frame_rgb):
        """NumPy 配列（RGB）をプレビューラベルに表示する。"""
        try:
            import cv2
            from PIL import Image, ImageTk
            src_h, src_w = frame_rgb.shape[:2]
            w = max(200, self._preview_frame.winfo_width() - 20)
            h = max(150, self._preview_frame.winfo_height() - 60)
            if src_w > w or src_h > h:
                scale = min(w / src_w, h / src_h)
                new_w = max(1, int(src_w * scale))
                new_h = max(1, int(src_h * scale))
                frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            photo = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
            self._preview_label.config(image=photo, text="")
            self._preview_label.image = photo
        except ImportError:
            self._preview_label.config(
                text="プレビュー不可\n(Pillow 未インストール)", image="", fg="#9e9088")
        except Exception as e:
            self._preview_label.config(text=f"プレビューエラー:\n{e}", image="", fg="#9e9088")

    # ──────────────────────────────────────────────────────────
    #  動画プレーヤー
    # ──────────────────────────────────────────────────────────

    def _build_player_controls(self, parent, surface, surface2):
        """再生・停止・コマ送り・シーク UI を row=2 に構築する。"""
        ctrl = tk.Frame(parent, bg=surface2)
        ctrl.grid(row=2, column=0, sticky="ew", pady=(2, 0))
        ctrl.columnconfigure(0, weight=1)

        # フレーム情報ラベル（左寄せ）
        self._player_info_label = tk.Label(
            ctrl,
            text="-- / --  |  0.00s / 0.00s",
            bg=surface2, fg="#444444",
            font=(FONT_MONO, 10),
        )
        self._player_info_label.grid(row=0, column=0, sticky="w", padx=8, pady=(4, 0))

        # シークスライダー
        self._player_slider = ttk.Scale(
            ctrl, from_=0, to=100, orient="horizontal",
            command=self._player_on_slider_moved,
        )
        self._player_slider.grid(row=1, column=0, sticky="ew", padx=6, pady=2)

        # ボタン行
        btn_row = tk.Frame(ctrl, bg=surface2)
        btn_row.grid(row=2, column=0, pady=(0, 4))

        self._player_prev_btn = ttk.Button(
            btn_row, text="◀◀", width=4,
            command=lambda: self._player_step(-1),
        )
        self._player_prev_btn.pack(side="left", padx=2)

        self._player_play_btn = ttk.Button(
            btn_row, text="▶ 再生", width=7,
            command=self._player_play,
        )
        self._player_play_btn.pack(side="left", padx=2)

        self._player_stop_btn = ttk.Button(
            btn_row, text="■ 停止", width=7,
            command=self._player_stop,
        )
        self._player_stop_btn.pack(side="left", padx=2)

        self._player_next_btn = ttk.Button(
            btn_row, text="▶▶", width=4,
            command=lambda: self._player_step(1),
        )
        self._player_next_btn.pack(side="left", padx=2)

        # 初期状態：コントロール無効
        self._player_set_controls_state("disabled")

    # ──────────────────────────────────────────────────────────
    #  波形連動表示
    # ──────────────────────────────────────────────────────────

    def _build_waveform_area(self, parent, surface, surface2):
        """波形連動表示エリアを row=3 に構築する。"""
        wf_lf = ttk.LabelFrame(parent, text="波形連動表示", padding=4)
        wf_lf.grid(row=3, column=0, sticky="ew", pady=(4, 0))

        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except ImportError:
            tk.Label(
                wf_lf,
                text="波形表示には matplotlib が必要です (pip install matplotlib)",
                bg=surface2, fg="#9e9088", font=(FONT_UI, 10),
            ).pack(pady=6)
            return

        fig = Figure(figsize=(5, 1.5), dpi=80)
        fig.patch.set_facecolor(surface)
        ax = fig.add_subplot(111)
        ax.set_facecolor(surface)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("time (s)", fontsize=7)
        ax.text(
            0.5, 0.5, "Run analysis to show waveform",
            transform=ax.transAxes,
            ha="center", va="center", fontsize=9, color="#9e9088",
        )
        fig.tight_layout(pad=0.6)

        canvas = FigureCanvasTkAgg(fig, master=wf_lf)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        self._waveform_fig = fig
        self._waveform_ax = ax
        self._waveform_canvas = canvas

        canvas.mpl_connect("button_press_event", self._on_waveform_click)

        tk.Label(
            wf_lf,
            text="波形グラフをクリックすると、その時刻へ動画を移動します。",
            bg=surface, fg="#6b5f52", font=(FONT_UI, 10),
        ).pack(anchor="w", padx=4, pady=(0, 2))

    def _load_waveform_from_csv(self, frames_csv_path: str, task: str):
        """frames.csv を読み取り専用で参照し、波形グラフを描画する。"""
        if self._waveform_canvas is None:
            return

        col = WAVEFORM_COL.get(task, "wrist_y_px_sm")
        self._waveform_col = col

        time_arr: list = []
        val_arr: list = []
        outlier_idx_set: set = set()
        try:
            with open(frames_csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
                missing = [c for c in ("time_s", col) if c not in headers]
                if missing:
                    self._log_write(f"[波形] 必要な列が見つかりません: {missing}")
                    self._waveform_loaded = False
                    return
                has_outlier_flag = "outlier_flag" in headers
                for row in reader:
                    try:
                        t = float(row["time_s"])
                        is_outlier = (
                            has_outlier_flag
                            and int(float(row.get("outlier_flag", 0))) == 1
                        )
                        if is_outlier:
                            outlier_idx_set.add(len(time_arr))
                            v = float("nan")
                        else:
                            v = float(row[col])
                        time_arr.append(t)
                        val_arr.append(v)
                    except (ValueError, TypeError):
                        continue
        except Exception as e:
            self._log_write(f"[波形] frames.csv 読み込みエラー: {e}")
            self._waveform_loaded = False
            return

        # 境界効果対策：外れ値フレームの前後 ±5 フレームも NaN に拡張（表示専用）
        BOUNDARY_EXPAND = 1
        n_total = len(val_arr)
        for idx in outlier_idx_set:
            for offset in range(-BOUNDARY_EXPAND, BOUNDARY_EXPAND + 1):
                neighbor = idx + offset
                if 0 <= neighbor < n_total:
                    val_arr[neighbor] = float("nan")
        n_masked = sum(1 for v in val_arr if v != v)

        if not time_arr:
            self._log_write("[波形] 有効なデータがありませんでした。")
            self._waveform_loaded = False
            return

        self._waveform_time_arr = time_arr

        ax = self._waveform_ax
        ax.clear()
        ax.plot(time_arr, val_arr, color="#2d5a8e", linewidth=0.8, alpha=0.9)
        ax.set_xlabel("time (s)", fontsize=7)
        ax.set_ylabel(col, fontsize=6)
        ax.tick_params(labelsize=7)
        ax.set_title(f"Waveform: {col}", fontsize=8, pad=2)
        ax.set_xlim(time_arr[0], time_arr[-1])

        self._waveform_cursor_line = ax.axvline(
            x=time_arr[0], color="#e03030", linewidth=1.2, alpha=0.85,
        )

        self._waveform_fig.tight_layout(pad=0.6)
        self._waveform_canvas.draw()
        self._waveform_loaded = True
        mask_note = f"  外れ値マスク: {n_masked}フレーム（境界拡張±{BOUNDARY_EXPAND}含む）" if n_masked > 0 else ""
        self._log_write(
            f"[波形] 表示完了 — 列: {col}  フレーム数: {len(time_arr)}{mask_note}"
        )

    def _update_waveform_cursor(self, frame_num: int):
        """現在フレームに対応する時刻へ縦線カーソルを更新する（波形全体は再描画しない）。"""
        if not self._waveform_loaded or self._waveform_cursor_line is None:
            return
        current_time = frame_num / self._player_fps if self._player_fps > 0 else 0.0
        self._waveform_cursor_line.set_xdata([current_time, current_time])
        self._waveform_canvas.draw_idle()

    def _on_waveform_click(self, event):
        """波形グラフクリック時に動画をその時刻へシークする。"""
        if not self._waveform_loaded:
            return
        if self._player_cap is None:
            return
        if event.inaxes is not self._waveform_ax:
            return
        if event.xdata is None:
            return
        if self._player_fps <= 0:
            self._log_write("[波形クリック] FPS が不正のためシークできません。")
            return
        self._player_stop()
        frame_num = int(round(float(event.xdata) * self._player_fps))
        frame_num = max(0, min(self._player_total_frames - 1, frame_num))
        self._player_seek_to(frame_num)

    def _player_reset(self):
        """再生を停止し VideoCapture を解放する。動画変更・終了時に呼ぶ。"""
        self._player_stop()
        if self._player_cap is not None:
            try:
                self._player_cap.release()
            except Exception:
                pass
            self._player_cap = None
        self._player_total_frames = 0
        self._player_current_frame = 0
        self._player_fps = 30.0
        self._player_set_controls_state("disabled")
        self._player_info_label.config(text="-- / --  |  0.00s / 0.00s")
        self._player_slider_busy = True
        self._player_slider.config(to=100)
        self._player_slider.set(0)
        self._player_slider_busy = False
        self._update_screenshot_btn_state()

    def _player_init(self, video_path: str, total_frames: int, fps: float):
        """動画が読み込み可能と確認された後、メインスレッドから呼んでプレーヤーを初期化する。"""
        import cv2
        self._player_reset()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return
        self._player_cap = cap
        self._player_total_frames = max(1, total_frames)
        self._player_fps = fps if fps > 0 else 30.0
        self._player_current_frame = 0
        self._player_slider_busy = True
        self._player_slider.config(to=self._player_total_frames - 1)
        self._player_slider.set(0)
        self._player_slider_busy = False
        self._player_update_info()
        self._player_set_controls_state("normal")
        self._update_screenshot_btn_state()

    def _player_set_controls_state(self, state: str):
        for w in (self._player_play_btn, self._player_stop_btn,
                  self._player_prev_btn, self._player_next_btn,
                  self._player_slider):
            w.config(state=state)

    def _player_update_info(self):
        cur   = self._player_current_frame
        total = self._player_total_frames
        fps   = self._player_fps
        cur_t   = cur   / fps if fps > 0 else 0.0
        total_t = total / fps if fps > 0 else 0.0
        self._player_info_label.config(
            text=f"Frame {cur + 1} / {total}  |  {cur_t:.2f}s / {total_t:.2f}s"
        )

    def _player_play(self):
        if self._player_cap is None or self._player_playing:
            return
        import cv2
        # 末尾に達していたら先頭から再生
        if self._player_current_frame >= self._player_total_frames - 1:
            self._player_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._player_current_frame = 0
        self._player_playing = True
        self._player_loop()

    def _player_stop(self):
        self._player_playing = False
        if self._player_after_id is not None:
            try:
                self.after_cancel(self._player_after_id)
            except Exception:
                pass
            self._player_after_id = None

    def _player_loop(self):
        """after() で繰り返し呼ばれる再生ループ。"""
        if not self._player_playing or self._player_cap is None:
            return
        import cv2
        ret, frame = self._player_cap.read()
        if not ret:
            self._player_playing = False
            self._player_current_frame = max(0, self._player_total_frames - 1)
            self._player_update_info()
            return
        self._player_current_frame = (
            int(self._player_cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        )
        # BGR のまま先にリサイズ → 小画像で色変換（処理量を削減）
        pw = max(200, self._preview_frame.winfo_width() - 20)
        ph = max(150, self._preview_frame.winfo_height() - 60)
        fh, fw = frame.shape[:2]
        scale = min(pw / fw, ph / fh)
        nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
        small_rgb = cv2.cvtColor(
            cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR),
            cv2.COLOR_BGR2RGB,
        )
        self._display_frame_preview(small_rgb)
        # スライダー・情報・カーソルは3フレームに1回更新（tkinter負荷を軽減）
        if self._player_current_frame % 3 == 0:
            self._player_slider_busy = True
            self._player_slider.set(self._player_current_frame)
            self._player_slider_busy = False
            self._player_update_info()
            self._update_waveform_cursor(self._player_current_frame)
        delay = max(1, int(1000 / self._player_fps))
        self._player_after_id = self.after(delay, self._player_loop)

    def _player_step(self, delta: int):
        """1 フレームずつ前後移動。"""
        if self._player_cap is None:
            return
        self._player_stop()
        target = max(0, min(self._player_total_frames - 1,
                            self._player_current_frame + delta))
        self._player_seek_to(target)

    def _player_seek_to(self, frame_num: int):
        """指定フレームへシークして表示する。"""
        import cv2
        if self._player_cap is None:
            return
        frame_num = max(0, min(self._player_total_frames - 1, frame_num))
        self._player_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self._player_cap.read()
        if ret:
            self._player_current_frame = frame_num
            pw = max(200, self._preview_frame.winfo_width() - 20)
            ph = max(150, self._preview_frame.winfo_height() - 60)
            fh, fw = frame.shape[:2]
            scale = min(pw / fw, ph / fh)
            nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
            small_rgb = cv2.cvtColor(
                cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR),
                cv2.COLOR_BGR2RGB,
            )
            self._display_frame_preview(small_rgb)
            self._player_slider_busy = True
            self._player_slider.set(frame_num)
            self._player_slider_busy = False
            self._player_update_info()
            self._update_waveform_cursor(frame_num)

    def _player_on_slider_moved(self, val):
        """ユーザーがスライダーを操作したときのコールバック。"""
        if self._player_slider_busy or self._player_cap is None:
            return
        self._player_stop()
        self._player_seek_to(int(float(val)))

    # ──────────────────────────────────────────────────────────
    #  HEVC → H.264 自動変換（バックグラウンドスレッド内で呼ぶこと）
    # ──────────────────────────────────────────────────────────

    def _ensure_readable_video_bg(self, video_path: str) -> "str | None":
        """OpenCV で読めない動画（HEVC/H.265 等）を H.264 MP4 に変換する。
        バックグラウンドスレッドから呼ぶこと（UI はブロックしない）。
        成功時は使用すべきパス文字列、失敗時は None を返す。"""
        # PyAV でコーデックを直接確認。HEVC なら OpenCV テストをスキップして即変換
        # （OpenCV は1フレーム目だけ読めるHEVC動画を誤って「読める」と判定し、
        #   サブプロセスで全フレーム読み込み時にフリーズする問題を回避するため）
        is_hevc = False
        try:
            import av as _av_check
            with _av_check.open(video_path) as _f:
                is_hevc = _f.streams.video[0].codec_context.name.lower() in (
                    "hevc", "h265", "hvc1")
        except Exception:
            pass

        if not is_hevc:
            cap, opened, timed_out = self._open_cap_with_timeout(video_path, timeout=8.0)
            if cap:
                try:
                    cap.release()
                except Exception:
                    pass
            if opened:
                return video_path
            reason = "タイムアウト（HEVC/H.265の可能性）" if timed_out else "読み込みエラー"
        else:
            reason = "HEVC（H.265）形式を検出"

        self.after(0, self._log_write,
                   f"[変換] OpenCV で読めない動画を検出（{reason}）。H.264 に自動変換します...")
        try:
            import av  # type: ignore
        except ImportError:
            self.after(0, lambda: messagebox.showerror(
                "ライブラリ不足",
                "この動画は HEVC（H.265）形式のため、変換が必要です。\n"
                "変換には PyAV ライブラリが必要ですが、インストールされていません。\n\n"
                "【インストール手順】\n"
                "1. ターミナル（Mac）またはコマンドプロンプト（Windows）を開く\n"
                "2. 以下のコマンドを実行する:\n\n"
                f"   {sys.executable} -m pip install av\n\n"
                "3. インストール完了後、アプリを再起動して再度お試しください。\n\n"
                "※ Windows の場合は、別途 ffmpeg のインストールが必要な場合があります。\n"
                "  詳細は README.md の「動画形式について」をご参照ください。"
            ))
            return None

        import itertools as _it
        import shutil as _shutil
        import tempfile as _tempfile
        from pathlib import Path as _Path

        src = _Path(video_path)
        dst = src.with_name(src.stem + "_h264.mp4")

        # 最終保存先：非ASCII文字が含まれる場合はアプリフォルダ直下に変更
        if not str(dst).isascii():
            dst = APP_DIR / dst.name

        # 変換済みファイルが既に存在し、十分なサイズで OpenCV で読める場合はスキップ
        # （1MB未満は変換途中の不完全ファイルと判断して再変換する）
        if dst.exists() and dst.stat().st_size > 1024 * 1024:
            cap, opened, _ = self._open_cap_with_timeout(str(dst), timeout=8.0)
            if cap:
                try:
                    cap.release()
                except Exception:
                    pass
            if opened:
                self.after(0, self._log_write,
                           f"[変換スキップ] 変換済みファイルを使用: {dst.name}")
                return str(dst)

        # PyAV(読み込み) + OpenCV(書き込み) でHEVC→H.264変換
        # PyAVのlibx264エンコードはmacOS環境で失敗するケースがあるため
        # 書き込みはOpenCV(VideoToolbox)を使用する
        import uuid as _uuid
        import cv2 as _cv2
        tmp_path = _Path(_tempfile.gettempdir()) / f"apraxia_{_uuid.uuid4().hex}.mp4"
        writer = None
        try:
            with av.open(str(src)) as inp:
                in_stream = inp.streams.video[0]
                rate = float(in_stream.average_rate or 30)

                frames_iter = inp.decode(video=0)
                first_frame = next(frames_iter, None)
                if first_frame is None:
                    raise RuntimeError("動画からフレームを取得できません")
                width  = (first_frame.width  // 2) * 2
                height = (first_frame.height // 2) * 2

                # macOSはavc1(H.264)、フォールバックはmp4v
                for fourcc_str in ("avc1", "mp4v"):
                    fourcc = _cv2.VideoWriter_fourcc(*fourcc_str)
                    writer = _cv2.VideoWriter(
                        str(tmp_path), fourcc, rate, (width, height))
                    if writer.isOpened():
                        break
                    writer.release()
                    writer = None
                if writer is None:
                    raise RuntimeError("H.264エンコーダーが利用できません")

                for frame in _it.chain([first_frame], frames_iter):
                    img = frame.to_ndarray(format="bgr24")
                    if img.shape[1] != width or img.shape[0] != height:
                        img = _cv2.resize(img, (width, height))
                    writer.write(img)

            writer.release()
            writer = None

            # 変換成功 → 最終保存先へ移動
            _shutil.move(str(tmp_path), str(dst))
            tmp_path = None

            dst_name = dst.name
            self.after(0, self._log_write,
                       f"[変換完了] {dst_name} として保存しました")
            self.after(0, lambda: self._video_label.config(
                text=f"選択済: {dst_name}（H.264 変換済）"))
            return str(dst)
        except Exception as e:
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
            err = str(e)
            self.after(0, self._log_write, f"[変換エラー] {err}")
            self.after(0, lambda: messagebox.showerror("動画変換エラー", err))
            return None

    def _run_analysis(self, task: str, pid: str, set_id: str,
                      trial_id: str, out_dir_str: str):
        # HEVC 等で OpenCV が読めない場合は H.264 に変換してからパスを差し替える
        # （このメソッドはすでにバックグラウンドスレッドで動いているので UI はブロックしない）
        video_path = self._ensure_readable_video_bg(self._video_path)
        if video_path is None:
            self.after(0, self._reset_btn)
            return

        try:
            cmd = [
                PYTHON, "-u", "-m", "apraxia_analysis.main",
                "--task",           task,
                "--video",          video_path,
                "--pose_model",     self._pose_model_var.get(),
                "--out_dir",        out_dir_str,
                "--side",           self._side_var.get(),
                "--participant_id", pid,
                "--set_id",         set_id,
                "--trial_id",       trial_id,
                "--cue_frame",      self._cue_var.get().strip() or "0",
            ]
            if task in ("byebye", "comehere"):
                cmd += ["--hand_model", self._hand_model_var.get()]

            proc = subprocess.Popen(
                cmd,
                cwd=str(APP_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            for line in proc.stdout:
                self.after(0, self._log_write, line.rstrip())

            proc.wait()

            if proc.returncode == 0:
                self.after(0, self._on_analysis_done, task, out_dir_str)
            else:
                self.after(0, self._log_write,
                           f"\n[エラー] 解析が異常終了しました（コード: {proc.returncode}）")
                # Mac：非ASCIIパスが原因の可能性をヒントとして表示
                if sys.platform != "win32":
                    checks = [
                        ("動画ファイル",   video_path),
                        ("Poseモデル",     self._pose_model_var.get()),
                        ("出力先フォルダ", out_dir_str),
                        ("アプリフォルダ", str(APP_DIR)),
                    ]
                    problem = [f"・{label}:\n  {path}"
                               for label, path in checks if path and not path.isascii()]
                    if problem:
                        hint = "\n".join(problem)
                        self.after(0, self._log_write,
                                   "\n[ヒント] 以下のパスに日本語や全角文字が含まれています。\n"
                                   "これが原因の可能性があります。\n"
                                   "英数字のみのパスに変更してお試しください。\n" + hint)
                self.after(0, self._reset_btn)

        except FileNotFoundError:
            self.after(0, self._log_write,
                       f"\n[エラー] Python が見つかりません:\n{PYTHON}")
            self.after(0, self._reset_btn)
        except Exception as e:
            self.after(0, self._log_write, f"\n[例外] {e}")
            self.after(0, self._reset_btn)

    def _on_analysis_done(self, task: str, out_dir_str: str):
        self._last_trial_out = out_dir_str
        self._last_task = task
        out_dir = Path(out_dir_str)
        self._log_write("")
        self._log_write("=== 解析完了 ===")
        self._log_write(f"出力先: {out_dir_str}")

        # ── summary.csv を読み込んで表示 ──
        summary_csv = out_dir / "summary.csv"
        if summary_csv.exists():
            try:
                with open(summary_csv, newline="", encoding="utf-8") as f:
                    row = next(csv.DictReader(f), {})

                keys = SUMMARY_KEYS_COMMON + (
                    SUMMARY_KEYS_HAMMER if task == "hammer"
                    else SUMMARY_KEYS_BYEBYE_COMEHERE
                ) + SUMMARY_KEYS_CENTRAL5

                label_dict = {
                    **LABEL_COMMON,
                    **(LABEL_HAMMER if task == "hammer" else LABEL_BYEBYE_COMEHERE),
                    **LABEL_CENTRAL5,
                }

                lines = ["─" * 40]
                for k in keys:
                    v = row.get(k, "")
                    if v not in ("", None, "nan"):
                        # 数値は小数点4桁に整形
                        try:
                            fv = float(v)
                            v = f"{fv:.4f}" if "." in v else v
                        except ValueError:
                            pass
                        label = label_dict.get(k)
                        if label:
                            lines.append(f"{label}（{k}）：{v}")
                        else:
                            lines.append(f"{k}：{v}")

                lines.append("─" * 40)
                lines.append(f"CSV保存先: {out_dir_str}/")
                self._show_result("\n".join(lines))

            except Exception as e:
                self._show_result(f"summary.csv 読み込みエラー:\n{e}")

        # ── 波形PNG表示 ──
        pngs = sorted(out_dir.glob("waveform_*.png"))
        if pngs:
            self._show_waveform(str(pngs[0]))

        # ── 波形連動表示ロード（frames.csv 読み取り専用） ──
        frames_csv = out_dir / "frames.csv"
        if frames_csv.exists():
            self._load_waveform_from_csv(str(frames_csv), task)
        else:
            self._log_write("[波形] frames.csv が見つかりません。波形連動表示は無効です。")

        # ── HEVC 変換済みファイルへの自動切替 ──
        # 解析中に HEVC→H.264 変換が行われた場合、変換後ファイルにプレイヤーを切替える
        if self._video_path:
            src = Path(self._video_path)
            h264_name = src.stem + "_h264.mp4"
            # 変換先候補: 元ファイルと同じディレクトリ、または非ASCII時は APP_DIR
            candidates = [src.with_name(h264_name), APP_DIR / h264_name]
            converted_path: "str | None" = None
            for c in candidates:
                if c.exists() and str(c) != self._video_path:
                    converted_path = str(c)
                    break
            if converted_path:
                self._log_write(f"[プレイヤー] H.264変換済みファイルに切替: {Path(converted_path).name}")
                self._player_reset()
                self._video_path = converted_path
                self._video_label.config(
                    text=f"選択済: {Path(converted_path).name}（H.264 変換済）",
                    fg="#1a1612",
                )
                self._preview_label.config(image="", text="動画情報を読み込み中...", fg="#9e9088")
                self._preview_label.image = None
                self._video_info_label.config(text="", fg="#555555")
                threading.Thread(
                    target=self._load_video_preview,
                    args=(converted_path,),
                    daemon=True,
                ).start()

        self._reset_btn()
        self._update_screenshot_btn_state()

    def _reset_btn(self):
        self._running = False
        self._analyze_btn.configure(state="normal", text="解析開始 ▶")

    # ──────────────────────────────────────────────────────────
    #  オーバーレイ動画生成
    # ──────────────────────────────────────────────────────────

    def _start_overlay(self):
        if self._overlay_running:
            messagebox.showinfo("作成中", "オーバーレイ動画を作成中です。完了をお待ちください。")
            return

        if not self._video_path:
            messagebox.showerror("エラー", "動画ファイルを選択してください。")
            return
        if not Path(self._video_path).exists():
            messagebox.showerror("エラー", f"動画ファイルが見つかりません:\n{self._video_path}")
            return

        if not self._last_trial_out:
            messagebox.showerror(
                "エラー",
                "解析結果フォルダが見つかりません。\n"
                "先に解析を実行してください。",
            )
            return

        frames_csv = Path(self._last_trial_out) / "frames.csv"
        if not frames_csv.exists():
            messagebox.showerror(
                "エラー",
                f"frames.csv が見つかりません:\n{frames_csv}\n\n"
                "先に解析を実行してください。",
            )
            return

        self._overlay_running = True
        self._overlay_btn.configure(state="disabled", text="オーバーレイ作成中…")
        self._log_write("")
        self._log_write("=== オーバーレイ動画作成開始 ===")

        threading.Thread(
            target=self._make_overlay,
            args=(self._video_path, str(frames_csv), self._last_trial_out, self._last_task),
            daemon=True,
        ).start()

    def _make_overlay(self, video_path: str, frames_csv_path: str,
                      out_dir_str: str, task: str):
        try:
            import cv2  # type: ignore
        except ImportError:
            self.after(0, self._log_write, "[エラー] opencv-python がインストールされていません。")
            self.after(0, self._reset_overlay_btn)
            return

        try:
            # ── 動画の読み込み確認（HEVC対応） ──
            readable_path = self._ensure_readable_video_bg(video_path)
            if readable_path is None:
                self.after(0, self._log_write, "[エラー] 動画を読み込めませんでした。")
                self.after(0, self._reset_overlay_btn)
                return

            cap = cv2.VideoCapture(readable_path)
            if not cap.isOpened():
                self.after(0, self._log_write, f"[エラー] 動画を開けませんでした:\n{readable_path}")
                self.after(0, self._reset_overlay_btn)
                return

            fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            # ── frames.csv 読み込み（読み取りのみ） ──
            frame_data: dict[int, dict] = {}
            with open(frames_csv_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    try:
                        fidx = int(float(row.get("frame_idx", -1)))
                    except (ValueError, TypeError):
                        continue
                    frame_data[fidx] = row

            # ── 出力パス ──
            video_stem = Path(video_path).stem
            out_path = Path(out_dir_str) / f"overlay_{video_stem}.mp4"

            for fourcc_str in ("avc1", "mp4v"):
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
                if writer.isOpened():
                    break
                writer.release()
                writer = None

            if writer is None:
                self.after(0, self._log_write, "[エラー] 動画エンコーダーを初期化できませんでした。")
                self.after(0, self._reset_overlay_btn)
                return

            # ── ランドマーク色定義 ──
            C_SHOULDER   = (255, 180,  50)   # 水色系
            C_ELBOW      = ( 50, 220, 255)   # 黄色系
            C_WRIST      = ( 50, 160, 255)   # オレンジ系
            C_INDEX      = ( 80,  80, 255)   # 赤系
            C_HAND_EXTRA = (180,  80, 255)   # 紫系（byebye/comehere 追加点）
            C_BONE       = (200, 200, 200)   # 骨格線：薄灰
            R_JOINT      = max(6, int(min(width, height) * 0.012))  # 解像度比例半径
            T_BONE       = max(2, R_JOINT // 3)

            def _px(row: dict, xk: str, yk: str):
                """float変換。NaN・変換失敗時は None を返す。"""
                try:
                    x = float(row[xk])
                    y = float(row[yk])
                    if x != x or y != y:   # NaN check
                        return None
                    return (int(round(x)), int(round(y)))
                except (KeyError, ValueError, TypeError):
                    return None

            def _line(img, p1, p2):
                if p1 and p2:
                    cv2.line(img, p1, p2, C_BONE, T_BONE, cv2.LINE_AA)

            def _circle(img, pt, color):
                if pt:
                    cv2.circle(img, pt, R_JOINT, color, -1, cv2.LINE_AA)
                    cv2.circle(img, pt, R_JOINT, (30, 30, 30), 1, cv2.LINE_AA)

            # ── フレームループ ──
            cap = cv2.VideoCapture(readable_path)
            frame_idx = 0
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break

                row = frame_data.get(frame_idx)
                if row is not None:
                    sh  = _px(row, "shoulder_x_px",        "shoulder_y_px")
                    el  = _px(row, "elbow_x_px",           "elbow_y_px")
                    wr  = _px(row, "wrist_x_px_raw",       "wrist_y_px_raw")
                    idx = _px(row, "index_x_px",           "index_y_px")

                    # 骨格線（共通）
                    _line(bgr, sh, el)
                    _line(bgr, el, wr)
                    _line(bgr, wr, idx)

                    # 関節点（共通）
                    _circle(bgr, sh,  C_SHOULDER)
                    _circle(bgr, el,  C_ELBOW)
                    _circle(bgr, wr,  C_WRIST)
                    _circle(bgr, idx, C_INDEX)

                    # byebye / comehere: 手指追加点
                    if task in ("byebye", "comehere"):
                        hw  = _px(row, "hand_wrist_x_px",       "hand_wrist_y_px")
                        mcp = _px(row, "index_mcp_x_px",        "index_mcp_y_px")
                        pip = _px(row, "index_pip_x_px",        "index_pip_y_px")
                        tip = _px(row, "hand_index_tip_x_px",   "hand_index_tip_y_px")

                        _line(bgr, hw, mcp)
                        _line(bgr, mcp, pip)
                        _line(bgr, pip, tip)

                        _circle(bgr, hw,  C_HAND_EXTRA)
                        _circle(bgr, mcp, C_HAND_EXTRA)
                        _circle(bgr, pip, C_HAND_EXTRA)
                        _circle(bgr, tip, C_HAND_EXTRA)

                    # フレーム情報テキスト
                    try:
                        t_s = float(row.get("time_s", frame_idx / fps))
                    except (ValueError, TypeError):
                        t_s = frame_idx / fps

                    info_lines = [
                        f"Frame: {frame_idx}",
                        f"Time:  {t_s:.2f} s",
                        f"Task:  {task}",
                    ]
                    font_face  = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = max(0.5, min(width, height) / 1000.0)
                    thickness  = max(1, int(font_scale * 2))
                    pad        = int(height * 0.015)
                    line_h     = int((cv2.getTextSize("A", font_face, font_scale, thickness)[0][1]) * 1.8)
                    for i, txt in enumerate(info_lines):
                        y = pad + line_h * (i + 1)
                        cv2.putText(bgr, txt, (pad, y), font_face, font_scale,
                                    (0, 0, 0),       thickness + 2, cv2.LINE_AA)
                        cv2.putText(bgr, txt, (pad, y), font_face, font_scale,
                                    (255, 255, 255), thickness,     cv2.LINE_AA)

                writer.write(bgr)
                if frame_idx % 50 == 0:
                    self.after(0, self._log_write, f"  {frame_idx}フレーム処理中...")
                frame_idx += 1

            cap.release()
            writer.release()

            self.after(0, self._log_write, f"=== オーバーレイ動画作成完了 ===")
            self.after(0, self._log_write, f"保存先: {out_path}")

        except Exception as e:
            self.after(0, self._log_write, f"[エラー] オーバーレイ動画の作成中に例外が発生しました:\n{e}")

        finally:
            self.after(0, self._reset_overlay_btn)

    def _reset_overlay_btn(self):
        self._overlay_running = False
        self._overlay_btn.configure(state="normal", text="オーバーレイ動画を作成")

    # ──────────────────────────────────────────────────────────
    #  解析前チェック
    # ──────────────────────────────────────────────────────────

    def _run_pre_check(self) -> list:
        """解析前チェックを実行し結果リストを返す。
        各要素: {"status": "ok"|"warn"|"stop", "message": str}
        """
        results = []
        task = self._task_var.get()

        # ── 1. 動画ファイル ──
        video_exists = False
        if not self._video_path:
            results.append({"status": "stop",
                            "message": "❌ 動画ファイル：未選択"})
        elif not Path(self._video_path).exists():
            results.append({"status": "stop",
                            "message": "❌ 動画ファイル：見つかりません"})
        else:
            results.append({"status": "ok",
                            "message": "✅ 動画ファイル：OK"})
            video_exists = True

        # ── 2. 動画読み込み ──
        if video_exists:
            if self._player_cap is not None:
                results.append({"status": "ok",
                                "message": "✅ 動画読み込み：OK"})
            elif self._video_is_hevc is True:
                results.append({"status": "warn",
                                "message": "⚠️ 動画読み込み：HEVC形式のためOpenCVでは開けません。H.264変換後に解析されます"})
            elif self._video_is_hevc is None:
                results.append({"status": "warn",
                                "message": "⚠️ 動画読み込み：動画情報を読み込み中です。しばらく待ってから再実行してください"})
            else:
                results.append({"status": "stop",
                                "message": "❌ 動画読み込み：失敗"})

        # ── 3. FPS取得 ──
        if video_exists:
            EXPECTED_FPS = 60.0
            if self._player_cap is not None and self._player_fps > 0:
                fps = self._player_fps
                if abs(fps - EXPECTED_FPS) < 0.5:
                    results.append({"status": "ok",
                                    "message": f"✅ FPS取得：{fps:.2f} fps"})
                else:
                    results.append({"status": "warn",
                                    "message": f"⚠️ FPS：{fps:.2f} fps（{EXPECTED_FPS:.0f}fps想定と異なりますが、取得FPSに基づいて解析します）"})
            elif self._video_is_hevc is True:
                results.append({"status": "warn",
                                "message": "⚠️ FPS取得：HEVC形式のためFPS未取得。H.264変換後に解析されます"})
            elif self._video_is_hevc is None:
                results.append({"status": "warn",
                                "message": "⚠️ FPS取得：動画情報を読み込み中です"})
            else:
                results.append({"status": "stop",
                                "message": "❌ FPS取得：失敗"})

        # ── 4. フレーム数取得 ──
        if video_exists:
            if self._player_cap is not None and self._player_total_frames > 0:
                results.append({"status": "ok",
                                "message": f"✅ フレーム数取得：{self._player_total_frames} frames"})
            elif self._video_is_hevc is True:
                results.append({"status": "warn",
                                "message": "⚠️ フレーム数取得：HEVC形式のため未取得。H.264変換後に解析されます"})
            elif self._video_is_hevc is None:
                results.append({"status": "warn",
                                "message": "⚠️ フレーム数取得：動画情報を読み込み中です"})
            else:
                results.append({"status": "stop",
                                "message": "❌ フレーム数取得：失敗"})

        # ── 5. Poseモデル ──
        pose = self._pose_model_var.get().strip()
        if not pose:
            results.append({"status": "stop",
                            "message": "❌ Poseモデル：未設定"})
        elif not Path(pose).exists():
            results.append({"status": "stop",
                            "message": "❌ Poseモデル：見つかりません"})
        else:
            results.append({"status": "ok",
                            "message": "✅ Poseモデル：OK"})

        # ── 6. Handモデル ──
        hand = self._hand_model_var.get().strip()
        if task == "hammer":
            if hand and Path(hand).exists():
                results.append({"status": "ok",
                                "message": "✅ Handモデル：設定あり。ただしhammerでは使用しません"})
            else:
                results.append({"status": "ok",
                                "message": "✅ Handモデル：hammerでは不要"})
        elif task in ("byebye", "comehere"):
            if not hand:
                results.append({"status": "stop",
                                "message": f"❌ Handモデル：{task} では必要です"})
            elif not Path(hand).exists():
                results.append({"status": "stop",
                                "message": "❌ Handモデル：見つかりません"})
            else:
                results.append({"status": "ok",
                                "message": "✅ Handモデル：OK"})
        else:
            results.append({"status": "warn",
                            "message": "⚠️ Handモデル：タスク未選択のため判定不可"})

        # ── 7. 出力先フォルダ ──
        out_dir_str = self._out_dir_var.get().strip()
        if not out_dir_str:
            results.append({"status": "stop",
                            "message": "❌ 出力先フォルダ：未設定"})
        else:
            try:
                Path(out_dir_str).mkdir(parents=True, exist_ok=True)
                results.append({"status": "ok",
                                "message": "✅ 出力先フォルダ：OK"})
            except Exception:
                results.append({"status": "stop",
                                "message": "❌ 出力先フォルダ：使用できません"})

        # ── 8. 日本語・全角パス ──
        path_checks = [
            ("動画ファイル",   self._video_path),
            ("Poseモデル",     self._pose_model_var.get()),
            ("出力先フォルダ", self._out_dir_var.get()),
            ("アプリフォルダ", str(APP_DIR)),
        ]
        if task in ("byebye", "comehere") and self._hand_model_var.get():
            path_checks.append(("Handモデル", self._hand_model_var.get()))
        problem = [(lbl, p) for lbl, p in path_checks if p and not p.isascii()]
        if not problem:
            results.append({"status": "ok",
                            "message": "✅ 日本語・全角パス：なし"})
        elif sys.platform == "win32":
            detail = "、".join(lbl for lbl, _ in problem)
            results.append({"status": "stop",
                            "message": f"❌ 日本語・全角パス：WindowsではMediaPipeが読み込めない可能性があります（{detail}）"})
        else:
            detail = "、".join(lbl for lbl, _ in problem)
            results.append({"status": "warn",
                            "message": f"⚠️ 日本語・全角パス：macOSでは注意扱いです（{detail}）"})

        # ── 9. Cueフレーム ──
        try:
            cue_val = int(self._cue_var.get().strip())
        except ValueError:
            cue_val = 0
        if cue_val == 0:
            results.append({"status": "ok",
                            "message": "✅ Cueフレーム：0"})
        else:
            results.append({"status": "warn",
                            "message": f"⚠️ Cueフレーム：{cue_val}（通常は0を使用します）"})

        # ── 10. HEVC判定 ──
        if video_exists:
            if self._video_is_hevc is True:
                results.append({"status": "warn",
                                "message": "⚠️ HEVC判定：HEVC形式です。H.264へ変換される可能性があります"})
            elif self._video_is_hevc is False:
                results.append({"status": "ok",
                                "message": "✅ HEVC判定：通常動画"})
            else:
                results.append({"status": "warn",
                                "message": "⚠️ HEVC判定：判定できませんでした"})

        # ── 11. タスク選択 ──
        if task in TASKS:
            results.append({"status": "ok",
                            "message": f"✅ タスク選択：{task}"})
        else:
            results.append({"status": "stop",
                            "message": "❌ タスク選択：未選択"})

        return results

    def _update_pre_check_label(self, results: list):
        """チェック結果に基づき要約ラベルと詳細ボタンを更新する。"""
        if self._precheck_summary_label is None:
            return
        statuses = {r["status"] for r in results}
        if "stop" in statuses:
            self._precheck_summary_label.config(
                text="解析前チェック：停止項目あり", foreground="#cc0000")
        elif "warn" in statuses:
            self._precheck_summary_label.config(
                text="解析前チェック：注意あり", foreground="#b85c00")
        else:
            self._precheck_summary_label.config(
                text="解析前チェック：OK", foreground="#2e7d2e")
        if self._precheck_detail_btn is not None:
            self._precheck_detail_btn.config(state="normal")

    def _show_pre_check_detail(self, results: list):
        """解析前チェック詳細を別ウィンドウに表示する。"""
        if not results:
            return
        dlg = tk.Toplevel(self)
        dlg.title("解析前チェック詳細")
        dlg.resizable(True, False)

        frame = ttk.Frame(dlg, padding=(12, 8, 12, 4))
        frame.pack(fill="both", expand=True)

        text = scrolledtext.ScrolledText(
            frame,
            width=74, height=min(len(results) + 4, 22),
            font=(FONT_MONO, 12),
            bg="#f8f6f2", fg="#1a1612",
            relief="flat",
            state="normal",
            wrap="word",
        )
        text.pack(fill="both", expand=True)
        for r in results:
            text.insert("end", r["message"] + "\n")
        text.config(state="disabled")

        ttk.Button(dlg, text="閉じる", command=dlg.destroy).pack(pady=(4, 10))

        dlg.update_idletasks()
        w = dlg.winfo_reqwidth()
        h = dlg.winfo_reqheight()
        x = self.winfo_rootx() + (self.winfo_width() - w) // 2
        y = self.winfo_rooty() + (self.winfo_height() - h) // 2
        dlg.geometry(f"+{x}+{y}")

    # ──────────────────────────────────────────────────────────
    #  スクリーンショット保存
    # ──────────────────────────────────────────────────────────

    def _update_screenshot_btn_state(self):
        if self._screenshot_btn is None:
            return
        ok = (
            self._player_cap is not None
            and self._waveform_loaded
            and bool(self._last_trial_out)
        )
        self._screenshot_btn.config(state="normal" if ok else "disabled")

    def _save_screenshot(self):
        if self._player_cap is None or not self._waveform_loaded or not self._last_trial_out:
            messagebox.showerror("エラー", "動画と波形データが必要です。")
            return

        try:
            import cv2
        except ImportError:
            messagebox.showerror("エラー", "opencv-python がインストールされていません。")
            return
        try:
            import io
            from PIL import Image, ImageDraw
        except ImportError:
            messagebox.showerror("エラー", "Pillow がインストールされていません。")
            return

        frame_num = self._player_current_frame

        # 現在フレームを取得し cap 位置を元に戻す
        self._player_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, bgr = self._player_cap.read()
        self._player_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        if not ret:
            messagebox.showerror("エラー", "フレームを取得できませんでした。")
            return

        video_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        # 波形図をメモリに保存
        buf = io.BytesIO()
        self._waveform_fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        waveform_img = Image.open(buf).copy()
        buf.close()

        # 動画を高さ最大 400px に縮小
        VIDEO_MAX_H  = 400
        CANVAS_MIN_W = 800
        vw, vh = video_img.size
        video_img = video_img.resize(
            (max(1, int(vw * VIDEO_MAX_H / vh)), VIDEO_MAX_H), Image.LANCZOS)
        video_w = video_img.size[0]

        # キャンバス幅 = max(動画幅, 800px)
        total_w = max(video_w, CANVAS_MIN_W)

        # 動画がキャンバス幅より狭ければキャンバス幅に合わせて拡大
        if video_w < total_w:
            video_img = video_img.resize(
                (total_w, max(1, int(VIDEO_MAX_H * total_w / video_w))), Image.LANCZOS)
        video_w, video_h = video_img.size

        # 波形をキャンバス幅に合わせてリサイズ（縦横比を保持）
        ww, wh = waveform_img.size
        waveform_img = waveform_img.resize(
            (total_w, max(1, int(wh * total_w / ww))), Image.LANCZOS)
        waveform_h = waveform_img.size[1]

        # 縦積み合成（情報バー → 動画 → 波形）
        INFO_H  = 28
        total_h = INFO_H + video_h + waveform_h
        canvas  = Image.new("RGB", (total_w, total_h), (40, 40, 40))
        draw    = ImageDraw.Draw(canvas)

        fps    = self._player_fps
        time_s = frame_num / fps if fps > 0 else 0.0
        col    = WAVEFORM_COL.get(self._last_task, "")
        info   = (
            f"Task: {self._last_task}  |  "
            f"Frame: {frame_num}  |  "
            f"Time: {time_s:.2f} s  |  col: {col}"
        )
        draw.text((8, 7), info, fill=(220, 220, 200))

        canvas.paste(video_img,    (0, INFO_H))
        canvas.paste(waveform_img, (0, INFO_H + video_h))

        # 保存先（重複時はカウンタを付加）
        stem      = Path(self._video_path).stem if self._video_path else "video"
        base_name = f"screenshot_{stem}_frame{frame_num:04d}"
        out_dir   = Path(self._last_trial_out)
        out_path  = out_dir / f"{base_name}.png"
        counter   = 1
        while out_path.exists():
            out_path = out_dir / f"{base_name}_{counter}.png"
            counter += 1

        try:
            canvas.save(str(out_path), format="PNG")
            self._log_write(f"[スクリーンショット] 保存: {out_path.name}")
        except Exception as e:
            messagebox.showerror("保存エラー", f"スクリーンショットの保存に失敗しました:\n{e}")

    # ──────────────────────────────────────────────────────────
    #  設定の保存・復元
    # ──────────────────────────────────────────────────────────

    def _save_config(self):
        _save_cfg({
            "task":       self._task_var.get(),
            "side":       self._side_var.get(),
            "pose_model": self._pose_model_var.get(),
            "hand_model": self._hand_model_var.get(),
            "out_dir":    self._out_dir_var.get(),
        })

    def _restore_values(self):
        cfg = self._cfg
        if "task"       in cfg: self._task_var.set(cfg["task"])
        if "side"       in cfg: self._side_var.set(cfg["side"])
        if "pose_model" in cfg: self._pose_model_var.set(cfg["pose_model"])
        if "hand_model" in cfg: self._hand_model_var.set(cfg["hand_model"])
        if "out_dir"    in cfg: self._out_dir_var.set(cfg["out_dir"])

    def _apply_default_models(self):
        """models/ フォルダに既定のモデルファイルがあれば未設定の入力欄にセットする。"""
        models_dir = APP_DIR / "models"
        pose_default = models_dir / "pose_landmarker_full.task"
        hand_default = models_dir / "hand_landmarker.task"
        if not self._pose_model_var.get() and pose_default.exists():
            self._pose_model_var.set(str(pose_default))
        if not self._hand_model_var.get() and hand_default.exists():
            self._hand_model_var.set(str(hand_default))

    def _on_close(self):
        self._save_config()
        self._player_reset()
        self.destroy()


# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = ApraxiaApp()
    app.mainloop()
