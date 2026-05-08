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
    "rhythm_cv_selected10",
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

        # ── 動画プレーヤー状態 ──
        self._player_cap = None          # cv2.VideoCapture（再生中は開いたまま）
        self._player_total_frames: int = 0
        self._player_fps: float = 30.0
        self._player_current_frame: int = 0
        self._player_playing: bool = False
        self._player_after_id = None
        self._player_slider_busy: bool = False

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

        # ログ
        log_lf = ttk.LabelFrame(parent, text="ログ", padding=4)
        log_lf.grid(row=3, column=0, sticky="ew", pady=(8, 0))

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

        # ── オーバーレイ動画作成ボタン ──
        self._overlay_btn = ttk.Button(
            parent, text="オーバーレイ動画を作成",
            command=self._start_overlay,
        )
        self._overlay_btn.pack(fill="x", pady=(0, 8))

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
            msg = "⚠ HEVC（H.265）形式の可能性があります\n解析時に自動的にH.264へ変換されます"
            self.after(0, lambda: self._video_info_label.config(text=msg, fg="#b85c00"))
            self.after(0, lambda: self._preview_label.config(
                text="プレビュー不可（HEVC形式）", fg="#9e9088"))
            return

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
            from PIL import Image, ImageTk
            img = Image.fromarray(frame_rgb)
            self._preview_frame.update_idletasks()
            w = max(200, self._preview_frame.winfo_width() - 20)
            h = max(150, self._preview_frame.winfo_height() - 60)
            img.thumbnail((w, h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
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
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._display_frame_preview(frame_rgb)
        self._player_slider_busy = True
        self._player_slider.set(self._player_current_frame)
        self._player_slider_busy = False
        self._player_update_info()
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
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._display_frame_preview(frame_rgb)
            self._player_slider_busy = True
            self._player_slider.set(frame_num)
            self._player_slider_busy = False
            self._player_update_info()

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
                )

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
                        lines.append(f"{k:<38}: {v}")

                lines.append("─" * 40)
                lines.append(f"CSV保存先: {out_dir_str}/")
                self._show_result("\n".join(lines))

            except Exception as e:
                self._show_result(f"summary.csv 読み込みエラー:\n{e}")

        # ── 波形PNG表示 ──
        pngs = sorted(out_dir.glob("waveform_*.png"))
        if pngs:
            self._show_waveform(str(pngs[0]))

        self._reset_btn()

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
