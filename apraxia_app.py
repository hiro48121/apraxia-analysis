#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apraxia_app.py  —  Apraxia Analysis Desktop App (最小構成)

起動方法 (ターミナル):
  ~/Desktop/hammer_project/.venv/bin/python apraxia_app.py

起動方法 (ダブルクリック):
  apraxia_app.command  をダブルクリック

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

        # ログ
        log_lf = ttk.LabelFrame(parent, text="ログ", padding=4)
        log_lf.grid(row=2, column=0, sticky="ew", pady=(8, 0))

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

    def _start_analysis(self):
        if self._running:
            messagebox.showinfo("解析中", "現在解析中です。完了をお待ちください。")
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

        # PyAV/libav はスペースや特殊文字を含むパスへの書き込みに失敗する場合があるため、
        # まず /tmp 以下の単純なパスに変換してから最終保存先へ移動する
        tmp_fd, tmp_path_str = _tempfile.mkstemp(suffix=".mp4", prefix="apraxia_")
        tmp_path = _Path(tmp_path_str)
        try:
            import os as _os
            _os.close(tmp_fd)

            with av.open(str(src)) as inp:
                in_stream = inp.streams.video[0]

                # フレームレート正規化（0や不正値は30fpsにフォールバック）
                rate = in_stream.average_rate
                if not rate or float(rate) <= 0:
                    rate = 30

                # 解像度はストリームヘッダではなく最初のフレームから取得
                # （HEVCではヘッダの値が不正なことがあるため）
                frames_iter = inp.decode(video=0)
                first_frame = next(frames_iter, None)
                if first_frame is None:
                    raise RuntimeError("動画からフレームを取得できません")
                # libx264 は偶数解像度が必要
                width  = (first_frame.width  // 2) * 2
                height = (first_frame.height // 2) * 2

                with av.open(str(tmp_path), "w") as out:
                    out_stream = out.add_stream("libx264", rate=rate)
                    out_stream.width   = width
                    out_stream.height  = height
                    out_stream.pix_fmt = "yuv420p"
                    for frame in _it.chain([first_frame], frames_iter):
                        frame = frame.reformat(
                            width=width, height=height, format="yuv420p")
                        for pkt in out_stream.encode(frame):
                            out.mux(pkt)
                    for pkt in out_stream.encode():
                        out.mux(pkt)

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
            # 一時ファイルが残っている場合は削除
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
        self.destroy()


# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = ApraxiaApp()
    app.mainloop()
