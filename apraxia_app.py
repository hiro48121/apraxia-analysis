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
PYTHON = str(Path.home() / "Desktop" / "hammer_project" / ".venv" / "bin" / "python")

# このファイルがある場所 = apraxia_analysis/ の親ディレクトリ
APP_DIR = Path(__file__).parent

# 設定の保存先（ホームディレクトリに隠しファイル）
CONFIG_PATH = Path.home() / ".apraxia_app_config.json"

# ─────────────────────────────────────────────────────────────
#  定数
# ─────────────────────────────────────────────────────────────

TASKS = ["hammer", "byebye", "comehere"]
SIDES = ["Left", "Right"]

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
        style.configure("TLabel",         background=BG, font=("Helvetica Neue", 13))
        style.configure("TButton",        font=("Helvetica Neue", 13), padding=5)
        style.configure("TRadiobutton",   background=BG, font=("Helvetica Neue", 13))
        style.configure("TEntry",         font=("Helvetica Neue", 13), padding=4)
        style.configure("TLabelframe",    background=BG)
        style.configure("TLabelframe.Label",
                        background=BG, font=("Helvetica Neue", 12, "bold"))

        # アクセントボタン
        style.configure("Accent.TButton",
                        background=ACCENT, foreground="white",
                        font=("Helvetica Neue", 14, "bold"), padding=10)
        style.map("Accent.TButton",
                  background=[("active", "#1e4070"), ("disabled", "#9ab0cc")],
                  foreground=[("disabled", "#e0e8f4")])

        # ── ヘッダー ──
        header = tk.Frame(self, bg=HEADER, height=50)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text="APRAXIA ANALYSIS", bg=HEADER, fg="#f5f2ee",
                 font=("Courier", 15, "bold")).pack(side="left", padx=22, pady=12)
        tk.Label(header, text="v1.0", bg="#2a2a2a", fg="#888",
                 font=("Courier", 10)).pack(side="left", pady=12)

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
        # マウスホイールでスクロール
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-1 * e.delta / 120), "units"))

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
            font=("Helvetica Neue", 13),
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
            font=("Helvetica Neue", 12),
        )
        self._preview_label.pack(expand=True, fill="both")

        # ログ
        log_lf = ttk.LabelFrame(parent, text="ログ", padding=4)
        log_lf.grid(row=2, column=0, sticky="ew", pady=(8, 0))

        self._log = scrolledtext.ScrolledText(
            log_lf, height=8,
            font=("Courier", 11),
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
            value=str(Path.home() / "Desktop" / "apraxia_results"))
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
            font=("Courier", 11),
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
    #  HEVC → H.264 自動変換（バックグラウンドスレッド内で呼ぶこと）
    # ──────────────────────────────────────────────────────────

    def _ensure_readable_video_bg(self, video_path: str) -> "str | None":
        """OpenCV で読めない動画（HEVC/H.265 等）を H.264 MP4 に変換する。
        バックグラウンドスレッドから呼ぶこと（UI はブロックしない）。
        成功時は使用すべきパス文字列、失敗時は None を返す。"""
        import cv2
        cap = cv2.VideoCapture(video_path)
        opened = cap.isOpened()
        if opened:
            ret, _ = cap.read()
            opened = ret
        cap.release()
        if opened:
            return video_path  # 読めるのでそのまま使う

        self.after(0, self._log_write,
                   "[変換] OpenCV で読めない動画を検出（HEVC/H.265 の可能性）。"
                   "H.264 に自動変換します...")
        try:
            import av  # type: ignore
        except ImportError:
            self.after(0, lambda: messagebox.showerror(
                "ライブラリ不足",
                "HEVC 動画の変換に PyAV が必要です。\n\n"
                "ターミナルで以下を実行してください:\n"
                f"{PYTHON} -m pip install av"
            ))
            return None

        from pathlib import Path as _Path
        src = _Path(video_path)
        dst = src.with_name(src.stem + "_h264.mp4")

        try:
            with av.open(str(src)) as inp:
                with av.open(str(dst), "w") as out:
                    in_stream = inp.streams.video[0]
                    out_stream = out.add_stream(
                        "libx264", rate=in_stream.average_rate)
                    out_stream.width   = in_stream.width
                    out_stream.height  = in_stream.height
                    out_stream.pix_fmt = "yuv420p"
                    for frame in inp.decode(video=0):
                        for pkt in out_stream.encode(frame):
                            out.mux(pkt)
                    for pkt in out_stream.encode():
                        out.mux(pkt)

            dst_name = dst.name
            self.after(0, self._log_write,
                       f"[変換完了] {dst_name} として保存しました")
            self.after(0, lambda: self._video_label.config(
                text=f"選択済: {dst_name}（H.264 変換済）"))
            return str(dst)
        except Exception as e:
            err = str(e)
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
                PYTHON, "-m", "apraxia_analysis.main",
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

    def _on_close(self):
        self._save_config()
        self.destroy()


# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = ApraxiaApp()
    app.mainloop()
