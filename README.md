# Apraxia Analysis

上肢失行症の動作解析ツール。
スマートフォンで撮影した動画から MediaPipe でキーポイントを抽出し、
繰り返し動作のサイクル指標（周期・振れ幅・ROM・リズム変動係数など）を自動算出する。

## 対応タスク

| タスク | 動作 | 使用モデル |
|--------|------|-----------|
| `hammer` | 打鍵動作（ハンマー振り） | Pose |
| `byebye` | 手を振る動作（バイバイ） | Pose + Hand |
| `comehere` | 手招き動作（おいでおいで） | Pose + Hand |

## 出力ファイル

| ファイル | 内容 |
|---------|------|
| `frames.csv` | フレームごとのキーポイント座標・速度・角度 |
| `cycles.csv` | 検出された各サイクルの指標 |
| `summary.csv` | 試行全体のサマリ指標（周期・CV・ROM 等） |
| `waveform_*.png` | 手首 / 指先の変位波形グラフ |

## 主な解析指標

全タスク共通：
- **start_to_onset_s** : Cue から動作開始までの時間（秒）
- **cycle_time_mean_s_selected10** : 選択サイクルの平均サイクル時間（秒）
- **rhythm_cv_selected10** : リズム変動係数（SD / mean）
- **waveform_mean_corr_10** : 選択サイクルの波形類似度・平均相関
- **waveform_pass_10** : 波形類似度の合格判定（1=合格）

hammer 固有：
- **hit_time_mean_s** : 打鍵時間・平均（秒）
- **lift_time_mean_s** : 振り上げ時間・平均（秒）
- **direction_deg_abs_mean** : 運動方向角度・平均（度）
- **traj_len_px_mean** : 軌道長・平均（ピクセル）
- **vmax_px_s_mean** : 最大速度・平均（ピクセル/秒）
- **shoulder/elbow/wrist_deg_range_mean** : 肩・肘・手関節可動域・平均（度）

byebye / comehere 固有：
- **selected10_area_px2_mean_over_cycles** : 面積・平均（ピクセル²）
- **selected10_traj_len_px_mean_over_cycles** : 軌道長・平均（ピクセル）
- **selected10_max_speed_px_s_mean_over_cycles** : 最大速度・平均（ピクセル/秒）
- **shoulder/elbow/wrist/index_mcp_deg_range_mean** : 肩・肘・手関節・手指MP関節可動域・平均（度）

## ファイル構成

```
.
├── apraxia_app.py            # デスクトップGUI（tkinter）★現在の推奨起動方法
├── apraxia_app.command       # ダブルクリック起動用ランチャー（macOS 専用）
├── apraxia_app.bat           # ダブルクリック起動用ランチャー（Windows 専用）
├── apraxia_analysis/         # 解析モジュール（パッケージ）
│   ├── main.py               # CLI エントリポイント
│   ├── core/                 # 全タスク共通モジュール
│   │   ├── math_utils.py     # 信号処理・サイクル検出・角度計算など共通ユーティリティ
│   │   └── video_extractor.py# MediaPipe による座標抽出（全タスク共通）
│   └── tasks/
│       ├── hammer.py         # hammer タスク固有ロジック
│       ├── byebye.py         # byebye タスク固有ロジック
│       └── comehere.py       # comehere タスク固有ロジック
└── video_to_*_metrics.py     # 【旧版・参照用】タスク別スタンドアロンスクリプト
                               # 現在は apraxia_app.py + apraxia_analysis/ を使用
                               # 削除せず参照用として保持
```

## 環境構築

Python 3.10 以上を推奨。

```bash
pip install mediapipe opencv-python numpy pandas scipy matplotlib Pillow av
```

MediaPipe のモデルファイル（`.task`）は別途ダウンロードし、`models/` フォルダに配置してください：
- [pose_landmarker_full.task](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
- [hand_landmarker.task](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)

```
apraxia-analysis-main/
└── models/
    ├── pose_landmarker_full.task   ← Pose モデル
    └── hand_landmarker.task        ← Hand モデル（byebye / comehere 用）
```

`models/` フォルダに上記のファイル名で配置すると、アプリ起動時に自動でセットされます。

## 使い方

### A. デスクトップGUI（推奨）

```bash
python apraxia_app.py
```

ダブルクリックで起動するランチャーも用意されています：

| OS | ファイル | 操作 |
|----|---------|------|
| macOS | `apraxia_app.command` | ターミナルから `chmod +x apraxia_app.command` を実行後、ダブルクリック |
| Windows | `apraxia_app.bat` | ダブルクリック |

> **macOS の初回起動時**: Gatekeeper によりブロックされる場合は、右クリック →「開く」を選択してください。

### B. コマンドライン

```bash
# hammer タスク
python -m apraxia_analysis.main \
  --task hammer \
  --video /path/to/video.mov \
  --pose_model /path/to/pose_landmarker_full.task \
  --out_dir /path/to/output \
  --participant_id P001 --set_id 1 --trial_id 1

# byebye / comehere タスク（Hand モデルも必要）
python -m apraxia_analysis.main \
  --task byebye \
  --video /path/to/video.mov \
  --pose_model /path/to/pose_landmarker_full.task \
  --hand_model /path/to/hand_landmarker.task \
  --out_dir /path/to/output \
  --participant_id P001 --set_id 1 --trial_id 1
```

## サイクル選択アルゴリズム

1. 手首速度のベースライン（中央値 + k×MAD）を閾値として動作区間を自動検出
2. カスタム実装の局所極値検出（プロミネンス閾値・最小間隔・振れ幅フィルタ）でサイクルを検出
3. `target_cycles`（デフォルト10）個の連続サイクルを全窓で走査し、最適な窓を選択
   - **hammer / byebye** : サイクル時間の変動係数（CV）が最小の窓を選択
   - **comehere** : 波形類似度（平均相関）が最大の窓を優先し、同等の場合は CV 最小を選択
4. 選択ブロックの波形相関（各サイクル vs ブロック平均波形）で一致性を評価し `waveform_pass_10` を出力

## 動画形式について

### 対応形式

MOV / MP4 / AVI など主要な形式に対応しています。

### HEVC（H.265）動画について

iPhoneで撮影した動画は HEVC（H.265）形式の場合があります。  
HEVC動画は **解析時に自動的に H.264 形式へ変換** されます。変換には **PyAV** が必要です。

動画選択直後にプレビューエリアへ以下のメッセージが表示された場合、HEVC形式と判定されています：

> ⚠ HEVC（H.265）形式の可能性があります　解析時に自動的にH.264へ変換されます

### PyAV のインストール手順

**Mac / Windows 共通：**

```bash
pip install av
```

**Windows でエラーが出る場合：**

PyAV は内部で ffmpeg を使用しています。Windows 環境でインストールに失敗する場合は、先に ffmpeg をインストールしてください。

1. [ffmpeg 公式サイト](https://ffmpeg.org/download.html) からインストーラーをダウンロード
2. インストール後、`ffmpeg` にパスを通す（システム環境変数の `Path` に追加）
3. 改めて `pip install av` を実行

## Cueフレーム運用方針

本研究では、全課題・全試行において動画開始時点をCueフレームとし、`cue_frame = 0` を基本設定とする。

撮影開始後、約3秒間静止してから検者の合図で動作を開始する手順に統一している。

`start_to_onset_s` は、動画開始時点から解析上検出された動作開始時点までの時間として算出される。

## 注意事項

- 動画・解析結果ファイル（CSV / PNG）はリポジトリに含まれていません
- MediaPipe モデルファイル（`.task`）はサイズが大きいため含まれていません
- 解析数値の計算ロジックは `apraxia_analysis/tasks/` 内の各タスクファイルに実装されています
