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

- **cycle_time_mean_s** : 平均サイクル時間（秒）
- **rhythm_cv** : リズム変動係数（SD / mean）
- **waveform_pass_10** : 選択10サイクルの波形一致判定（1=合格）
- **rom_elbow_deg_mean** : 平均肘関節可動域（hammer）
- **x_range_px_mean_over_cycles** : 横方向振れ幅の平均（byebye / comehere）
- **start_to_onset_s** : Cue から動作開始までの時間（byebye / comehere）

## ファイル構成

```
.
├── apraxia_app.py            # デスクトップGUI（tkinter）
├── apraxia_app.command       # macOS ダブルクリック起動用ランチャー
├── apraxia_analysis/         # 解析モジュール（パッケージ）
│   ├── main.py               # CLI エントリポイント
│   ├── core/
│   │   ├── math_utils.py     # 信号処理・サイクル検出ユーティリティ
│   │   └── video_extractor.py# MediaPipe による座標抽出
│   └── tasks/
│       ├── hammer.py         # hammer タスク固有ロジック
│       ├── byebye.py         # byebye タスク固有ロジック
│       └── comehere.py       # comehere タスク固有ロジック
└── video_to_*_metrics.py     # タスク別スタンドアロンスクリプト（旧版）
```

## 環境構築

Python 3.10 以上を推奨。

```bash
pip install mediapipe opencv-python numpy pandas scipy matplotlib Pillow
```

MediaPipe のモデルファイル（`.task`）は別途ダウンロードしてください：
- [pose_landmarker_full.task](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
- [hand_landmarker.task](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)

## 使い方

### A. デスクトップGUI（推奨）

```bash
python apraxia_app.py
```

macOS の場合は `apraxia_app.command` をダブルクリックでも起動できます。

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

1. 動作区間を速度ベースで自動検出
2. 局所極値からサイクルを検出（scipy.signal.find_peaks）
3. `target_cycles`（デフォルト10）個の連続サイクルを全窓で試し、**サイクル時間CVが最小の窓**を選択
4. 選択ブロックの波形相関（各サイクル vs 平均波形）で一致性を評価

## 注意事項

- 動画・解析結果ファイル（CSV / PNG）はリポジトリに含まれていません
- MediaPipe モデルファイル（`.task`）はサイズが大きいため含まれていません
- 解析数値の計算ロジックは `apraxia_analysis/tasks/` 内の各タスクファイルに実装されています
