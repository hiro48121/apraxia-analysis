# Apraxia Analysis

上肢失行症の動作解析ツール。
スマートフォンで撮影した動画から MediaPipe でキーポイントを抽出し、
繰り返し動作のサイクル指標（周期・振れ幅・ROM・リズム変動係数など）を自動算出する。

---

## 対応タスク

| タスク | 動作 | 使用モデル |
|--------|------|-----------|
| `hammer` | 打鍵動作（ハンマー振り） | Pose |
| `byebye` | 手を振る動作（バイバイ） | Pose + Hand |
| `comehere` | 手招き動作（おいでおいで） | Pose + Hand |

---

## 出力ファイル

解析結果は以下のサブフォルダ構成で保存されます：

```
出力先フォルダ/
└── {参加者ID}/
    └── {タスク}/
        └── set{セットID}_trial{試行ID}/
            ├── frames.csv
            ├── cycles.csv
            ├── summary.csv
            ├── waveform_*.png
            └── overlay_*.mp4   ← オーバーレイ動画作成時のみ生成
```

| ファイル | 内容 |
|---------|------|
| `frames.csv` | フレームごとのキーポイント座標・速度・角度 |
| `cycles.csv` | 検出された各サイクルの指標 |
| `summary.csv` | 試行全体のサマリ指標（周期・CV・ROM 等） |
| `waveform_*.png` | 変位波形グラフ（hammer：手首、byebye/comehere：示指） |
| `overlay_*.mp4` | 骨格ランドマークを重畳したオーバーレイ動画 |

---

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

---

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

---

## 環境構築

Python 3.10 以上を推奨。

```bash
pip install mediapipe opencv-python numpy pandas scipy matplotlib Pillow av
```

> **matplotlib** は GUI の波形連動表示機能に必要です。インストールされていない場合、波形グラフは表示されません（解析自体は実行されます）。

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

---

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

#### GUI の機能概要

**左パネル**

| 機能 | 説明 |
|------|------|
| 動画選択・プレビュー | 解析対象の動画ファイルを選択。選択後に先頭フレームをプレビュー表示 |
| 動画プレーヤー | 再生・停止・コマ送り・シークスライダーで動画を確認 |
| 波形連動表示 | 解析完了後に波形グラフを表示。動画の再生位置に連動して縦線カーソルが動く |
| ログ | 解析の進捗・エラーメッセージをリアルタイム表示 |

**右パネル**

| 機能 | 説明 |
|------|------|
| タスク・側の選択 | hammer / byebye / comehere、Left / Right を選択 |
| 患者情報入力 | 参加者ID・セットID・試行ID・Cueフレームを入力 |
| モデルファイル設定 | Pose / Hand モデルのパスを指定 |
| 出力先フォルダ設定 | 解析結果の保存先を指定 |
| 解析前チェック | 解析開始ボタン押下時に動画・モデル・FPS・HEVC・パス・Cueフレームなどを自動確認。停止項目があれば解析を止め、詳細ウィンドウで理由を表示 |
| 解析開始ボタン | 解析をバックグラウンドで実行 |
| オーバーレイ動画作成ボタン | 解析後に骨格ランドマークを重畳した動画を生成 |
| スクリーンショット保存ボタン | 現在の動画フレームと波形グラフを1枚のPNGとして出力フォルダに保存（解析完了後に有効化） |
| 解析結果サマリ | `summary.csv` の主要指標を解析完了後に日本語ラベル付きで表示（形式：`日本語ラベル（列名）：値`） |

#### 波形連動表示について

解析完了後、左パネルの「波形連動表示」エリアに波形グラフが自動表示されます。

- **横軸**：時刻（秒）
- **縦軸**：タスクに応じた変位
  - hammer：`wrist_y_px_sm`（垂直方向の平滑化手首位置）
  - comehere：`index_y_px_sm`（垂直方向の平滑化示指位置）
  - byebye：`index_x_px_sm`（水平方向の平滑化示指位置）
- **赤い縦線**：動画の現在再生位置に対応するカーソル
- 外れ値フレーム（`outlier_flag = 1`）および前後 ±1 フレームはマスクされ、線が途切れて表示されます
- `frames.csv` は読み取り専用で参照しており、ファイルは変更されません
- 波形グラフをクリックすると、クリックした時刻に対応するフレームへ動画がシークします

#### オーバーレイ動画について

解析完了後に「オーバーレイ動画を作成」ボタンを押すと、骨格ランドマーク（肩・肘・手首・指先）を重畳した動画（`overlay_*.mp4`）が出力フォルダに生成されます。

タスクによって表示内容が異なります：

| タスク | 表示内容 |
|--------|---------|
| `hammer` | 肩・肘・手首・示指先端（Pose ランドマーク） |
| `byebye` / `comehere` | 上記に加え、Hand ランドマーク（手首・示指 MCP・PIP・先端）も表示 |

既存の `summary.csv`・`cycles.csv`・`frames.csv`・waveform PNG はオーバーレイ動画作成時に変更されません。

#### 解析前チェックについて

解析開始ボタンを押すと、解析実行前に以下の項目が自動確認されます。

| チェック項目 | 停止（❌） | 注意（⚠️） | OK（✅） |
|---|---|---|---|
| 動画ファイル | 未選択・存在しない | — | 存在する |
| 動画読み込み | 読み込み失敗 | HEVC形式（変換対応） | 読み込み可 |
| FPS取得 | 取得失敗 | 60fps想定と異なる | 取得成功 |
| フレーム数取得 | 取得失敗 | HEVC形式 | 取得成功 |
| Poseモデル | 未設定・見つからない | — | 存在する |
| Handモデル | byebye/comehere で未設定 | — | OK（hammerは不要） |
| 出力先フォルダ | 使用不可 | — | OK |
| 日本語・全角パス | Windows で検出 | macOS で検出 | なし |
| Cueフレーム | — | 0以外 | 0 |
| HEVC判定 | — | HEVC形式または判定不可 | 通常動画 |
| タスク選択 | 未選択 | — | 選択済み |

- **停止項目あり**：解析は開始されません。詳細ウィンドウで理由を確認してください。
- **注意のみ**：追加の確認ダイアログなしで解析を続行します。
- 「詳細」ボタンから詳細ウィンドウをいつでも開けます（非モーダル）。
- HEVC形式の動画は自動的にH.264へ変換されるため、HEVC検出は停止理由になりません。

#### スクリーンショット保存について

解析完了後、「スクリーンショット保存」ボタンが有効化されます。ボタンを押すと、現在プレーヤーに表示されている動画フレームと波形グラフを1枚のPNG画像として出力フォルダに保存します。

- ファイル名：`screenshot_{動画名}_frame{フレーム番号}.png`
- 保存先：解析結果フォルダ（`set{セットID}_trial{試行ID}/` 直下）
- レイアウト：上段に情報バー（タスク・フレーム番号・時刻・波形列名）、中段に動画フレーム、下段に波形グラフ

---

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

---

## サイクル選択アルゴリズム

1. 手首速度のベースライン（中央値 + k×MAD）を閾値として動作区間を自動検出
2. カスタム実装の局所極値検出（プロミネンス閾値・最小間隔・振れ幅フィルタ）でサイクルを検出
3. `target_cycles`（デフォルト10）個の連続サイクルを全窓で走査し、最適な窓を選択
   - **hammer / byebye** : サイクル時間の変動係数（CV）が最小の窓を選択
   - **comehere** : 波形類似度（平均相関）が最大の窓を優先し、同等の場合は CV 最小を選択
4. 選択ブロックの波形相関（各サイクル vs ブロック平均波形）で一致性を評価し `waveform_pass_10` を出力

---

## 動画形式について

### 対応形式

MOV / MP4 / AVI など主要な形式に対応しています。

### HEVC（H.265）動画について

iPhoneで撮影した動画は HEVC（H.265）形式の場合があります。
HEVC動画は **解析時に自動的に H.264 形式へ変換** されます。変換には **PyAV** が必要です。

動画選択直後にプレビューエリアへ以下のいずれかのメッセージが表示された場合、HEVC形式と判定されています：

> ⚠ HEVC（H.265）形式です　解析時に自動的にH.264へ変換されます

（PyAV がインストール済みの場合：コーデックを直接確認して確定表示）

> ⚠ HEVC（H.265）形式の可能性があります　解析時に自動的にH.264へ変換されます

（PyAV 未インストールまたは確認失敗の場合：タイムアウトにより推定表示）

HEVC形式の動画は、解析時にH.264形式へ自動変換される場合があります。変換後のファイルは `{元ファイル名}_h264.mp4` として元動画と同じフォルダに保存されます（パスに日本語が含まれる場合はアプリフォルダ直下に保存）。次回以降は変換済みファイルを直接選択することを推奨します。

動画変換により、ランドマーク推定や算出指標にごく軽微な差が生じる可能性があります。同一データの比較や再解析では、同一形式の動画を用いることを推奨します。

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

---

## Cueフレーム運用方針

本研究では、全課題・全試行において動画開始時点をCueフレームとし、`cue_frame = 0` を基本設定とする。

撮影開始後、約3秒間静止してから検者の合図で動作を開始する手順に統一している。

`start_to_onset_s` は、動画開始時点から解析上検出された動作開始時点までの時間として算出される。

> **注意**：`cue_frame` を 0 以外に設定した状態で解析を開始すると確認ダイアログが表示されます。「このまま解析する」「0 に戻す」「キャンセル」の3択から選択してください。

---

## トラブルシューティング

| 症状 | 原因・対処 |
|------|----------|
| HEVC動画が開けない / プレビューが表示されない | PyAV が未インストールの可能性。`pip install av` を実行してください（Windows では ffmpeg のインストールが先に必要な場合あり） |
| 解析開始後にアプリがフリーズする | HEVC動画を変換せずに解析しようとしている可能性。PyAV をインストールするか、あらかじめ H.264 形式に変換した動画を使用してください |
| `モデルファイルが見つかりません` エラー | `models/` フォルダに `pose_landmarker_full.task`（および byebye/comehere では `hand_landmarker.task`）が配置されているか確認してください |
| 解析が異常終了する（Windowsのみ） | パスに日本語・全角文字が含まれていないか確認してください。含まれている場合は半角英数字のみのパスに変更してください |
| 解析が異常終了する（Mac） | ログに日本語パスに関するヒントが表示されている場合は同様にパスを確認してください |
| オーバーレイ動画が作成できない | 解析を先に実行してから「オーバーレイ動画を作成」ボタンを押してください。`opencv-python` が未インストールの場合は `pip install opencv-python` を実行してください |

---

## 注意事項

- 動画・解析結果ファイル（CSV / PNG / MP4）はリポジトリに含まれていません
- MediaPipe モデルファイル（`.task`）はサイズが大きいため含まれていません
- 解析数値の計算ロジックは `apraxia_analysis/tasks/` 内の各タスクファイルに実装されています
- **日本語・全角文字を含むパスは使用しないでください**：動画ファイル・モデルファイル・出力先フォルダ・アプリフォルダのいずれかに日本語や全角文字が含まれると、Windows 環境では MediaPipe が正常に動作しない場合があります。半角英数字のみのパスを推奨します（Mac 環境でも解析が異常終了する場合は同様に確認してください）。
