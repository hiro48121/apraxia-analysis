#!/bin/bash
# apraxia_app.command
# macOS でダブルクリックするとターミナルが開いてアプリが起動します

# このスクリプトがある場所に移動
cd "$(dirname "$0")"

# Python インタープリタのパス
PYTHON="$HOME/Desktop/hammer_project/.venv/bin/python"

# Python の存在確認
if [ ! -f "$PYTHON" ]; then
    echo "========================================"
    echo "エラー: Python が見つかりません"
    echo "  $PYTHON"
    echo ""
    echo "PYTHON= の行を編集して正しいパスを指定してください"
    echo "========================================"
    read -p "Enterキーで終了..."
    exit 1
fi

echo "Apraxia Analysis App を起動します..."
"$PYTHON" apraxia_app.py

# アプリが終了したらターミナルを自動で閉じる
# （エラーが出た場合はコメントアウトしてください）
# exit 0
