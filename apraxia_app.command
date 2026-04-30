#!/bin/bash
# apraxia_app.command
# macOS でダブルクリックするとターミナルが開いてアプリが起動します

# このスクリプトがある場所に移動
cd "$(dirname "$0")"

# Python インタープリタのパス（優先順）
# 1. スクリプトと同じフォルダの .venv を使用
# 2. システムの python3 を使用
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    echo "========================================"
    echo "エラー: Python が見つかりません"
    echo ""
    echo "以下のいずれかをお試しください："
    echo "  1. .venv/bin/python が存在するか確認"
    echo "  2. python3 をインストール"
    echo "========================================"
    read -p "Enterキーで終了..."
    exit 1
fi

echo "Apraxia Analysis App を起動します..."
"$PYTHON" apraxia_app.py
