@echo off
chcp 65001 > nul
cd /d "%~dp0"

rem Python インタープリタのパス（優先順）
rem 1. スクリプトと同じフォルダの .venv を使用
rem 2. システムの python を使用
if exist ".venv\Scripts\python.exe" (
    set PYTHON=.venv\Scripts\python.exe
) else (
    where python > nul 2>&1
    if %errorlevel% == 0 (
        set PYTHON=python
    ) else (
        echo ========================================
        echo エラー: Python が見つかりません
        echo.
        echo 以下のいずれかをお試しください：
        echo   1. .venv\Scripts\python.exe が存在するか確認
        echo   2. Python をインストール
        echo ========================================
        pause
        exit /b 1
    )
)

echo Apraxia Analysis App を起動します...
%PYTHON% apraxia_app.py
pause
