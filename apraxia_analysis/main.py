#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""apraxia_analysis/main.py

統合 CLI エントリポイント。
--task hammer / byebye / comehere で解析タスクを切り替える。

使用例:
  python -m apraxia_analysis.main --task hammer   --video ... --pose_model ... --out_dir ...
  python -m apraxia_analysis.main --task byebye   --video ... --pose_model ... --hand_model ... --out_dir ...
  python -m apraxia_analysis.main --task comehere --video ... --pose_model ... --hand_model ... --out_dir ...
"""

from __future__ import annotations

import argparse
import sys


def main():
    # --task だけをここで取り出し、残りの引数はそのままタスク側へ渡す。
    top = argparse.ArgumentParser(add_help=False)
    top.add_argument("--task", choices=["hammer", "byebye", "comehere"], required=True,
                     help="解析タスクを選択: hammer / byebye / comehere")
    known, remaining = top.parse_known_args()

    if known.task == "hammer":
        from .tasks.hammer import run_hammer
        return run_hammer(remaining)
    elif known.task == "byebye":
        from .tasks.byebye import run_byebye
        return run_byebye(remaining)
    else:
        from .tasks.comehere import run_comehere
        return run_comehere(remaining)


if __name__ == "__main__":
    import traceback
    import faulthandler
    faulthandler.enable()
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
