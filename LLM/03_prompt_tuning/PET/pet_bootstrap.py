# coding:utf-8
"""
PET パス誘導（標準ライブラリのみ）。

任意のスクリプト先頭で次のように使う（本モジュールを先に import できなくても可）：

    import sys
    from pathlib import Path
    for _r in Path(__file__).resolve().parents:
        if (_r / "pet_bootstrap.py").is_file():
            if str(_r) not in sys.path:
                sys.path.insert(0, str(_r))
            break
    from pet_bootstrap import bootstrap
    bootstrap(__file__)

より短い同等の書き方（ProjectConfig.py を直接探す）は bootstrap() 実装を参照。
LLM 環境で一度 pip install -e . してもよい。
その後は: from pet_bootstrap import bootstrap; bootstrap(__file__)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, os.PathLike]


def get_project_root(start_file: Optional[PathLike] = None) -> Path:
    """start_file から上方向に ProjectConfig.py を含む PET ルートを探す。"""
    start = Path(start_file or __file__).resolve()
    for parent in [start.parent, *start.parents]:
        if (parent / "ProjectConfig.py").is_file():
            return parent
    raise RuntimeError("Cannot find PET project root (ProjectConfig.py)")


def bootstrap(start_file: Optional[PathLike] = None) -> str:
    """PET ルートを sys.path に挿入（冪等）し、ルートパス文字列を返す。"""
    root = get_project_root(start_file)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    os.environ.setdefault("PET_ROOT", root_str)
    return root_str


# 旧名との互換
ensure_project_root = bootstrap


if __name__ == "__main__":
    print(bootstrap(__file__))
