# coding:utf-8
# 必要なツールパッケージをインポートする
import torch
import torch.nn as nn
import sys
# sys.path.append('..')

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from glm_config import ProjectConfig

pc = ProjectConfig()


# データ型を torch.float32 に変換する
class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def second2time(seconds: int):
    """
    秒を「時:分:秒」形式に変換する。

    Args:
        seconds (int): _description_
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)


def save_model(model, cur_save_dir: str):
    """
    現在のモデルを保存する。
    Args:
        cur_save_path (str): 保存先パス。
    """
    if pc.use_lora:
        # PEFT モデルは LoRA adapter のみを保存し、完全な 6B ベースモデルの複製・保存を避ける。
        model.save_pretrained(cur_save_dir)
    else:
        model.save_pretrained(cur_save_dir)
