# coding:utf-8
# 导入必备工具包
import torch
import torch.nn as nn
import sys
# sys.path.append('..')

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from glm_config import ProjectConfig

pc = ProjectConfig()


# 将数据类型转换为：torch.float32
class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def second2time(seconds: int):
    """
    将秒转换成时分秒。

    Args:
        seconds (int): _description_
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)


def save_model(model, cur_save_dir: str):
    """
    存储当前模型。
    Args:
        cur_save_path (str): 存储路径。
    """
    if pc.use_lora:
        # PEFT 模型只保存 LoRA adapter，避免复制和保存完整 6B 基础模型。
        model.save_pretrained(cur_save_dir)
    else:
        model.save_pretrained(cur_save_dir)
