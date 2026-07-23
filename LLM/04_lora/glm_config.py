# -*- coding:utf-8 -*-
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent


class ProjectConfig(object):
    def __init__(self):
        # 训练和推理所使用的计算设备：有可用 GPU 时使用第一张 GPU，否则使用 CPU。
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # 使用标准 ChatGLM2-6B；训练时由 bitsandbytes 动态量化为 4-bit。
        # 如果已经下载到本地，也可以改为完整模型目录的绝对路径。
        self.pre_model = "THUDM/chatglm2-6b"
        # 训练数据集文件路径。
        self.train_path = str(BASE_DIR / "data" / "mixed_train_dataset.jsonl")
        # 验证数据集文件路径，用于训练过程中评估模型效果。
        self.dev_path = str(BASE_DIR / "data" / "mixed_dev_dataset.jsonl")
        # 是否启用 LoRA（低秩适配）方式进行参数高效微调。
        self.use_lora = True
        # 标准 6B LoRA 不同时启用 P-Tuning。
        self.use_ptuning = False
        # LoRA 低秩矩阵的秩；值越大可训练参数越多，模型拟也合能力和显存占用通常越高。
        self.lora_rank = 8
        # 单个训练批次包含的样本数量。
        self.batch_size = 1
        # 完整遍历训练数据集的次数。
        self.epochs = 1
        # 优化器的初始学习率，控制每次参数更新的步长。
        self.learning_rate = 3e-5
        # 权重衰减系数，用于 L2 正则化以缓解过拟合；0 表示不启用。
        self.weight_decay = 0
        # 学习率预热阶段占总训练步数的比例。
        self.warmup_ratio = 0.06
        # 8GB 显存先使用较短序列；显存仍不足时可继续降为 96/48。
        self.max_source_seq_len = 128
        self.max_target_seq_len = 64
        # 每隔多少个训练步记录一次损失等训练日志。
        self.logging_steps = 10
        # 每隔多少个训练步保存一次模型检查点。
        self.save_freq = 200
        # P-Tuning 使用的连续前缀 token 数量。
        self.pre_seq_len = 128
        # 是否使用 MLP 对前缀向量进行投影；仅在启用 P-Tuning 时生效。
        self.prefix_projection = False
        # LoRA adapter、tokenizer 及训练检查点的保存目录。
        self.save_dir = str(BASE_DIR / "lora_checkpoints")


if __name__ == '__main__':
    pc = ProjectConfig()
    print(pc.save_dir)
