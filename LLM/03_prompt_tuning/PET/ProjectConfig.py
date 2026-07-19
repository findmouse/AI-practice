# coding:utf-8
import sys
from pathlib import Path

import torch

# import 時にプロジェクトルートを sys.path へ追加し、data_handle / utils を import できるようにする
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class ProjectConfig(object):
    def __init__(self):
        # プロジェクトルート（本ファイルのあるディレクトリ）。サブディレクトリから実行しても相対パスが壊れないようにする
        root = _ROOT
        # GPU を使うか
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # 事前学習 BERT モデルのパス
        self.root = root
        self.pre_model = root / 'bert-base-chinese'
        self.checkpoint = root / 'checkpoints' / 'model_best'
        self.train_path = root / 'data' / 'train.txt'
        self.dev_path = root / 'data' / 'dev.txt'
        self.prompt_file = root / 'data' / 'prompt.txt'
        self.verbalizer = root / 'data' / 'verbalizer.txt'
        self.max_seq_len = 128
        self.batch_size = 4
        self.learning_rate = 5e-5
        # 重み減衰（正則化、過学習抑制）
        self.weight_decay = 0
        # ウォームアップ比率（ウォームアップステップ数の算出に使用）
        self.warmup_ratio = 0.06
        self.max_label_len = 2
        self.epochs = 1
        self.logging_steps = 10
        self.valid_steps = 20
        self.save_dir = root / 'checkpoints'


if __name__ == '__main__':
    pc = ProjectConfig()
    print(pc.train_path)
    print(pc.dev_path)
