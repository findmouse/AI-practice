# -*- coding: utf-8 -*-
import os
import torch


class ParameterConfig:
    def __init__(self):
        # プロジェクトのルートディレクトリ（本ファイルがあるディレクトリ）
        self.project_root = os.path.dirname(os.path.abspath(__file__))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab_path = os.path.join(self.project_root, 'vocab', 'vocab.txt')
        self.train_txt_path = os.path.join(self.project_root, 'data', 'medical_train.txt')
        self.valid_txt_path = os.path.join(self.project_root, 'data', 'medical_valid.txt')
        self.train_path = os.path.join(self.project_root, 'data', 'medical_train.pkl')
        self.valid_path = os.path.join(self.project_root, 'data', 'medical_valid.pkl')
        self.config_json = os.path.join(self.project_root, 'config', 'config.json')
        self.save_model_path = os.path.join(self.project_root, 'save_model')
        self.pretrained_model = ''
        self.save_samples_path = os.path.join(self.project_root, 'sample')
        self.loss_history_path = os.path.join(self.project_root, 'loss_history.json')
        self.loss_curve_path = os.path.join(self.project_root, 'loss_curve.png')

        self.ignore_index = -100
        self.max_history_len = 3
        self.max_len = 300
        self.repetition_penalty = 1.2
        self.topk = 8
        self.topp = 0.0
        self.batch_size = 4
        self.epochs = 2
        self.loss_step = 50
        self.lr = 2.6e-5
        self.eps = 1.0e-09
        self.max_grad_norm = 2.0
        self.gradient_accumulation_steps = 1
        self.warmup_steps = 100
        # 動作確認時は正の整数で学習サンプル数を制限。None は全データを使用
        self.max_train_samples = None
        self.max_valid_samples = None


if __name__ == '__main__':
    pc = ParameterConfig()
    print(pc.train_path)
    print(pc.device)
    print(torch.cuda.device_count())
