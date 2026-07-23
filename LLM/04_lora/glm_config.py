# -*- coding:utf-8 -*-
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent


class ProjectConfig(object):
    def __init__(self):
        # 学習・推論に使用する計算デバイス：利用可能な GPU があれば1枚目の GPU を使用し、なければ CPU を使用する。
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # 標準の ChatGLM2-6B を使用する。学習時は bitsandbytes によって動的に 4-bit 量子化される。
        # すでにローカルにダウンロード済みの場合は、完全なモデルディレクトリの絶対パスに変更してもよい。
        self.pre_model = "THUDM/chatglm2-6b"
        # 学習データセットファイルのパス。
        self.train_path = str(BASE_DIR / "data" / "mixed_train_dataset.jsonl")
        # 検証データセットファイルのパス。学習中のモデル評価に使用する。
        self.dev_path = str(BASE_DIR / "data" / "mixed_dev_dataset.jsonl")
        # LoRA（低ランク適応）によるパラメータ効率の良いファインチューニングを有効にするかどうか。
        self.use_lora = True
        # 標準の 6B LoRA では P-Tuning を同時に有効化しない。
        self.use_ptuning = False
        # LoRA 低ランク行列のランク。値が大きいほど学習可能パラメータが増え、モデルの表現力と VRAM 使用量も通常大きくなる。
        self.lora_rank = 8
        # 1つの学習バッチに含まれるサンプル数。
        self.batch_size = 1
        # 学習データセットを一通り走査する回数（エポック数）。
        self.epochs = 1
        # オプティマイザの初期学習率。パラメータ更新のステップ幅を制御する。
        self.learning_rate = 3e-5
        # 重み減衰係数。L2 正則化により過学習を緩和する。0 は無効を意味する。
        self.weight_decay = 0
        # 学習率のウォームアップ段階が総学習ステップ数に占める割合。
        self.warmup_ratio = 0.06
        # VRAM 8GB ではまず短めのシーケンス長を使用する。それでも不足する場合は 96/48 までさらに下げてよい。
        self.max_source_seq_len = 128
        self.max_target_seq_len = 64
        # 何ステップごとに損失などの学習ログを記録するか。
        self.logging_steps = 10
        # 何ステップごとにモデルのチェックポイントを保存するか。
        self.save_freq = 200
        # P-Tuning で使用する連続プレフィックス token 数。
        self.pre_seq_len = 128
        # プレフィックスベクトルを MLP で射影するかどうか。P-Tuning 有効時のみ効果がある。
        self.prefix_projection = False
        # LoRA adapter、tokenizer、および学習チェックポイントの保存ディレクトリ。
        self.save_dir = str(BASE_DIR / "lora_checkpoints")


if __name__ == '__main__':
    pc = ProjectConfig()
    print(pc.save_dir)
