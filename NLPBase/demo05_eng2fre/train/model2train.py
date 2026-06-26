import json
import os
import random
import sys
import time
from datetime import datetime
from typing import List

# 複数ライブラリの重複ロードによるプロセス異常終了の防止
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

# CUI環境（バックエンドがない環境）でもグラフ保存を可能にする設定
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 自作モジュール用のインポートパス追加設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.AttentionDecoder import AttentionDecoder
from models.EncoderGRU import EncoderGRU
from utils.dataset import TranslationDataset, get_data, PAD_TOKEN, collate_fn

# 定数・ハイパーパラメータの定義
MAX_LENGTH = 10  # 扱う文章の最大単語数（記号含む）
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 5
TEACHER_FORCING_RATIO = 0.5
PRINT_INTERVAL = 1000  # ログ出力を行うステップ間隔
PLOT_INTERVAL = 100  # 損失記録・グラフ描画を行うステップ間隔

# 特殊トークンの定義
SOS_TOKEN = 0
EOS_TOKEN = 1

# 実行デバイスの設定（GPUが利用可能であればCUDA、不可であればCPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# パス関連の定義（プロジェクトルート基準）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
IMG_DIR = os.path.join(PROJECT_ROOT, "img")
SAVE_MODEL_DIR = os.path.join(PROJECT_ROOT, "save_model")
LOSS_IMG_PATH = os.path.join(IMG_DIR, "eng2french_loss.png")
LOSS_TXT_PATH = os.path.join(IMG_DIR, "eng2french_loss.txt")
TRAIN_META_PATH = os.path.join(SAVE_MODEL_DIR, "train_meta.json")


def train_step(
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        encoder: nn.Module,
        decoder: nn.Module,
        encoder_optimizer: optim.Optimizer,
        decoder_optimizer: optim.Optimizer,
        criterion: nn.Module,
) -> float:
    """1ミニバッチ分の訓練（順伝播、逆伝播、パラメータ更新）を実行します。

    Args:
        batch_x (torch.Tensor): 入力（英語）のIDテンソル [バッチサイズ, シーケンス長]
        batch_y (torch.Tensor): 正解（フランス語）のIDテンソル [バッチサイズ, シーケンス長]
        encoder (nn.Module): エンコーダーモデル
        decoder (nn.Module): デコーダーモデル
        encoder_optimizer (optim.Optimizer): エンコーダー用最適化器
        decoder_optimizer (optim.Optimizer): デコーダー用最適化器
        criterion (nn.Module): 損失関数 (nn.NLLLoss)

    Returns:
        float: このミニバッチにおけるステップ平均の損失値
    """
    batch_size = batch_x.size(0)
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    # 1. エンコーダーの順伝播処理
    init_hidden = encoder.init_hidden(batch_size)
    encoder_output, encoder_hidden = encoder(batch_x, init_hidden)

    # アテンションのKey/Value用にエンコーダー出力を固定長（MAX_LENGTH）にパディング
    seq_len = min(batch_x.size(1), MAX_LENGTH)
    encoder_outputs_padded = torch.zeros(
        batch_size, MAX_LENGTH, encoder.hidden_size, device=device
    )
    encoder_outputs_padded[:, :seq_len] = encoder_output[:, :seq_len]

    # デコーダーの初期状態設定（エンコーダーの最終隠れ状態を引き継ぐ）
    decoder_hidden = encoder_hidden
    decoder_input = torch.full((batch_size, 1), SOS_TOKEN, dtype=torch.long, device=device)

    target_length = batch_y.size(1)
    use_teacher_forcing = random.random() < TEACHER_FORCING_RATIO
    loss = 0.0

    # 2. デコーダーの構造（Teacher Forcingの有無による分岐）
    if use_teacher_forcing:
        # Teacher Forcingあり: 次ステップの入力として常に「実際の正解単語」を与える
        for di in range(target_length):
            output_probs, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs_padded)
            loss += criterion(output_probs, batch_y[:, di])
            decoder_input = batch_y[:, di].detach().unsqueeze(1)
    else:
        # Teacher Forcingなし: 次ステップの入力として「自身が予測した単語」を与える
        for di in range(target_length):
            output_probs, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs_padded)
            loss += criterion(output_probs, batch_y[:, di])
            top_idx = output_probs.argmax(dim=-1, keepdim=True)
            decoder_input = top_idx.detach()

    # 3. 逆伝播およびパラメータの更新
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    # ターゲットのシーケンス長で正規化した損失値を返す
    return loss.item() / target_length


def save_train_meta(epoch_idx: int, plot_loss_list: List[float]) -> None:
    """訓練のメタ情報をJSON形式で保存します。"""
    meta = {
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "epoch": epoch_idx,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs_total": EPOCHS,
        "latest_avg_loss": plot_loss_list[-1] if plot_loss_list else None,
        "encoder_path": os.path.join(SAVE_MODEL_DIR, f"ai20_encoder_{epoch_idx}.pth"),
        "decoder_path": os.path.join(SAVE_MODEL_DIR, f"ai20_decoder_{epoch_idx}.pth"),
    }
    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
    with open(TRAIN_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"メタ情報を保存しました: {os.path.abspath(TRAIN_META_PATH)}")


def save_loss_plot(plot_loss_list: List[float], img_path: str = LOSS_IMG_PATH) -> None:
    """損失の推移データをテキストに記録し、グラフ画像として保存します。"""
    os.makedirs(IMG_DIR, exist_ok=True)

    # 損失データをテキストファイルに書き出し
    with open(LOSS_TXT_PATH, "w", encoding="utf-8") as f:
        for loss_value in plot_loss_list:
            f.write(f"{loss_value:.6f}\n")

    # グラフの作成と保存
    plt.figure(figsize=(8, 4))
    plt.plot(plot_loss_list)
    plt.xlabel(f"Step (every {PLOT_INTERVAL} batches)")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig(img_path, dpi=150)
    plt.close()

    if not os.path.isfile(img_path):
        raise RuntimeError(f"損失グラフ画像の保存に失敗しました: {img_path}")

    print(f"損失グラフを保存しました: {os.path.abspath(img_path)}")
    print(f"損失データを保存しました: {os.path.abspath(LOSS_TXT_PATH)}")


def train_model() -> List[float]:
    """モデルの訓練全体を制御するメイン関数です。"""
    # データの読み込み
    eng_vocab, fra_vocab, eng_vocab_size, fra_vocab_size, _, _, pairs = get_data()
    dataset = TranslationDataset(pairs)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    hidden_size = 256

    # エンコーダーの初期化と設定
    encoder = EncoderGRU(eng_vocab_size, hidden_size).to(device)
    encoder.train()

    # アテンション付きデコーダーの初期化と設定
    decoder = AttentionDecoder(fra_vocab_size, hidden_size).to(device)
    decoder.train()

    # 最適化器（Optimizer）の設定
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)

    # 損失関数の設定（パディングトークンは損失計算から除外）
    criterion = nn.NLLLoss(ignore_index=PAD_TOKEN)

    plot_loss_list = []

    # エポックループ開始
    for epoch_idx in range(1, EPOCHS + 1):
        print_loss_total = 0.0
        plot_loss_total = 0.0
        start_time = time.time()

        # ミニバッチループ開始
        for step, (batch_x, batch_y) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch_idx}/{EPOCHS}"), start=1):
            loss = train_step(
                batch_x, batch_y, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion
            )
            print_loss_total += loss
            plot_loss_total += loss

            # 指定ステップごとにコンソールへログを出力
            if step % PRINT_INTERVAL == 0:
                avg_print_loss = print_loss_total / PRINT_INTERVAL
                print_loss_total = 0.0
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print(
                    f"Epoch: {epoch_idx} | Step: {step} | Loss: {avg_print_loss:.4f} | Time: {elapsed_time:.1f}s"
                )

            # 指定ステップごとにグラフ用損失データを記録
            if step % PLOT_INTERVAL == 0:
                avg_plot_loss = plot_loss_total / PLOT_INTERVAL
                plot_loss_list.append(avg_plot_loss)
                plot_loss_total = 0.0

        # 端数ステップの損失を処理
        remainder = step % PLOT_INTERVAL
        if plot_loss_total > 0 and remainder != 0:
            plot_loss_list.append(plot_loss_total / remainder)

        # モデルのチェックポイント（重み）保存
        os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
        encoder_path = os.path.join(SAVE_MODEL_DIR, f"ai20_encoder_{epoch_idx}.pth")
        decoder_path = os.path.join(SAVE_MODEL_DIR, f"ai20_decoder_{epoch_idx}.pth")

        torch.save(encoder.state_dict(), encoder_path)
        torch.save(decoder.state_dict(), decoder_path)
        print(f"エンコーダーモデルを保存しました: {os.path.abspath(encoder_path)}")
        print(f"デコーダーモデルを保存しました: {os.path.abspath(decoder_path)}")

        # 進行メタデータとグラフの保存
        save_train_meta(epoch_idx, plot_loss_list)
        save_loss_plot(plot_loss_list)

    return plot_loss_list


if __name__ == "__main__":
    train_model()
