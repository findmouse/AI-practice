"""
歌詞テキスト生成（RNN）

Jay Chou の歌詞コーパス（中国語）を用いて、単語単位の次トークン予測モデルを学習し、
開始語から続きのテキストを生成する。
"""

from __future__ import annotations

from pathlib import Path

import jieba
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# パス & デバイス
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "jaychou_lyrics.txt"
MODEL_PATH = BASE_DIR / "model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# ハイパーパラメータ
# ---------------------------------------------------------------------------
EMBED_DIM = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 1
SEQ_LEN = 64          # 長い系列ほど GPU 演算密度が上がる
BATCH_SIZE = 256      # 旧 2 → 顕存に余裕があるため大幅増
EPOCHS = 10
# batch 拡大に合わせて学習率も上げる（おおよそ線形スケールの控えめ版）
LEARNING_RATE = 1e-3
ADAM_BETAS = (0.9, 0.99)
NUM_WORKERS = 4       # DataLoader 並列読み込み
USE_AMP = True        # CUDA 混合精度（fp16）で高速化

# 生成時のサンプリング設定
TEMPERATURE = 0.8
TOP_K = 50
TOP_P = 0.9


# ---------------------------------------------------------------------------
# 語彙構築
# ---------------------------------------------------------------------------
def build_vocab(data_path: Path = DATA_PATH):
    """歌詞ファイルを読み込み、語彙とコーパス ID 列を構築する。"""
    all_words: list[list[str]] = []

    # 1. データ読み込み & 分かち書き（中国語: jieba）
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            all_words.append(jieba.lcut(line))

    # 2. 出現順を保ったまま高速に重複除去
    flat_words = [w for words in all_words for w in words]
    unique_words = list(dict.fromkeys(flat_words))
    if " " not in unique_words:
        unique_words.append(" ")

    # 3. 単語 → インデックス辞書
    word2index = {word: i for i, word in enumerate(unique_words)}

    # 4. テキストを ID 列へ変換（行末に空白トークンを付与）
    space_id = word2index[" "]
    corpus_id: list[int] = []
    for words in all_words:
        corpus_id.extend(word2index[w] for w in words)
        corpus_id.append(space_id)

    return unique_words, word2index, corpus_id


# ---------------------------------------------------------------------------
# データセット
# ---------------------------------------------------------------------------
class LyricsDataset(Dataset):
    """固定長シーケンスの (入力, 次トークン) ペアを返すデータセット。"""

    def __init__(self, corpus_id: list[int], num_char: int):
        # 事前に LongTensor 化して __getitem__ の毎回変換コストを削減
        self.corpus_id = torch.tensor(corpus_id, dtype=torch.long)
        self.num_char = num_char
        self.word_count = len(self.corpus_id)
        self.num = max(self.word_count - num_char - 1, 0)

    def __len__(self) -> int:
        return self.num

    def __getitem__(self, idx: int):
        x = self.corpus_id[idx : idx + self.num_char]
        y = self.corpus_id[idx + 1 : idx + 1 + self.num_char]
        return x, y


# ---------------------------------------------------------------------------
# モデル
# ---------------------------------------------------------------------------
class TextGenerator(nn.Module):
    """Embedding → RNN → Linear による次トークン予測モデル。"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = EMBED_DIM,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self.out = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(
        self, inputs: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # inputs: (batch, seq) → embeds: (batch, seq, embed_dim)
        embeds = self.embed(inputs)
        # RNN は (seq, batch, feature) を期待するため転置
        out, hidden = self.rnn(embeds.transpose(0, 1), hidden)
        # (seq * batch, hidden) → 語彙サイズのロジット
        logits = self.out(out.reshape(-1, self.hidden_size))
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device = DEVICE) -> torch.Tensor:
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


# ---------------------------------------------------------------------------
# チェックポイント
# ---------------------------------------------------------------------------
def save_checkpoint(
    model: TextGenerator,
    unique_words: list[str],
    path: Path = MODEL_PATH,
) -> None:
    """重みと語彙をまとめて保存する（再学習なしで推論可能にする）。"""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "unique_words": unique_words,
            "config": {
                "embed_dim": EMBED_DIM,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
            },
        },
        path,
    )
    print(f"チェックポイントを保存しました: {path}")


def load_checkpoint(
    path: Path = MODEL_PATH,
    device: torch.device = DEVICE,
) -> tuple[TextGenerator, list[str], dict[str, int]]:
    """チェックポイントからモデルと語彙を復元する。"""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    unique_words: list[str] = ckpt["unique_words"]
    word2index = {word: i for i, word in enumerate(unique_words)}
    config = ckpt.get("config", {})
    model = TextGenerator(
        vocab_size=len(unique_words),
        embed_dim=config.get("embed_dim", EMBED_DIM),
        hidden_size=config.get("hidden_size", HIDDEN_SIZE),
        num_layers=config.get("num_layers", NUM_LAYERS),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    return model, unique_words, word2index


# ---------------------------------------------------------------------------
# 学習
# ---------------------------------------------------------------------------
def train(
    dataset: LyricsDataset,
    model: TextGenerator,
    unique_words: list[str],
    device: torch.device = DEVICE,
) -> None:
    """クロスエントロピー損失でモデルを学習し、チェックポイントを保存する。"""
    model.to(device)
    model.train()

    # 入力サイズが固定なので cuDNN オートチューンを有効化
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
    )
    use_amp = USE_AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    loader_kwargs: dict = {
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "drop_last": True,
        "pin_memory": device.type == "cuda",
        "num_workers": NUM_WORKERS if device.type == "cuda" else 0,
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    dataloader = DataLoader(dataset, **loader_kwargs)
    print(
        f"train config: batch={BATCH_SIZE}, seq={SEQ_LEN}, "
        f"amp={use_amp}, workers={loader_kwargs['num_workers']}"
    )

    for epoch in range(EPOCHS):
        loss_sum = 0.0
        n_batches = 0

        for x, y in dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            h0 = model.init_hidden(batch_size=BATCH_SIZE, device=device)
            y_flat = y.transpose(0, 1).contiguous().view(-1)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, _ = model(x, h0)
                loss = criterion(logits, y_flat)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item()
            n_batches += 1

        avg_loss = loss_sum / max(n_batches, 1)
        print(f"epoch {epoch + 1}/{EPOCHS}  loss={avg_loss:.6f}")

    save_checkpoint(model, unique_words)


# ---------------------------------------------------------------------------
# サンプリング
# ---------------------------------------------------------------------------
def sample_next_token(
    logits: torch.Tensor,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
    top_p: float = TOP_P,
) -> int:
    """temperature + top-k + top-p で次トークンをサンプリングする。"""
    # logits: (vocab,) または (1, vocab) など → 1 次元へ
    logits = logits.reshape(-1).float()

    if temperature <= 0:
        return int(torch.argmax(logits))

    logits = logits / temperature

    # top-k: 上位 k 以外をマスク
    if top_k > 0 and top_k < logits.numel():
        topk_values, _ = torch.topk(logits, top_k)
        min_keep = topk_values[-1]
        logits = logits.masked_fill(logits < min_keep, float("-inf"))

    # top-p (nucleus): 累積確率が p を超える尾部をマスク
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)

        # 累積が top_p を超えた位置以降を落とす（先頭は必ず残す）
        remove_mask = cumulative > top_p
        remove_mask[1:] = remove_mask[:-1].clone()
        remove_mask[0] = False

        indices_to_remove = sorted_indices[remove_mask]
        logits = logits.clone()
        logits[indices_to_remove] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


# ---------------------------------------------------------------------------
# 推論（テキスト生成）
# ---------------------------------------------------------------------------
def predict(
    start_word: str,
    length: int,
    model_path: Path = MODEL_PATH,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
    top_p: float = TOP_P,
    device: torch.device = DEVICE,
) -> str:
    """チェックポイントを読み込み、開始語から続きを生成して返す。"""
    model, unique_words, word2index = load_checkpoint(model_path, device=device)
    model.eval()

    if start_word not in word2index:
        raise KeyError(f"開始語 '{start_word}' が語彙にありません。")

    word_index = word2index[start_word]
    hidden = model.init_hidden(batch_size=1, device=device)
    words_list: list[str] = [start_word]

    with torch.no_grad():
        for _ in range(length):
            inp = torch.tensor([[word_index]], device=device)
            logits, hidden = model(inp, hidden)
            # 1 ステップ分のロジットのみ使用
            word_index = sample_next_token(
                logits[-1],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            words_list.append(unique_words[word_index])

    text = "".join(words_list)
    print(text)
    return text


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"device: {DEVICE}")
    print(f"data:   {DATA_PATH}")

    # --- 学習（必要なときだけ語彙・データセット・モデルを構築）---
    # unique_words, word2index, corpus_id = build_vocab()
    # dataset = LyricsDataset(corpus_id, SEQ_LEN)
    # model = TextGenerator(len(unique_words))
    # print(f"語彙サイズ: {len(unique_words)}, コーパス長: {len(corpus_id)}")
    # train(dataset, model, unique_words)

    # --- 生成（チェックポイントから復元するため、上記の構築は不要）---
    predict("青春", length=50)
