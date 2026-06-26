import os
import re
from typing import Dict, List, Tuple

# ネットワーク構造および関数構築のためのPyTorchライブラリ
import torch
from torch.utils.data import DataLoader, Dataset

# 実行デバイスの設定（GPUが利用可能であればCUDA、不可であればCPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 特殊トークンの定義
SOS_TOKEN = 0  # Start Of Sequence (開始フラグ)
EOS_TOKEN = 1  # End Of Sequence (終了フラグ)
PAD_TOKEN = 2  # Padding (パディングフラグ)

# データファイルの参照パス（実行ディレクトリに依存しない絶対パス）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "eng-fra-v2.txt")


def clean_text(text: str) -> str:
    """テキストのクリーニング処理を行います。

    小文字化、前後の空白削除、および記号の前にスペースを挿入する前処理を実行します。

    Args:
        text (str): 処理対象の文字列

    Returns:
        str: クリーニング済みの文字列
    """
    text = text.lower().strip()
    # 記号（.!?）の直前にスペースを挿入する正規表現
    text = re.sub(r"([.!?])", r" \1", text)
    # 英字と主要な記号以外の文字を除外（スペースに置換）
    text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)
    return text


def load_vocab_and_pairs() -> Tuple[
    Dict[str, int],
    Dict[str, int],
    int,
    int,
    Dict[int, str],
    Dict[int, str],
    List[List[str]],
]:
    """テキストデータを読み込み、英語とフランス語の対訳ペアおよび各単語辞書を構築します。

    Returns:
        tuple: 以下の要素を含むタプル
            - eng_vocab (dict): 英語の単語 -> IDの辞書
            - fra_vocab (dict): フランス語の単語 -> IDの辞書
            - eng_vocab_size (int): 英語の総単語数（ボキャブラリサイズ）
            - fra_vocab_size (int): フランス語の総単語数（ボキャブラリサイズ）
            - fra_idx2word (dict): フランス語のID -> 単語の逆引き辞書
            - eng_idx2word (dict): 英語のID -> 単語の逆引き辞書
            - pairs (list): クリーニング済みの対訳データリスト [[eng1, fra1], [eng2, fra2], ...]
    """
    # ファイルの読み込み（例外処理を考慮）
    try:
        with open(DATA_PATH, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError as e:
        print(f"エラー: データファイルが見つかりません。パスを確認してください: {DATA_PATH}")
        raise e

    # タブ区切りのテキストを分割し、各行をクリーニングして対訳ペアのリストを作成
    pairs = []
    for line in lines:
        columns = line.rstrip("\n").split("\t")
        if len(columns) < 2:
            continue
        pairs.append([clean_text(columns[0]), clean_text(columns[1])])

    # 初期単語辞書の定義（特殊トークンをあらかじめ登録）
    eng_vocab = {"SOS": SOS_TOKEN, "EOS": EOS_TOKEN, "PAD": PAD_TOKEN}
    eng_vocab_size = len(eng_vocab)

    fra_vocab = {"SOS": SOS_TOKEN, "EOS": EOS_TOKEN, "PAD": PAD_TOKEN}
    fra_vocab_size = len(fra_vocab)

    # 対訳ペアをループ処理し、未登録の単語を辞書に追加
    for pair in pairs:
        # 英語の単語登録
        for word in pair[0].split(" "):
            if word and (word not in eng_vocab):
                eng_vocab[word] = eng_vocab_size
                eng_vocab_size += 1

        # フランス語の単語登録
        for word in pair[1].split(" "):
            if word and (word not in fra_vocab):
                fra_vocab[word] = fra_vocab_size
                fra_vocab_size += 1

    # IDから単語を検索するための逆引き辞書（インデックスマップ）を作成
    eng_idx2word = {v: k for k, v in eng_vocab.items()}
    fra_idx2word = {v: k for k, v in fra_vocab.items()}

    return (
        eng_vocab,
        fra_vocab,
        eng_vocab_size,
        fra_vocab_size,
        fra_idx2word,
        eng_idx2word,
        pairs,
    )


# グローバル展開用のデータロード処理
(
    eng_vocab,
    fra_vocab,
    eng_vocab_size,
    fra_vocab_size,
    fra_idx2word,
    eng_idx2word,
    pairs,
) = load_vocab_and_pairs()


def get_data():
    """ボキャブラリ、逆引き辞書、対訳ペアをまとめて返します。"""
    return (
        eng_vocab,
        fra_vocab,
        eng_vocab_size,
        fra_vocab_size,
        fra_idx2word,
        eng_idx2word,
        pairs,
    )


class TranslationDataset(Dataset):
    """対訳ペアデータをPyTorch Tensorに変換して管理するカスタムデータセットクラスです。"""

    def __init__(self, data_pairs: List[List[str]]):
        """初期化処理

        Args:
            data_pairs (list): 対訳データのリスト
        """
        super().__init__()
        self.data_pairs = data_pairs
        self.total_samples = len(data_pairs)

    def __len__(self) -> int:
        """データセットの総サンプル数を返します。"""
        return self.total_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """指定されたインデックスのサンプルを取得し、テンソルに変換して返します。

        Args:
            index (int): 取得対象サンプルのインデックス

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 英語とフランス語のIDシーケンス（Tensor）
        """
        # インデックスに基づくデータの抽出
        eng_text = self.data_pairs[index][0]
        fra_text = self.data_pairs[index][1]

        # 英語テキストの単語をIDに変換し、末尾にEOSトークンを追加
        # 辞書に存在しない単語対策として .get() を使用（安全策）
        eng_ids = [eng_vocab.get(word, PAD_TOKEN) for word in eng_text.split()]
        eng_ids.append(EOS_TOKEN)
        tensor_x = torch.tensor(eng_ids, dtype=torch.long)

        # フランス語テキストの単語をIDに変換し、末尾にEOSトークンを追加
        fra_ids = [fra_vocab.get(word, PAD_TOKEN) for word in fra_text.split()]
        fra_ids.append(EOS_TOKEN)
        tensor_y = torch.tensor(fra_ids, dtype=torch.long)

        return tensor_x, tensor_y


MyPairDataset = TranslationDataset
PAD_token = PAD_TOKEN


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """ミニバッチ内のシーケンス長を揃えるためのパディング処理を行います（DataLoader用カスタム関数）。

    Args:
        batch (list): Datasetから取得した (tensor_x, tensor_y) のリスト

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: パディング済みのミニバッチデータ
    """
    xs, ys = zip(*batch)

    # バッチ内の一番長いシーケンス長を取得
    max_x_len = max(len(x) for x in xs)
    max_y_len = max(len(y) for y in ys)

    # パディングトークン（PAD_TOKEN）で初期化した固定長テンソルを作成
    batch_x = torch.full((len(batch), max_x_len), PAD_TOKEN, dtype=torch.long)
    batch_y = torch.full((len(batch), max_y_len), PAD_TOKEN, dtype=torch.long)

    # 各サンプルのデータを固定長テンソルにコピー
    for i, (x, y) in enumerate(zip(xs, ys)):
        batch_x[i, : len(x)] = x
        batch_y[i, : len(y)] = y

    return batch_x, batch_y


def get_dataloader(batch_size: int = 64) -> DataLoader:
    """データローダーを生成して返します。

    Args:
        batch_size (int, optional): ミニバッチのサイズ. Defaults to 64.

    Returns:
        DataLoader: PyTorchのDataLoaderインスタンス
    """
    dataset = TranslationDataset(pairs)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # 疎通確認用の処理（デバッグ時は必要に応じて有効化）
    for i, (batch_x, batch_y) in enumerate(dataloader):
        # print(f"batch_x --> {batch_x}")
        # print(f"batch_x.shape --> {batch_x.shape}")
        # print(f"batch_y --> {batch_y}")
        # print(f"batch_y.shape --> {batch_y.shape}")
        break

    return dataloader
