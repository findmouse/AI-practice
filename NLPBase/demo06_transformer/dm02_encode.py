# -*- coding: utf-8 -*-
"""
システム名：自然言語処理・基盤AIシステム
モジュール名：transformer_encoder
作成日：2026/06/26
作成者：
修正履歴：

"""

import copy
import torch.nn.functional as F
from dm01_input import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def subsequent_mask(size: int) -> torch.Tensor:
    """自動回帰モデル用（Look-ahead）下三角マスク行列を生成する関数。

    デコーダが未来のトークンをアテンションで参照しないように隠蔽します。
    """
    sub_mask = np.triu(m=np.ones((1, size, size)), k=1).astype('uint8')
    return torch.from_numpy(1 - sub_mask)


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
              mask: torch.Tensor = None, dropout: nn.Dropout = None):
    """Scaled Dot-Product Attention（スケールド・ドットプロダクト・アテンション）計算関数。"""
    d_k = query.size(-1)
    # 類似度スコアの算出
    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(d_k)

    # マスクが存在する場合、遮蔽エリアを非常に小さな値（-1e9）で埋めます
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # ソフトマックスによる正規化
    p_atten = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_atten = dropout(p_atten)

    return torch.matmul(p_atten, value), p_atten


def clones(model: nn.Module, N: int) -> nn.ModuleList:
    """同一モジュールのディープコピーをN個生成するユーティリティ関数。"""
    return nn.ModuleList([copy.deepcopy(model) for _ in range(N)])


class MutiHeadAttention(nn.Module):
    """Multi-Head Attention（マルチヘッド・アテンション）レイヤークラス。"""

    def __init__(self, embed_dim: int, head: int, dropout_p: float = 0.1):
        super().__init__()
        assert embed_dim % head == 0, "embed_dim は head で割り切れる必要があります。"
        self.d_k = embed_dim // head
        self.head = head
        self.embed_dim = embed_dim

        # W_q, W_k, W_v と最後の射影用の計4つの線形層
        self.linears = clones(nn.Linear(embed_dim, embed_dim), 4)
        self.dropout = nn.Dropout(p=dropout_p)
        self.atten = None

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask is not None:
            mask = mask.unsqueeze(0)

        batch_size = query.size(0)

        # 線形変換とヘッド分割の実行 [バッチ, 長さ, ヘッド数, d_k] -> [バッチ, ヘッド数, 長さ, d_k]
        query, key, value = [
            linear_layer(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
            for linear_layer, x in zip(self.linears, (query, key, value))
        ]

        # 分割されたヘッド単位でスケールド・ドットプロダクト・アテンションを実行
        x, self.atten = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 各ヘッドの結合（コンカテネーション）
        atten_x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_k * self.head)

        # 最後の線形変換を出力して返却
        return self.linears[-1](atten_x)


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Networks（位置毎の前向き全結合層）クラス。"""

    def __init__(self, d_model: int, d_ff: int, dropout_p: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class LayerNorm(nn.Module):
    """レイヤー正規化（Layer Normalization）レイヤークラス。"""

    def __init__(self, feature: int, eps: float = 1e-6):
        super().__init__()
        self.a = nn.Parameter(torch.ones(feature))
        self.b = nn.Parameter(torch.zeros(feature))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b


class SublayerConnection(nn.Module):
    """残差接続（Residual Connection）およびレイヤー正規化を制御する構造クラス（Pre-LN方式）。"""

    def __init__(self, size: int, dropout_p: float = 0.1):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        # 正規化を適用してからサブレイヤー処理を行うPre-LN構造
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """単一のエンコーダ・レイヤークラス。"""

    def __init__(self, size: int, self_attn: MutiHeadAttention, feed_forward: FeedForward, dropout: float = 0.1):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.ff = feed_forward
        self.sublayers = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # 1. 自己アテンション（Self-Attention）サブレイヤー
        x = self.sublayers[0](x, lambda t: self.self_attn(t, t, t, mask))
        # 2. ポジション毎の前向き全結合（FeedForward）サブレイヤー
        return self.sublayers[1](x, self.ff)


class Encoder(nn.Module):
    """エンコーダレイヤーをN個スタックした、エンコーダ本体クラス。"""

    def __init__(self, layer: EncoderLayer, N: int):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


def test_encoder() -> torch.Tensor:
    """【テスト用】エンコーダブロック全体の統合テスト。"""
    x = torch.tensor([[100, 2, 412, 508],
                      [491, 998, 1, 221]])
    vocab_size, d_model = 1000, 512
    my_embed = Embeddings(vocab_size=vocab_size, d_model=d_model)
    my_position = PositionEncoding(d_model, dropout=0.1, max_len=1000)
    position_x = my_position(my_embed(x))

    my_attention = MutiHeadAttention(embed_dim=512, head=8, dropout_p=0.1)
    my_ff = FeedForward(d_model=512, d_ff=1024)
    mask = torch.zeros(8, 4, 4)

    my_encoder_layer = EncoderLayer(size=512, self_attn=my_attention, feed_forward=my_ff, dropout=0.1)
    my_encoder = Encoder(layer=my_encoder_layer, N=6)

    encoder_result = my_encoder(position_x, mask)
    print(f"[INFO] test_encoder - 最終出力形状: {encoder_result.shape}")
    return encoder_result


if __name__ == '__main__':
    test_encoder()
