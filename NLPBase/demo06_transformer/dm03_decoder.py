# -*- coding: utf-8 -*-
"""
システム名：自然言語処理・基盤AIシステム
モジュール名：transformer_decoder
作成日：2026/06/27
作成者：
修正履歴：

"""

from dm02_encode import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DecoderLayer(nn.Module):
    """単一のデコーダ・レイヤークラス。

    自己アテンション、ソース・デコーダ間クロスアテンション、FFNの3層のサブレイヤーを持ちます。
    """

    def __init__(self, size: int, self_attn: MutiHeadAttention,
                 src_attn: MutiHeadAttention, feed_forward: FeedForward, dropout_p: float):
        """初期化処理。

        Args:
            size (int): 次元数
            self_attn (MutiHeadAttention): デコーダ側自己アテンション
            src_attn (MutiHeadAttention): ソース（エンコーダ出力）に対するアテンション（Q!=K=V）
            feed_forward (FeedForward): ポジション毎の前向き全結合層
            dropout_p (float): ドロップアウト率
        """
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.ff = feed_forward
        self.sublayers = clones(SublayerConnection(size, dropout_p=dropout_p), 3)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                source_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """順伝播処理。"""
        # 1. ターゲット側自己アテンション（未来情報へのアクセスを隠蔽するマスクを適用）
        x = self.sublayers[0](x, lambda t: self.self_attn(t, t, t, target_mask))

        # 2. エンコーダ出力（memory）に対するソース・デコーダ間アテンションの実行
        x = self.sublayers[1](x, lambda t: self.src_attn(t, memory, memory, source_mask))

        # 3. 前向き全結合サブレイヤーの適用
        return self.sublayers[2](x, self.ff)


class Decoder(nn.Module):
    """デコーダレイヤーをN個スタックした、デコーダ本体クラス。"""

    def __init__(self, layer: DecoderLayer, N: int):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                source_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


def test_decoder() -> torch.Tensor:
    """【テスト用】デコーダブロック全体の統合テスト。"""
    x = torch.tensor([[10, 30, 1, 2],
                      [25, 9, 10, 10]])
    vocab_size, d_model = 1000, 512
    my_embed = Embeddings(vocab_size=vocab_size, d_model=d_model)
    my_position = PositionEncoding(d_model, dropout=0.1, max_len=1000)
    position_x = my_position(my_embed(x))

    my_attention = MutiHeadAttention(embed_dim=512, head=8, dropout_p=0.1)
    self_attn = copy.deepcopy(my_attention)
    src_attn = copy.deepcopy(my_attention)
    my_ff = FeedForward(d_model=512, d_ff=1024)

    # 前段エンコーダのモック出力を取得
    encoder_result = test_encoder()

    mask = torch.zeros(8, 4, 4)
    source_mask = target_mask = mask

    my_decoderlayer = DecoderLayer(size=512, self_attn=self_attn,
                                   src_attn=src_attn, feed_forward=my_ff, dropout_p=0.1)
    my_decoder = Decoder(my_decoderlayer, 6)

    decoder_result = my_decoder(position_x, encoder_result, source_mask, target_mask)
    print(f"[INFO] test_decoder - 最終出力形状: {decoder_result.shape}")
    return decoder_result


if __name__ == '__main__':
    test_decoder()
