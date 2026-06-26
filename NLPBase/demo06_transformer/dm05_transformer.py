# -*- coding: utf-8 -*-
"""
システム名：自然言語処理・基盤AIシステム
モジュール名：transformer_model
作成日：2026/06/28
作成者：
修正履歴：

"""

import torch
import torch.nn as nn

# 現場の共通モジュール（前フェーズで作成した汎用アルゴリズムクラスを想定）
# ※ 実環境の配備構成に合わせてインポートパスは適宜調整してください。
from dm01_input import *
from dm02_encode import *
from dm03_decoder import *
from dm04_generator import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class EncoderDecoder(nn.Module):
    """Transformer標準のエンコーダ・デコーダ統合モデルクラス。

    入力シーケンス（Source）を潜在表現に符号化し、それを基にターゲットシーケンス（Target）を確率予測します。
    """

    def __init__(self, encoder: Encoder, decoder: Decoder,
                 src_embed: nn.Sequential, tgt_embed: nn.Sequential, generator: Generator):
        """初期化処理。

        Args:
            encoder (Encoder): 符号化ネットワーク（エンコーダブロック）
            decoder (Decoder): 復号化ネットワーク（デコーダブロック）
            src_embed (nn.Sequential): ソース側 埋め込み層 + 位置エンコーディング
            tgt_embed (nn.Sequential): ターゲット側 埋め込み層 + 位置エンコーディング
            generator (Generator): 最終線形変換 + 確率分布生成レイヤー
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, source: torch.Tensor, target: torch.Tensor,
                source_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """順伝播（フォワード処理）。

        Args:
            source (torch.Tensor): 入力トークンIDテンソル [バッチサイズ, 入力シーケンス長]
            target (torch.Tensor): 出力トークンIDテンソル [バッチサイズ, 出力シーケンス長]
            source_mask (torch.Tensor): パディング等を除外するためのソース側マスク [バッチ, シーケンス長, シーケンス長]
            target_mask (torch.Tensor): 未来のトークン隠蔽用（Look-ahead）ターゲット側マスク [バッチ, シーケンス長, シーケンス长]

        Returns:
            torch.Tensor: 各語彙に対する対数確率分布 [バッチサイズ, 出力シーケンス长, 語彙数]
        """
        # 1. ソースシーケンスの埋め込み表現および位置情報の付与
        encoder_embed_x = self.src_embed(source)

        # 2. エンコーダによるコンテキストベクトルの抽出
        encoder_output = self.encoder(encoder_embed_x, source_mask)

        # 3. ターゲットシーケンス（正解シフトデータ等）の埋め込み表現および位置情報の付与
        decoder_embed_x = self.tgt_embed(target)

        # 4. デコーダによるクロスアテンションおよびトークン予測潜在ベクトルの生成
        decoder_output = self.decoder(decoder_embed_x, encoder_output, source_mask, target_mask)

        # 5. ジェネレータによる語彙次元へのマッピングとLog-Softmaxの計算
        final_output = self.generator(decoder_output)

        print(f"[INFO] EncoderDecoder - 順伝播処理完了。出力形状: {final_output.shape}")
        return final_output


def make_model(source_vocab: int, target_vocab: int,
               N: int = 6, d_model: int = 512, d_ff: int = 1024,
               head: int = 8, dropout_p: float = 0.1) -> EncoderDecoder:
    """汎用Transformer（Encoder-Decoder）モデルを構築し、パラメータを初期化するファクトリ関数。

    Args:
        source_vocab (int): 入力側の総語彙数（Embeddingサイズ）
        target_vocab (int): 出力側の总語彙数（Generatorサイズ）
        N (int): エンコーダおよびデコーダレイヤーのスタック（積層）数
        d_model (int): 隠れ層（単語埋め込み）の次元数
        d_ff (int): ポジション毎の前向き全結合層（FeedForward）の中間層次元数
        head (int): マルチヘッドアテンションのヘッド分割数
        dropout_p (float): ドロップアウト率

    Returns:
        EncoderDecoder: Xavier一様分布で重みが初期化されたモデルオブジェクト
    """
    # ネットワークレイヤーの複製（ディープコピー）用に関数オブジェクトを抽出
    clone_layer = copy.deepcopy

    # 共通となるアテンション機構および前向き全結合層（FFN）のインスタンス化
    attn = MutiHeadAttention(embed_dim=d_model, head=head, dropout_p=dropout_p)
    ff = FeedForward(d_model=d_model, d_ff=d_ff, dropout_p=dropout_p)

    # 各トークン埋め込み層（トークナイズ後の表現層）と共通位置エンコーディングの定義
    encode_embed = Embeddings(vocab_size=source_vocab, d_model=d_model)
    decode_embed = Embeddings(vocab_size=target_vocab, d_model=d_model)
    position = PositionEncoding(d_model=d_model, dropout=dropout_p, max_len=2000)

    # 出力生成レイヤーの定義
    generator = Generator(d_model=d_model, vocab_size=target_vocab)

    # モデル全体のコンポーネント結合
    model = EncoderDecoder(
        encoder=Encoder(EncoderLayer(d_model, clone_layer(attn), clone_layer(ff), dropout_p), N),
        decoder=Decoder(DecoderLayer(d_model, clone_layer(attn), clone_layer(attn), clone_layer(ff), dropout_p), N),
        src_embed=nn.Sequential(encode_embed, clone_layer(position)),
        tgt_embed=nn.Sequential(decode_embed, clone_layer(position)),
        generator=generator
    )

    # パラメータの初期化処理（2次元以上の重みテンソルに対してXavier初期化を一括適用）
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    print(f"[SUCCESS] Transformer汎用モデルの初期化が正常に完了しました。")
    return model


if __name__ == '__main__':
    # 単体動作検証テストの実行
    transformer_model = make_model(source_vocab=1000, target_vocab=2000)

    # 模擬入力データの生成（例：[バッチサイズ=2, シーケンス長=4]）
    mock_source = torch.tensor([[1, 2, 3, 4],
                                [5, 6, 7, 8]])

    mock_target = torch.tensor([[11, 22, 33, 44],
                                [55, 66, 77, 88]])

    # 空のダミーマスクの生成 [マルチヘッド数8, シーケンス長4, シーケンス長4]
    mock_source_mask = mock_target_mask = torch.zeros(8, 4, 4)

    # モデル推論テスト
    prediction = transformer_model(mock_source, mock_target, mock_source_mask, mock_target_mask)
    print(f"[SUCCESS] テストスクリプト正常終了。最終出力形状: {prediction.shape}")
