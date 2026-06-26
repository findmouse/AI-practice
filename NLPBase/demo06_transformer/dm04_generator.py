# -*- coding: utf-8 -*-
"""
システム名：自然言語処理・基盤AIシステム
モジュール名：transformer_generator
作成日：2026/06/28
作成者：
修正履歴：

"""
from dm03_decoder import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Generator(nn.Module):
    """デコーダの出力を語彙空間へ投影し、トークン予測確率を出力するジェネレータ層クラス。"""

    def __init__(self, d_model: int, vocab_size: int):
        """初期化処理。

        Args:
            d_model (int): 隠れ層・デコーダの出力次元数
            vocab_size (int): 予測対象のターゲット総語彙数
        """
        super().__init__()
        # 最終的な語彙数へのマッピングを行う線形射影層
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播処理。

        最後の次元に対して対数ソフトマックス（Log-Softmax）を計算します。
        """
        return F.log_softmax(self.out(x), dim=-1)


def test_generator():
    """ジェネレータ層の動作確認用単体テスト。"""
    decoder_result = test_decoder()
    my_generator = Generator(d_model=512, vocab_size=1000)
    output = my_generator(decoder_result)
    print(f"[SUCCESS] Generator出力形状: {output.shape}")


if __name__ == '__main__':
    test_generator()
