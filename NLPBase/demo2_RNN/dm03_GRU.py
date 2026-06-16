import torch
import torch.nn as nn


def dm_gru_use():
    """
    PyTorchのnn.GRUモジュールの基本動作を確認するためのデモ関数。
    入力テンソルと初期隠れ状態（h0）の次元、および出力されるテンソルの
    形状（Shape）の整合性を検証します。
    """
    # ==========================================
    # 1. GRUモデルの定義
    # ==========================================
    # input_size:  入力テンソルの埋め込み次元 (Embedding Dimension)
    # hidden_size: 隠れ層の出力次元 (Hidden State Dimension)
    # num_layers:  GRUのレイヤー数
    gru = nn.GRU(input_size=5, hidden_size=6, num_layers=1)

    # ==========================================
    # 2. 入力データの作成
    # ==========================================
    # 形状の定義: (sequence_length, batch_size, input_size)
    # - sequence_length (系列長/トークン数): 2
    # - batch_size (バッチサイズ): 3
    # - input_size (埋め込み次元): 5
    # ※注: 'input'はPythonの組み込み関数名であるため、現場の規約に従い変数名を変更
    input_tensor = torch.randn(2, 3, 5)

    # ==========================================
    # 3. 初期隠れ状態（h0）の設定
    # ==========================================
    # 形状の定義: (num_layers * num_directions, batch_size, hidden_size)
    # ※今回は単方向（num_directions=1）のため、第1次元は num_layers と等しくなります。
    # - num_layers: 1
    # - batch_size: 3
    # - hidden_size: 6
    h0 = torch.randn(1, 3, 6)

    # ==========================================
    # 4. モデルへの入力と推論
    # ==========================================
    # 期待される出力のShape:
    # - output: (sequence_length, batch_size, hidden_size) -> (2, 3, 6)
    # - hn:     (num_layers, batch_size, hidden_size)     -> (1, 3, 6)
    # ※GRUはLSTMと異なり、セル状態(cn)を持たないため、戻り値はhnのみとなります。
    output, hn = gru(input_tensor, h0)

    # 各テンソルの値とShapeを出力（デバッグ・ログ確認用）
    print(f"--- Output (各ステップの隠れ状態) ---\nShape: {output.shape}\n{output}\n")
    print(f"--- hn (最終ステップの隠れ状態) ---\nShape: {hn.shape}\n{hn}\n")


if __name__ == '__main__':
    dm_gru_use()