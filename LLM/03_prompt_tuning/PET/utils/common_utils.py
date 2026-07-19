# coding:utf-8
# 必要なツールキットをインポート
import torch
from rich import print


def mlm_loss(logits, mask_positions, sub_mask_labels,
             cross_entropy_criterion, device):
    """
    指定位置の mask token の出力と label の間の cross entropy loss を計算する。

    Args:
        logits (torch.tensor): モデルの生出力 -> (batch, seq_len, vocab_size)
        mask_positions (torch.tensor): mask token の位置 -> (batch, mask_label_num)
        sub_mask_labels (list): mask token の sub label。label ごとに sub_label 数が異なるため可変長 list,
                                    e.g. -> [
                                        [[2398, 3352]],
                                        [[2398, 3352], [3819, 3861]]
                                    ]
        cross_entropy_criterion (CrossEntropyLoss): CE Loss 計算器
        device (str): cpu または gpu

    Returns:
        torch.tensor: CE Loss
    """
    vocab_size = logits.size(-1)
    losses = []
    for single_value in zip(logits, sub_mask_labels, mask_positions):
        single_logits = single_value[0]
        single_sub_mask_labels = single_value[1]
        single_mask_positions = single_value[2]

        # single_mask_logits の形状：(mask_label_num, vocab_size)
        single_mask_logits = single_logits[single_mask_positions]

        # 子ラベル数に合わせて single_mask_logits を複製:
        # 形状 --> (sub_label_num, mask_label_num, vocab_size)
        single_mask_logits = single_mask_logits.repeat(len(single_sub_mask_labels), 1,
                                                       1)

        # 形状変更：(sub_label_num * mask_label_num, vocab_size)
        # モデル予測結果
        single_mask_logits = single_mask_logits.reshape(-1, vocab_size)

        # 形状：(sub_label_num, mask_label_num)
        single_sub_mask_labels = torch.LongTensor(single_sub_mask_labels).to(device)

        # 形状：(sub_label_num * mask_label_num)
        single_sub_mask_labels = single_sub_mask_labels.reshape(-1, 1).squeeze()

        if not single_sub_mask_labels.size():  # 単一 token 時の次元欠落を補正
            single_sub_mask_labels = single_sub_mask_labels.unsqueeze(dim=0)

        cur_loss = cross_entropy_criterion(single_mask_logits, single_sub_mask_labels)
        losses.append(cur_loss)

    if not losses:
        raise ValueError("空のバッチでは loss を計算できません。")
    return torch.stack(losses).mean()


def convert_logits_to_ids(
        logits: torch.tensor,
        mask_positions: torch.tensor):
    """
    LM の語彙確率分布（logits）から、mask_position 位置の
    token logits を token id に変換する。

    Args:
        logits (torch.tensor): model output -> (batch, seq_len, vocab_size)
        mask_positions (torch.tensor): mask token の位置 -> (batch, mask_label_num)

    Returns:
        torch.LongTensor: 各 mask position で最大確率の推論 token -> (batch, mask_label_num)
    """
    label_length = mask_positions.size()[1]  # ラベル長
    batch_size, seq_len, vocab_size = logits.size()

    mask_positions_after_reshaped = []

    for batch, mask_pos in enumerate(mask_positions.detach().cpu().numpy().tolist()):
        for pos in mask_pos:
            mask_positions_after_reshaped.append(batch * seq_len + pos)

    # logits 形状：(batch_size * seq_len, vocab_size)
    logits = logits.reshape(batch_size * seq_len, -1)

    # mask_logits 形状：(batch * label_num, vocab_size)
    mask_logits = logits[mask_positions_after_reshaped]

    # predict_tokens 形状：(batch * label_num)
    predict_tokens = mask_logits.argmax(dim=-1)
    # 形状変更後：(batch, label_num)
    predict_tokens = predict_tokens.reshape(-1, label_length)

    return predict_tokens


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logits = torch.randn((2, 10, 20)).to(device)
    mask_positions = torch.tensor([[5, 6], [5, 6]]).to(device)
    sub_mask_labels = [[[2, 5]],
                       [[3, 2], [1, 8]]
                       ]

    cross_entropy_criterion = torch.nn.CrossEntropyLoss()

    predict_tokens = convert_logits_to_ids(logits, mask_positions)

    print(f"predict_tokens-->{predict_tokens}")
