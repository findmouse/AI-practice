# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F


def caculate_loss(logit, target, pad_idx, smoothing=False):
    """交差エントロピー損失を手動計算する（GPT2 内部の labels 損失ロジックと一致）。"""
    if smoothing:
        logit = logit[..., :-1, :].contiguous().view(-1, logit.size(2))
        target = target[..., 1:].contiguous().view(-1)
        eps = 0.1
        n_class = logit.size(-1)
        one_hot = torch.zeros_like(logit).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(logit, dim=1)
        non_pad_mask = target.ne(pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).mean()
    else:
        logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
        labels = target[..., 1:].contiguous().view(-1)
        loss = F.cross_entropy(logit, labels, ignore_index=pad_idx)
    return loss


def calculate_acc(logit, labels, ignore_index=-100):
    """pad 以外の位置におけるトークン予測精度を計算する。"""
    logit = logit[:, :-1, :].contiguous().view(-1, logit.size(-1))
    labels = labels[:, 1:].contiguous().view(-1)
    _, pred = logit.max(dim=-1)
    non_pad_mask = labels.ne(ignore_index)
    n_correct = pred.eq(labels).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    return n_correct, n_word
