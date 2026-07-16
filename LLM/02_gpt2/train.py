# -*- coding: utf-8 -*-
import json
import os
from datetime import datetime

import torch
from transformers import BertTokenizerFast, GPT2Config, GPT2LMHeadModel
from transformers import get_linear_schedule_with_warmup

from data_preprocess.dataloader import get_dataloader
from data_preprocess.preprocess import preprocess
from functions_tools import calculate_acc
from parameter_config import ParameterConfig
from plot_loss import plot_loss


def ensure_pkl_data(params):
    """pkl が存在しない場合、txt から前処理して自動生成する。"""
    if not os.path.exists(params.train_path):
        print(f'{params.train_path} が見つかりません。学習データの前処理を開始します...')
        preprocess(params.train_txt_path, params.train_path, params.vocab_path)
    if not os.path.exists(params.valid_path):
        print(f'{params.valid_path} が見つかりません。検証データの前処理を開始します...')
        preprocess(params.valid_txt_path, params.valid_path, params.vocab_path)


def train_epoch(model, train_dataloader, optimizer, scheduler, epoch, args):
    model.train()
    device = args.device
    ignore_index = args.ignore_index
    epoch_start_time = datetime.now()
    total_loss = 0
    epoch_correct_num, epoch_total_num = 0, 0

    for batch_idx, (input_ids, labels) in enumerate(train_dataloader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        outputs = model(input_ids=input_ids, labels=labels)
        logits = outputs.logits
        loss = outputs.loss.mean()

        batch_correct_num, batch_total_num = calculate_acc(
            logits, labels, ignore_index=ignore_index
        )
        batch_acc = batch_correct_num / batch_total_num if batch_total_num else 0
        epoch_correct_num += batch_correct_num
        epoch_total_num += batch_total_num
        total_loss += loss.item()

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (batch_idx + 1) % args.loss_step == 0:
            lr = scheduler.get_last_lr()[0]
            print(
                'batch {} of epoch {}, loss {:.4f}, batch_acc {:.4f}, lr {:.2e}'.format(
                    batch_idx + 1,
                    epoch + 1,
                    loss.item() * args.gradient_accumulation_steps,
                    batch_acc,
                    lr,
                )
            )

        del input_ids, labels, outputs

    epoch_mean_loss = total_loss / max(len(train_dataloader), 1)
    epoch_mean_acc = epoch_correct_num / epoch_total_num if epoch_total_num else 0
    print(
        'epoch {}: loss {:.4f}, predict_acc {:.4f}'.format(
            epoch + 1, epoch_mean_loss, epoch_mean_acc
        )
    )

    # 10 epoch ごと、または最後の epoch で追加保存
    if epoch % 10 == 0 or epoch + 1 == args.epochs:
        model_path = os.path.join(args.save_model_path, 'epoch{}'.format(epoch + 1))
        os.makedirs(model_path, exist_ok=True)
        model.save_pretrained(model_path)
        print('saved checkpoint: {}'.format(model_path))

    print('time for one epoch: {}'.format(datetime.now() - epoch_start_time))
    return epoch_mean_loss


def validate_epoch(model, validate_dataloader, epoch, args):
    print('start validating')
    model.eval()
    device = args.device
    epoch_start_time = datetime.now()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for input_ids, labels in validate_dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss.mean()
            total_loss += loss.item()
            num_batches += 1
            del input_ids, labels, outputs

    epoch_mean_loss = total_loss / max(num_batches, 1)
    print('validate epoch {}: loss {:.4f}'.format(epoch + 1, epoch_mean_loss))
    print('time for validating one epoch: {}'.format(datetime.now() - epoch_start_time))
    return epoch_mean_loss


def save_loss_history(args, train_losses, validate_losses):
    history = {
        'train_losses': train_losses,
        'validate_losses': validate_losses,
    }
    with open(args.loss_history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f'損失履歴を保存しました: {args.loss_history_path}')


def train(model, train_dataloader, validate_dataloader, args):
    t_total = max(
        len(train_dataloader) // args.gradient_accumulation_steps * args.epochs, 1
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total,
    )

    print('starting training')
    train_losses, validate_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            args=args,
        )
        train_losses.append(train_loss)

        validate_loss = validate_epoch(
            model=model,
            validate_dataloader=validate_dataloader,
            epoch=epoch,
            args=args,
        )
        validate_losses.append(validate_loss)

        if validate_loss < best_val_loss:
            best_val_loss = validate_loss
            model_path = os.path.join(args.save_model_path, 'min_ppl_model')
            os.makedirs(model_path, exist_ok=True)
            model.save_pretrained(model_path)
            print('saving current best model for epoch {}'.format(epoch + 1))

        # 各 epoch ごとに保存し、途中で損失の推移を確認できるようにする
        save_loss_history(args, train_losses, validate_losses)

    print('train_losses:', train_losses)
    print('validate_losses:', validate_losses)
    try:
        plot_loss(show=False)
    except Exception as e:
        print(f'損失曲線の描画に失敗しました（後で plot_loss.py を手動実行できます）: {e}')


def main():
    params = ParameterConfig()
    print(f'使用デバイス: {params.device}')

    os.makedirs(params.save_model_path, exist_ok=True)
    ensure_pkl_data(params)

    tokenizer = BertTokenizerFast(
        vocab_file=params.vocab_path,
        sep_token='[SEP]',
        pad_token='[PAD]',
        cls_token='[CLS]',
    )

    if params.pretrained_model:
        model = GPT2LMHeadModel.from_pretrained(params.pretrained_model)
    else:
        model_config = GPT2Config.from_json_file(params.config_json)
        model = GPT2LMHeadModel(config=model_config)

    model = model.to(params.device)
    print(f'model.vocab_size={model.config.vocab_size}, tokenizer.vocab_size={tokenizer.vocab_size}')
    assert model.config.vocab_size == tokenizer.vocab_size

    num_parameters = sum(p.numel() for p in model.parameters())
    print(f'モデルのパラメータ総数: {num_parameters}')

    train_dataloader, validate_dataloader = get_dataloader(
        params.train_path,
        params.valid_path,
        batch_size=params.batch_size,
        max_len=params.max_len,
        max_train_samples=params.max_train_samples,
        max_valid_samples=params.max_valid_samples,
    )
    train(model, train_dataloader, validate_dataloader, params)


if __name__ == '__main__':
    main()
