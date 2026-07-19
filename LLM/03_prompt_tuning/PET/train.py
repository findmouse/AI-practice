# coding:utf-8
import sys
from pathlib import Path

for _root in Path(__file__).resolve().parents:
    if (_root / "pet_bootstrap.py").is_file():
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        break
else:
    raise RuntimeError("Cannot find PET project root (pet_bootstrap.py)")

from pet_bootstrap import bootstrap

bootstrap(__file__)

import time

import torch
from rich import print
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer, get_scheduler

from ProjectConfig import ProjectConfig
from data_handle.data_loader import get_data
from utils.common_utils import convert_logits_to_ids, mlm_loss
from utils.metric_utils import ClassEvaluator
from utils.verbalizer import Verbalizer

pc = ProjectConfig()


def model2train():
    tokenizer = AutoTokenizer.from_pretrained(
        str(pc.pre_model), local_files_only=True
    )
    weight_files = ("model.safetensors", "pytorch_model.bin")
    if any((pc.pre_model / name).is_file() for name in weight_files):
        model = AutoModelForMaskedLM.from_pretrained(
            str(pc.pre_model), local_files_only=True
        )
    else:
        print(
            "[yellow]警告: bert-base-chinese に学習済み重みがありません。"
            "デモ用にランダム初期化して学習します。[/yellow]"
        )
        config = AutoConfig.from_pretrained(
            str(pc.pre_model), local_files_only=True
        )
        model = AutoModelForMaskedLM.from_config(config)
    verbalizer = Verbalizer(verbalizer_file=pc.verbalizer,
                            tokenizer=tokenizer,
                            max_label_len=pc.max_label_len)

    # 重み減衰は関数を滑らかにするためだが、bias と LayerNorm の重みは滑らかさに影響しない。
    # これらはスケール／平行移動のみなので重み減衰は不要。
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": pc.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=pc.learning_rate)
    model.to(pc.device)

    train_dataloader, dev_dataloader = get_data(tokenizer)
    # エポック数から最大学習ステップを算出し、scheduler が lr を動的調整できるようにする
    num_update_steps_per_epoch = len(train_dataloader)
    # 総学習ステップ数。学習率スケジューラが lr の変化を決めるために使う
    max_train_steps = pc.epochs * num_update_steps_per_epoch
    warm_steps = int(pc.warmup_ratio * max_train_steps)  # ウォームアップ段階のステップ数
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )

    loss_list = []
    tic_train = time.time()
    metric = ClassEvaluator()
    criterion = torch.nn.CrossEntropyLoss()
    global_step, best_f1 = 0, float("-inf")
    print('学習開始：')
    for epoch in range(pc.epochs):
        for batch in train_dataloader:
            logits = model(input_ids=batch['input_ids'].to(pc.device),
                           token_type_ids=batch['token_type_ids'].to(pc.device),
                           attention_mask=batch['attention_mask'].to(pc.device)).logits

            # 正解ラベル
            mask_labels = batch['mask_labels'].numpy().tolist()
            sub_labels = verbalizer.batch_find_sub_labels(mask_labels)
            sub_labels = [ele['token_ids'] for ele in sub_labels]

            loss = mlm_loss(logits,
                            batch['mask_positions'].to(pc.device),
                            sub_labels,
                            criterion,
                            pc.device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            loss_list.append(float(loss.cpu().detach()))
            global_step += 1
            if global_step % pc.logging_steps == 0:
                time_diff = time.time() - tic_train
                loss_avg = sum(loss_list) / len(loss_list)
                print("global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                      % (global_step, epoch, loss_avg, pc.logging_steps / time_diff))
                tic_train = time.time()

            if global_step % pc.valid_steps == 0 or global_step == max_train_steps:
                acc, precision, recall, f1, class_metrics = evaluate_model(model,
                                                                           metric,
                                                                           dev_dataloader,
                                                                           tokenizer,
                                                                           verbalizer)

                print("Evaluation precision: %.5f, recall: %.5f, F1: %.5f" % (precision, recall, f1))
                if f1 > best_f1:
                    print(
                        f"best F1 performance has been updated: {best_f1:.5f} --> {f1:.5f}"
                    )
                    print(f'Each Class Metrics are: {class_metrics}')
                    best_f1 = f1
                    pc.checkpoint.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(pc.checkpoint))
                    tokenizer.save_pretrained(str(pc.checkpoint))
                tic_train = time.time()
    print('学習終了')


def evaluate_model(model, metric, data_loader, tokenizer, verbalizer):
    """
    検証セットで現在のモデルの学習効果を評価する。

    Args:
        model: 現在のモデル
        metric: 評価指標クラス
        data_loader: 検証セットの dataloader
        tokenizer: トークナイザ
        verbalizer: Verbalizer
    """
    model.eval()
    metric.reset()

    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            logits = model(input_ids=batch['input_ids'].to(pc.device),
                           token_type_ids=batch['token_type_ids'].to(pc.device),
                           attention_mask=batch['attention_mask'].to(pc.device)).logits
            mask_labels = batch['mask_labels'].numpy().tolist()  # (batch, label_num)
            for i in range(len(mask_labels)):  # ラベル中の [PAD] token を除去
                while tokenizer.pad_token_id in mask_labels[i]:
                    mask_labels[i].remove(tokenizer.pad_token_id)

            # id を文字へ変換
            mask_labels = [''.join(tokenizer.convert_ids_to_tokens(t)) for t in mask_labels]

            # (batch, label_num)
            predictions = convert_logits_to_ids(logits,
                                                batch['mask_positions']).cpu().numpy().tolist()

            # 子ラベルが属する親ラベルを探す
            predictions = verbalizer.batch_find_main_label(predictions)
            predictions = [ele['label'] for ele in predictions]
            metric.add_batch(pred_batch=predictions, gold_batch=mask_labels)
    eval_metric = metric.compute()
    model.train()

    return eval_metric['accuracy'], eval_metric['precision'], \
        eval_metric['recall'], eval_metric['f1'], \
        eval_metric['class_metrics']


if __name__ == '__main__':
    model2train()
