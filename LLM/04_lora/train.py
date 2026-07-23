import os
import time
import copy
import argparse
import sys
from functools import partial
import peft
import torch
# autocast は PyTorch の混合精度技術で、数値精度を保ちながら学習速度の向上と VRAM 使用量の削減を実現できる。
# この混合精度学習は、CPU 環境では効果を発揮しない。
from torch.cuda.amp import autocast as autocast
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    get_scheduler,
)
from common_utils import *
from data_handle.data_loader import *
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from glm_config import *

pc = ProjectConfig()


def patch_chatglm_tokenizer(tokenizer):
    """ChatGLM のリモート tokenizer を transformers 5.x の padding_side 引数に対応させる。"""
    cls = tokenizer.__class__
    if getattr(cls, "_pad_compat_patched", False):
        return tokenizer

    original_pad = cls._pad

    def _pad(self, *args, padding_side=None, **kwargs):
        if padding_side is not None:
            self.padding_side = padding_side
        return original_pad(self, *args, **kwargs)

    cls._pad = _pad
    cls._pad_compat_patched = True
    return tokenizer


def model2train():
    if pc.device == 'cpu':
        raise RuntimeError("ChatGLM2-6B QLoRA 训练需要支持 CUDA 的 NVIDIA GPU。")

    # ChatGLM2 のリモートモデルコードは transformers 4.x を前提としている。transformers 5.x では
    # 量子化重みを読み込む前に、旧モデルの未初期化な tied-weight メタデータが存在している必要がある。
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}

    tokenizer = AutoTokenizer.from_pretrained(pc.pre_model, trust_remote_code=True)
    tokenizer = patch_chatglm_tokenizer(tokenizer)

    config = AutoConfig.from_pretrained(pc.pre_model, trust_remote_code=True)
    config.max_length = getattr(config, "max_length", config.seq_length)
    config.use_cache = False
    # RTX 2080 は BF16 に対応していないため、NF4 4-bit 格納・FP16 計算・二重量子化を使用する。
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 標準の ChatGLM2-6B 重みから bitsandbytes 4-bit モデルを動的に構築する。
    # プロジェクトに元々あった chatglm2-6b-int4 は QuantizedLinear が PEFT でサポートされないため使用できない。
    model = AutoModel.from_pretrained(pc.pre_model,
                                      config=config,
                                      trust_remote_code=True,
                                      dtype=torch.float16,
                                      quantization_config=quantization_config,
                                      device_map={"": 0},
                                      low_cpu_mem_usage=True,
                                      )

    # 4-bit ベース重みを凍結し、gradient checkpointing と LoRA 学習のためにモデルを準備する。
    model = peft.prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    # キャッシュを行わず、メモリ使用量を削減する
    model.config.use_cache = False
    peft_config = peft.LoraConfig(
        task_type=peft.TaskType.CAUSAL_LM,
        inference_mode=False,
        r=pc.lora_rank,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query_key_value"],
        bias="none",
    )
    model = peft.get_peft_model(model, peft_config)

    # 4-bit モデルはすでに device_map によって GPU に配置されているため、再度 model.to(...) を呼んではいけない。
    model.train()
    model.print_trainable_parameters()

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": pc.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=pc.learning_rate)
    # model.to(pc.device)
    #
    train_dataloader, dev_dataloader = get_data(tokenizer)
    # エポック数から最大学習ステップ数を算出し、scheduler が動的に学習率を調整できるようにする
    num_update_steps_per_epoch = len(train_dataloader)
    # 総学習ステップ数を指定する。学習率スケジューラはこれを用いて学習率の変化パターンを決定し、学習全体を通して学習率が適切に調整されるようにする
    max_train_steps = pc.epochs * num_update_steps_per_epoch
    warm_steps = int(pc.warmup_ratio * max_train_steps)  # ウォームアップ段階の学習ステップ数
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )
    #
    loss_list = []
    tic_train = time.time()
    global_step, best_eval_loss = 0, float('inf')
    for epoch in range(1, pc.epochs + 1):
        for batch in train_dataloader:
            if pc.use_lora:
                with autocast():
                    loss = model(
                        input_ids=batch['input_ids'].to(dtype=torch.long, device=pc.device),
                        labels=batch['labels'].to(dtype=torch.long, device=pc.device)
                    ).loss
            else:
                loss = model(
                    input_ids=batch['input_ids'].to(dtype=torch.long, device=pc.device),
                    labels=batch['labels'].to(dtype=torch.long, device=pc.device)
                ).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            loss_list.append(float(loss.cpu().detach()))

            global_step += 1
            if global_step % pc.logging_steps == 0:
                time_diff = time.time() - tic_train
                loss_avg = sum(loss_list) / len(loss_list)
                print("global step %d ( %02.2f%% ) , epoch: %d, loss: %.5f, speed: %.2f step/s, ETA: %s" % (global_step,
                                                                                                            global_step / max_train_steps * 100,
                                                                                                            epoch,
                                                                                                            loss_avg,
                                                                                                            pc.logging_steps / time_diff,
                                                                                                            second2time(
                                                                                                                int(max_train_steps - global_step) / (
                                                                                                                        pc.logging_steps / time_diff))))
                tic_train = time.time()

            if global_step % pc.save_freq == 0:
                cur_save_dir = os.path.join(pc.save_dir, "model_%d" % global_step)
                save_model(model, cur_save_dir)
                tokenizer.save_pretrained(cur_save_dir)
                print(f'Model has saved at {cur_save_dir}.')

                eval_loss = evaluate_model(model, dev_dataloader)

                print("Evaluation Loss: %.5f" % (eval_loss))
                if eval_loss < best_eval_loss:
                    print(f"Min eval loss has been updated: {best_eval_loss:.5f} --> {eval_loss:.5f}")
                    best_eval_loss = eval_loss
                    cur_save_dir = os.path.join(pc.save_dir, "model_best")
                    save_model(model, cur_save_dir)
                    tokenizer.save_pretrained(cur_save_dir)
                    print(f'Best model has saved at {cur_save_dir}.')
                tic_train = time.time()

    final_save_dir = os.path.join(pc.save_dir, "model_final")
    save_model(model, final_save_dir)
    tokenizer.save_pretrained(final_save_dir)
    print(f'Final LoRA adapter has saved at {final_save_dir}.')


def evaluate_model(model, dev_dataloader):
    """
    テストセット上で現在のモデルの学習効果を評価する。

    Args:
        model: 現在のモデル
        data_loader: テストセットの dataloader
    """
    model.eval()
    loss_list = []
    with torch.no_grad():
        for batch in dev_dataloader:
            if pc.use_lora:
                with autocast():
                    loss = model(
                        input_ids=batch['input_ids'].to(dtype=torch.long, device=pc.device),
                        labels=batch['labels'].to(dtype=torch.long, device=pc.device)
                    ).loss
            else:
                loss = model(
                    input_ids=batch['input_ids'].to(dtype=torch.long, device=pc.device),
                    labels=batch['labels'].to(dtype=torch.long, device=pc.device)
                ).loss
            loss_list.append(float(loss.cpu().detach()))
    model.train()
    return sum(loss_list) / len(loss_list)


if __name__ == '__main__':
    model2train()
