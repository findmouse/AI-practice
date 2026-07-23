import os
import time
import copy
import argparse
import sys
from functools import partial
import peft
import torch
# autocast是PyTorch中一种混合精度的技术，可在保持数值精度的情况下提高训练速度和减少显存占用。
# 该方法混合精度训练，如果在CPU环境中不起任何作用
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
    """让 ChatGLM 远程 tokenizer 兼容 transformers 5.x 的 padding_side 参数。"""
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

    # ChatGLM2 的远程模型代码基于 transformers 4.x。transformers 5.x
    # 要求旧模型没有初始化的 tied-weight 元数据在加载量化权重前存在。
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}

    tokenizer = AutoTokenizer.from_pretrained(pc.pre_model, trust_remote_code=True)
    tokenizer = patch_chatglm_tokenizer(tokenizer)

    config = AutoConfig.from_pretrained(pc.pre_model, trust_remote_code=True)
    config.max_length = getattr(config, "max_length", config.seq_length)
    config.use_cache = False
    # RTX 2080 不支持 BF16，使用 NF4 4-bit 存储、FP16 计算和双重量化。
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 从标准 ChatGLM2-6B 权重动态构建 bitsandbytes 4-bit 模型。
    # 不能使用项目原来的 chatglm2-6b-int4，其 QuantizedLinear 不受 PEFT 支持。
    model = AutoModel.from_pretrained(pc.pre_model,
                                      config=config,
                                      trust_remote_code=True,
                                      dtype=torch.float16,
                                      quantization_config=quantization_config,
                                      device_map={"": 0},
                                      low_cpu_mem_usage=True,
                                      )

    # 冻结 4-bit 基础权重，并为梯度检查点和 LoRA 训练准备模型。
    model = peft.prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    # 不进行缓存，减少内存
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

    # 4-bit 模型已由 device_map 放到 GPU，不能再次调用 model.to(...)。
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
    # 根据训练轮数计算最大训练步数，以便于scheduler动态调整lr
    num_update_steps_per_epoch = len(train_dataloader)
    # 指定总的训练步数，它会被学习率调度器用来确定学习率的变化规律，确保学习率在整个训练过程中得以合理地调节
    max_train_steps = pc.epochs * num_update_steps_per_epoch
    warm_steps = int(pc.warmup_ratio * max_train_steps)  # 预热阶段的训练步数
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
    在测试集上评估当前模型的训练效果。

    Args:
        model: 当前模型
        data_loader: 测试集的dataloader
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
