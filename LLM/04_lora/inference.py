import time
import sys
from pathlib import Path

import torch
import peft
from transformers import AutoConfig, AutoTokenizer, AutoModel, PreTrainedModel

# ChatGLM2 のリモートモデルコードは transformers 4.x を前提としている。transformers 5.x では
# 重みを読み込む前に、旧モデルの未初期化な tied-weight メタデータが存在している必要がある。
if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    PreTrainedModel.all_tied_weights_keys = {}

sys.path.append(str(Path(__file__).resolve().parent.parent))
from glm_config import ProjectConfig

pc = ProjectConfig()


# torch.set_default_tensor_type(torch.cuda.HalfTensor)


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


def patch_chatglm_generation_mixin(model):
    """
    ChatGLM2 のリモートコードは `_update_model_kwargs_for_generation` を上書きしており、
    その内部で呼ばれる `_extract_past_from_model_output` は旧版 transformers の
    GenerationMixin にあった private メソッドで、transformers 5.x では削除されている。
    ここで最小限の実装を補っている。
    """
    cls = type(model)
    if hasattr(cls, "_extract_past_from_model_output"):
        return model

    def _extract_past_from_model_output(self, outputs, standardize_cache_format=False):
        if "past_key_values" in outputs:
            return outputs.past_key_values
        if "mems" in outputs:
            return outputs.mems
        if "past_buckets_states" in outputs:
            return outputs.past_buckets_states
        return None

    cls._extract_past_from_model_output = _extract_past_from_model_output
    return model


def inference(
        model,
        tokenizer,
        instuction: str,
        sentence: str
):
    """
    モデル推論関数。

    Args:
        instuction (str): _description_
        sentence (str): _description_

    Returns:
        _type_: _description_
    """
    with torch.no_grad():
        input_text = f"Instruction: {instuction}\n"
        if sentence:
            input_text += f"Input: {sentence}\n"
        input_text += f"Answer: "
        batch = tokenizer(input_text, return_tensors="pt")
        out = model.generate(
            input_ids=batch["input_ids"].to(device),
            max_new_tokens=max_new_tokens,
            temperature=0,
            # ChatGLM2 のカスタム forward は past_key_values を依然として tuple 形式で読み書きしており、
            # transformers 5.x の新しい DynamicCache オブジェクトと互換性がないため、ここではキャッシュを無効化して回避する。
            use_cache=False
        )
        out_text = tokenizer.decode(out[0])
        answer = out_text.split('Answer: ')[-1]
        return answer


if __name__ == '__main__':
    from rich import print

    device = 'cuda:0'
    max_new_tokens = 300
    model_path = "./lora_checkpoints/model_best"
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    tokenizer = patch_chatglm_tokenizer(tokenizer)

    # model_path 配下には LoRA adapter のみが保存されている（common_utils.save_model 参照）。
    # そのため先にベースモデルを読み込み、その後 adapter を装着する必要がある。
    config = AutoConfig.from_pretrained(pc.pre_model, trust_remote_code=True)
    config.max_length = getattr(config, "max_length", config.seq_length)
    # transformers 5.x の generate() は DynamicCache を構築する際、新しい命名規則に従って
    # config.num_hidden_layers を読み取るが、ChatGLM2 のカスタム config では依然として num_layers という名前のままである。
    config.num_hidden_layers = getattr(config, "num_hidden_layers", config.num_layers)

    base_model = AutoModel.from_pretrained(
        pc.pre_model,
        config=config,
        trust_remote_code=True
    ).half().to(device)
    # ChatGLM の __init__ は config.max_length を借用して max_sequence_length を設定しているだけだが、
    # transformers 5.x は生成パラメータが model.config に残ることを許さないため、読み込み後に削除する。
    if hasattr(base_model.config, "max_length"):
        del base_model.config.max_length
    base_model = patch_chatglm_generation_mixin(base_model)

    model = peft.PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    samples = [
        {
            'instruction': "现在你是一个非常厉害的SPO抽取器。",
            "input": "下面这句中包含了哪些三元组，用json列表的形式回答，不要输出除json外的其他答案。\n\n73获奖记录人物评价：黄磊是一个特别幸运的演员，拍第一部戏就碰到了导演陈凯歌，而且在他的下一部电影《夜半歌声》中演对手戏的张国荣、吴倩莲、黎明等都是著名的港台演员。",
        },
        {
            'instruction': "你现在是一个很厉害的阅读理解器，严格按照人类指令进行回答。",
            "input": "下面子中的主语是什么类别，输出成列表形式。\n\n第N次入住了，就是方便去客户那里哈哈。还有啥说的"
        }
    ]

    start = time.time()
    for i, sample in enumerate(samples):
        res = inference(
            model,
            tokenizer,
            sample['instruction'],
            sample['input']
        )
        print(f'res {i}: ')
        print(res)
    print(f'Used {round(time.time() - start, 2)}s.')
