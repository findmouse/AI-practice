import time
import sys
from pathlib import Path

import torch
import peft
from transformers import AutoConfig, AutoTokenizer, AutoModel, PreTrainedModel

# ChatGLM2 的远程模型代码基于 transformers 4.x，而 transformers 5.x
# 要求旧模型没有初始化的 tied-weight 元数据在加载权重前存在。
if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    PreTrainedModel.all_tied_weights_keys = {}

sys.path.append(str(Path(__file__).resolve().parent.parent))
from glm_config import ProjectConfig

pc = ProjectConfig()


# torch.set_default_tensor_type(torch.cuda.HalfTensor)


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


def patch_chatglm_generation_mixin(model):
    """
    ChatGLM2 的远程代码重写了 `_update_model_kwargs_for_generation`，其内部调用的
    `_extract_past_from_model_output` 是老版 transformers GenerationMixin 的私有方法，
    在 transformers 5.x 中已被移除，这里补回最小实现。
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
    模型 inference 函数。

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
            # ChatGLM2 的自定义 forward 仍以 tuple 形式读写 past_key_values，
            # 与 transformers 5.x 新的 DynamicCache 对象不兼容，这里关闭缓存规避。
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

    # model_path 下只保存了 LoRA adapter（见 common_utils.save_model），
    # 因此需要先加载基础模型，再挂载 adapter。
    config = AutoConfig.from_pretrained(pc.pre_model, trust_remote_code=True)
    config.max_length = getattr(config, "max_length", config.seq_length)
    # transformers 5.x 的 generate() 在构建 DynamicCache 时按新命名规范读取
    # config.num_hidden_layers，而 ChatGLM2 的自定义 config 仍叫 num_layers。
    config.num_hidden_layers = getattr(config, "num_hidden_layers", config.num_layers)

    base_model = AutoModel.from_pretrained(
        pc.pre_model,
        config=config,
        trust_remote_code=True
    ).half().to(device)
    # ChatGLM 的 __init__ 只是借用 config.max_length 来设置 max_sequence_length，
    # 但 transformers 5.x 不允许生成参数残留在 model.config 上，加载完就清掉。
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
