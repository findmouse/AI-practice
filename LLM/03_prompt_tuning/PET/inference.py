import time
from typing import List

import torch
from rich import print
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

from ProjectConfig import ProjectConfig
from utils.verbalizer import Verbalizer
from data_handle.template import HardTemplate
from data_handle.data_preprocess import convert_example
from utils.common_utils import convert_logits_to_ids


pc = ProjectConfig()


def load_resources():
    """推論に必要なモデル、tokenizer、verbalizer、template を読み込む。"""
    checkpoint_files = ("model.safetensors", "pytorch_model.bin")
    has_checkpoint = pc.checkpoint.is_dir() and any(
        (pc.checkpoint / name).is_file() for name in checkpoint_files
    )

    if has_checkpoint:
        model_source = pc.checkpoint
        tokenizer_source = pc.checkpoint
    else:
        model_source = None
        tokenizer_source = pc.pre_model
        print(
            "[yellow]警告: checkpoints/model_best がありません。"
            "デモ実行用にランダム初期化モデルを使用します。予測結果には意味がありません。[/yellow]"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_source), local_files_only=True
    )
    if model_source is None:
        config = AutoConfig.from_pretrained(str(pc.pre_model), local_files_only=True)
        model = AutoModelForMaskedLM.from_config(config)
    else:
        model = AutoModelForMaskedLM.from_pretrained(
            str(model_source), local_files_only=True
        )

    model.to(pc.device).eval()
    verbalizer = Verbalizer(
        verbalizer_file=pc.verbalizer,
        tokenizer=tokenizer,
        max_label_len=pc.max_label_len,
    )
    prompt = pc.prompt_file.read_text(encoding="utf-8").splitlines()[0].strip()
    hard_template = HardTemplate(prompt=prompt)
    return model, tokenizer, verbalizer, hard_template


def inference(contents: List[str]):
    """
    推論関数。元の文を入力し、mask label の予測値を返す。

    Args:
        contents (List[str]): 元の文のリスト。
    """
    if not contents:
        return []

    model, tokenizer, verbalizer, hard_template = load_resources()
    with torch.inference_mode():
        start_time = time.time()
        examples = {'text': contents}
        tokenized_output = convert_example(
            examples,
            tokenizer,
            hard_template=hard_template,
            max_seq_len=128,
            max_label_len=pc.max_label_len,
            train_mode=False,
            return_tensor=True
        )
        logits = model(input_ids=tokenized_output['input_ids'].to(pc.device),
                    token_type_ids=tokenized_output['token_type_ids'].to(pc.device),
                    attention_mask=tokenized_output['attention_mask'].to(pc.device)).logits
        predictions = convert_logits_to_ids(logits, tokenized_output['mask_positions']).cpu().numpy().tolist()  # (batch, label_num)

        # 子ラベルが属する親ラベルを探す
        predictions = verbalizer.batch_find_main_label(predictions)

        predictions = [ele['label'] for ele in predictions]
        used = time.time() - start_time
        print(f'Used {used:.3f}s on {pc.device}.')
        return predictions


if __name__ == '__main__':
    contents = [
        '天台很好看，躺在躺椅上很悠闲，因为活动所以我觉得性价比还不错，适合一家出行，特别是去迪士尼也蛮近的，下次有机会肯定还会再来的，值得推荐',
        '环境，设施，很棒，周边配套设施齐全，前台小姐姐超级漂亮！酒店很赞，早餐不错，服务态度很好，前台美眉很漂亮。性价比超高的一家酒店。强烈推荐',
        "物流超快，隔天就到了，还没用，屯着出游的时候用的，听方便的，占地小",
        "福行市来到无早集市，因为是喜欢的面包店，所以跑来集市看看。第一眼就看到了，之前在微店买了小刘，这次买了老刘，还有一直喜欢的巧克力磅蛋糕。好奇老板为啥不做柠檬磅蛋糕了，微店一直都是买不到的状态。因为不爱碱水硬欧之类的，所以期待老板多来点其他小点，饼干一直也是大爱，那天好像也没看到",
        "服务很用心，房型也很舒服，小朋友很喜欢，下次去嘉定还会再选择。床铺柔软舒适，晚上休息很安逸，隔音效果不错赞，下次还会来"
    ]
    print("以下のレビュー文について、それぞれ所属カテゴリを出力してください：")
    res = inference(contents)
    new_dict = {}
    for i in range(len(contents)):
        new_dict[contents[i]] = res[i]
    print(new_dict)
