# coding:utf-8
import json
import re

import torch
from rich import print
from transformers import AutoModel, AutoTokenizer

# 定義されたエンティティとそれに属する属性（スキーマ）
schema = {
    '金融': ['日期', '股票名称', '开盘价', '收盘价', '成交量'],
}
MODEL_NAME = "zai-org/chatglm-6b-int4"
IE_PATTERN = "{}\n\n提取上述句子中{}的实体，并按照JSON格式输出，上述句子中不存在的信息用['原文中未提及']来表示，多个值之间用','分隔。"

# モデル参照用のFew-shot例文データ
ie_examples = {
    '金融': [
        {
            'content': '2023-01-10，股市震荡。股票古哥-D[EOOE]美股今日开盘价100美元，一度飙升至105美元，随后回落至98美元，最终以102美元收盘，成交量达到520000。',
            'answers': {
                '日期': ['2023-01-10'],
                '股票名称': ['古哥-D[EOOE]美股'],
                '开盘价': ['100美元'],
                '收盘价': ['102美元'],
                '成交量': ['520000'],
            }
        }
    ]
}


def init_prompts() -> dict:
    """プロンプトの初期化を行い、インコンテキストラーニング（In-Context Learning）用の情報抽出データを生成する。

    Returns:
        dict: 以下のキーを持つ辞書。
            - 'ie_pre_history' (list): モデルに事前に入力する情報抽出の対話履歴リスト。
    """
    ie_pre_history = [
        (
            "现在你需要帮助我完成信息抽取任务，当我给你一个句子时，你需要帮我抽取出句子中实体信息，并按照JSON的格式输出，上述句子中没有的信息用['原文中未提及']来表示，多个值之间用','分隔。",
            '好的，请输入您的句子。'
        )
    ]

    for _type, example_list in ie_examples.items():
        for example in example_list:
            sentence = example['content']
            properties_str = ', '.join(schema[_type])
            schema_str_list = f'“{_type}”({properties_str})'

            sentence_with_prompt = IE_PATTERN.format(sentence, schema_str_list)

            ie_pre_history.append((
                f'{sentence_with_prompt}',
                f"{json.dumps(example['answers'], ensure_ascii=False)}"
            ))

    return {'ie_pre_history': ie_pre_history}


def clean_response(response: str):
    """モデルから返された文字列からJSON部分を抽出し、辞書オブジェクトにクリーニングする。

    Args:
        response (str): モデルから返された生の応答テキスト。

    Returns:
        dict | str: パースに成功した場合は辞書オブジェクト、失敗した場合は生の文字列。
    """
    if '```json' in response:
        res = re.findall('```json(.*?)```', response, re.DOTALL)
        if len(res) > 0 and res[0]:
            response = res[0]
        response = response.replace('、', '，')
    try:
        return json.loads(response.strip())
    except:
        return response


def inference(sentences: list, custom_settings: dict) -> None:
    """与えられた複数のテキストに対して、ChatGLMモデルを用いて情報抽出（エンティティ抽出）の推論を行う。

    Args:
        sentences (list[str]): 抽出対象となるテキスト（中国語）のリスト。
        custom_settings (dict): `init_prompts` 関数によって生成された、事前プロンプトを含む辞書。

    Returns:
        None: 結果はコンソールに直接出力されます。
    """
    for sentence in sentences:
        cls_res = '金融'
        if cls_res not in schema:
            exit()
        properties_str = ', '.join(schema[cls_res])
        schema_str_list = f'“{cls_res}”({properties_str})'
        sentence_with_prompt = IE_PATTERN.format(sentence, schema_str_list)

        ie_res, _ = my_model.chat(tokenizer, sentence_with_prompt, history=custom_settings['ie_pre_history'])
        ie_res = clean_response(ie_res)

        print(f"sentence:-->{sentence}")
        print(f"answer:-->{ie_res}\n")


if __name__ == '__main__':
    # トークナイザーとモデルの読み込み
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    my_model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).half().cuda().eval()

    # テスト用の中国語テキストリスト
    sentences = [
        '2023-02-15，寓意吉祥的节日，股票佰笃[BD]美股开盘价10美元，虽然经历了波动，但最终以13美元收盘，成交量微幅增加至460,000，投资者情绪较为平稳。',
        '2023-04-05，市场迎来轻松氛围，股票盘古(0021)开盘价23元，尽管经历了波动，但最终以26美元收盘，成交量缩小至310,000，投资者保持观望态度。',
    ]

    # プロンプトの設定を初期化して推論を実行
    custom_settings = init_prompts()
    inference(sentences, custom_settings)