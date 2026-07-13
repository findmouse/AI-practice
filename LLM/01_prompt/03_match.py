# coding:utf-8
import torch
from rich import print
from transformers import AutoModel, AutoTokenizer

# 意味が類似しているペア、類似していないペアの例文（Few-shot用）
examples = {
    '是': [
        ('公司ABC发布了季度财报，显示盈利增长。', '财报披露，公司ABC利润上升。'),
    ],
    '不是': [
        ('黄金价格下跌，投资者抛售。', '外汇市场交易额创下新高。'),
        ('央行降息，刺激经济增长。', '新能源技术的创新。')
    ]
}
MODEL_NAME = "zai-org/chatglm-6b-int4"


def init_prompts() -> dict:
    """プロンプトの初期化を行い、インコンテキストラーニング（In-Context Learning）用のテキスト一致データを生成する。

    Returns:
        dict: 以下のキーを持つ辞書。
            - 'pre_history' (list): モデルに事前に入力する類似度判定の対話履歴リスト。
    """
    pre_history = [
        (
            '现在你需要帮助我完成文本匹配任务，当我给你两个句子时，你需要回答我这两句话语义是否相似。只需要回答是否相似，不要做多余的回答。',
            '好的，我将只回答”是“或”不是“。'
        )
    ]

    for key, sentence_pairs in examples.items():
        for sentence_pair in sentence_pairs:
            sentence1, sentence2 = sentence_pair
            pre_history.append((f'句子一：{sentence1}\n句子二：{sentence2}\n上面两句话是相似的语义吗？', key))

    return {'pre_history': pre_history}


def inference(sentence_pairs: list, custom_settings: dict) -> None:
    """与えられた複数の文ペアに対して、ChatGLMモデルを用いて意味的類似性の推論を行う。

    Args:
        sentence_pairs (list[tuple[str, str]]): 判定対象となる文ペア（中国語）のリスト。
        custom_settings (dict): `init_prompts` 関数によって生成された、事前プロンプトを含む辞書。

    Returns:
        None: 結果はコンソールに直接出力されます。
    """
    for sentence_pair in sentence_pairs:
        sentence1, sentence2 = sentence_pair
        # 入力文ペアをプロンプト形式に整形
        sentence_with_prompt = f'句子一: {sentence1}\n句子二: {sentence2}\n上面两句话是相似的语义吗？'

        # 推論の実行
        response, history = my_model.chat(
            tokenizer,
            sentence_with_prompt,
            history=custom_settings['pre_history']
        )

        # 結果の出力
        print(f'>>> [bold bright_red]sentence: {sentence_pair}')
        print(f'>>> [bold bright_green]inference answer: {response}\n')


if __name__ == '__main__':
    # トークナイザーとモデルの読み込み
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    my_model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).half().cuda().eval()

    # テスト用の中国語文ペアリスト
    sentence_pairs = [
        ('股票市场今日大涨，投资者乐观。', '持续上涨的市场让投资者感到满意。'),
        ('油价大幅下跌，能源公司面临挑战。', '未来智能城市的建设趋势愈发明显。'),
        ('利率上升，影响房地产市场。', '高利率对房地产有一定冲击。'),
    ]

    # プロンプトの設定を初期化して推論を実行
    custom_settings = init_prompts()
    inference(sentence_pairs, custom_settings)