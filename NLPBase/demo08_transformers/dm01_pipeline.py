import os

# TensorFlowの警告を抑制するため環境変数を設定
os.environ["USE_TF"] = "0"

import numpy as np
from transformers import pipeline


def execute_text_classification() -> None:
    """感情分析（Text Classification）の検証用関数

    指定されたモデルを用いて、日本語テキストの感情分析を行います。
    """
    # 感情分析パイプラインの初期化
    pipe = pipeline(
        "text-classification", model="tabularisai/multilingual-sentiment-analysis"
    )

    # テキストの分類実行（デモ用に日本語テキストに変更）
    sentence = "この製品は本当に素晴らしいです。大満足しています。"
    result = pipe(sentence)

    # 実行結果の出力
    print(f"[INFO] 感情分析 実行結果: {result}")


def execute_feature_extraction() -> None:
    """特徴量抽出（Feature Extraction）の検証用関数

    汎用的な日本語BERTモデルを用いて、テキストから特徴量を抽出します。
    """
    # 特徴量抽出パイプラインの初期化（汎用的な日本語BERTモデルに変更）
    pipe = pipeline(
        task="feature-extraction", model="cl-tohoku/bert-base-japanese-whole-word-masking"
    )

    # 特徴量抽出の実行
    sentence = "人生はどこから始めるべきか"
    result = pipe(sentence)

    # 実行結果および形状（Shape）の出力
    print(f"[INFO] 特徴量抽出 戻り値の型: {type(result)}")
    print(f"[INFO] 特徴量抽出 データの形状(Shape): {np.array(result).shape}")


def execute_fill_mask() -> None:
    """穴埋めタスク（Fill-Mask）の検証用関数

    [MASK] トークンを含むテキストの穴埋め予測を行います。
    """
    # 穴埋めパイプラインの初期化
    pipe = pipeline(
        task="fill-mask", model="cl-tohoku/bert-base-japanese-whole-word-masking"
    )

    # 穴埋め予測の実行（[MASK]を用いて予測）
    sentence = "私は明日、[MASK]の家に行ってご飯を食べます。"
    result = pipe(sentence)

    print(f"[INFO] 穴埋めタスク 実行結果: {result}")


def execute_question_answering() -> None:
    """質問応答（QA / Reading Comprehension）の検証用関数

    与えられたコンテキスト（文脈）に基づいて、質問に対する回答を抽出します。
    """
    # 質問応答パイプラインの初期化（日本語対応モデルに変更）
    pipe = pipeline(
        task="question-answering",
        model="takehika/xlm-roberta-ja-jaquad-qa",
    )

    context = """
    太郎は東京に住んでいます。
    彼はシステムエンジニアです。
    趣味はバスケットボールです。
    """

    questions = [
        "太郎はどこに住んでいますか？",
        "太郎の職業は何ですか？",
        "太郎の趣味は何ですか？",
    ]

    examples = [
        {
            "context": context,
            "question": q
        }
        for q in questions
    ]

    results = pipe(examples)

    for q, r in zip(questions, results):
        print(q)
        print(r["answer"])
        # print(r["score"])


def execute_summarization() -> None:
    """テキスト要約（Summarization）の検証用関数

    英文のテキストを受け取り、その要約文を自動生成します。
    """
    # テキスト要約パイプラインの初期化
    pipe = pipeline(task="summarization", model="sshleifer/distilbart-cnn-12-6")

    # 対象テキスト（BERTの概要説明文）
    sentence = (
        "BERT is a transformers model pretrained on a large corpus of English data "
        "in a self-supervised fashion. This means it was pretrained on the raw texts "
        "only, with no humans labelling them in any way (which is why it can use lots "
        "of publicly available data) with an automatic process to generate inputs and "
        "labels from those texts. More precisely, it was pretrained with two objectives:"
        "Masked language modeling (MLM): taking a sentence, the model randomly masks 15% "
        "of the words in the input then run the entire masked sentence through the model "
        "and has to predict the masked words. This is different from traditional recurrent "
        "neural networks (RNNs) that usually see the words one after the other, or from "
        "autoregressive models like GPT which internally mask the future tokens. It allows "
        "the model to learn a bidirectional representation of the sentence.Next sentence "
        "prediction (NSP): the models concatenates two masked sentences as inputs during "
        "pretraining. Sometimes they correspond to sentences that were next to each other "
        "in the original text, sometimes not. The model then has to predict if the two "
        "sentences were following each other or not."
    )

    result = pipe(sentence)
    print(f"[INFO] テキスト要約 実行結果: {result}")


# 日本語の固有表現抽出のモデル
def execute_named_entity_recognition() -> None:
    """固有表現抽出（NER）の検証用関数

    テキスト内から人名や地名などの固有表現を抽出します。
    """
    sentence = "鈴井は4月の陽気の良い日に、鈴をつけて北海道のトムラウシへと登った"
    # 固有表現抽出パイプラインの初期化（日本語対応モデルに変更）
    pipe = pipeline(
        task="token-classification",
        model="tsmatz/xlm-roberta-ner-japanese",
        aggregation_strategy="simple",
    )

    # 固有表現抽出の実行
    result = pipe(sentence)
    # print(f"[INFO] 固有表現抽出 実行結果: {result}")
    for i in result:
        print(i)


if __name__ == "__main__":
    # 必要に応じて各関数のコメントアウトを解除して実行してください
    # execute_text_classification()
    # execute_feature_extraction()
    # execute_fill_mask()
    execute_question_answering()
    # execute_summarization()
    # execute_named_entity_recognition()
