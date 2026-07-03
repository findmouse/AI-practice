import os

# TensorFlowの警告を抑制するため環境変数を設定
os.environ["USE_TF"] = "0"

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoModelForQuestionAnswering
from transformers import AutoModelForSeq2SeqLM, AutoModelForTokenClassification

from rich import print


# 感情分析
def dm01_text_classification():
    # トークナイザーを読み込む
    my_tokerizer = AutoTokenizer.from_pretrained("tabularisai/multilingual-sentiment-analysis")

    # モデルを読み込む
    my_model = AutoModelForSequenceClassification.from_pretrained("tabularisai/multilingual-sentiment-analysis")

    # テキストの分類実行（デモ用に日本語テキストに変更）
    sentence = "この製品は本当に素晴らしいです。大満足しています。"

    result2 = my_tokerizer.encode(text=sentence, return_tensors='pt', padding="max_length", truncation=True,
                                  max_length=20)
    # print(f"result2-->{result2}")
    # print(f"type(result2)-->{type(result2)}")
    my_model.eval()
    output = my_model(result2)
    # print(f"output-->{output.logits}")
    index1 = torch.argmax(output.logits, dim=-1)
    print(index1)


# 特徴抽出
def dm02_feature_extraction():
    # トークナイザーを読み込む
    my_tokerizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")

    # モデルを読み込む
    my_model = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")

    # テキストの分類実行（デモ用に日本語テキストに変更）
    sentence = ["君はだれ？", "人生はどこから始めるべきか"]

    result = my_tokerizer.encode_plus(
        sentence,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=20,
    )

    # エンコード後のテキストをモデルへ入力する
    my_model.eval()
    output = my_model(**result)
    # print(f"output--{output}")
    # print(f"output.last_hidden_state-->{output.last_hidden_state.shape}")
    # print(f"output.pooler_output-->{output.pooler_output.shape}")


# マスク補完
def dm03_fill_mask():
    # トークナイザーを読み込む
    my_tokerizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")

    # モデルを読み込む
    my_model = AutoModelForMaskedLM.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")

    # テキストの分類実行（デモ用に日本語テキストに変更）
    sentence = "私は明日、[MASK]の家に行ってご飯を食べます。"

    # データをトークナイザーに渡し、モデル入力形式へ変換する
    result2 = my_tokerizer.encode_plus(sentence, return_tensors="pt")
    # print(f"result2-->{result2}")
    # print(f"type(result2)-->{type(result2)}")
    my_model.eval()
    output = my_model(**result2)
    # print(f"output.logits.shape-->{output.logits.shape}")
    temp = output.logits[0, sentence.index("[")]
    # print(f"temp-->{temp}")
    idx = torch.argmax(temp, dim=-1).item()
    # print(f"idx-->{idx}")
    print(f"最も可能性が高い単語：{my_tokerizer.convert_ids_to_tokens([idx])}")
    # index1 = torch.argmax(output.logits, dim=-1)
    # print(index1)


# 質問応答
def dm04_qa():
    # トークナイザーを読み込む
    my_tokerizer = AutoTokenizer.from_pretrained("takehika/xlm-roberta-ja-jaquad-qa")

    # モデルを読み込む
    my_model = AutoModelForQuestionAnswering.from_pretrained("takehika/xlm-roberta-ja-jaquad-qa")

    # 文脈と質問の定義
    context = "私は太郎です。システムエンジニアをしています。趣味はバスケットボールです。"
    questions = ["私は誰ですか？", "私の職業は何ですか？", "私の趣味は何ですか？"]

    for question in questions:
        result2 = my_tokerizer.encode_plus(question, context, return_tensors='pt', max_length=100)
        # print(f'result2-->{result2}')
        # print(f'result2.input_ids-->{result2.input_ids.shape}')

        my_model.eval()
        output = my_model(**result2)
        # print(f"output-->{output}")
        # print(f"output.start_logits-->{output.start_logits.shape}")
        # print(f"output.end_logits-->{output.end_logits.shape}")
        # start_logitsの予測確率が最大となるインデックスを取得

        start_idx = torch.argmax(output.start_logits, dim=-1).item()
        # print(f"start_idx-->{start_idx}")

        # end_logitsの予測確率が最大となるインデックスを取得

        end_idx = torch.argmax(output.end_logits, dim=-1).item()
        # print(f"end_idx-->{end_idx}")
        # 実際の回答へ変換
        answer_ids = result2["input_ids"][0][start_idx:end_idx + 1]
        answer = my_tokerizer.decode(
            answer_ids,
            skip_special_tokens=True
        )
        print(question, answer)


# 要約
def dm05_summary():
    # トークナイザーを読み込む
    my_tokerizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

    # モデルを読み込む
    my_model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")

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
    # データをエンコードする
    inputs = my_tokerizer.encode_plus(sentence, return_tensors="pt")
    # print(f"inputs-->{inputs}")
    my_model.eval()
    # outputs = my_model.generate(**inputs)
    outputs = my_model.generate(inputs.input_ids)
    print(f"output-->{outputs}")
    print(f"output.shape-->{outputs.shape}")

    # デコード
    result = my_tokerizer.decode(outputs[0])
    print(f"result-->{result}")
    print("*" * 80)
    result2 = my_tokerizer.decode(outputs[0], skip_special_tokens=True)
    print(f"result2-->{result2}")
    print("*" * 80)
    result3 = my_tokerizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(f"result3-->{result3}")


# 固有表現抽出（NER）
def dm06_ner():
    # トークナイザーを読み込む
    my_tokerizer = AutoTokenizer.from_pretrained("tsmatz/xlm-roberta-ner-japanese")

    # モデルを読み込む
    my_model = AutoModelForTokenClassification.from_pretrained("tsmatz/xlm-roberta-ner-japanese")
    # データを準備する
    sentence = "鈴井は4月の陽気の良い日に、鈴をつけて北海道のトムラウシへと登った"

    # 設定ファイルを読み込む
    my_config = AutoConfig.from_pretrained("tsmatz/xlm-roberta-ner-japanese")
    # print(f'my_config-->{my_config}')
    # print(f'my_config-->{my_config.id2label}')
    # データをテンソル化する
    inputs = my_tokerizer.encode_plus(sentence, return_tensors='pt')
    # print(f'inputs-->{inputs}')
    # データをモデルへ入力する
    my_model.eval()

    outputs = my_model(**inputs).logits
    print(f"outputs-->{outputs.shape}")
    my_tokens = my_tokerizer.convert_ids_to_tokens(inputs.input_ids[0])
    print(f'my_tokens-->{my_tokens}')
    output = []
    # 各トークンに対して予測を行う
    for token, value in zip(my_tokens, outputs[0]):
        print(f"token-->{token}")
        print(f"value.shape-->{value.shape}")
        if token in my_tokerizer.all_special_tokens:
            continue
        # valueの中で確率が最大となるインデックスを取得
        idx = torch.argmax(value, dim=-1).item()
        print(f'idx-->{idx}')
        output.append((token, my_config.id2label[idx]))

    print(f'output-->{output}')


if __name__ == '__main__':
    dm01_text_classification()
    dm02_feature_extraction()
    dm03_fill_mask()
    dm04_qa()
    dm05_summary()
    dm06_ner()