# 导入工具包
import os

# TensorFlowの警告を抑制するため環境変数を設定
os.environ["USE_TF"] = "0"

from transformers import BertTokenizer, BertForMaskedLM
import torch
from rich import print


def dm01_fill_mask():
    # トークナイザーを読み込む
    my_tokerizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")

    # モデルを読み込む
    my_model = BertForMaskedLM.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")

    # テキストの分類実行（デモ用に日本語テキストに変更）
    sentence = "私は明日[MASK]の家に行ってご飯を食べます。"

    # データをトークナイザーに渡し、モデル入力形式へ変換する
    result2 = my_tokerizer.encode_plus(sentence, return_tensors="pt")
    print(f"result2-->{result2}")
    print(f"type(result2)-->{type(result2)}")
    my_model.eval()
    output = my_model(**result2).logits
    print(f"output.logits.shape-->{output.shape}")
    temp = output[0][5]
    idx = torch.argmax(temp, dim=-1).item()
    token = my_tokerizer.convert_ids_to_tokens(idx)
    print(token)


if __name__ == '__main__':
    dm01_fill_mask()