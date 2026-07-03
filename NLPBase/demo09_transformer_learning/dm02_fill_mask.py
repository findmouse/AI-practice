import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
import time
import torch.nn as nn
from rich import print
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'hfl/chinese-bert-wwm-ext'

# トークナイザーを読み込む
bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModelForMaskedLM.from_pretrained(model_name)
bert_model = bert_model.to(device)


def collate_fn2(data):
    sents = [i["text"] for i in data]
    inputs = bert_tokenizer.batch_encode_plus(
        sents,
        truncation=True,
        padding="max_length",
        max_length=32,
        return_tensors='pt'
    )

    # print(inputs[:1])
    input_ids = inputs["input_ids"]
    # print(f"input_ids-->{input_ids.shape}")
    token_type_ids = inputs["token_type_ids"]
    # print(f"token_type_ids-->{token_type_ids.shape}")
    attention_mask = inputs["attention_mask"]
    # print(f"attention_mask-->{attention_mask.shape}")

    # 各サンプルの16番目の位置を取り出し、Masked Language Modeling（MLM）学習用のラベルとする
    labels = input_ids[:, 16].view(-1).clone()
    input_ids[:, 16] = bert_tokenizer.mask_token_id

    # print(f'input_ids-->{input_ids}')
    # print(labels)

    return input_ids, token_type_ids, attention_mask, labels


def test_dataset():
    # 学習データを読み込む
    train_dataset = load_dataset('csv', data_files='./data/train.csv', split='train')

    # print(f"train_dataset-->{train_dataset[:2]}")
    # print(f"train_dataset-->{train_dataset}")

    # textの長さが32を超えるデータのみを抽出する
    new_train_dataset = train_dataset.filter(lambda x: len(x['text']) > 32)

    # print(f"new_train_dataset-->{new_train_dataset}")

    # DataLoaderを生成する
    train_dataloader = DataLoader(
        new_train_dataset,
        batch_size=8,
        collate_fn=collate_fn2,
        shuffle=True,
        drop_last=True
    )

    for inputs_ids, token_type_ids, attention_mask, labels in train_dataloader:
        print(f'inputs_ids.shape-->{inputs_ids.shape}')
        print(f'token_type_ids.shape-->{token_type_ids.shape}')
        print(f'attention_mask.shape-->{attention_mask.shape}')
        print(f'labels-->{labels}')
        break


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert_model = bert_model

    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_output = self.bert_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        # print(f"bert_output-->{bert_output}")

        # 各サンプルの16番目の[MASK]位置に対応する予測結果を取得する: [batch_size, vocab_size]
        return bert_output.logits[:, 16, :]


def test_model():
    # 学習データを読み込む
    train_dataset = load_dataset('csv', data_files='./data/train.csv', split='train')

    # print(f"train_dataset-->{train_dataset[:2]}")
    # print(f"train_dataset-->{train_dataset}")

    # textの長さが32を超えるデータのみを抽出する
    new_train_dataset = train_dataset.filter(lambda x: len(x['text']) > 32)

    # print(f"new_train_dataset-->{new_train_dataset}")

    # DataLoaderを生成する
    train_dataloader = DataLoader(
        new_train_dataset,
        batch_size=8,
        collate_fn=collate_fn2,
        shuffle=True,
        drop_last=True
    )

    my_model = MyModel()
    my_model = my_model.to(device)

    for inputs_id, token_type_ids, attention_mask, labels in train_dataloader:
        inputs_id = inputs_id.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        output = my_model(inputs_id, token_type_ids, attention_mask)
        print(f"output.shape-->{output.shape}")
        break


def model2train():
    # モデルを生成する
    my_model = MyModel()
    my_model = my_model.to(device)

    # オプティマイザを生成する
    my_adamw = AdamW(my_model.parameters(), lr=2e-5)

    # 損失関数を生成する
    my_crossentropy = nn.CrossEntropyLoss()

    # Datasetを読み込む
    train_dataset = load_dataset('csv', data_files='./data/train.csv', split='train')
    train_dataset = train_dataset.filter(lambda x: len(x["text"]) > 32)

    # モデルを学習モードに設定する（事前学習済みモデルを利用するため）
    my_model.train()

    # エポック数を設定する
    epochs = 5

    # 学習開始
    for epoch_idx in range(1, epochs + 1):
        # DataLoaderを生成する
        my_dataloader = DataLoader(
            train_dataset,
            batch_size=128,
            collate_fn=collate_fn2,
            shuffle=True,
            drop_last=True
        )

        global start_time
        start_time = time.time()

        # データを反復処理する
        for i, (inputs_ids, token_type_ids, attention_mask, labels) in enumerate(tqdm(my_dataloader)):
            inputs_ids = inputs_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            output = my_model(inputs_ids, token_type_ids, attention_mask)

            # print(f"output-->{output}")

            # 損失を計算する
            my_loss = my_crossentropy(output, labels)

            # 勾配を初期化する
            my_adamw.zero_grad()

            # 逆伝播
            my_loss.backward()

            # パラメータを更新する
            my_adamw.step()

            # print(f"labels-->{labels}")

            # ログを出力する
            if i % 20 == 0:
                temp = torch.argmax(output, dim=-1)

                # print(f"temp-->{temp}")

                acc = (temp == labels).sum().item() / len(labels)

                print('エポック:%d イテレーション:%d 損失:%.6f 正解率:%.3f 時間:%d' % (
                    epoch_idx, i, my_loss.item(), acc, (int)(time.time() - start_time)))

                start_time = time.time()

        # モデルを保存する
        torch.save(my_model.state_dict(), './save_model/ai20_text_fill_mask_%d.bin' % epoch_idx)


def model2test():
    # テスト用Datasetを読み込む
    test_dataset = load_dataset('csv', data_files='./data/test.csv', split='train')
    test_dataset = test_dataset.filter(lambda x: len(x["text"]) > 32)

    # 学習済みモデルを読み込む
    my_model = MyModel()
    my_model.load_state_dict(torch.load('./save_model/ai20_text_fill_mask_5.bin', map_location=device))
    my_model.to(device)

    # テスト用DataLoaderを生成する
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=8,
        collate_fn=collate_fn2,
        shuffle=True,
        drop_last=True
    )

    # モデルを評価モードに設定する
    my_model.eval()

    # 評価用変数
    correct = 0  # 正しく予測したサンプル数
    total = 0  # 評価済みサンプル総数

    for i, (inputs_ids, token_type_ids, attention_mask, labels) in enumerate(tqdm(test_dataloader)):
        # input_ids-->[8, 32]
        inputs_ids = inputs_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            output = my_model(inputs_ids, token_type_ids, attention_mask)

        # 予測確率が最大となるクラスのインデックスを取得する output-->[8, vocab_size]
        temp_idx = torch.argmax(output, dim=-1)

        correct = correct + (temp_idx == labels).sum().item()
        total = total + len(labels)

        if i % 25 == 0:
            print(correct / total, end=" ")

            # 各バッチの先頭サンプルを出力し、目視で予測結果を確認する
            first_tokens = bert_tokenizer.decode(inputs_ids[0])
            print(first_tokens, end=' ')
            print('予測値:', bert_tokenizer.decode(temp_idx[0]), "正解値:", bert_tokenizer.decode(labels[0]))


if __name__ == '__main__':
    # test_dataset()
    # test_model()
    # model2train()
    model2test()
