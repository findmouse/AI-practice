import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import time
import torch.nn as nn
from rich import print
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'hfl/chinese-bert-wwm-ext'

# トークナイザーを読み込む
bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name)
bert_model = bert_model.to(device)


class MyDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        train_dataset = load_dataset('csv', data_files=data_path, split='train')

        # 44文字を超えるサンプルのみを抽出する
        self.train_dataset = train_dataset.filter(lambda x: len(x['text']) > 44)
        self.sample_len = len(self.train_dataset)

    def __len__(self):
        return self.sample_len

    def __getitem__(self, item):
        label = 1
        sequence = self.train_dataset[item]['text']
        sent1 = sequence[:22]
        sent2 = sequence[22:44]

        # 負例を生成する
        if random.randint(0, 1) == 0:
            j = random.randint(0, self.sample_len - 1)
            sent2 = self.train_dataset[j]['text'][22:44]
            label = 0

        return sent1, sent2, label


def collate_fn3(data):
    print(f'data-->{data}')

    sents = [i[:2] for i in data]
    labels = [i[-1] for i in data]

    # print(sents)
    # print(labels)

    # テキストを数値化する
    inputs = bert_tokenizer.batch_encode_plus(
        sents,
        padding="max_length",
        max_length=50,
        truncation=True,
        return_tensors='pt'
    )

    # print(f"inputs-->{inputs}")

    input_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']
    attention_mask = inputs['attention_mask']
    labels = torch.tensor(labels, dtype=torch.long)

    return input_ids, token_type_ids, attention_mask, labels


def test_dataset():
    my_dataset = MyDataset(data_path='./data/train.csv')

    # print(len(my_dataset))
    # sent1, sent2, label = my_dataset[0]
    # print(f'sent1-->{sent1}')
    # print(f'sent2-->{sent2}')
    # print(f'label-->{label}')

    my_dataloader = DataLoader(
        my_dataset,
        batch_size=4,
        collate_fn=collate_fn3,
        drop_last=True,
        shuffle=True
    )

    for input_ids, token_type_ids, attention_mask, labels in my_dataloader:
        print(f"input_ids-->{input_ids.shape}")
        print(f"token_type_ids-->{token_type_ids.shape}")
        print(f"attention_mask-->{attention_mask.shape}")
        print(f"labels-->{labels}")
        break


# モデルを定義する
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 768はBERTモデルの出力次元
        self.linear = nn.Linear(768, 2)

    def forward(self, input_ids, token_type_ids, attention_mask):
        with torch.no_grad():
            bert_output = bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

        # print(f'bert_output-->{bert_output}')
        # print(f'bert_output.last_hidden_state.shape-->{bert_output.last_hidden_state.shape}')
        # print(f'bert_output.pooler_output.shape-->{bert_output.pooler_output.shape}')

        # 文分類を行うため、pooler_output（CLSトークンに対応する文全体の特徴ベクトル）を直接利用する
        # output-->[8,2]
        output = self.linear(bert_output.pooler_output)

        return output


def test_model():
    my_dataset = MyDataset(data_path='./data/train.csv')

    # print(len(my_dataset))
    # sent1, sent2, label = my_dataset[0]
    # print(f'sent1-->{sent1}')
    # print(f'sent2-->{sent2}')
    # print(f'label-->{label}')

    my_dataloader = DataLoader(
        my_dataset,
        batch_size=4,
        collate_fn=collate_fn3,
        drop_last=True,
        shuffle=True
    )

    my_model = MyModel()
    my_model = my_model.to(device)

    for input_ids, token_type_ids, attention_mask, labels in my_dataloader:
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)

        output = my_model(input_ids, token_type_ids, attention_mask)

        # print(f"output-->{output}")
        break


def model2train():
    # モデルを生成する
    my_model = MyModel()
    my_model = my_model.to(device)

    # オプティマイザを生成する
    my_adamw = AdamW(my_model.parameters(), lr=5e-4)

    # 損失関数を生成する
    my_crossentropy = nn.CrossEntropyLoss()

    # 学習用Datasetを生成する
    train_dataset = MyDataset(data_path='./data/train.csv')

    # 事前学習済みモデルのパラメータは更新しない
    for param in bert_model.parameters():
        param.requires_grad_(False)

    # モデルを学習モードに設定する（事前学習済みモデルを利用するため）
    my_model.train()

    # エポック数を設定する
    epochs = 3

    # 学習開始
    for epoch_idx in range(1, epochs + 1):
        # DataLoaderを生成する
        my_dataloader = DataLoader(
            train_dataset,
            batch_size=8,
            collate_fn=collate_fn3,
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
            if i % 5 == 0:
                temp = torch.argmax(output, dim=-1)

                # print(f"temp-->{temp}")

                acc = (temp == labels).sum().item() / len(labels)

                # print('エポック:%d イテレーション:%d 損失:%.6f 正解率:%.3f 時間:%d' % (
                #     epoch_idx, i, my_loss.item(), acc, (int)(time.time() - start_time)))

                start_time = time.time()

        # モデルを保存する
        torch.save(my_model.state_dict(), './save_model/ai20_nsp_%d.bin' % epoch_idx)


if __name__ == '__main__':
    # test_dataset()
    # test_model()
    model2train()
