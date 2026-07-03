import torch
from datasets import load_dataset
from transformers import AutoTokenizer, BertModel
import time
import torch.nn as nn
from rich import print
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# トークナイザーを読み込む
bert_tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

bert_model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
bert_model = bert_model.to(device)


# Datasetを読み込む
def dm_load_dataset():
    # 学習データを読み込む
    # splitを指定しない場合はDatasetDictを返し、splitを指定するとDataset（全データを辞書形式で保持するリスト）が返される
    train_dataset = load_dataset('csv', data_files='./data/train.csv', split='train')
    print(f'train_dataset-->{train_dataset}')
    print(f'train_dataset-->{train_dataset[0]}')
    print(f'train_dataset-->{train_dataset[0, 3]}')

    # 検証データを読み込む
    # splitを指定しない場合はDatasetDictを返し、splitを指定するとDataset（全データを辞書形式で保持するリスト）が返される
    valid_dataset = load_dataset('csv', data_files='./data/validation.csv', split='train')
    print(f'valid_dataset-->{valid_dataset}')
    print(f'valid_dataset-->{valid_dataset[0]}')
    print(f'valid_dataset-->{valid_dataset[0, 3]}')

    # テストデータを読み込む
    # splitを指定しない場合はDatasetDictを返し、splitを指定するとDataset（全データを辞書形式で保持するリスト）が返される
    test_dataset = load_dataset('csv', data_files='./data/test.csv', split='train')
    print(f'test_dataset-->{test_dataset}')
    print(f'test_dataset-->{test_dataset[0]}')
    print(f'test_dataset-->{test_dataset[0, 3]}')


def collate_fn1(data):
    sents = [i['text'] for i in data]
    labels = [i['label'] for i in data]
    inputs = bert_tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents, truncation=True, padding="max_length",
                                              max_length=300, return_tensors='pt', return_length=True)

    # print(f"inputs-->{inputs}")
    inputs_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']
    attention_mask = inputs['attention_mask']
    labels = torch.tensor(labels, dtype=torch.long)
    return inputs_ids, token_type_ids, attention_mask, labels


# Datasetの動作確認
def dm02_test_dataset():
    train_dataset = load_dataset('csv', data_files='./data/train.csv', split='train')
    # 上記のDatasetオブジェクトをさらにDataLoaderでラップする
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True, collate_fn=collate_fn1)
    # for inputs_ids, token_type_ids, attention_mask, labels in train_dataloader:
    #     print(f'inputs_ids.shape-->{inputs_ids.shape}')
    #     print(f'token_type_ids.shape-->{token_type_ids.shape}')
    #     print(f'attention_mask.shape-->{attention_mask.shape}')
    #     print(f'labels.shape-->{labels.shape}')
    #     break
    return train_dataloader


# 二値分類モデルを定義する
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 768はBERTモデルの出力次元
        self.linear = nn.Linear(768, 2)

    def forward(self, input_ids, token_type_ids, attention_mask):
        with torch.no_grad():
            bert_output = bert_model(input_ids=input_ids, attention_mask=attention_mask,
                                     token_type_ids=token_type_ids)

        # print(f'bert_output-->{bert_output}')
        # print(f'bert_output.last_hidden_state.shape-->{bert_output.last_hidden_state.shape}')
        # print(f'bert_output.pooler_output.shape-->{bert_output.pooler_output.shape}')
        # 文分類を行うため、pooler_output（CLSトークンに対応する文全体の特徴ベクトル）を直接利用する
        # output-->[8,2]
        output = self.linear(bert_output.pooler_output)

        return output


def dm03_test_model():
    my_model = MyModel()
    train_dataloader = dm02_test_dataset()
    for inputs_ids, token_type_ids, attention_mask, labels in train_dataloader:
        # print(f'inputs_ids.shape-->{inputs_ids.shape}')
        # print(f'token_type_ids.shape-->{token_type_ids.shape}')
        # print(f'attention_mask.shape-->{attention_mask.shape}')
        # print(f'labels.shape-->{labels.shape}')
        output = my_model(inputs_ids, token_type_ids, attention_mask)
        print(f"output-->{output}")
        break

# モデルの訓練
def model2train():
    # モデルを生成する
    my_model = MyModel()
    my_model = my_model.to(device)
    # オプティマイザを生成する
    my_adamw = AdamW(my_model.parameters(), lr=5e-4)
    # 損失関数を生成する
    my_crossentropy = nn.CrossEntropyLoss()
    # Datasetを読み込む
    train_dataset = load_dataset('csv', data_files='./data/train.csv', split='train')
    # 事前学習済みモデルのパラメータは更新しない
    for param in bert_model.parameters():
        param.requires_grad_(False)
    # モデルを学習モードに設定する（事前学習済みモデルを利用するため）
    my_model.train()
    # エポック数を設定する
    epochs = 1

    # 学習開始
    for epoch_idx in range(1, epochs + 1):
        # DataLoaderを生成する
        my_dataloader = DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn1, shuffle=True, drop_last=True)
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
        torch.save(my_model.state_dict(), './save_model/ai20_text_class_%d.bin' % epoch_idx)


# DataLoaderを生成する


def model2test():
    # テスト用Datasetを読み込む
    test_dataset = load_dataset('csv', data_files='./data/test.csv', split='train')
    # 学習済みモデルを読み込む
    my_model = MyModel()
    my_model.load_state_dict(torch.load('./save_model/ai20_text_class_1.bin', map_location=device))
    my_model.to(device)
    # テスト用DataLoaderを生成する
    test_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=collate_fn1, shuffle=True, drop_last=True)
    # モデルを評価モードに設定する
    my_model.eval()
    # 評価用変数
    correct = 0  # 正しく予測したサンプル数
    total = 0  # 評価済みサンプル総数

    for i, (inputs_ids, token_type_ids, attention_mask, labels) in enumerate(tqdm(test_dataloader)):
        # input_ids-->[8, 300]
        inputs_ids = inputs_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            output = my_model(inputs_ids, token_type_ids, attention_mask)
        # 予測確率が最大となるクラスのインデックスを取得する output-->[8,2]
        temp_idx = torch.argmax(output, dim=-1)
        correct = correct + (temp_idx == labels).sum().item()
        total = total + len(labels)

        if i % 5 == 0:
            print(correct / total, end=" ")
            # 各バッチの先頭サンプルを出力し、目視で予測結果を確認する
            first_tokens = bert_tokenizer.decode(inputs_ids[0], skip_special_tokens=True)
            print(first_tokens, end=' ')
            print('予測値:', temp_idx[0], "正解値:", labels[0])


if __name__ == '__main__':
    # dm_load_dataset()
    # dm02_test_dataset()
    # dm03_test_model()
    # model2train()
    model2test()