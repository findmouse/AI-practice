# PET Prompt Tuning Demo

中国語レビューを `[MASK]` によって分類する、小規模な PET（Pattern-Exploiting Training）デモです。

## ディレクトリ構成

```text
PET/
├─ data/                 # 学習・検証データ、prompt、verbalizer
├─ data_handle/          # テンプレート処理とデータローダー
├─ utils/                # loss、評価指標、verbalizer
├─ bert-base-chinese/    # tokenizer とモデル設定
├─ checkpoints/          # train.py が生成（Git 管理外）
├─ ProjectConfig.py      # パスと学習パラメータ
├─ train.py              # 学習入口
└─ inference.py          # 推論入口
```

## セットアップ

```powershell
conda activate LLM
python -m pip install -e .
```

## 事前学習済みモデルのダウンロード

事前学習済みの中国語 BERT モデルは、以下からダウンロードできます。

- [google-bert/bert-base-chinese（Hugging Face）](https://huggingface.co/google-bert/bert-base-chinese)

モデルの設定、tokenizer、`model.safetensors` または `pytorch_model.bin` などの重みファイルをダウンロードし、プロジェクト直下の `bert-base-chinese/` に配置してください。

## 実行

PET ディレクトリ以外をカレントディレクトリにしても実行できます。

```powershell
python C:\workspace\AI-practice\LLM\03_prompt_tuning\PET\train.py
python C:\workspace\AI-practice\LLM\03_prompt_tuning\PET\inference.py
```

`checkpoints/model_best` がない場合、推論は構造確認用のランダム初期化モデルへフォールバックします。この場合、処理は最後まで動きますが予測結果に意味はありません。意味のある推論には、事前学習済み重みを `bert-base-chinese` に配置して学習するか、学習済みの `checkpoints/model_best` を用意してください。

現在の小規模な `train.txt` には「手机」「电器」クラスが含まれていません。データローダーはこの状態を警告します。これらも評価対象にする場合は、各クラスの学習例を追加してください。

## データ形式

- `data/train.txt`, `data/dev.txt`: `親ラベル<TAB>レビュー本文`
- `data/prompt.txt`: `{MASK}` と `{textA}` を含むテンプレート
- `data/verbalizer.txt`: `親ラベル<TAB>子ラベル1,子ラベル2,...`
