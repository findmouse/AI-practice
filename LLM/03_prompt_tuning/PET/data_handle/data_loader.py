# coding:utf-8
import sys
import warnings
from functools import partial
from pathlib import Path

for _root in Path(__file__).resolve().parents:
    if (_root / "pet_bootstrap.py").is_file():
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        break
else:
    raise RuntimeError("Cannot find PET project root (pet_bootstrap.py)")

from pet_bootstrap import bootstrap

bootstrap(__file__)

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, default_data_collator

from ProjectConfig import ProjectConfig
from data_handle.data_preprocess import convert_example
from data_handle.template import HardTemplate

pc = ProjectConfig()


def get_data(tokenizer=None):
    """学習・検証データローダーを構築する。"""
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            str(pc.pre_model), local_files_only=True
        )

    prompt = pc.prompt_file.read_text(encoding='utf8').splitlines()[0].strip()
    hard_template = HardTemplate(prompt=prompt)
    dataset = load_dataset(
        'text',
        data_files={'train': str(pc.train_path), 'dev': str(pc.dev_path)},
    )
    train_labels = {
        row['text'].split('\t', 1)[0] for row in dataset['train'] if '\t' in row['text']
    }
    dev_labels = {
        row['text'].split('\t', 1)[0] for row in dataset['dev'] if '\t' in row['text']
    }
    missing_labels = sorted(dev_labels - train_labels)
    if missing_labels:
        warnings.warn(
            "検証データにのみ存在するラベルがあります: "
            + ", ".join(missing_labels)
            + "。これらのクラスは正しく学習できません。",
            stacklevel=2,
        )
    new_func = partial(convert_example,
                       tokenizer=tokenizer,
                       hard_template=hard_template,
                       max_seq_len=pc.max_seq_len,
                       max_label_len=pc.max_label_len)

    dataset = dataset.map(new_func, batched=True)

    train_dataset = dataset["train"]
    dev_dataset = dataset["dev"]
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  collate_fn=default_data_collator,
                                  batch_size=pc.batch_size)
    dev_dataloader = DataLoader(dev_dataset,
                                collate_fn=default_data_collator,
                                batch_size=pc.batch_size)
    return train_dataloader, dev_dataloader


if __name__ == '__main__':
    train_dataloader, dev_dataloader = get_data()
    print(len(train_dataloader))
    print(len(dev_dataloader))
    for i, value in enumerate(train_dataloader):
        print(i)
        print(value)
        print(value['input_ids'].dtype)
        break
