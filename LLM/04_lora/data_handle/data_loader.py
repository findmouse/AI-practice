# coding:utf-8
from functools import partial

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, default_data_collator

from data_handle.data_preprocess import convert_example
from glm_config import ProjectConfig

pc = ProjectConfig()


def get_data(tokenizer):
    dataset = load_dataset('text', data_files={'train': pc.train_path,
                                               'dev': pc.dev_path})
    # partical函数的作用是将函数：convert_example里面的3个参数：tokenizer,max_source_seq_len,
    # max_target_seq_len长度固定，也就是说函数:new_func在使用的时候，只需要传入函数：convert_example的其他的一个没有被固定的变量：example
    # 就可以了，因为在这里，其他的参数已经被固定了。
    new_func = partial(convert_example,
                       tokenizer=tokenizer,
                       max_source_seq_len=pc.max_source_seq_len,
                       max_target_seq_len=pc.max_target_seq_len)

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
    tokenizer = AutoTokenizer.from_pretrained(pc.pre_model, trust_remote_code=True)
    train_dataloader, dev_dataloader = get_data(tokenizer)
    print(len(train_dataloader))
    print(len(dev_dataloader))
    for i, value in enumerate(train_dataloader):
        print(value)
        print(value['input_ids'].shape)
        print(value['labels'].shape)
        break
