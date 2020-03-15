# coding=utf-8
# author=yphacker

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from conf import config
from conf import model_config_bert


class MyDataset(Dataset):

    def __init__(self, df, mode='train', task='1'):
        self.mode = mode
        self.task = task
        self.tokenizer = BertTokenizer.from_pretrained(model_config_bert.pretrain_model_path)
        self.pad_idx = self.tokenizer.pad_token_id
        self.x_data = []
        self.y_data = []
        for i, row in df.iterrows():
            x, y = self.row_to_tensor(self.tokenizer, row)
            self.x_data.append(x)
            self.y_data.append(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def contact(self, str1, str2):
        if pd.isnull(str2):
            return str1
        return str1 + str2

    def row_to_tensor(self, tokenizer, row):
        if self.task == '0':
            text = row['content']
        elif self.task == '1':
            text = self.contact(row['content'], row['comment_2c'])
        else:
            text = self.contact(row['content'], row['comment_all'])
        x_encode = tokenizer.encode(text)
        if len(x_encode) > config.max_seq_len[self.task]:
            text_len = int(config.max_seq_len[self.task] / 2)
            x_encode = x_encode[:text_len] + x_encode[-text_len:]
        else:
            padding = [0] * (config.max_seq_len[self.task] - len(x_encode))
            x_encode += padding
        x_tensor = torch.tensor(x_encode, dtype=torch.long)
        if self.mode == 'test':
            y_tensor = torch.tensor([0] * len(config.label_columns), dtype=torch.long)
        else:
            # y_data = row[config.label_columns]
            y_data = row['label']
            y_tensor = torch.tensor(y_data, dtype=torch.long)
        return x_tensor, y_tensor

    def __len__(self):
        return len(self.y_data)


if __name__ == "__main__":
    pass
