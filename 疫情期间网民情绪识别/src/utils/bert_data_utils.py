# coding=utf-8
# author=yphacker


import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from conf import config
from conf import model_config_bert
from utils.data_utils import clean_name, clean_huati


class MyDataset(Dataset):

    def __init__(self, df, mode='train'):
        self.mode = mode
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

    def row_to_tensor(self, tokenizer, row):
        text = row["微博中文内容"]
        x_encode = tokenizer.encode(text, max_length=config.max_seq_len)
        padding = [0] * (config.max_seq_len - len(x_encode))
        x_encode += padding
        x_tensor = torch.tensor(x_encode, dtype=torch.long)
        if self.mode == 'test':
            y_tensor = torch.tensor([0], dtype=torch.long)
        else:
            y_tensor = torch.tensor(config.label2id[row['情感倾向']], dtype=torch.long)
        return x_tensor, y_tensor

    def __len__(self):
        return len(self.y_data)


if __name__ == "__main__":
    pass
