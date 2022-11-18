#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/11/18 17:02
# @Author: lionel
from torch import nn
from torch.utils import data


class CNewsDataset(data.Dataset):
    def __init__(self, file_path, mode='cut', max_seq_len=128):
        self.datas = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) != 2:
                    continue
                tag, text = tuple(fields)
                self.datas.append((tag, text))
        self.mode = mode
        self.max_seq_len = max_seq_len

    def __getitem__(self, item):
        tag, text = self.datas[item]
        if self.mode == 'cut':
            if len(text) > self.max_seq_len:
                text = text[:self.max_seq_len]
        return tag, text

    def __len__(self):
        return len(self.datas)


if __name__ == '__main__':
    cnews = CNewsDataset('/tmp/cnews/train.csv')
