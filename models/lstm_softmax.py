#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/11/18 21:36
# @Author: lionel
import torch
from torch import nn
from transformers import BertTokenizer

from models.model import BilstmEncoder


class BilistmSoftmax(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, num_classes):
        super(BilistmSoftmax, self).__init__()
        self.encoder = BilstmEncoder(vocab_size, embedding_size, hidden_size, num_layers)
        self.fc = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, token_ids, seq_lens):
        encoder_outputs = self.encoder(token_ids, seq_lens)
        logits = self.fc(torch.mean(encoder_outputs, dim=1))

        return logits


if __name__ == '__main__':
    bert_model_path = '/tmp/chinese-roberta-wwm-ext'
    texts = ['原告：张三', '被告：李四伍']
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    encoded_outputs = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False)
    token_ids = encoded_outputs['input_ids']
    bs = BilistmSoftmax(tokenizer.vocab_size, embedding_size=200, hidden_size=100, num_layers=1, num_classes=10)
    a = bs(token_ids, torch.LongTensor([5, 6]))
    print(a)
