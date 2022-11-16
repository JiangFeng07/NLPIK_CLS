#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/11/16 23:25
# @Author: lionel
from torch import nn
from transformers import BertModel, BertTokenizer


class BertSoftmax(nn.Module):
    def __init__(self, encoder, num_classes):
        super(BertSoftmax, self).__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.fc = nn.Linear(encoder.config.hidden_size, num_classes)

    def forward(self, token_ids, token_type_ids, attention_mask):
        text_encoder = self.encoder(input_ids=token_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[
            0]
        cls_token_hidden_state = text_encoder[:, 0, :]
        logits = self.fc(cls_token_hidden_state)
        return logits


if __name__ == '__main__':
    bert_model_path = '/tmp/chinese-roberta-wwm-ext'
    bert_model = BertModel.from_pretrained(bert_model_path)
    bs = BertSoftmax(encoder=bert_model, num_classes=10)
    texts = ['原告：张三', '被告：李四伍']
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    encoded_outputs = tokenizer(texts, return_tensors='pt', padding=True)
    token_ids, token_type_ids, attention_mask = encoded_outputs['input_ids'], encoded_outputs['token_type_ids'], \
                                                encoded_outputs['attention_mask']
    a = bs(token_ids, token_type_ids, attention_mask)
