#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/11/16 23:25
# @Author: lionel
from torch import nn


class BertSoftmax(nn.Module):
    def __init__(self, encoder, num_classes):
        super(BertSoftmax, self).__init__()
        self.encoder = encoder
        self.hidden_size = encoder.config.hidden_size
        self.num_classes = num_classes
        self.fc = nn.Linear()
