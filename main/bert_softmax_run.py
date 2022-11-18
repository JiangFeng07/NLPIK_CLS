#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/11/18 17:12
# @Author: lionel
import argparse
import os

import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from torch.utils import data

from datasets.cnews import CNewsDataset
from models.bert_softmax import BertSoftmax

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collect_cn(batch):
    labels, texts = zip(*batch)
    encoded_outputs = tokenizer(texts, return_tensors='pt', padding=True)
    token_ids, token_type_ids, attention_mask = encoded_outputs['input_ids'], encoded_outputs['token_type_ids'], \
                                                encoded_outputs['attention_mask']
    label_ids = torch.LongTensor([label2id[label] for label in labels]).to(device)

    return token_ids, token_type_ids, attention_mask, label_ids


def metric(valid_loader, model):
    correct_num, predict_num, gold_num = 0, 0, 0
    with tqdm(total=len(valid_loader), desc='模型验证进度条') as pbar:
        for index, batch in enumerate(valid_loader):
            token_ids, token_type_ids, attention_mask, gold_labels = batch
            pred_labels = model(token_ids, token_type_ids, attention_mask)
            pred_labels = torch.argmax(pred_labels, dim=1)
            predict_num += len(pred_labels)
            gold_num += len(gold_labels)
            correct_num += int(torch.sum(pred_labels == gold_labels))
            pbar.update()

    print("correct_num: {:3d}, predict_num: {:3d}, gold_num: {:3d}".format(correct_num, predict_num, gold_num))
    precision = correct_num / (predict_num + 1e-10)
    recall = correct_num / (gold_num + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall + 1e-10)
    print('f1: {:4.4f}, precision: {:4.4f}, recall: {:4.4f}'.format(f1_score, precision, recall))
    return precision, recall, f1_score


def train():
    train_dataset = CNewsDataset(os.path.join(args.file_path, 'train.csv'))
    dev_dataset = CNewsDataset(os.path.join(args.file_path, 'dev.csv'))
    train_loader = data.DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collect_cn)
    valid_loader = data.DataLoader(dev_dataset, batch_size=8, collate_fn=collect_cn)

    bert_model = BertModel.from_pretrained(args.bert_model_path)
    model = BertSoftmax(encoder=bert_model, num_classes=len(label2id)).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    best_f1_score = 0.0
    early_epochs = 0
    for epoch in range(args.epochs):
        model.train()
        with tqdm(total=len(train_loader), desc='模型训练进度条') as pbar:
            for batch_idx, batch in enumerate(train_loader):
                input_ids, token_type_ids, attention_mask, labels = batch
                optimizer.zero_grad()
                logits = model(input_ids, token_type_ids, attention_mask)
                loss = torch.nn.CrossEntropyLoss()(logits, labels)
                pbar.set_postfix({'loss': '{0:1.5f}'.format(float(loss))})
                pbar.update()
                loss.backward()
                optimizer.step()
                scheduler.step()
        model.eval()
        with torch.no_grad():
            precision, recall, f1_score = metric(valid_loader, model)
            if f1_score > best_f1_score:
                torch.save(model.state_dict(), args.model_path)
                best_f1_score = f1_score
                early_epochs = 0
            else:
                early_epochs += 1

            if early_epochs > 3:  # 连续三个epoch，验证集f1_score没有提升，训练结束
                print('验证集f1_score连续三个epoch没有提升，训练结束')
                break
        print('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', help='训练数据路径', type=str, default='/tmp/cnews')
    parser.add_argument('--bert_model_path', help='预训练模型路径', type=str, default='/tmp/chinese-roberta-wwm-ext')
    parser.add_argument('--epochs', help='训练轮数', type=int, default=1)
    parser.add_argument('--dropout', help='', type=float, default=0.5)
    parser.add_argument('--embedding_size', help='', type=int, default=100)
    parser.add_argument('--batch_size', help='', type=int, default=100)
    parser.add_argument('--hidden_size', help='', type=int, default=200)
    parser.add_argument('--num_layers', help='', type=int, default=1)
    parser.add_argument('--lr', help='学习率', type=float, default=1e-3)
    parser.add_argument('--mode', help='长文本处理方式', type=str, default='cut')
    parser.add_argument('--model_path', help='模型存储路径', type=str, default='/tmp/cnews_cls.pt')
    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
    label2id, id2label = {}, {}
    with open(os.path.join(args.file_path, 'label.csv'), 'r', encoding='utf-8') as f:
        for line in f:
            label2id[line.strip()] = len(label2id)
            id2label[len(id2label)] = line.strip()
    train()
