# coding=utf-8
# author=yphacker

import gc
import os
import time
import argparse
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from conf import config
from conf import model_config_bert as model_config
from model.bert import Model
from utils.bert_data_utils import MyDataset
from utils.model_utils import get_score

torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate(model, val_loader, criterion):
    model.eval()
    data_len = 0
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_len = len(batch_y)
            # batch_len = len(batch_y.size(0))
            data_len += batch_len
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            probs = model(batch_x)
            loss = criterion(probs, batch_y)
            total_loss += loss.item()

    return total_loss / data_len, get_score(total_loss / data_len)


def train(train_data, val_data, fold_idx=None):
    train_dataset = MyDataset(train_data, 'train', task)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataset = MyDataset(val_data, 'val', task)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False)

    model = Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config.learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

    if fold_idx is None:
        print('start')
        model_save_path = os.path.join(config.model_path, '{}_task{}.bin'.format(model_name, task))
    else:
        print('start fold: {}'.format(fold_idx + 1))
        model_save_path = os.path.join(config.model_path, '{}_task{}_fold{}.bin'.format(model_name, task, fold_idx))

    best_val_score = 0
    last_improved_epoch = 0
    adjust_lr_num = 0
    for cur_epoch in range(config.epochs_num):
        start_time = int(time.time())
        model.train()
        print('epoch:{}, step:{}'.format(cur_epoch + 1, len(train_loader)))
        cur_step = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            # probs = nn.Softmax(logits)
            train_loss = criterion(logits, batch_y)
            train_loss.backward()
            optimizer.step()

            cur_step += 1
            if cur_step % config.train_print_step == 0:
                msg = 'the current step: {0}/{1}, train loss: {2:>5.2}, train score: {3:>6.2%}'
                print(msg.format(cur_step, len(train_loader), train_loss.item(), get_score(train_loss.item())))
        val_loss, val_score = evaluate(model, val_loader, criterion)
        if val_score >= best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), model_save_path)
            improved_str = '*'
            last_improved_epoch = cur_epoch
        else:
            improved_str = ''
        msg = 'the current epoch: {0}/{1}, val loss: {2:>5.2}, val score: {3:>6.2%}, cost: {4}s {5}'
        end_time = int(time.time())
        print(msg.format(cur_epoch + 1, config.epochs_num, val_loss, val_score,
                         end_time - start_time, improved_str))
        if cur_epoch - last_improved_epoch >= model_config.patience_epoch:
            if adjust_lr_num >= model_config.adjust_lr_num:
                print("No optimization for a long time, auto stopping...")
                break
            print("No optimization for a long time, adjust lr...")
            scheduler.step()
            last_improved_epoch = cur_epoch  # 加上，不然会连续更新的
            adjust_lr_num += 1
    del model
    gc.collect()

    if fold_idx is not None:
        model_score[fold_idx] = best_val_score


def get_label(row):
    if row['ncw_label'] == 1:
        return 0
    if row['fake_label'] == 1:
        return 1
    if row['real_label'] == 1:
        return 2


def predict():
    comment_dict = {
        0: '0c',
        1: '2c',
        2: 'all'
    }
    fake_prob_label = defaultdict(list)
    real_prob_label = defaultdict(list)
    ncw_prob_label = defaultdict(list)
    test_df = pd.read_csv(config.test_path)
    test_df.fillna({'content': ''}, inplace=True)
    test_dataset = MyDataset(test_df, 'test')
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    for i in range(3):
        model = Model().to(device)
        # model.load_state_dict(torch.load(config.model_save_path))
        # os.path.join(config.model_path, '{}_task{}.bin'.format(model_name, task))
        os.path.join(config.model_path, '{}.bin'.format(model_name, task))
        model.eval()
        with torch.no_grad():
            for batch_x, _ in tqdm(test_loader):
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                # _, preds = torch.max(probs, 1)
                probs = torch.softmax(logits, 1)
                # probs_data = probs.cpu().data.numpy()
                fake_prob_label[i] += [p[0].item() for p in probs]
                real_prob_label[i] += [p[1].item() for p in probs]
                ncw_prob_label[i] += [p[2].item() for p in probs]
    submission = pd.read_csv(config.sample_submission_path)
    for i in range(3):
        submission['fake_prob_label_{}'.format(comment_dict[i])] = fake_prob_label[i]
        submission['real_prob_label_{}'.format(comment_dict[i])] = real_prob_label[i]
        submission['ncw_prob_label_{}'.format(comment_dict[i])] = ncw_prob_label[i]
    submission.to_csv('submission.csv', index=False)


def main(op):
    if op == 'train':
        train_df = pd.read_csv('../data/train.csv')
        train_df.dropna(subset=['content'], inplace=True)
        train_df['label'] = train_df.apply(lambda x: get_label(x), axis=1)
        print(train_df['label'].value_counts())
        # train_df = train_df[:1000]
        if args.mode == 1:
            x = train_df['content'].values
            y = train_df['label'].values
            skf = StratifiedKFold(n_splits=config.n_splits, random_state=0, shuffle=True)
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x, y)):
                train(train_df.iloc[train_idx], train_df.iloc[val_idx], fold_idx)
            score = 0
            score_list = []
            for fold_idx in range(config.n_splits):
                score += model_score[fold_idx]
                score_list.append('{:.4f}'.format(model_score[fold_idx]))
            print('val score:{}, avg val score:{:.4f}'.format(','.join(score_list), score / config.n_splits))
        else:
            train_data, val_data = train_test_split(train_df, shuffle=True, test_size=0.1)
            print('train:{}, val:{}'.format(train_data.shape[0], val_data.shape[0]))
            train(train_data, val_data)
    elif op == 'eval':
        pass
    elif op == 'predict':
        predict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--operation", default='train', type=str, help="operation")
    parser.add_argument("-b", "--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("-e", "--epochs_num", default=16, type=int, help="epochs num")
    parser.add_argument("-m", "--model_name", default='bert', type=str, help="model select")
    parser.add_argument("-mode", "--mode", default=1, type=int, help="train mode")
    parser.add_argument("-task", "--task", default='0', type=str, help="task")
    args = parser.parse_args()
    config.batch_size = args.batch_size
    config.epochs_num = args.epochs_num
    model_name = args.model_name
    task = args.task

    model_score = dict()
    main(args.operation)
