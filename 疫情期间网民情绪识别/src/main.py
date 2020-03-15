# coding=utf-8
# author=yphacker

import gc
import os
import time
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from conf import config
from conf import model_config_bert as model_config
from model.bert import Model
from utils.bert_data_utils import MyDataset
from utils.model_utils import FocalLoss

torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate(model, val_loader, criterion):
    model.eval()
    data_len = 0
    total_loss = 0
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_len = len(batch_y)
            # batch_len = len(batch_y.size(0))
            data_len += batch_len
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            # probs = torch.softmax(logits)
            loss = criterion(logits, batch_y)
            total_loss += loss.item()
            _, preds = torch.max(logits, 1)
            y_true_list += batch_y.cpu().numpy().tolist()
            y_pred_list += preds.cpu().numpy().tolist()

    return total_loss / data_len, f1_score(y_true_list, y_pred_list, average="macro")


def train(train_data, val_data, fold_idx=None):
    train_dataset = MyDataset(train_data, 'train')
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataset = MyDataset(val_data, 'val')
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False)

    model = Model().to(device)
    criterion = nn.CrossEntropyLoss()
    # 11 5 3
    # weight = torch.tensor([1, 2.5, 4], dtype=torch.float)
    # criterion = nn.CrossEntropyLoss(weight.to(device))
    # criterion = FocalLoss(0.5)
    # criterion = F1_Loss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config.learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

    if fold_idx is None:
        print('start')
        model_save_path = os.path.join(config.model_path, '{}.bin'.format(model_name))
    else:
        print('start fold: {}'.format(fold_idx + 1))
        model_save_path = os.path.join(config.model_path, '{}_fold{}.bin'.format(model_name, fold_idx))

    best_val_score = 0
    last_improved_epoch = 0
    adjust_lr_num = 0
    y_true_list = []
    y_pred_list = []
    for cur_epoch in range(config.epochs_num):
        start_time = int(time.time())
        model.train()
        print('epoch:{}, step:{}'.format(cur_epoch + 1, len(train_loader)))
        cur_step = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            # probs = torch.softmax(logits)
            train_loss = criterion(logits, batch_y)
            train_loss.backward()
            optimizer.step()

            cur_step += 1
            _, train_preds = torch.max(logits, 1)
            y_true_list += batch_y.cpu().numpy().tolist()
            y_pred_list += train_preds.cpu().numpy().tolist()
            if cur_step % config.train_print_step == 0:
                train_score = f1_score(y_true_list, y_pred_list, average="macro")
                msg = 'the current step: {0}/{1}, train loss: {2:>5.2}, train score: {3:>6.2%}'
                print(msg.format(cur_step, len(train_loader), train_loss.item(), train_score))
                y_true_list = []
                y_pred_list = []
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


def predict():
    model = Model().to(device)
    model.load_state_dict(torch.load(config.model_save_path))
    test_df = pd.read_csv(config.test_path)
    texts = test_df['微博中文内容'].tolist()
    test_df.dropna(axis=0, inplace=True)
    test_dataset = MyDataset(test_df, 'test')
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    model.eval()
    tmp = []
    preds_list = []
    with torch.no_grad():
        for batch_x, _ in tqdm(test_loader):
            batch_x = batch_x.to(device)
            probs = model(batch_x)
            # pred = torch.argmax(output, dim=1)
            _, preds = torch.max(probs, 1)
            tmp += [p.item() for p in preds]
    cnt = 0
    for text in texts:
        if pd.isnull(text):
            # 直接赋值为0
            preds_list.append(0)
        else:
            preds_list.append(config.id2label[tmp[cnt]])
            cnt += 1
    submission = pd.read_csv(config.sample_submission_path)
    submission['y'] = preds_list
    submission.to_csv('../data/submission.csv', index=False)


def main(op):
    if op == 'train':
        train_df = pd.read_csv('../data/train.csv')
        train_df = train_df[train_df['情感倾向'].isin(['0', '1', '-1'])]
        train_df.dropna(subset=['微博中文内容'], inplace=True)
        print(train_df['情感倾向'].value_counts())
        # train_df = train_df[:1000]
        if args.mode == 1:
            x = train_df['微博中文内容'].values
            y = train_df['情感倾向'].values
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
    args = parser.parse_args()
    config.batch_size = args.batch_size
    config.epochs_num = args.epochs_num
    model_name = args.model_name

    model_score = dict()
    main(args.operation)
