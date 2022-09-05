# -*- coding: utf-8 -*-
# ----------------------------------------------------#
#   基于BiLstm-Attension训练
# ----------------------------------------------------#
import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy, Recall
from transformers import AdamW
from data_helper import get_x_y, MyDataset
from utils import save_checkpoint_state
from model import LSTM_Attention
# 设置value的显示长度为200，默认为50
pd.set_option('max_colwidth', 200)
# 显示所有列，把行显示设置成最大
pd.set_option('display.max_columns', None)
# 显示所有行，把列显示设置成最大
pd.set_option('display.max_rows', None)
if __name__ == '__main__':
    epochs = 20
    # 涨幅
    INCREASE = 0.04
    # 基于
    INPUT_WINDOW = 32
    # 预测
    PREDICT_WINDOW = 1
    batch_size = 200
    num_workers = 4
    shuffle = True
    pin_memory = True
    lr = 1e-3
    dice_weight = 1
    class_names = ["跌", "涨"]
    test_size = 0.3
    # 日志目录及写入器
    current_time = datetime.now().strftime('%y%m%d_%H%M%S')
    log_dir = os.path.join('runs', current_time)
    writer = SummaryWriter(log_dir=log_dir)
    # 获取运行硬件
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
    xs, ys = get_x_y('20220721[0]',INPUT_WINDOW,PREDICT_WINDOW,INCREASE)
    n, l, _ = xs.shape
    xs = xs.reshape(n * l, -1)
    scaler = preprocessing.StandardScaler()
    k = scaler.fit_transform(xs[:, :-1])
    j = xs[:, -1:]
    xs = np.concatenate((k, j), axis=1)
    # xs = xs
    xs = xs.reshape(n, l, -1)
    xs = xs[:, :, :-1]
    # scaler = StandardScaler().fit(X)
    # xs = scaler.fit_transform(xs)
    n_class = len(np.unique(ys))
    ar, num = np.unique(ys, return_counts=True)
    x_train, x_val, y_train, y_val = train_test_split(xs, ys, test_size=test_size, shuffle=True)
    #   制作DataLoader

    train_set = MyDataset(x_train, y_train)
    val_set = MyDataset(x_val, y_val)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=shuffle,
                              pin_memory=pin_memory
                              )
    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=shuffle,
                            pin_memory=pin_memory
                            )
    # 模型
    # model = TransAm(s_len=INPUT_WINDOW, feature_size=12, num_layers=1, dropout=0.3, nhead=12, num_class=2,
    #                 device=device).to(device)
    
    model = LSTM_Attention(10, 256, 1, 2).to(device)
    model = model.to(device)
    # 梯度优化器
    # 损失
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
    # lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    evaluate_dict = dict()
    mean_iou = 0
    val_epoch_min_loss = sys.maxsize
    criterion = nn.CrossEntropyLoss().to(device)
    for epoch in range(epochs):
        now_epoch = epoch + 1
        print(f'[epoch]{now_epoch}')
        model.train()
        loss_total = 0
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            # 正则
            loss_total += loss.item()
            # 反向传播，计算当前梯度
            # loss = loss.requires_grad_()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mean_loss = loss_total / len(train_loader)
        print('[训练损失]', mean_loss)
        print('-' * 50)
        writer.add_scalar('训练/损失', mean_loss, now_epoch)
        lr_scheduler.step()
        with torch.no_grad():
            test_acc_en = Accuracy(num_classes=2)  # Accuracy
            test_rcl_en = Recall(num_classes=2)  # Recall
            model.eval()
            val_epoch_loss = 0
            for i, (x, y) in enumerate(val_loader):
                x = x.to(device)
                y = y.to(device)
                outputs = model(x)
                outputs_one = outputs.argmax(axis=1)
                loss = criterion(outputs, y)
                loss_value = loss.item()
                val_epoch_loss += loss_value
                test_acc_en.update(outputs_one.cpu(), y.cpu())
                test_rcl_en.update(outputs_one.cpu(), y.cpu())

            mean_val_epoch_loss = val_epoch_loss / len(val_loader)

            print('[验证损失]', mean_val_epoch_loss)
            print('[准确率]', test_acc_en.compute())
            print('[召回率]', test_rcl_en.compute())
            print('-' * 50)
            writer.add_scalar('训练/损失', mean_val_epoch_loss, now_epoch)
            test_acc_en.reset()
            test_rcl_en.reset()
            # 保存模型
            if mean_val_epoch_loss < val_epoch_min_loss:
                save_checkpoint_state(f'checkpoint/{model.model_type}_{current_time}.pth', now_epoch, model, optimizer, lr_scheduler,
                                      scaler)
                val_epoch_min_loss = mean_val_epoch_loss
