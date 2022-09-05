# -*- coding: utf-8 -*-
# ----------------------------------------------------#
#   基于Transfomer训练
# ----------------------------------------------------#
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn import preprocessing
from sqlalchemy import create_engine
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from model import ModestCattle
from utils import save_checkpoint

# 设置value的显示长度为200，默认为50
pd.set_option('max_colwidth', 200)
# 显示所有列，把行显示设置成最大
pd.set_option('display.max_columns', None)
# 显示所有行，把列显示设置成最大
pd.set_option('display.max_rows', None)
import datetime
import baostock as bs


def get_all_date(start_date, end_date=datetime.datetime.now().strftime('%Y-%m-%d')):
    #### 登陆系统 ####
    lg = bs.login()
    code = 'sh.000001'
    rs = bs.query_history_k_data_plus(code, "date",
                                      start_date=start_date, end_date=end_date,
                                      frequency="d", adjustflag="3")

    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data()[0])

    #### 登出系统 ####
    bs.logout()
    return data_list


import pymysql

pymysql.install_as_MySQLdb()
# 创建连接
# conn = pymysql.connect(
#     host='127.0.0.1',
#     user='root',
#     password='',
#     database='ai_stock'
# )
conn = create_engine('mysql+mysqldb://root:@localhost:3306/ai_stock?charset=utf8')
all_stock_df = pd.read_csv('data/all_stock_20220721_delete.csv')


def get_detail(start_date, end_date):
    cursor = conn.cursor()
    # 遍历code
    # k_features = np.array([])
    # k_features = np.empty((1,64,13))
    k_features = []
    targets = []
    tricks = []
    # Pandas 查出所有k_model trick,通过code 分组
    for index, code in enumerate(all_stock_df['code']):
        sql = f'''SELECT open,high,low,close,preclose,volume,amount,turn,pctChg,up_007_target FROM k_model where code="{code}" and date <= "{end_date}" and date >= "{start_date}"'''
        k_df = pd.read_sql_query(sql, conn)
        if k_df.empty or len(k_df) < 64:
            continue
        try:
            target = k_df['up_007_target'].values[-1]
        except:
            continue
        k_df = k_df.drop('up_007_target', axis=1)
        trick_sql = f'''SELECT open,high,low,close,volume,amount FROM trick_model where  date = "{end_date}" and code="{code}" and time <="14:55"'''
        trick_df = pd.read_sql_query(trick_sql, conn)
        if trick_df.empty or len(trick_df) < 47:
            continue
        k_features.append(k_df.values.astype(np.float32))
        tricks.append(trick_df.values.astype(np.float32))
        targets.append(target)
    k_features, tricks, targets = np.array(k_features), np.array(tricks), np.array(targets)
    return k_features, tricks, targets


class StockDataset(Dataset):
    def __init__(self, k_features, tricks_features, targets):
        self.k_features = k_features
        self.tricks_features = tricks_features
        self.targets = targets.astype(np.int64)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.k_features[index], self.tricks_features[index], self.targets[index]


def str2date(date_str):
    return datetime.datetime.date(datetime.datetime.strptime(date_str, '%Y-%m-%d'))

def biaozhunhua(features):
    n, h, w = features.shape
    features = features.reshape(n * h, w)
    scaler = preprocessing.StandardScaler()
    features = scaler.fit_transform(features)
    features = features.reshape(n , h, w)
    return features,scaler
def beibiaozhunhua(features,scaler):
    n, h, w = features.shape
    features = features.reshape(n * h, w)
    features = scaler.transform(features)
    features = features.reshape(n , h, w)
    return features


if __name__ == '__main__':
    epochs = 20
    # 涨幅
    INCREASE = 0.07
    # 基于
    INPUT_WINDOW = 64
    # 预测
    PREDICT_WINDOW = 1
    batch_size = 50
    num_workers = 4
    shuffle = True
    pin_memory = True
    lr = 1e-3
    dice_weight = 1
    class_names = ["跌", "涨"]
    test_size = 0.3
    # 日志目录及写入器
    current_time = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    # log_dir = os.path.join('runs', current_time)
    # writer = SummaryWriter(log_dir=log_dir)
    # 获取运行硬件
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
    all_date = get_all_date('2022-01-04')
    end = len(all_date)
    train_size = int(end * 0.9)
    train_date = all_date[:train_size]
    start_time = time.time()
    #
    data01 = {
        0:
            {
                'k_features': [],
                'tricks_features': [],
            },
        1:
            {
                'k_features': [],
                'tricks_features': []
            }
    }
    k_features_val = []
    tricks_features_val = []
    targets_val = []
    end = len(all_date)
    for index, code in enumerate(all_stock_df['code']):
        k_sql = f'''SELECT date,open,high,low,close,preclose,volume,amount,turn,pctChg,up_007_target FROM k_model where code="{code}" and date <= "{all_date[-1]}" and date >= "{all_date[0]}"'''
        k_df = pd.read_sql_query(k_sql, conn)
        trick_sql = f'''SELECT date,open,high,low,close,volume,amount FROM trick_model where code="{code}" and time <= "14:55" and date <= "{all_date[-1]}" and date >= "{all_date[0]}"'''
        trick_df = pd.read_sql_query(trick_sql, conn)
        start = 0
        while start + INPUT_WINDOW < end:
            start_date = str2date(all_date[start])
            end_date = str2date(all_date[start + INPUT_WINDOW - 1])
            k_df_item = k_df[(k_df['date'] >= start_date) & (k_df['date'] <= end_date)]
            if len(k_df_item) == INPUT_WINDOW:
                target = k_df_item['up_007_target'].values[-1]
                trick_df_item = trick_df[trick_df['date'] == end_date]
                if len(trick_df_item) == 47:
                    if not np.isnan(target):
                        k_df_item = k_df_item.drop(['up_007_target', 'date'], axis=1)
                        trick_df_item = trick_df_item.drop(['date'], axis=1)
                        if str(end_date) in train_date:
                            data01[target]['k_features'].append(k_df_item.values.astype(np.float32))
                            data01[target]['tricks_features'].append(trick_df_item.values.astype(np.float32))
                        else:
                            k_features_val.append(k_df_item.values.astype(np.float32))
                            tricks_features_val.append(trick_df_item.values.astype(np.float32))
                            targets_val.append(target)

            start += 1
    end_time = time.time()
    print("time cost:", end_time - start_time, "s")
    len_1 = len(data01[1]['tricks_features'])
    k_features = np.array(data01[1]['k_features'] + data01[0]['k_features'][:len_1])
    tricks_features = np.array(data01[1]['tricks_features'] + data01[0]['tricks_features'][:len_1])
    targets = np.concatenate((np.ones(len_1),np.zeros(len_1)),axis=0)
    k_features,k_features_scaler = biaozhunhua(k_features)
    tricks_features,tricks_features_scaler = biaozhunhua(tricks_features)
    train_dataset = StockDataset(np.array(k_features), np.array(tricks_features), np.array(targets))
    # 验证集
    k_features_val = beibiaozhunhua(np.array(k_features_val), k_features_scaler)
    tricks_features_val = beibiaozhunhua(np.array(tricks_features_val), tricks_features_scaler)
    val_dataset = StockDataset(k_features_val, tricks_features_val, np.array(targets_val))
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=shuffle,
                              pin_memory=pin_memory
                              )
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=shuffle,
                            pin_memory=pin_memory
                            )
    # # 模型
    model = ModestCattle(num_layers=1, dropout=0.3, n_class=2, k_len=64, k_feature_size=9, trick_len=47,
                         trick_feature_size=6).to(device)
    # # 梯度优化器
    # # 损失
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params)
    scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
    # lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    val_epoch_min_loss = sys.maxsize
    criterion = nn.CrossEntropyLoss().to(device)
    for epoch in range(epochs):
        now_epoch = epoch + 1
        print(f'[epoch] {now_epoch}')
        model.train()
        loss_total = 0
        for i, (k_x, trick_x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            k_x = k_x.to(device)
            trick_x = trick_x.to(device)
            target = target.to(device)
            outputs = model(k_x, trick_x)
            loss = criterion(outputs, target)
            #         # 正则
            #         # loss = norm(model,loss)
            loss_total += loss.item()
            #         # 反向传播，计算当前梯度
            #         # loss = loss.requires_grad_()
            loss.backward()
            optimizer.step()
        mean_loss = loss_total / len(train_loader)
        print('-' * 50)
        print('[训练损失]', mean_loss)
        scheduler.step()
        with torch.no_grad():
            test_acc_en = Accuracy(num_classes=2)  # Accuracy
            model.eval()
            val_epoch_loss = 0
            for i, (k_x, trick_x, target) in enumerate(val_loader):
                k_x = k_x.to(device)
                trick_x = trick_x.to(device)
                target = target.to(device)
                outputs = model(k_x, trick_x)
                outputs_one = outputs.argmax(axis=1)
                loss = criterion(outputs, target)
                val_epoch_loss += loss.item()
                test_acc_en.update(outputs_one.cpu(), target.cpu())
                # test_rcl_en.update(outputs_one.cpu(), y.cpu())
            #
            mean_val_epoch_loss = val_epoch_loss / len(val_loader)
            #
            print('[验证损失]', mean_val_epoch_loss)
            zhunquelv = test_acc_en.compute()
            print('[准确率]', zhunquelv)
            if mean_val_epoch_loss < val_epoch_min_loss:
                save_checkpoint(f'checkpoint/{model.model_type}_{current_time}_{zhunquelv}.pth', model,k_features_scaler,tricks_features_scaler)
                val_epoch_min_loss = mean_val_epoch_loss
