# -*- coding: utf-8 -*-
# ----------------------------------------------------#
#   数据加载
# ----------------------------------------------------#
import numpy as np
import pandas as pd
from datasets import Dataset


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]





def get_x_y(date, input_window, predict_window, increase):
    '''
    获取样本
    :param date: 产生日期
    :param input_window: 基于日期长度
    :param predict_window: 预测未来天数
    :param increase: 涨幅
    :return:
    '''
    xs_1 = []
    xs_0 = []
    ys_1 = []
    ys_0 = []
    all_stock_df = pd.read_csv('data/all_stock_20220721.csv', encoding='gbk')
    for code in all_stock_df['code']:
        # print(code)
        df = pd.read_csv(f"data/{date}/{code}.csv")
        if df.isnull().values.any():
            # print('有缺失:',code)
            continue
        columns_all = df.columns
        columns_need = columns_all[2:]

        df_len = df.shape[0]
        is_up = list()

        for i in range(df_len):
            # i+1 到 i+7 的最大值
            if i < df_len - predict_window:
                if max((df['high'][i + 1:i + 1 + predict_window]) - df['close'][i]) / df['close'][i] >= increase:
                    is_up.append(1)
                else:
                    is_up.append(0)
            else:
                is_up.append(None)
        df['is_up'] = is_up
        df = df.dropna(axis=0, subset=["is_up"])
        df['is_up'] = df['is_up'].astype('int')

        df_len = df.shape[0]

        for i in range(df_len - input_window):
            index = i + input_window - 1
            iu = df['is_up']
            if df['is_up'][index] == 1:
                xs_1.append(df[columns_need][i:i + input_window].values.tolist())
                ys_1.append(df['is_up'][index])
            else:
                xs_0.append(df[columns_need][i:i + input_window].values.tolist())
                ys_0.append(df['is_up'][index])
    xs = xs_1 + xs_0[:len(xs_1)]
    ys = ys_1 + ys_0[:len(ys_1)]

    xs = np.array(xs).astype(np.float32)
    ys = np.array(ys, dtype=int)
    return xs, ys

def get_trick(date, input_window, predict_window, increase):
    all_stock_df = pd.read_csv('data/all_stock_20220721_delete.csv')
    labels_1 = []
    labels_0 = []
    features_1 = []
    features_0 = []
    for code in all_stock_df['code']:
        df = pd.read_csv(f"data/{date}/{code}.csv")
        if df.shape[0] == 0:
            # print('有缺失:',code)
            continue
        columns_all = df.columns
        columns_need = columns_all[3:]
        values = df[columns_need].values
        row, col = df.shape
        stride = 1 * 48
        window = (input_window+1) * 48
        p_window = predict_window*48
        y = 0
        while y + window <= row:
            now_data = values[y:y + window, :]
            max_value = max(now_data[-p_window:, 1])
            buy_value = now_data[-(p_window+1), 3]
            return_rate = (max_value - buy_value) / buy_value
            if return_rate >= increase:
                labels_1.append(1)
                features_1.append(now_data[:-49])
            else:
                labels_0.append(0)
                features_0.append(now_data[:-49])

            y += stride
    features = features_1 + features_0[:len(features_1)]
    labels = labels_1 + labels_0[:len(labels_1)]
    return np.array(features),np.array(labels)
if __name__ == '__main__':
    pass