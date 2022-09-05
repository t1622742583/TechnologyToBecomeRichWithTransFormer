# 加载模型
import datetime
import baostock as bs
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import TransAm
from utils import get_checkpoint_state
INPUT_WINDOW = 64
if torch.cuda.is_available():
    print('有显卡')
    device = torch.device("cuda")
    torch.cuda.empty_cache()
else:
    print('无显卡')
    device = torch.device("cpu")
model_path = 'checkpoint/Transformer_220729_224910_0.7949901223182678.pth'
model = TransAm(s_len=INPUT_WINDOW, feature_size=12, num_layers=1, dropout=0.3, nhead=12, num_class=2,
                    device=device).to(device)
model = model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(params)
scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
model, epoch, optimizer, scheduler, scaler = get_checkpoint_state(model_path, model, optimizer, scheduler)
model.eval()

#### 登陆系统 ####
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:' + lg.error_code)
print('login respond  error_msg:' + lg.error_msg)
all_stock_df = pd.read_csv('data/all_stock_20220721.csv', encoding='gbk')
# code = "sh.166019"
now_date = datetime.datetime.now().strftime('%Y-%m-%d')
zhang = []
f = open(f"up_stock/{now_date}.txt", "w")
for code in all_stock_df['code']:
    rs = bs.query_history_k_data_plus(code,
                                      "open,high,low,close,preclose,volume,amount,turn,pctChg,pctChg",
                                      start_date='2022-03-21',
                                      end_date=now_date,
                                      frequency="d", adjustflag="3")

    #### 打印结果集 ####
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    end_data = result.tail(INPUT_WINDOW)
    if end_data.shape[0] < INPUT_WINDOW:
        continue
    try:
        values = end_data.values.astype(np.float32)
        values = scaler.transform(values)
        now_data = torch.from_numpy(values).to(device).unsqueeze(0)
        print(code)
        outputs = model(now_data)
        outputs_one = int(outputs.argmax(axis=1))
        if outputs_one == 1:
            print(code, '涨起来')
            f.write(code + '\n')
    except:
        continue


#### 登出系统 ####
bs.logout()
f.close()
