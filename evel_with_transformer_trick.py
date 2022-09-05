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
INPUT_WINDOW = 32*48
if torch.cuda.is_available():
    print('有显卡')
    device = torch.device("cuda")
    torch.cuda.empty_cache()
else:
    print('无显卡')
    device = torch.device("cpu")
model_path = 'checkpoint/Transformer_220730_183925_best.pth'
h = 1535
w = 6
model = TransAm(s_len=h, feature_size=w, num_layers=1, dropout=0.3, nhead=w, num_class=2,
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
drop_indexs = []
for index,code in enumerate(all_stock_df['code']):
    print(code)
    result = pd.read_csv(f'data/20220730/5/{code}.csv', encoding='gbk')
    if result.shape[0] == 0:
        drop_indexs.append(index)
    if result.shape[0] < INPUT_WINDOW:
        continue
    try:
        columns_all = result.columns
        columns_need = columns_all[3:]
        values = result[columns_need].values
        end_data = values[-INPUT_WINDOW:-1]
        values = end_data.astype(np.float32)
        values = scaler.transform(values)
        now_data = torch.from_numpy(values).to(device).unsqueeze(0)

        outputs = model(now_data)
        outputs_one = int(outputs.argmax(axis=1))
        if outputs_one == 1:
            print(code, '涨起来')
            f.write(code + '\n')
    except:
        continue
bs.logout()
f.close()
# all_stock_df.drop(index = drop_indexs,inplace = True)
# all_stock_df.to_csv('data/all_stock_20220721_delete.csv',index=False)
#### 登出系统 ####

