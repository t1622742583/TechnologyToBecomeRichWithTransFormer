import datetime

import baostock as bs
import pandas as pd

#### 登陆系统 ####
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)
now_data = datetime.datetime.now().strftime('%Y-%m-%d')
code = 'sh.000001'
rs = bs.query_history_k_data_plus(code,"date",
                                      start_date='2022-01-21', end_date=now_data,
                                      frequency="d", adjustflag="3")
    # rs = bs.query_history_k_data_plus(code,
    #                                   "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg,pctChg,isST",
    #                                   start_date='2022-01-21',
    #                                   end_date=now_data,
    #                                   frequency="20", adjustflag="3")
print('query_history_k_data_plus respond error_code:' + rs.error_code)
print('query_history_k_data_plus respond  error_msg:' + rs.error_msg)

#### 打印结果集 ####
data_list = []
while (rs.error_code == '0') & rs.next():
    # 获取一条记录，将记录合并在一起

    data_list.append(rs.get_row_data()[0])
#### 登出系统 ####
bs.logout()