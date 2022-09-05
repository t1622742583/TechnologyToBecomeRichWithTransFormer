import datetime

import baostock as bs
import pandas as pd

#### 登陆系统 ####
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)
all_stock_df = pd.read_csv('data/all_stock_20220721_delete.csv')
# code = "sh.166019"
now_data = datetime.datetime.now().strftime('%Y-%m-%d')
drop_indexs = []
for index,code in enumerate(all_stock_df['code']):
    print(index,code)
    # code = 'sh.600076'
    rs = bs.query_history_k_data_plus(code,
                                      "date,time,code,open,high,low,close,volume,amount",
                                      start_date='2022-01-21', end_date=now_data,
                                      frequency="5", adjustflag="3")
    # rs = bs.query_history_k_data_plus(code,
    #                                   "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg,pctChg,isST",
    #                                   start_date='2022-01-21',
    #                                   end_date=now_data,
    #                                   frequency="20", adjustflag="3")
    if rs.error_code != '10004012':
        print('query_history_k_data_plus respond error_code:' + rs.error_code)
        print('query_history_k_data_plus respond  error_msg:' + rs.error_msg)

        #### 打印结果集 ####
        data_list = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())
        result = pd.DataFrame(data_list, columns=rs.fields)

        #### 结果集输出到csv文件 ####
        result.to_csv(f"data/20220730/5/{code}.csv", index=False)

        print(code, '成功！')
    else:
        drop_indexs.append(index)



all_stock_df.drop(index = drop_indexs,inplace = True)
all_stock_df.to_csv('data/all_stock_20220721_delete.csv',index=False)
#### 登出系统 ####
bs.logout()