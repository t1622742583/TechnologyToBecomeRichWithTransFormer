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
import pymysql
from pymysql import IntegrityError
mysql_conn = pymysql.connect(host= '127.0.0.1', port= 3306, user= 'root', password= '', db= 'ai_stock')

for index,code in enumerate(all_stock_df['code']):
    print(index,code)
    # code = 'sh.600076'
    rs = bs.query_history_k_data_plus(code,
                                      "date,time,code,open,high,low,close,volume,amount",
                                      start_date='2022-01-01', end_date=now_data,
                                      frequency="5", adjustflag="3")
    if rs.error_code != '10004012':
        print('query_history_k_data_plus respond error_code:' + rs.error_code)
        print('query_history_k_data_plus respond  error_msg:' + rs.error_msg)

        #### 打印结果集 ####
        data_list = []
        while (rs.error_code == '0') & rs.next():
            row = rs.get_row_data()
            row[1] = row[1][8:10]+':'+row[1][10:12]
            # row[1] = '09:56:49.000000'
            row = tuple(row)
            sql = f"INSERT INTO trick_model (date,time,code,open,high,low,close,volume,amount) VALUES {row}"
            try:
                with mysql_conn.cursor() as cursor:
                    cursor.execute(sql)
            except IntegrityError:
                print('重复,跳过..')
            mysql_conn.commit()
    else:
        drop_indexs.append(index)
#### 登出系统 ####
bs.logout()