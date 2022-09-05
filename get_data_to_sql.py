import datetime

import baostock as bs
import pandas as pd
import pymysql
from pymysql import IntegrityError

mysql_conn = pymysql.connect(host= '127.0.0.1', port= 3306, user= 'root', password= '', db= 'ai_stock')

#### 登陆系统 ####
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)
all_stock_df = pd.read_csv('data/all_stock_20220721.csv', encoding='gbk')
# code = "sh.166019"
start_date = '2022-01-01'
now_data = datetime.datetime.now().strftime('%Y-%m-%d')
drop_indexs = []
for index,code in enumerate(all_stock_df['code']):
    # print(index,code)
    # code = 'sh.600076'
    rs = bs.query_history_k_data_plus(code,
                                      "code,date,open,high,low,close,preclose,volume,amount,turn,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST,adjustflag,tradestatus",
                                      start_date=start_date, end_date=now_data,
                                      frequency="d", adjustflag="3")
    print('query_history_k_data_plus respond error_code:' + rs.error_code)
    print('query_history_k_data_plus respond  error_msg:' + rs.error_msg)

    #### 打印结果集 ####
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    result['up_007_target'] = (result['high'].astype(float) - result['preclose'].astype(float))/result['preclose'].astype(float)
    result['up_007_target'] = result['up_007_target'].apply(lambda x: '1' if x>0.07 else '0')
    result['up_007_target'] = result['up_007_target'].shift(-1)
    # result['up_007_target'] = result['up_007_target'].fillna('None')
    for row in result.values:
        row = tuple(row)
        sql = f"INSERT INTO k_model (code,date,open,high,low,close,preclose,volume,amount,turn,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST,adjustflag,tradestatus,up_007_target) VALUES {row}"
        sql = sql.replace('nan','NULL')
        try:
            with mysql_conn.cursor() as cursor:
                cursor.execute(sql)
        except IntegrityError:
            print('重复,跳过..')
        mysql_conn.commit()


#### 登出系统 ####
bs.logout()
mysql_conn.close()