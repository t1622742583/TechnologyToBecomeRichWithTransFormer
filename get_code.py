import baostock as bs
import pandas as pd

#### 登陆系统 ####
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

#### 获取证券信息 ####
rs = bs.query_all_stock(day="2022-07-20")
print('query_all_stock respond error_code:'+rs.error_code)
print('query_all_stock respond  error_msg:'+rs.error_msg)

#### 打印结果集 ####
data_list = []
while (rs.error_code == '0') & rs.next():
    # 获取一条记录，将记录合并在一起
    nm = rs.get_row_data()[-1]
    # print(nm)
    if nm.endswith('指数'):
        print(True)
        continue
    if nm == '':
        continue
    data_list.append(rs.get_row_data())

result = pd.DataFrame(data_list, columns=rs.fields)

#### 结果集输出到csv文件 ####
result.to_csv("data/all_stock.csv", encoding="gbk", index=False)
# print(result)

#### 登出系统 ####
bs.logout()