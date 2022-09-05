import time
import pymysql
from torch.nn import Parameter

pymysql.install_as_MySQLdb()
import pandas as pd
import torch
# from transformers import BertTokenizer, BertForSequenceClassification
# from transformers import pipeline
#
# finbert = BertForSequenceClassification.from_pretrained('ProsusAI/finbert', num_labels=3)
# tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
# nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)
#
#
# start_time = time.time()
# sentences = ["Don\'t fall, please"]
# results = nlp(sentences)
# print(results)
# end_time = time.time()
# print("time cost:", end_time - start_time, "s")
#
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# sentence = 'Don\'t fall, please'
# tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
# model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert', num_labels=3)
# start_time = time.time()
# inputs = tokenizer(sentence, return_tensors="pt").to(device)
# model = model.to(device)
# outputs = model(**inputs)
# end_time = time.time()
# print("time cost:", end_time - start_time, "s")
from sqlalchemy import create_engine

conn = create_engine('mysql+mysqldb://root:@localhost:3306/ai_stock?charset=utf8')
code = 'sh.600004'
start_date = '2022-01-04'
end_date = '2022-04-12'
sql = f'''SELECT text FROM public_opinion_model where code="{code}"'''
k_df = pd.read_sql_query(sql, conn)
# w_ih = Parameter(torch.empty((gate_size, layer_input_size), **factory_kwargs))