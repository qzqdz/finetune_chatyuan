import pandas as pd

data = pd.read_csv(r'D:\Desktop\p3\立场数据集\sen_chinese_nlpcc_2016\evasampledata4-TaskAA.txt', delimiter='\t', header=0)
# data.columns = ['ID', 'TARGET', 'TEXT', 'STANCE']

print(data.head(10))
print(data.shape)
print(data.columns)# print(data.head(10))
print(data.shape)
print(data.columns)
# print(data.head(10))

data.to_excel('evasampledata4-TaskAA.xlsx', index=False)