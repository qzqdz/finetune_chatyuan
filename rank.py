import json

import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from sklearn.neighbors import NearestNeighbors
from infer import *
import pandas as pd

import os

'''
以下是一些常见距离度量方法的计算公式：

欧几里得距离（Euclidean distance）:

$d_{euclidean}(A, B) = \sqrt{\sum_{i=1}^n (A_i - B_i)^2}$

曼哈顿距离（Manhattan distance）:

$d_{manhattan}(A, B) = \sum_{i=1}^n |A_i - B_i|$

余弦相似度（Cosine similarity）:

$sim_{cosine}(A, B) = \frac{\sum_{i=1}^n A_i B_i}{\sqrt{\sum_{i=1}^n A_i^2} \sqrt{\sum_{i=1}^n B_i^2}}$

余弦距离（Cosine distance）:

$d_{cosine}(A, B) = 1 - sim_{cosine}(A, B)$

Jaccard相似度（Jaccard similarity）:

$sim_{jaccard}(A, B) = \frac{|A \cap B|}{|A \cup B|}$

Jaccard距离（Jaccard distance）:

$d_{jaccard}(A, B) = 1 - sim_{jaccard}(A, B)$

Pearson相关系数（Pearson correlation coefficient）:

$r_{pearson}(A, B) = \frac{\sum_{i=1}^n (A_i - \bar{A})(B_i - \bar{B})}{\sqrt{\sum_{i=1}^n (A_i - \bar{A})^2} \sqrt{\sum_{i=1}^n (B_i - \bar{B})^2}}$

其中，$A$和$B$是两个向量，$n$是向量的维数，$A_i$和$B_i$是向量$A$和$B$中的第$i$个元素，$\bar{A}$和$\bar{B}$分别是向量$A$和$B$的均值。

请注意，这些计算公式涉及向量和集合操作。在实际应用中，您可能需要根据数据类型和结构选择合适的公式。


更改    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(train_embeddings)中的metric，是否能达到以上不同量度的选择？

是的，通过更改NearestNeighbors中的metric参数，您可以使用不同的距离度量方法。scikit-learn库提供了许多内置的距离度量方法，例如：

"euclidean"：欧几里得距离（L2距离）
"manhattan"：曼哈顿距离（L1距离）
"cosine"：余弦距离（注意这是距离，而不是相似度，距离等于1减去余弦相似度）

'''


# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化句子编码器
model_name = r"E:\model\white_model\bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoder = AutoModel.from_pretrained(model_name).to(device)  # 将模型移动到 GPU

def embed(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)  # 将输入数据移动到 GPU
    with torch.no_grad():
        outputs = encoder(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu()  # 将结果移回 CPU

# 用于计算文本样本的嵌入
def compute_embeddings(text_samples):
    embeddings = []
    for sample in text_samples:
        embeddings.append(embed(sample).detach().numpy())  # 添加 detach() 方法
    return embeddings

# 用于选择k个最相似的样本
def select_knn_examples(x_test, train_data, k):
    train_prompts, train_targets = zip(*train_data)
    train_embeddings = compute_embeddings(train_prompts)
    test_embedding = embed(x_test).unsqueeze(0).numpy()  # 将结果转换为 NumPy 数组

    # 使用kNN查找最近邻
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(train_embeddings)
    distances, indices = nbrs.kneighbors(test_embedding)
    # 返回选定的示例
    selected_examples = [train_data[i] for i in indices.flatten()]
    return selected_examples


# 读取数据，文件格式为xlsx,表头为text和label，
def read_data1(file_path,label_dict):
    data = pd.read_excel(file_path)
    data = data.values.tolist()
    # 处理成[(text1, label1), (text2, label2)]的格式
    data = [(x[0], x[1]) for x in data]  # 将列表解拆
    # 由于x[1]是0，1标签，该函数还需要读取列表，实现0，1到标签的映射
    data = [(x[0], label_dict[x[1]]) for x in data]
    return data

# 读取数据，文件格式为txt,有两个文件：text.txt和label.txt,这些文件每行都是一个样本
def read_data2(file_path, label_dict, file_name=None):
    if file_name is None:
        file_name = ["text.txt", "label.txt"]
    # 基于os,实现路径的拼接，读取text和label
    path_a = os.path.join(file_path, file_name[0])
    print(path_a)
    path_b = os.path.join(file_path, file_name[1])
    if os.path.exists(path_a) == False:
        print("file a not exist")
        return
    if os.path.exists(path_b) == False:
        print("file b not exist")
        return
    # 读取text和label
    text = open(path_a, "r", encoding="utf-8").readlines()
    label = open(path_b, "r", encoding="utf-8").readlines()
    data = [(text[i].strip(), label_dict[int(label[i].strip())]) for i in range(len(text))]
    return data


# val_test = "your_test_prompt_here"
# train_data = [("prompt_1", "target_1"), ("prompt_2", "target_2"), ("prompt_3", "target_3"), ("prompt_4", "target_4"), ("prompt_5", "target_5")]  # D_T


label_dict = {0: 'non-hate', 1: 'hate'}
# 根据read_data,取得train_data
train_data = read_data2(r"E:\data\tweet\data\hate",label_dict,file_name=["val_text.txt", "val_labels.txt"])
# train_data划分为train_data和val_data,3:7
train_data, val_data = train_test_split(train_data, test_size=0.3, random_state=42)


# K近邻数量
k = min(3, len(train_data)-1)
save_file = 'knn_prompt.json'
print(f"Using {k} nearest neighbors.")
print("-------------------------")
# 循环，对于val_data中的每个样本，都使用K远邻算法，选择最相近的样本
for i, (x_test, y_test) in enumerate(val_data):
    knn_examples = select_knn_examples(x_test, train_data, k)
    # 将这些例子连接起来
    # input_text = "".join([f"```{x}```;{y}. " for x, y in knn_examples]) + f'So: ```{x_test}```?'
    input_text = [f"```{x}```;{y}. " for x, y in knn_examples]
    input_text.append(f'So: ```{x_test}```?')
    # 将以上结果，组织成字典格式：{"id": 1, "paragraph": [{"q": "从南京到上海的路线", "a": ["你好，南京到上海的路线如下"]}]}
    # 请注意，这里的paragraph是一个列表，因为一个样本可能有多个最相近的样本,这里的id是一个整数，从1开始，每次加1,这里的q是一个字符串，是一个问题,里的a是一个列表，是一个答案
    # 打印以上格式的字典
    print({"id": i+1, "paragraph": [{"q": input_text, "a": [y_test]}]})
    print("-------------------------")
    # 以“a”的方式打开一个text文件，写入以上输入
    with open(save_file, "a", encoding="utf-8") as f:
        # 将字典转化为json格式，写入文件
        f.write(json.dumps({"id": i+1, "paragraph": [{"q": " ".join(input_text), "a": [y_test]}]}) + "\n")
        # f.write(str({"id": i+1, "paragraph": [{"q": " ".join(input_text), "a": [y_test]}]}) + "\n")






# 在这里，您可以将`input_text`作为GPT-3模型的输入，获取预测结果
# 例如，您可以使用OpenAI API，或者使用您自己的GPT-3模型




# print('-------------------------')
# print(input_text)
# print('-------------------------')


# input_text = "帮我对比几款GPU，列出详细参数对比，并且给出最终结论"
# input_list = [input_text]
#
# for i, input_text in enumerate(input_list):
#   input_text = "用户：" + input_text + "\n小元："
#   print(f"示例{i}".center(50, "="))
#
#   output_text = answer(input_text)
#   print(f"{input_text}{output_text}")