import json

import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from sklearn.neighbors import NearestNeighbors
from infer import *
import pandas as pd
import numpy as np
import os
import json
import ast
import torch
from transformers import AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration

class topic_knn_prompt_generator:
    def __init__(self, model_path, knowledge_lst, max_length=512, threshold_rate=None, device="cuda", k=5, knn_metric="euclidean",topic=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.encoder = AutoModel.from_pretrained(model_path).to(device)
        self.device = device
        self.max_length = max_length
        self.k = k
        self.threshold_rate = threshold_rate
        self.knowledge_lst = knowledge_lst
        self.topic = "\n".join(topic)
        train_embeddings = self.compute_embeddings(self.knowledge_lst)
        self.nbrs = NearestNeighbors(n_neighbors=self.k, metric=knn_metric).fit(train_embeddings)


    def __call__(self, X):
        test_embedding = self.embed(X).unsqueeze(0).numpy()
        distances, indices = self.nbrs.kneighbors(test_embedding)
        # print(distances, indices)
        # [[2.98318761 3.55258593 3.67806278]] [[6 3 5]]
        selected_examples = [self.knowledge_lst[i] for i in indices.flatten()]

        if self.threshold_rate is None:
            # return selected_examples
            return self.generate_prompt(X, selected_examples)
        else:
            # 获取第一个样本的距离，即最小距离
            min_distance = distances[0][0]
            # 筛选距离小于阈值的样本
            selected_examples_filtered = []
            for i in range(len(selected_examples)):
                if distances[0][i] <= self.threshold_rate*min_distance:
                    selected_examples_filtered.append(selected_examples[i])

            # return selected_examples_filtered
            return self.generate_prompt(X, selected_examples_filtered)

    def generate_prompt(self, text, knn_examples):
        knowledge = "\n".join(knn_examples)
        prompt = f'''1.话题：```\n{self.topic}\n```\n2.相关知识：```\n{knowledge}\n```\n3.该言论表明了什么态度？（支持/反对/均不是）：\n```\n{text}\n```\n'''
        return prompt



    def compute_embeddings(self, text_samples):
        embeddings = []
        for sample in text_samples:
            embeddings.append(self.embed(sample).detach().numpy())
        return embeddings

    def embed(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)
        with torch.no_grad():
            outputs = self.encoder(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().cpu()


    def topic_knn_prompt_generator(self,df):
        # 初始化知识列表
        knowledge_lst = []

        # 遍历每一行
        for _, row in df.iterrows():
            topic_ori = row['topic_ori']
            core_word = row['core_word']
            syn = row['syn']
            po = row['po']
            topic = row['topic']

            # 根据topic_ori，core_word生成
            if syn.get('status') == 'ok':
                for syn_word in syn.get('ret', []):
                    knowledge_lst.append((topic_ori, syn_word))

            # 根据topic_ori,po生成
            if po.get('status') == 'ok':
                for po_item in po.get('ret', []):
                    knowledge_lst.append((topic_ori, po_item[0], po_item[1]))

            # 根据topic_ori,topic生成
            for topic_description in topic.split('\n'):
                knowledge_lst.append((topic_ori, topic_description))

        return knowledge_lst


class topic_knn_prompt_generator_from_df(topic_knn_prompt_generator):
    def __init__(self, model_path, knowledge_df, max_length=512, threshold_rate=None, device="cuda", k=5, knn_metric="euclidean"):
        knowledge_lst, topic = self.generate_knowledge(knowledge_df)
        super().__init__(model_path, knowledge_lst, max_length, threshold_rate, device, k, knn_metric, topic)

    def generate_knowledge(self, df):
        knowledge_lst = []
        topic = []

        # 根据core_word，syn生成
        for i in range(len(df)):
            syn_dict = ast.literal_eval(df['syn'][i])
            if syn_dict['status'] == 'ok':
                for syn in syn_dict['ret']:
                    knowledge_lst.append((df['core_word'][i], syn))

        # 根据topic_ori,po生成
        for i in range(len(df)):
            po_dict = ast.literal_eval(df['po'][i])
            if po_dict['status'] == 'ok':
                for po in po_dict['ret']:
                    knowledge_lst.append((df['topic_ori'][i], po[0], po[1]))

        # 根据topic_ori,topic生成
        for i in range(len(df)):
            topic_lines = df['topic'][i].split('\n')
            for line in topic_lines:
                topic.append(line)
            topic.append(df['topic_ori'][i])
        # 打印结果
        print(knowledge_lst)
        print(list(set(topic)))

        return knowledge_lst, list(set(topic))


def answer(text, tokenizer, model, sample=True, top_p=1, temperature=0.7):
    '''sample：是否抽样。生成任务，可以设置为True;
    top_p：0-1之间，生成的内容越多样'''
    text = preprocess(text)
    encoding = tokenizer(text=[text], truncation=True, padding=True, max_length=768, return_tensors="pt").to(
        device)
    if not sample:
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=512,
                             num_beams=1, length_penalty=0.6)
    else:
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=512,
                             do_sample=True, top_p=top_p, temperature=temperature, no_repeat_ngram_size=3)
    out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
    return postprocess(out_text[0])

def generate_prompt(tup, model, tokenizer):
    if len(tup) == 2:
        prompt = f"'{tup[0]}' 是什么？它和 '{tup[1]}' 有什么关系？"
    elif len(tup) == 3:
        prompt = f"'{tup[0]}' 和 '{tup[2]}' 在 '{tup[1]}' 方面有什么关系？"
    else:
        raise ValueError("Tuple must be of length 2 or 3.")

    return answer(prompt, model=model, tokenizer=tokenizer)




def main():
    data_path = 'D:\\Desktop\\p3\\立场数据集\\sen_chinese_nlpcc_2016\\kg_Iphone_SE.xlsx'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder_model_path = r"E:\model\white_model\bert"
    LLM_path = r"E:/model/white_model/chatyuan"


    # Load pre-trained model and tokenizer
    yuan_tokenizer = T5Tokenizer.from_pretrained(LLM_path)
    yuan_model = T5ForConditionalGeneration.from_pretrained(LLM_path)
    yuan_model.to(device)

    # Load knowledge dataframe
    df = pd.read_excel(data_path)

    # Initialize topic_knn_prompt_generator_from_df
    k = 3
    KPG = topic_knn_prompt_generator_from_df(model_path=encoder_model_path, device=device, k=k, knowledge_df=df)
    knowledge_lst = KPG.knowledge_lst
    knowledge_lst_sen = []

    # Generate prompts for each tuple
    for tup in knowledge_lst:
        knowledge_lst_sen.append(generate_prompt(tup, yuan_model, yuan_tokenizer))
        print(knowledge_lst_sen[-1])


    # Save mapping to json file
    knowledge_dict = dict(zip(knowledge_lst, knowledge_lst_sen))
    with open('knowledge_dict.json', 'w', encoding='utf-8') as f:
        json.dump(knowledge_dict, f, ensure_ascii=False)

'''
话题：```
{topic}
```
相关知识：```
{knowledge}
```
该言论表明了什么态度？（支持/反对/均不是）：
```
{text}
```
'''

def main1():


    # knowledge_lst = ['我是一个好人', '我是一个坏人', '房学花wsw花', '房学gds花花', '房学花s2花', '房方式学花花', '房3fc学花花', '房�']
    # topic = ['房学花花1', '房学花花2', '房学花花3']
    # model_path = r"E:\model\white_model\bert"
    # k = 3
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # KPG = topic_knn_prompt_generator(model_path=model_path, device=device, k=k, knowledge_lst=knowledge_lst,topic=topic)

    # x_test = '房学花花'
    # knn_examples = KPG(x_test)
    # print(knn_examples)

    yuan_tokenizer = T5Tokenizer.from_pretrained("E:/model/white_model/chatyuan")
    yuan_model = T5ForConditionalGeneration.from_pretrained("E:/model/white_model/chatyuan")

    # 修改colab笔记本设置为gpu，推理更快
    device = "cuda" if torch.cuda.is_available() else "cpu" # 如果你没有GPU，可以使用'cpu'
    yuan_model.to(device)



    df = pd.read_excel('D:\\Desktop\\p3\\立场数据集\\sen_chinese_nlpcc_2016\\kg_Iphone_SE.xlsx')
    # 在这里，我假设你已经将模型文件保存在了'model_path'变量中
    model_path = r"E:\model\white_model\bert"
    k = 3  # 这是你要找的最近邻居的数量

    # 创建topic_knn_prompt_generator_from_df对象
    KPG = topic_knn_prompt_generator_from_df(model_path=model_path, device=device, k=k, knowledge_df=df)


    # 使用示例
    # knowledge_lst = [('智能手机', '智能手机（电子设备）'), ('智能手机', '智能手机'), ('智能手机', '智能手机（2021年网络电影）'), ('处理器', '中央处理器'), ('iPhone', 'iphone（苹果公司发布的电子产品系列）'), ('iPhone', 'iphone（美国苹果公司研发的智能手机系列）'),('iPhone SE', '世界名牌', '三星'), ('iPhone SE', '世界名牌', '华为')]


    knowledge_lst = KPG.knowledge_lst
    knowledge_lst_sen = []
    for tup in knowledge_lst:
        # 根据tup生成回答，回答存储于knowledge_lst_sen，并形成映射字典，字典会被保存在knowledge_dict.json文件中
        knowledge_lst_sen.append(generate_prompt(tup, yuan_model, yuan_tokenizer))
        print(knowledge_lst_sen[-1])

    # 回答存储于knowledge_lst_sen，并形成映射字典，字典会被保存在knowledge_dict.json文件中
    knowledge_dict = dict(zip(knowledge_lst, knowledge_lst_sen))
    with open('knowledge_dict.json', 'w', encoding='utf-8') as f:
        json.dump(knowledge_dict, f, ensure_ascii=False)



    # 测试一个句子
    # x_test = '房学花花'
    # prompt = KPG(x_test)


if __name__ == '__main__':
    main()