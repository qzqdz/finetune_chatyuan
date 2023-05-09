import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def calculate_perplexity(sentence, model, tokenizer):
    input_ids = tokenizer.encode(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    perplexity = torch.exp(loss).item()
    return perplexity

# 加载模型和分词器
model_name = r'E:\model\white_model\gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 示例句子
sentence = 'The quick brown fox jumps over the lazy dog.'

# 计算困惑度
perplexity = calculate_perplexity(sentence, model, tokenizer)
print(f'Perplexity: {perplexity}')
