from transformers import MarianMTModel, MarianTokenizer
import os

# 🔥 直接关闭代理，不走你的 7890，解决连接拒绝
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['TRANSFORMERS_OFFLINE'] = '0'

# 翻译模型（中 → 英）
model_name = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

chinese_text = "深度求索(DeepSeek),成立于2023年,专注于研究世界领先的通用人工智能底层模型与技术,挑战人工智能前沿性难题"
inputs = tokenizer(chinese_text, return_tensors="pt", truncation=True)
translated = model.generate(**inputs)
output = tokenizer.decode(translated[0], skip_special_tokens=True)

print("翻译结果:", output)