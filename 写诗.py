from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# 国内镜像，解决下载超时
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 加载写诗模型（GPT2 中文古诗）
tokenizer = AutoTokenizer.from_pretrained("uer/gpt2-chinese-poem")
model = AutoModelForCausalLM.from_pretrained("uer/gpt2-chinese-poem")

input_text = "春眠不觉晓"
inputs = tokenizer(input_text, return_tensors="pt")

# 生成诗句
outputs = model.generate(
    inputs["input_ids"],
    max_length=32,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id  # 修复警告
)

# 输出结果
print(tokenizer.decode(outputs[0], skip_special_tokens=True))