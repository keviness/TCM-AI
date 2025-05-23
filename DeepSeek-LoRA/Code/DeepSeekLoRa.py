# -*- coding: utf-8 -*-
"""deepseek_lora_finetune.py
使用LoRA对本地DeepSeek模型进行法律领域微调
"""

import torch  # PyTorch深度学习框架
from datasets import load_dataset  # 加载和处理数据集
from transformers import (
    AutoTokenizer,                # 自动加载分词器
    AutoModelForCausalLM,         # 自动加载因果语言模型
    BitsAndBytesConfig,           # 量化配置
    TrainingArguments,            # 训练参数配置
    Trainer                       # Huggingface训练器
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel  # LoRA相关工具

# =========================
# 1. 参数配置
# =========================
MODEL_PATH = "./models/deepseek-7b"          # 本地DeepSeek模型路径
DATA_PATH = "./data/legal_dataset.json"      # 训练数据路径
OUTPUT_DIR = "./lora_legal"                  # 微调后模型保存路径

# 4bit量化配置，节省显存
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                       # 启用4bit量化
    bnb_4bit_use_double_quant=True,          # 使用双量化
    bnb_4bit_quant_type="nf4",               # 量化类型
    bnb_4bit_compute_dtype=torch.bfloat16    # 计算精度
)

# LoRA参数配置
peft_config = LoraConfig(
    r=32,                                    # LoRA秩
    lora_alpha=64,                           # LoRA缩放因子
    target_modules=["q_proj", "v_proj", "k_proj"],  # 指定微调的模块
    lora_dropout=0.1,                        # LoRA dropout
    bias="none",                             # 不训练bias
    task_type="CAUSAL_LM",                   # 任务类型
    modules_to_save=["lm_head"]              # 保存输出头
)

# 训练参数配置
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,                   # 输出目录
    per_device_train_batch_size=2,           # 每卡batch size
    gradient_accumulation_steps=8,           # 梯度累计步数
    num_train_epochs=5,                      # 训练轮数
    learning_rate=2e-4,                      # 学习率
    logging_steps=50,                        # 日志打印步数
    fp16=True,                               # 混合精度训练
    optim="paged_adamw_32bit",               # 优化器
    save_strategy="epoch",                   # 保存策略
    report_to="none",                        # 不上报到外部平台
    ddp_find_unused_parameters=False,         # DDP参数
    gradient_checkpointing=True              # 显存优化
)

# =========================
# 2. 模型加载与准备
# =========================
def load_model_and_tokenizer():
    # 加载本地DeepSeek模型，自动适配设备，使用4bit量化
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token  # 设置pad_token，避免警告
    # 使模型适配k-bit训练
    model = prepare_model_for_kbit_training(model)
    # 应用LoRA适配器
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()  # 打印可训练参数信息
    return model, tokenizer

# =========================
# 3. 数据处理
# =========================
def format_legal_data(example):
    # 格式化法律问答数据为统一模板
    return {
        "text": f"法律咨询：{example['question']}\n背景：{example['context']}\n法律分析：{example['analysis']}\n结论：{example['answer']}<|endoftext|>"
    }

def tokenize_function(examples, tokenizer):
    # 对格式化文本进行分词和编码
    tokenized = tokenizer(
        examples["text"],                     # 输入文本
        truncation=True,                      # 截断超长文本
        max_length=1024,                      # 最大长度
        padding="max_length",                 # 填充到最大长度
        return_tensors="pt"                   # 返回PyTorch张量
    )
    return {
        "input_ids": tokenized["input_ids"],              # 输入ID
        "attention_mask": tokenized["attention_mask"],    # 注意力掩码
        "labels": tokenized["input_ids"].clone()          # 标签（自回归任务）
    }

def get_tokenized_dataset(tokenizer):
    import os
    if not os.path.exists(DATA_PATH):
        # 如果数据文件不存在，自动生成一条示例数据
        import json
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        demo_data = [
            {
                "question": "合同违约后如何主张损害赔偿？",
                "context": "甲乙双方签订买卖合同，甲方未按时交货。",
                "analysis": "根据《民法典》第五百七十七条，违约方应承担损害赔偿责任。",
                "answer": "可要求违约方赔偿实际损失及可得利益。"
            }
        ]
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            for item in demo_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    # 加载json数据集
    dataset = load_dataset("json", data_files=DATA_PATH)
    # 格式化数据
    formatted_dataset = dataset.map(format_legal_data)
    # 分词编码
    tokenized_dataset = formatted_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns = formatted_dataset["train"].column_names
    )
    return tokenized_dataset

# =========================
# 4. 训练
# =========================
def train(model, tokenizer, tokenized_dataset):
    # 创建Trainer对象，负责训练流程
    trainer = Trainer(
        model=model,                          # 训练模型
        args=training_args,                   # 训练参数
        train_dataset=tokenized_dataset["train"],  # 训练集
        data_collator=lambda data: {          # 数据整理函数
            "input_ids": torch.stack([d["input_ids"] for d in data]),
            "attention_mask": torch.stack([d["attention_mask"] for d in data]),
            "labels": torch.stack([d["input_ids"] for d in data])
        }
    )
    trainer.train()                           # 启动训练
    return model

# =========================
# 5. 保存与推理
# =========================
def save_model(model):
    # 保存LoRA微调后的模型
    model.save_pretrained(OUTPUT_DIR)

def generate_response(prompt, model, tokenizer):
    # 用微调后的模型生成法律问答响应
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=512,                   # 最多生成512个新token
        temperature=0.8,                      # 采样温度
        top_p=0.9,                            # nucleus采样
        do_sample=True                        # 启用采样
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)  # 解码输出

def merge_and_test(model, tokenizer):
    # 合并LoRA权重并进行推理测试
    merged_model = PeftModel.from_pretrained(model, OUTPUT_DIR).merge_and_unload()
    legal_query = "合同违约后如何主张损害赔偿？请结合民法典相关规定分析。"
    response = generate_response(legal_query, merged_model, tokenizer)
    print("法律咨询结果：\n", response)

# =========================
# 主流程
# =========================
def main():
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer()
    # 加载和处理数据集
    tokenized_dataset = get_tokenized_dataset(tokenizer)
    # 训练模型
    model = train(model, tokenizer, tokenized_dataset)
    # 保存模型
    save_model(model)
    # 合并权重并测试推理效果
    merge_and_test(model, tokenizer)

if __name__ == "__main__":
    main()