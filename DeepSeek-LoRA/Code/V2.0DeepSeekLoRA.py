from transformers import AutoModelForCausalLM, AutoTokenizer  # 导入模型和分词器
from transformers import Trainer, TrainingArguments           # 导入训练器和训练参数
from peft import LoraConfig, get_peft_model                  # 导入LoRA配置和应用函数
import torch                                                 # 导入PyTorch

# ================== 模型与分词器加载 ==================
model_name = "deepseek-ai/deepseek-mistral-7b"               # 指定模型名称
tokenizer = AutoTokenizer.from_pretrained(model_name)        # 加载分词器
model = AutoModelForCausalLM.from_pretrained(model_name)     # 加载预训练模型

# ================== LoRA参数配置 ==================
lora_config = LoraConfig(
    r=8,                                # 低秩矩阵的秩
    lora_alpha=32,                      # LoRA缩放因子
    lora_dropout=0.1,                   # dropout概率
    bias="none",                        # 不训练bias
    target_modules=["q_proj", "v_proj"] # 仅对部分层进行微调
)

# ================== 应用LoRA适配器 ==================
model = get_peft_model(model, lora_config)                   # 应用LoRA到模型
model.print_trainable_parameters()                           # 打印可训练参数信息

# ================== 训练参数设置 ==================
training_args = TrainingArguments(
    output_dir="./lora_model",           # 输出模型保存路径
    per_device_train_batch_size=4,       # 每个设备的batch size
    num_train_epochs=3,                  # 训练轮数
    save_steps=100,                      # 每100步保存一次
    logging_dir="./logs",                # 日志保存路径
)

# ================== 构造示例数据集 ==================
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer):
        # 示例问答对，实际应用请替换为真实领域数据
        self.samples = [
            {"input": "中医问诊：咳嗽有痰怎么办？", "output": "建议清热化痰，可用川贝枇杷膏。"},
            {"input": "中医问诊：失眠多梦如何调理？", "output": "建议养心安神，适当饮用酸枣仁汤。"}
        ]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)  # 返回样本数量

    def __getitem__(self, idx):
        sample = self.samples[idx]  # 获取第idx个样本
        prompt = sample["input"] + "\n答："  # 构造输入提示
        target = sample["output"]           # 目标输出
        text = prompt + target              # 拼接输入和输出
        encoding = self.tokenizer(
            text,
            truncation=True,                # 截断超长文本
            max_length=256,                 # 最大长度
            padding="max_length",           # 填充到最大长度
            return_tensors="pt"             # 返回PyTorch张量
        )
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}  # 去除batch维
        encoding["labels"] = encoding["input_ids"].clone()         # 标签与输入一致
        return encoding

my_train_dataset = SimpleDataset(tokenizer)  # 实例化数据集

# ================== 创建Trainer并训练 ==================
trainer = Trainer(
    model=model,                           # 训练模型
    args=training_args,                    # 训练参数
    train_dataset=my_train_dataset,        # 训练数据集
)
trainer.train()                            # 启动训练
