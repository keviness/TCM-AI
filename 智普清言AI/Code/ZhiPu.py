import json
from openai import OpenAI

# ========== 1. 读取API Key ==========
def load_api_key(config_path="config.json"):
    """从配置文件读取API Key"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            return config.get("api_key", "")
    except Exception as e:
        print(f"未能读取API Key: {e}")
        return ""

# ========== 2. 初始化OpenAI客户端 ==========
api_key = load_api_key()
client = OpenAI(
    base_url='https://api.siliconflow.cn/v1',
    api_key=api_key
)

# ========== 3. 发送流式对话请求 ==========
def chat_with_model(prompt):
    """发送用户输入并流式输出模型回复"""
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V2.5",
        messages=[
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    for chunk in response:
        if not chunk.choices:
            continue
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
        if hasattr(chunk.choices[0].delta, "reasoning_content") and chunk.choices[0].delta.reasoning_content:
            print(chunk.choices[0].delta.reasoning_content, end="", flush=True)
    print()  # 换行

# ========== 4. 主程序入口 ==========
if __name__ == "__main__":
    user_prompt = input("请输入您的问题：")
    chat_with_model(user_prompt)