import json
from openai import OpenAI
config_path = "D:/ProjectOpenSource/中医诊断AI/中医药膳deepseek-AI/Code/config.json"
# ========== 1. 读取API Key =========
def load_api_key(config_path=config_path):
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

# ========== 3. 构建系统提示词 ==========
system_prompt = (
    "你是一位专业的中医药膳专家。请主动询问用户的主诉、症状、现病史等相关信息，"
    "并根据用户输入，结合中医理论，提供个性化的中医药膳建议。/n"
    "输出内容需包含：/n"
    "1. 药膳名称/n"
    "2. 功能主治/n"
    "3. 原材料/n"
    "4. 制作方法/n"
    "5. 注意事项/n"
    "请用简明、专业的语言回答。"
)

# ========== 4. 交互式收集用户信息 ==========
def collect_user_info():
    """多轮交互收集用户主诉、症状、现病史"""
    user_info = []
    questions = [
        "请描述您的主诉（主要不适或需求）：",
        "请描述您的症状（如口干、咳嗽、失眠等）：",
        "请简要说明您的现病史或既往病史（如有慢性病、过敏等可补充）："
    ]
    for q in questions:
        ans = input(q)
        user_info.append(ans)
    return "/n".join([
        f"主诉：{user_info[0]}",
        f"症状：{user_info[1]}",
        f"现病史：{user_info[2]}"
    ])

# ========== 5. 调用模型并输出建议 ==========
def medicated_diet_consult():
    user_input = collect_user_info()
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V2.5",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
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

# ========== 6. 主程序入口 ==========
if __name__ == "__main__":
    medicated_diet_consult()