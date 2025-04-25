from jiayan import load_lm
from jiayan import CharHMMTokenizer

# 示例古文
text = '是故内圣外王之道，暗而不明，郁而不发，天下之人各为其所欲焉以自为方。'

# 加载语言模型
lm = load_lm('jiayan.klm')

# 初始化分词器
tokenizer = CharHMMTokenizer(lm)

# 对古文进行分词
tokens = list(tokenizer.tokenize(text))

# 输出分词结果
print("原文：", text)
print("分词结果：", '/'.join(tokens))