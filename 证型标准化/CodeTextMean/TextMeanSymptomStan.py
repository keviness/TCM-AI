import pandas as pd
from sentence_transformers import SentenceTransformer, util

UNCASE = 'C:/Users/gst-0123/Desktop/Projects/GSTAIProgect/证型标准化/Model/all-MiniLM-L6-v2'

def load_target_texts_from_excel(file_path, sheet_name, column_name):
    """
    从Excel文件中加载目标文本列表。
    
    :param file_path: Excel文件路径 (str)
    :param sheet_name: Sheet名称 (str)
    :param column_name: 列名称 (str)
    :return: 目标文本列表 (list of str)
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df[column_name].dropna().tolist()

def find_most_similar_text(input_text, target_texts):
    """
    计算输入文本与多个目标文本的语义相似度，并返回相似度最高的目标文本及其相似度。
    
    :param input_text: 输入文本 (str)
    :param target_texts: 目标文本列表 (list of str)
    :return: 与输入文本语义相似度最高的目标文本及其相似度 (tuple of str, float)
    """
    # 加载预训练的句子嵌入模型
    model = SentenceTransformer(UNCASE)
    
    # 计算输入文本和目标文本的嵌入
    input_embedding = model.encode(input_text, convert_to_tensor=True)
    target_embeddings = model.encode(target_texts, convert_to_tensor=True)
    
    # 计算相似度分数
    similarity_scores = util.pytorch_cos_sim(input_embedding, target_embeddings)
    
    # 找到相似度最高的目标文本
    highest_score_index = similarity_scores.argmax().item()
    most_similar_text = target_texts[highest_score_index]
    highest_score = similarity_scores[0, highest_score_index].item()
    
    return most_similar_text, highest_score

# 示例用法
if __name__ == "__main__":
    input_text = "心血亏虚证"
    excel_file_path = "C:/Users/gst-0123/Desktop/Projects/GSTAIProgect/证型标准化/Data/中医证型规范术语.xlsx"
    sheet_name = "证候术语"
    column_name = "中医证候名"
    
    # 从Excel文件加载目标文本
    target_texts = load_target_texts_from_excel(excel_file_path, sheet_name, column_name)
    
    result_text, result_score = find_most_similar_text(input_text, target_texts)
    print(f"与输入文本语义相似度最高的目标文本是: {result_text}，相似度为: {result_score:.4f}")
