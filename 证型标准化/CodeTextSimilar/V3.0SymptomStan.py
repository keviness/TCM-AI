import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- 文本相似度计算模块 --------------------
def find_most_similar_text(base_text, target_texts):
    """
    计算 base_text 与 target_texts 中每个文本的相似度，并返回相似度最高的文本及其相似度。
    :param base_text: 基准文本 (str)
    :param target_texts: 目标文本列表 (list of str)
    :return: 相似度最高的文本 (str) 和相似度值 (float)
    """
    # 将基准文本和目标文本合并为一个列表
    texts = [base_text] + target_texts

    # 使用 TfidfVectorizer 提取文本特征
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    # 计算基准文本与目标文本的余弦相似度
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # 找到相似度最高的文本
    max_index = similarity_scores.argmax()
    return target_texts[max_index], similarity_scores[max_index]

# -------------------- 数据加载模块 --------------------
def load_target_texts_from_excel(file_path, sheet_name, main_column, secondary_column):
    """
    从 Excel 文件的指定 sheet 中加载目标文本列表，并将两列字符串拼接为一列。

    :param file_path: Excel 文件路径 (str)
    :param sheet_name: sheet 名称 (str)
    :param main_column: 主列列名 (str)
    :param secondary_column: 次列列名 (str)
    :return: 拼接后的目标文本列表 (list of str) 和主列文本列表 (list of str)
    """
    # 读取 Excel 文件并提取指定列的数据
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df[secondary_column] = df[secondary_column].fillna("")  # 填充空值为 ""
    combined_texts = (df[main_column] + " " + df[secondary_column]).str.strip().tolist()
    main_texts = df[main_column].tolist()
    return combined_texts, main_texts

def load_base_texts_from_excel(file_path, sheet_name, column_name):
    """
    从 Excel 文件的指定 sheet 中加载基准文本列表。
    :param file_path: Excel 文件路径 (str)
    :param sheet_name: sheet 名称 (str)
    :param column_name: 列名 (str)
    :return: 基准文本列表 (list of str)
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df[column_name].dropna().tolist()

# -------------------- 文本分割模块 --------------------
def split_base_text(base_text, delimiters):
    """
    根据指定分隔符分割基准文本，返回分割后的文本列表，并确保每段文本包含“证”字符。
    :param base_text: 基准文本 (str)
    :param delimiters: 分隔符列表 (list of str)
    :return: 分割后的文本列表 (list of str)
    """
    for delimiter in delimiters:
        if delimiter in base_text:
            split_texts = base_text.split(delimiter)
            # 检查分割后的每段文本是否包含“证”，若无则补充
            return [text if "证" in text else text + "证" for text in split_texts]
    # 如果没有分隔符，直接检查并补充“证”
    return [base_text if "证" in base_text else base_text + "证"]

# -------------------- 相似度计算主逻辑 --------------------
def calculate_similarity(base_texts, combined_texts, main_texts):
    """
    对每个基准文本计算最相似的主列文本，并去重。
    :param base_texts: 基准文本列表 (list of str)
    :param combined_texts: 拼接后的目标文本列表 (list of str)
    :param main_texts: 主列文本列表 (list of str)
    :return: 相似度结果字典 (dict)
    """
    results = {}
    for text in base_texts:
        most_similar_combined_text, similarity = find_most_similar_text(text, combined_texts)
        most_similar_index = combined_texts.index(most_similar_combined_text)
        most_similar_main_text = main_texts[most_similar_index]
        # 去重：以主列文本为键，相似度为值
        if most_similar_main_text not in results or results[most_similar_main_text] < similarity:
            results[most_similar_main_text] = similarity
    return results

# -------------------- 主程序入口 --------------------
if __name__ == "__main__":
    # 定义 Excel 文件路径及相关参数
    base_text_file_path = "C:/Users/gst-0123/Desktop/Projects/GSTAIProgect/证型标准化/Data/GSTDataSource.xlsx"
    base_text_sheet_name = "Sheet1"
    base_text_column_name = "中医证型"
    
    target_text_file_path = "C:/Users/gst-0123/Desktop/Projects/GSTAIProgect/证型标准化/Data/中医证型规范术语.xlsx"
    target_text_sheet_name = "证候术语"
    main_column = "中医证候名"
    secondary_column = "中医证候名备选词"

    # 加载基准文本列表
    base_texts = load_base_texts_from_excel(base_text_file_path, base_text_sheet_name, base_text_column_name)
    #print("基准文本列表：", base_texts)
    # 加载目标文本列表
    combined_texts, main_texts = load_target_texts_from_excel(target_text_file_path, target_text_sheet_name, main_column, secondary_column)

    # 分割基准文本并计算相似度
    delimiters = ["|", "；", "，", "、", ";", ",", " "]
    resultsTextArray = []
    for base_text in base_texts:
        split_base_texts = split_base_text(base_text, delimiters)
        results = calculate_similarity(split_base_texts, combined_texts, main_texts)
        resultsText = "|".join([f"{text}" for text, similarity in results.items()])
        resultsTextArray.append(resultsText)

    # 将结果保存为 DataFrame
    result_df = pd.DataFrame({
        "基准文本": base_texts,
        "匹配结果": resultsTextArray
    })

    # 保存结果到 Excel 文件
    result_file_path = "C:/Users/gst-0123/Desktop/Projects/GSTAIProgect/证型标准化/Result/SimilarityResults.xlsx"
    result_df.to_excel(result_file_path)
    print(f"结果已保存至 {result_file_path}")
