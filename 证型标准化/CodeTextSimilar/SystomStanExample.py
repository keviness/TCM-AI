import pandas as pd
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import re

dataPath = 'C:/Users/gst-0123/Desktop/Projects/GSTAIProgect/证型标准化/Data/'
path = dataPath+'中医证型规范术语.xlsx'
filePath='C:/Users/gst-0123/Desktop/Projects/GSTAIProgect/证型标准化/Result/'
modelCachePath = filePath+'Model/'

# 数据准备模块
def tokenize(text):
    """自定义分词函数"""
    tokens = jieba.lcut(text)
    tokens = [t for t in tokens if t not in ['证', '型']]
    return ' '.join(tokens)

def load_data():
    """加载数据"""
    dataFrame = pd.read_excel(path)
    print()
    return dataFrame

# 模型训练模块
def train_model(df):
    """训练模型"""
    df['中医证型'] = df['中医证型'].apply(lambda x: tokenize(x))
    
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(df['中医证型'])
    y = df['证型术语']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    return clf, vectorizer

def save_model(clf, vectorizer):
    """保存模型"""
    with open(modelCachePath+'model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    
    with open(modelCachePath+'vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

# 模型加载与预测模块
def load_model():
    """加载本地模型"""
    with open(modelCachePath+'model.pkl', 'rb') as f:
        clf = pickle.load(f)
    
    with open(modelCachePath+'vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    return clf, vectorizer

def predict(text, clf, vectorizer):
    """对输入文本进行标准化预测"""
    if not text:
        return "输入不能为空"
    
    tokenized_text = tokenize(text)
    tfidf = vectorizer.transform([tokenized_text])
    prediction = clf.predict(tfidf)
    print("Prediction:", prediction)
    return prediction

# 主程序
def main():

    # 加载数据
    df = load_data()
    
    # 训练模型
    clf, vectorizer = train_model(df)
    
    # 保存模型
    save_model(clf, vectorizer)
    
    # 加载模型进行预测
    loaded_clf, loaded_vectorizer = load_model()
    
    # 示例输入
    input_text = "肺脾胃气虚寒"
    result = predict(input_text, loaded_clf, loaded_vectorizer)
    print(f"输入文本：{input_text}")
    print(f"标准化结果：{result}")

if __name__ == "__main__":
    main()
