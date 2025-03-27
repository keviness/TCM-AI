# 导入必要的库
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
import os
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import  OneHotEncoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'
UNCASED = 'C:/Users/gst-0123/Desktop/Projects/中医诊断AI/BERTModel/'
Model_cache_dir = 'C:/Users/gst-0123/Desktop/Projects/中医诊断AI/ModelCahe/'
#bert_config = BertConfig.from_pretrained(UNCASED)

# 定义数据集类
class TextDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        """
        初始化数据集。
        
        Args:
            input_ids (tensor): BERT编码后的输入ids。
            attention_mask (tensor): BERT的注意力掩码。
            labels (tensor): 文本的多标签。
        """
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        """
        返回数据集的长度。
        """
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        获取一个样本。
        
        Args:
            idx (int): 样本的索引。
            
        Returns:
            dict: 包含输入ids、注意力掩码和标签的字典。
        """
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

# 定义分类模型
class BERTClassifier(torch.nn.Module):
    def __init__(self, num_labels):
        """
        初始化BERT分类模型。
        
        Args:
            num_labels (int): 标签的数量。
        """
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(UNCASED)
        self.dropout = torch.nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        """
        前向传播。
        
        Args:
            input_ids (tensor): 输入ids。
            attention_mask (tensor): 注意力掩码。
            
        Returns:
            tensor: 分类结果。
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        outputs = self.classifier(pooled_output)
        return outputs

# 定义训练和预测类
class BERTTrainer:
    def __init__(self, model, device, num_epochs=5, learning_rate=2e-5, batch_size=10):
        """
        初始化训练器。
        
        Args:
            model: BERT分类模型。
            device (str): 设备类型，'cuda'或'cpu'。
            num_epochs (int): 训练的epoch数。
            learning_rate (float): 学习率。
            batch_size (int): 批次大小。
        """
        self.model = model
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.criterion = BCEWithLogitsLoss()

    def train(self, dataloader):
        """
        训练模型。
        
        Args:
            dataloader: 数据加载器。
        """
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            train_bar = tqdm(dataloader)
            for batch in train_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                train_bar.set_description('Epoch %i train' % epoch)
                self.optimizer.zero_grad()
                
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                #predict = outputs[1]
                #pred_labels = torch.argmax(predict, dim=1)   # 预测出的label
                #acc = torch.sum(pred_labels==labels.flatten().to(device)).item()/len(pred_labels)
                #acc = torch.sum(labels==outputs.flatten().to(device)).item()/len(labels)
                train_bar.set_postfix(loss=loss.item())
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}")

    def evaluate(self, dataloader):
        """
        评估模型。
        
        Args:
            dataloader: 测试数据加载器。
            
        Returns:
            tuple: 测试损失和准确率。
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                predicted = torch.sigmoid(outputs) > 0.5
                correct += (predicted == labels.bool()).all(dim=1).sum().item()
        
        accuracy = correct / len(dataloader.dataset)
        avg_loss = total_loss / len(dataloader)
        return avg_loss, accuracy

    def save_model(self, save_path=Model_cache_dir+'BestModel.pkl'):
        """
        保存模型。
        
        Args:
            save_path (str): 模型保存路径。
        """
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        torch.save(self.model.state_dict(), save_path)

# 定义预测器类
class BERTPredictor:
    def __init__(self, model, tokenizer):
        """
        初始化预测器。
        
        Args:
            model: 训练好的BERT分类模型。
            tokenizer: BERT tokenizer。
        """
        self.model = model
        self.tokenizer = tokenizer

    def predict(self, text):
        """
        使用训练好的模型进行预测。
        
        Args:
            text (str): 待分类的文本。
            
        Returns:
            list: 预测的标签。
        """
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=100
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        with torch.no_grad():
            #outputs = self.model(outputs)
            predictions = torch.sigmoid(outputs) > 0.5
            return predictions

def ReadExcelData(ExcelPath, tokenizer):
    data = pd.read_excel(ExcelPath, sheet_name='BianZheng').iloc[:1000,:]  # 修改为取前1000行数据
    #print("data:",data)
    texts = data['detection'].values.tolist()
    labels = data['syndrome'].values

    #print("texts:", type(texts))

    # 将标签转换为多标签列表
    # 假设每个标签之间用逗号分隔
    encoder = OneHotEncoder() # One-Hot编码
    multi_labels = encoder.fit_transform(labels.reshape((-1,1))).toarray()
    #print("multi_labels:", multi_labels)
    encoder.fit(labels.reshape((-1,1)))
    featureList = encoder.categories_[0]
    print("encoder.feature_names_in_:",featureList)
    
    multi_labels = torch.tensor(multi_labels, dtype=torch.float)
    num_labels = len(multi_labels[0])
    #print("texts:", texts)
    #print("multi_labels:", multi_labels)
    #print("num_labels:", num_labels)

    # 使用BERT tokenizer进行编码
    encoded_data = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=200
    )
    
    input_ids = encoded_data['input_ids']
    attention_mask = encoded_data['attention_mask']
    #print("input_ids:", input_ids)
    #print("attention_mask:", attention_mask)
    # 创建数据集和数据加载器
    dataset = TextDataset(input_ids, attention_mask, multi_labels)
    #print("dataset:",dataset)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    return dataloader, num_labels, featureList

# 使用示例
if __name__ == "__main__":
    
    # 准备数据
    ExcelPath = 'C:/Users/gst-0123/Desktop/Projects/中医诊断AI/TCM-ERNIE/Data/trainBianZheng.xlsx'
    tokenizer = BertTokenizer.from_pretrained(UNCASED)
    dataloader, num_labels, featureList = ReadExcelData(ExcelPath, tokenizer)
    print("num_labels:",num_labels)
    #print("dataloader:",dataloader)
    
    # 初始化模型、训练器和预测器
    model = BERTClassifier(num_labels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    trainer = BERTTrainer(model, device, num_epochs=5, batch_size=10)
    predictor = BERTPredictor(model, tokenizer)
    
    # 训练模型
    trainer.train(dataloader)
    
    # 评估模型
    
    test_loss, test_acc = trainer.evaluate(dataloader)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # 保存模型
    trainer.save_model()
    
    # 使用预测器进行预测
    #model = BertForSequenceClassification.from_pretrained(UNCASED, num_labels=num_labels) 
    model.load_state_dict(torch.load(Model_cache_dir+'BestModel.pkl',map_location=torch.device('cpu')))
    model.to(device)
    text = ["喘憋、胸闷、呼吸困难，严重时不能平卧入眠",
            "休息情况下也可出现喘憋、胸闷、呼吸困难",
            "阵发性胸闷胸痛，伴气短乏力，心慌汗出，",
            "左肩部及后背部疼痛，伴头晕头痛，心烦易怒，失眠健忘，双下肢沉重感，夜间时有干咳，口干不苦，小便正常，大便溏薄。"
            ]
    predicted_labels = predictor.predict(text)
    print(f"Predicted labels: {predicted_labels}")
    featureArray = np.array([featureList])
    featureArray = featureArray.repeat(predicted_labels.shape[0], axis=0)
    print("featureList:",featureArray)
    ResultArray = featureArray[predicted_labels]
    print("Predicted labels: ",ResultArray)

    #ResultList = [featureList[i] for i in range(len(predicted_labels)) if predicted_labels[i] == 1]
    #print("Predicted labels: ",[featureList[i] for i in range(len(predicted_labels)) if predicted_labels[i] == 1])
