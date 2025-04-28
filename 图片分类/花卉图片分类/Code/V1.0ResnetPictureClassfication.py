import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from PIL import Image
import matplotlib.pyplot as plt

# 配置参数
dataPath = 'C:/Users/gst-0123/Desktop/Projects/中医诊断AI/图片分类/花卉图片分类/Result'
cache_dir = "C:/Users/gst-0123/Desktop/Projects/中医诊断AI/图片分类/花卉图片分类/Cache"
os.makedirs(cache_dir, exist_ok=True)
best_model_path = os.path.join(cache_dir, 'best_model.pkl')
num_epochs = 10
batch_size = 32
img_size = (128, 128)
lr = 0.001

def get_data_loaders(data_path, img_size, batch_size):
    # 定义图像预处理方式
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    # 加载训练集和验证集
    train_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'train'), transform=transform)
    valid_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'valid'), transform=transform)
    # 构建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, train_dataset

def build_model(num_classes, device):
    # 加载预训练的ResNet18模型
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    # 替换最后一层为适应当前类别数
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model.to(device)

def train_one_epoch(model, loader, criterion, optimizer, device):
    # 单个epoch的训练过程
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    loader_tqdm = tqdm(loader)
    for images, labels in loader_tqdm:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        train_acc = correct / total if total > 0 else 0
        train_loss = running_loss / total if total > 0 else 0
        loader_tqdm.set_postfix({'Loss': f'{train_loss:.4f}', 'Acc': f'{train_acc:.4f}'})
    return train_loss, train_acc

def validate(model, loader, criterion, device):
    # 验证过程
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    loader_tqdm = tqdm(loader)
    with torch.no_grad():
        for images, labels in loader_tqdm:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_acc = correct / total if total > 0 else 0
            avg_test_loss = test_loss / total if total > 0 else 0
            loader_tqdm.set_postfix({'Loss': f'{avg_test_loss:.4f}', 'Acc': f'{test_acc:.4f}'})
    return avg_test_loss, test_acc

def train_model(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs, best_model_path):
    # 训练主循环，包含保存最优模型
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        val_loss, val_acc = validate(model, valid_loader, criterion, device)
        print(f"Valid Loss: {val_loss:.4f}, Valid Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with acc: {best_acc:.4f} at {best_model_path}")

def predict_image(model, image_path, class_names, device, img_size):
    # 对单张图片进行预测
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        pred_class = class_names[predicted.item()]
    return pred_class

def load_model(model_path, num_classes, device):
    # 加载保存的模型权重
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def main():
    # 主函数，初始化设备、数据、模型和训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader, train_dataset = get_data_loaders(dataPath, img_size, batch_size)
    class_names = train_dataset.classes

    # 检查是否存在已保存模型
    if os.path.exists(best_model_path):
        print(f"检测到已保存模型，正在加载: {best_model_path}")
        model = load_model(best_model_path, len(class_names), device)
    else:
        print("未检测到已保存模型，开始训练新模型。")
        model = build_model(len(class_names), device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        train_model(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs, best_model_path)
        print("训练完成，模型已保存。")

    # 用户图片预测，允许多次输入
    while True:
        user_img_path = input("请输入要预测的图片路径（输入q退出）：").strip()
        if user_img_path.lower() == 'q':
            print("已退出图片预测。")
            break
        if os.path.exists(user_img_path):
            try:
                img = Image.open(user_img_path)
                # 用matplotlib展示图片
                plt.imshow(img)
                plt.axis('off')
                plt.title("待预测图片")
                plt.show()
            except Exception as e:
                print(f"图片无法打开: {e}")
                continue
            pred_class = predict_image(model, user_img_path, class_names, device, img_size)
            print(f"图片 {user_img_path} 的预测类别为: {pred_class}")
        else:
            print("图片路径不存在，请检查后重试。")

if __name__ == "__main__":
    main()
