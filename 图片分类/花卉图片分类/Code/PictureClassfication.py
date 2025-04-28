import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

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

def main():
    # 主函数，初始化设备、数据、模型和训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader, train_dataset = get_data_loaders(dataPath, img_size, batch_size)
    model = build_model(len(train_dataset.classes), device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_model(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs, best_model_path)

if __name__ == "__main__":
    main()
