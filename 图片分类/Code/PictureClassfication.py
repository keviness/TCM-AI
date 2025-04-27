import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm  # 新增
import os  # 新增

dataPath = 'C:/Users/gst-0123/Desktop/Projects/中医诊断AI/图片分类/Result'  # 数据集路径

# 1. 数据预处理和加载
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# 假设数据集结构为: C:/Users/gst-0123/Desktop/Projects/中医诊断AI/图片分类/Result/train/class_x/xxx.png
train_dataset = datasets.ImageFolder(root=dataPath + '/train', transform=transform)
test_dataset = datasets.ImageFolder(root=dataPath + '/valid', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 2. 定义模型（使用预训练的ResNet18）
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))  # 输出类别数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 3. 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 新增：保存最佳模型相关变量
best_acc = 0.0
cache_dir = "C:/Users/gst-0123/Desktop/Projects/中医诊断AI/图片分类/Cache"
os.makedirs(cache_dir, exist_ok=True)
best_model_path = os.path.join(cache_dir, 'best_model.pkl')

# 4. 训练模型
num_epochs = 2  # 只训练2轮做演示
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    train_loader_tqdm = tqdm(train_loader)
    for images, labels in train_loader_tqdm:
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
        train_loader_tqdm.set_postfix({'Loss': f'{train_loss:.4f}', 'Acc': f'{train_acc:.4f}'})
    print(f"Epoch {epoch+1} finished. Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    # 验证模型
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    test_loader_tqdm = tqdm(test_loader)
    with torch.no_grad():
        for images, labels in test_loader_tqdm:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_acc = correct / total if total > 0 else 0
            avg_test_loss = test_loss / total if total > 0 else 0
            test_loader_tqdm.set_postfix({'Loss': f'{avg_test_loss:.4f}', 'Acc': f'{test_acc:.4f}'})
    print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # 新增：保存最佳模型
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved with acc: {best_acc:.4f} at {best_model_path}")
