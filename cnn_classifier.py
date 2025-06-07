import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np

# 1. 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 数据预处理与加载
transform = transforms.ToTensor()
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# 3. 从 train_data 中划分训练集和验证集（80%/20%）
indices = np.arange(len(train_data))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

train_subset = Subset(train_data, train_idx)
val_subset = Subset(train_data, val_idx)

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 4. 定义 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

# 5. 初始化模型
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 6. 训练模型
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # 验证准确率
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            preds = torch.argmax(model(X), dim=1)
            val_correct += (preds == y).sum().item()
            val_total += y.size(0)
    val_acc = val_correct / val_total

    # ✅ 测试准确率（每个 epoch 输出）
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            preds = torch.argmax(model(X), dim=1)
            test_correct += (preds == y).sum().item()
            test_total += y.size(0)
    test_acc = test_correct / test_total

    print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
