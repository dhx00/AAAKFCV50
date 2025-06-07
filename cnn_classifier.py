import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, random_split, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据划分
transform = transforms.ToTensor()
train_raw = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_raw = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
full_dataset = ConcatDataset([train_raw, test_raw])

n_total = len(full_dataset)
n_test = int(0.2 * n_total)
n_trainval = n_total - n_test
n_train = int(0.8 * n_trainval)
n_val = n_trainval - n_train
trainval_data, test_data = random_split(full_dataset, [n_trainval, n_test])
train_data, val_data = random_split(trainval_data, [n_train, n_val])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64)

# CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练
for epoch in range(5):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")

# 验证
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for X, y in val_loader:
        X, y = X.to(device), y.to(device)
        preds = model(X).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

print(f"Validation Accuracy: {correct / total:.4f}")
