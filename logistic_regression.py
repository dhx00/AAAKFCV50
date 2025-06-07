import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, random_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 数据转换与合并
transform = transforms.ToTensor()
train_raw = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_raw = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
full_dataset = ConcatDataset([train_raw, test_raw])

# 划分数据
n_total = len(full_dataset)
n_test = int(0.2 * n_total)
n_trainval = n_total - n_test
n_train = int(0.8 * n_trainval)
n_val = n_trainval - n_train
trainval_data, test_data = random_split(full_dataset, [n_trainval, n_test])
train_data, val_data = random_split(trainval_data, [n_train, n_val])

# 转为 NumPy
X_train = np.array([x[0].numpy().flatten() for x in train_data])
y_train = np.array([x[1] for x in train_data])
X_val = np.array([x[0].numpy().flatten() for x in val_data])
y_val = np.array([x[1] for x in val_data])

# Logistic 回归实验
for C in [0.01, 0.1, 1, 10]:
    model = LogisticRegression(C=C, solver='lbfgs', max_iter=2000)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"C = {C}, Validation Accuracy = {acc:.4f}")
