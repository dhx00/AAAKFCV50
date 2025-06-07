import numpy as np
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据
transform = transforms.ToTensor()
train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

# 数据展平
X = np.array([img.numpy().flatten() for img, _ in train_set])
y = np.array([label for _, label in train_set])

# 拆分训练集/验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 不同的正则化强度 C（C 越小，正则化越强）
for C in [0.01, 0.1, 1, 10]:
    clf = LogisticRegression(C=C, solver='lbfgs', max_iter=1000, multi_class='multinomial')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"C = {C}, Validation Accuracy = {acc:.4f}")
