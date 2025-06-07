import numpy as np
from torchvision import datasets, transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 图像预处理
transform = transforms.ToTensor()

# 加载数据
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# 从 train_data 中提取图像和标签
X_all = np.array([img.numpy().flatten() for img, _ in train_data])
y_all = np.array([label for _, label in train_data])

# 从 test_data 中提取图像和标签（最终测试集）
X_test = np.array([img.numpy().flatten() for img, _ in test_data])
y_test = np.array([label for _, label in test_data])

# 划分 train_data 成训练集和验证集（80/20）
X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

# 用不同的 k 训练并评估
for k in [1, 3, 5, 7, 9]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # 验证集准确率
    y_val_pred = knn.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)

    # 测试集准确率
    y_test_pred = knn.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"k = {k} | Validation Accuracy = {val_acc:.4f} | Test Accuracy = {test_acc:.4f}")
