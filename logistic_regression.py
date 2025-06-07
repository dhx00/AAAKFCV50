import numpy as np
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 设置数据转换
transform = transforms.ToTensor()

# 加载训练和测试数据
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# 展平图像为一维向量
X_all = np.array([img.numpy().flatten() for img, _ in train_data])
y_all = np.array([label for _, label in train_data])

X_test = np.array([img.numpy().flatten() for img, _ in test_data])
y_test = np.array([label for _, label in test_data])

# 从训练集中划分出 80% training 和 20% validation
X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

# 用不同的 C 训练模型并评估准确率
for C in [0.01, 0.1, 1, 10, 50]:
    clf = LogisticRegression(C=C, solver='lbfgs', max_iter=1000, multi_class='multinomial')
    clf.fit(X_train, y_train)

    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)

    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"C = {C}, Validation Accuracy = {val_acc:.4f}, Test Accuracy = {test_acc:.4f}")
