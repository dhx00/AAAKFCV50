import numpy as np
from torchvision import datasets, transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据并转换为Tensor
transform = transforms.ToTensor()
train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# 展平图像为一维数组
X = np.array([img.numpy().flatten() for img, _ in train_set])
y = np.array([label for _, label in train_set])

# 划分训练集和验证集（80%训练，20%验证）
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 用不同的 k 训练并输出准确率
for k in [1, 3, 5, 7, 9]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"k = {k}, Validation Accuracy = {acc:.4f}")
