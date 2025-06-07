from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from collections import Counter

# 设置图像转换（将PIL图像转为Tensor）
transform = transforms.ToTensor()

# 下载数据
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# 打印基本信息
print("训练集大小:", len(train_data))
print("测试集大小:", len(test_data))
print("单张图片形状:", train_data[0][0].shape)
print("第一个图像的标签:", train_data[0][1])

# 类别标签映射
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# 可视化前9张图像
figure = plt.figure(figsize=(8, 8))
for i in range(9):
    img, label = train_data[i]
    plt.subplot(3, 3, i + 1)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# 类别分布统计
label_counts = Counter([label for _, label in train_data])
print("\n每个类别的训练样本数量：")
for label, count in label_counts.items():
    print(f"{labels_map[label]}: {count}")
