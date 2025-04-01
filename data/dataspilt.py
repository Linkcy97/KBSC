import os, sys
os.chdir(sys.path[0])
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchvision.datasets import CIFAR100
from collections import defaultdict
from utils.config import config
from utils.utils import set_seed

config = config
set_seed(config.seed)

# 设置保存路径
send_dir = "./send"
recv_dir = "./receive"

# 创建文件夹
os.makedirs(send_dir, exist_ok=True)
os.makedirs(recv_dir, exist_ok=True)

# 1. 加载 CIFAR-100 数据集
dataset = CIFAR100(root="../../datasets/CIFAR100", train=False, download=False)

# 2. 按类别划分索引
class_to_indices = defaultdict(list)
for idx, (_, label) in enumerate(dataset):
    class_to_indices[label].append(idx)

# 3. 进行划分，并保存图片
for label, indices in class_to_indices.items():
    class_name = dataset.classes[label]  # 获取类别名称

    # 发送端: 保存前 一半
    for i, idx in enumerate(indices[:len(indices)//2]):
        img, _ = dataset[idx]
        filename = f"{i:04d}_{class_name}_{label:02d}.png"  # 生成文件名，如 0000_dog.png
        img.save(os.path.join(send_dir, filename))

    # 接收端: 保存后 一半
    for i, idx in enumerate(indices[len(indices)//2:]):
        img, _ = dataset[idx]
        filename = f"{i:04d}_{class_name}_{label:02d}.png"
        img.save(os.path.join(recv_dir, filename))

print(f"图片已保存至 {send_dir} 和 {recv_dir}")