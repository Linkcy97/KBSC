import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader
from dataset import ReceiveDataset, SendDataset, get_feature_loader
import clip
import faiss
import numpy as np
from utils.config import config
from utils.utils import set_seed

config = config
set_seed(config.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("../ViT-B-32.pt", device=device)


receive_dataset = ReceiveDataset("./receive")
receive_loader = DataLoader(receive_dataset, batch_size=128, shuffle=False)
send_dataset = SendDataset("./send")
send_loader = DataLoader(send_dataset, batch_size=128, shuffle=False)

_, _, val_receive_loader, _ = get_feature_loader(config)


re_image_features = []
re_image_paths = []
re_image_labels = []
with torch.no_grad():
    for images, labels, img_names in receive_loader:
        images = images.to(device)
        features = model.encode_image(images)
        if config.l2_norm:
            features /= features.norm(dim=-1, keepdim=True)
        re_image_features.append(features.cpu().numpy())
        re_image_labels.extend(labels.cpu().numpy())
        re_image_paths.extend(img_names)


# 转换为 numpy 数组
re_image_features = np.vstack(re_image_features).astype("float32")
# 使用 FAISS 建立索引
index = faiss.IndexFlatL2(re_image_features.shape[1])
index.add(re_image_features)

faiss.write_index(index, "./receive_index.faiss")
np.save("./receive_image_paths.npy", np.array(re_image_paths))
np.save("./receive_labels.npy", np.array(re_image_labels))

print(f"Test FAISS index build complete, total {len(re_image_paths)} images")


re_image_features = []
re_image_paths = []
re_image_labels = []
with torch.no_grad():
    for images, labels in val_receive_loader:
        features = images
        re_image_features.append(features.cpu().numpy())
        re_image_labels.extend(labels.cpu().numpy())


# 转换为 numpy 数组
re_image_features = np.vstack(re_image_features).astype("float32")
# 使用 FAISS 建立索引
index = faiss.IndexFlatL2(re_image_features.shape[1])
index.add(re_image_features)

faiss.write_index(index, "./val_receive_index.faiss")
np.save("./val_receive_labels.npy", np.array(re_image_labels))

print(f"Val FAISS index build complete, total {len(re_image_labels)} images")


se_image_features = []
se_image_paths = []
se_image_labels = []

with torch.no_grad():
    for images, labels, img_names in send_loader:
        images = images.to(device)
        features = model.encode_image(images)
        if config.l2_norm:
            features /= features.norm(dim=-1, keepdim=True)  
        se_image_features.append(features)
        se_image_labels.append(labels)
        se_image_paths.append(img_names)

se_image_paths = [item for sublist in se_image_paths for item in sublist]

torch.save({"features": torch.cat(se_image_features, dim=0),
            "labels": torch.cat(se_image_labels, dim=0),
            "path": se_image_paths},
            "./test_feature.pth")

print(f"Training dataset compelte, total {len(se_image_paths)} images")

