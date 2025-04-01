# -*- coding: utf-8 -*-   
# Author       : Chongyang Li
# Email        : lichongyang2016@163.com
# Date         : 2025-03-18 22:21:35
# LastEditors  : Chongyang Li
# LastEditTime : 2025-03-28 19:05:00
# FilePath     : /KBSC/com_without_net.py

import clip
import torch
from data.dataset import SendDataset
import faiss
from utils.channel import Channel
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.config import config
from utils.utils import set_seed

mse = nn.MSELoss()


# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("./ViT-B-32.pt", device=device)

rece_KB = faiss.read_index("./data/receive_index.faiss")
rece_img_paths = np.load("./data/receive_image_paths.npy", allow_pickle=True)
receive_labels = np.load("./data/receive_labels.npy")

send_dataset = SendDataset('./data/send')
send_loader = DataLoader(send_dataset, batch_size=256, shuffle=False)

config = config()
set_seed(config.seed)


channel = Channel(config)               # awgn or rayleigh
snr_list = [-7.,-6.,-5.,-4.,-2., 0.,2., 4.,5.,6., 7., 10.]

#  x--->y--->kb
#  x: transmit
#  y: receiver
#  kb: KB

def test_once(snr):
    correct = loss_x_y = cos_x_y = loss_y_kb = cos_y_kb = 0
    for iter, data in enumerate(send_loader):
        images, labels, filenames = data
        images = images.to(device)
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            if snr == 'no':
                f_noise = image_features
            else:
                f_noise = channel(image_features, snr)
        
        loss = mse(image_features, f_noise)
        loss_x_y += loss.item()*len(images)
        cos_x_y += F.cosine_similarity(image_features, f_noise).sum().item()

        D, I = rece_KB.search(f_noise.cpu().numpy().astype("float32"), 1)
        nearest_y = np.array([rece_KB.reconstruct(int(idx)) for idx in I[:, 0]])
        nearest_y = torch.from_numpy(nearest_y).to(device)

        # mse and cos compare to KB
        loss_y_kb += mse(f_noise, nearest_y).item()*len(images)
        cos_y_kb += F.cosine_similarity(f_noise, nearest_y).sum().item()  

        matched_labels = receive_labels[I[:, 0]]
        matched_paths = rece_img_paths[I[:, 0]]
        # 计算匹配准确率
        correct += (labels.numpy() == matched_labels).sum()

    # **最终准确率**
    mse_avg_x_y = loss_x_y / len(send_dataset)
    cos_avg_x_y = cos_x_y / len(send_dataset)
    mse_avg_y_kb = loss_y_kb / len(send_dataset)
    cos_avg_y_kb = cos_y_kb / len(send_dataset)
                            
    accuracy = correct / len(send_dataset) * 100
    if snr == 'no':
        print(f"SNR:{snr:<6} x<--->y MSE:{mse_avg_x_y:<10.4f} Cos:{cos_avg_x_y:<10.4f}" 
              f"x=y<->kb:MSE:{mse_avg_y_kb:<10.4f} Cos:{cos_avg_y_kb:<10.4f}"
              f"语义匹配准确率: {accuracy:.2f}% ")
    else:
        print(f"SNR:{snr:<6} x<--->y MSE:{mse_avg_x_y:<10.4f} Cos:{cos_avg_x_y:<10.4f}" 
              f"y<--->kb MSE:{mse_avg_y_kb:<10.4f} Cos:{cos_avg_y_kb:<10.4f}"
              f"语义匹配准确率: {accuracy:.2f}%")


if __name__ == '__main__':
    for snr in snr_list:
        test_once(snr)
    test_once('no')