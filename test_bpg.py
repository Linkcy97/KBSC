# -*- coding: utf-8 -*-   
# Author       : Chongyang Li
# Email        : lichongyang2016@163.com
# Date         : 2025-03-26 10:51:57
# LastEditors  : Chongyang Li
# LastEditTime : 2025-03-28 18:33:59
# FilePath     : /KBSC/test_bpg.py

import clip
import torch
from data.dataset import get_feature_loader, SendDataset
import faiss
from utils.utils import *
from torch.utils.data import DataLoader
from utils.config import config
from engine import *
import numpy as np
from model.net import FeatureTranser
import os

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("./ViT-B-32.pt", device=device)

rece_KB = faiss.read_index("./data/receive_index.faiss")
rece_img_paths = np.load("./data/receive_image_paths.npy", allow_pickle=True)
receive_labels = np.load("./data/receive_labels.npy")


def main(config):
    print('#----------GPU init----------#')
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    send_dataset = SendDataset(config.root_path + 'KBSC/data/bpg/awgn/0.5/10/100')
    test_loader = DataLoader(send_dataset, batch_size=config.v_batch, shuffle=False)

    criterion = config.criterion

    test_bpg(config, test_loader, criterion)


if __name__ == '__main__':
    config = config()
    main(config)