# -*- coding: utf-8 -*-   
# Author       : Chongyang Li
# Email        : lichongyang2016@163.com
# Date         : 2025-03-19 19:08:47
# LastEditors  : Chongyang Li
# LastEditTime : 2025-03-31 16:42:01
# FilePath     : /KBSC/test.py

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
    delete_empty_checkpoints('results')
    print('#----------Creating logger----------#')
    config.work_dir = 'results/' + 'AWGN_MSE_256' + '/'
    log_dir = os.path.join(config.work_dir, 'log')

    logger = get_logger('train', log_dir)

    print('#----------GPU init----------#')
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    _, _, _, test_loader = get_feature_loader(config)

    print('#----------Prepareing Model----------#')
    model = FeatureTranser(config).to(device)
    model.cuda()
    model.load_state_dict(torch.load(
        config.work_dir + 'checkpoints/best.pth'), strict=False)

    criterion = config.criterion
    test(config, test_loader, model, criterion, logger)


if __name__ == '__main__':
    config = config()
    main(config)