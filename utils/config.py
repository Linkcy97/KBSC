# -*- coding: utf-8 -*-   
# Author       : Chongyang Li
# Email        : lichongyang2016@163.com
# Date         : 2025-03-19 20:18:08
# LastEditors  : Chongyang Li
# LastEditTime : 2025-03-31 16:32:44
# FilePath     : /KBSC/utils/config.py

from datetime import datetime
import torch.nn as nn
import torch
from utils.utils import *

class config:
    seed = 1997
    chan_type = 'awgn'               # 'rayleigh' or 'awgn'
    original_feature = 512
    compressed_feature = 256
    train_snr = [-7., -4., 0., 4., 7., 10.]
    test_snr = [-7.,-6., -5., -4., -2., 0., 2., 4., 5., 6., 7., 10.]
    t_batch = 128
    v_batch = 256
    lr = 1e-4
    l2_norm = True
    
    criterion = Integrate_Loss()
    epoch = 100
    print_interval = 100
    early_stop = 20
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4
    # work_dir = 'results/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '/'
    work_dir = 'results/' + 'Rayleigh_MSE_128/'
    root_path = "/home/linkcy97/Code/Sematic_Communication/"

