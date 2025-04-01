# Knowledge Base based Semantic Image Transmission using CLIP

Pytorch implementation for 



## Introduction

This repository contains the implementation of a **knowledge base (KB) assisted semantic communication framework** for image transmission. The proposed method leverages **CLIP-extracted semantic features** and transmits compressed representations over a wireless channel. At the receiver, the reconstructed features are used to retrieve the most semantically similar image from a **FAISS-based knowledge base**, ensuring robust and efficient transmission.



## Python environment

```shell
clip==1.0
faiss_gpu==1.7.2
matplotlib==3.10.1
numpy==2.2.4
Pillow==11.1.0
thop==0.1.1.post2209072238
torch==2.1.1+cu118
torchvision==0.16.1+cu118
tqdm==4.66.4
```



## Usage

set *root_path* to your path

```shell
## run clip_cifar.py to generate training validation CLIP feature
python clip_cifar100.py
## run data/dataspilt.py and divides test dataset equally into sender and receiver
python data/dataspilt.py
## run build_index.py and build the FAISS knowledge base of receiver
python data/build_index.py
## setup the configuration in utils/config.py and then run train.py to train your model
python train.py
```

