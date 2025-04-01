import os,sys
os.chdir(sys.path[0])
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import faiss
from PIL import Image
import numpy as np
import random
import torch.backends.cudnn as cudnn
import shutil
import logging
import logging.handlers

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, f_compressed, f_original):
        # 计算余弦相似度
        cos_sim = F.cosine_similarity(f_compressed, f_original, dim=-1)
        # 使余弦相似度尽可能接近 1
        loss = 1 - cos_sim.mean()
        return loss
    
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, f_x, f_y):
        f_x = F.normalize(f_x, p=2, dim=-1)  # 归一化
        f_y = F.normalize(f_y, p=2, dim=-1)  # 归一化
        
        logits = torch.matmul(f_x, f_y.T) / self.temperature  # 计算相似度
        labels = torch.arange(f_x.shape[0]).to(f_x.device)  # 正确匹配的索引
        loss = F.cross_entropy(logits, labels)  # 对比损失
        return loss

class KL_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, f_x, f_y):
        # 计算 batch 维度上的均值和标准差
        mu_x, std_x = torch.mean(f_x, dim=0), torch.std(f_x, dim=0) + 1e-6
        mu_y, std_y = torch.mean(f_y, dim=0), torch.std(f_y, dim=0) + 1e-6
        
        # KL 散度计算
        kl_loss = torch.log(std_y / std_x) + (std_x**2 + (mu_x - mu_y)**2) / (2 * std_y**2) - 0.5
        return kl_loss.mean()

class Integrate_Loss(nn.Module):
    def __init__(self, alpha=1, beta=0.3, gamma=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.cosine_loss = CosineSimilarityLoss()
        self.contrastive_loss = ContrastiveLoss()
        self.mse_loss = nn.MSELoss()
        self.kl_loss = KL_Loss()
    
    def forward(self, f_x, f_y):
        loss_cosine = self.cosine_loss(f_x, f_y)
        loss_contrastive = self.contrastive_loss(f_x, f_y)
        loss_mse = self.mse_loss(f_x, f_y)
        loss_kl = self.kl_loss(f_x, f_y)
        loss = self.alpha * loss_mse
        return loss

def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True

def delete_empty_checkpoints(root_dir):
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if dir_name == 'checkpoints':
                # 检查 checkpoints 文件夹是否为空
                if not os.listdir(dir_path):
                    # 获取包含 checkpoints 文件夹的父文件夹路径
                    parent_dir = os.path.dirname(dir_path)
                    print(f"Deleting folder: {parent_dir}")
                    shutil.rmtree(parent_dir)


def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    info_handler.setFormatter(formatter)
    logger.addHandler(info_handler)
    return logger


def log_config_info(config, logger):
    log_info = f'#----------Config info----------#'
    logger.info(log_info)
    # 遍历所有属性
    for attr in dir(config):
        # 跳过特殊属性和方法
        if attr.startswith('__') or callable(getattr(config, attr)):
            continue
        value = getattr(config, attr)
        log_info = f'{attr}: {value}'
        logger.info(log_info)


from thop import profile		 ## 导入thop模块
def cal_params_flops(model, size, logger):
    input = torch.randn(1, size).cuda()
    flops, params = profile(model, inputs=(input,1.0))
    print('flops',flops/1e9)			## 打印计算量
    print('params',params/1e6)			## 打印参数量

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total/1e6))
    logger.info(f'flops: {flops/1e9}, params: {params/1e6}, Total params: : {total/1e6:.4f}')


def find_similar_image_path(query_img_path, top_k=1):
    """给定查询图片，找出 receive/ 目录中最相似的图片"""
    # 加载 FAISS 索引 & 预存的图片路径
    index = faiss.read_index("../data/receive_index.faiss")
    image_paths = np.load("../data/receive_image_paths.npy", allow_pickle=True)

    # 处理查询图片
    query_image = preprocess(Image.open(query_img_path)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        query_feature = model.encode_image(query_image).cpu().numpy().astype("float32")

    # 进行最近邻搜索
    D, I = index.search(query_feature, top_k)  # 计算最相似的 top_k 个结果

    # 返回匹配的图片路径
    return [image_paths[i] for i in I[0]]

def compare_images(img1, img2):
    """比较两张图片的余弦相似度"""
    img1 = preprocess(Image.open(img1)).unsqueeze(0).to(device)
    img2 = preprocess(Image.open(img2)).unsqueeze(0).to(device)

    with torch.no_grad():
        img1_feature = model.encode_image(img1)
        img1_feature /= img1_feature.norm(dim=-1, keepdim=True)
        img2_feature = model.encode_image(img2)
        img2_feature /= img2_feature.norm(dim=-1, keepdim=True)

    # 计算余弦相似度
    cos_sim = F.cosine_similarity(torch.from_numpy(img1_feature.cpu().numpy()),
                                   torch.from_numpy(img2_feature.cpu().numpy()), dim=-1)
    return cos_sim.item()

if __name__ == "__main__":
    # 加载 CLIP 模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("../ViT-B-32.pt", device=device)
    img1 = "../data/send/0000_bus_13.png"
    img2 = "../data/receive/0008_train_90.png"
    cos = compare_images(img1=img1, img2=img2)

    print(cos)