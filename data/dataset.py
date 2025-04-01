import os, sys
os.chdir(sys.path[0])
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader,random_split
from torchvision.datasets import CIFAR100
import torch

root_path = "/home/linkcy97/Code/Sematic_Communication/"
class SendDataset(Dataset):
    def __init__(self, send_dir):
        self.send_dir = send_dir
        self.image_files = os.listdir(send_dir)  # 获取所有图片文件
        self.transform = Compose([
                        Resize(224, interpolation=InterpolationMode.BICUBIC),
                        CenterCrop(224),
                        ToTensor(),
                        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.send_dir, img_name)
        image = Image.open(img_path)
        image = self.transform(image)
        label = int(img_name.split("_")[-1].split(".")[0])
        
        return image, label, img_name

class ReceiveDataset(Dataset):
    """ 接收端数据集，加载 receive/ 目录下的所有图片 """
    def __init__(self, send_dir):
        self.send_dir = send_dir
        self.image_files = os.listdir(send_dir)  # 获取所有图片文件
        self.transform = Compose([
                        Resize(224, interpolation=InterpolationMode.BICUBIC),
                        CenterCrop(224),
                        ToTensor(),
                        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.send_dir, img_name)
        image = Image.open(img_path)
        image = self.transform(image)
        label = int(img_name.split("_")[-1].split(".")[0])
        
        return image, label, img_name

class TrainDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path, map_location='cpu')
        self.features = data['features'].float()
        self.labels = data['labels'].float()
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class ValDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path, map_location='cpu')
        self.features = data['features'].float()
        self.labels = data['labels'].float()
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
class TestDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path, map_location='cpu')
        self.features = data['features'].float()
        self.labels = data['labels'].float()
        self.path = data['path']
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.path[idx]
    

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def get_loader(config):
    transform = Compose([
        Resize(224, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(224),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    cifar100 = CIFAR100(root=
            os.path.expanduser(root_path + "datasets/CIFAR100"),
            train=True, transform=transform)

    # Split CIFAR100 dataset into training and validation sets
    train_size = int(0.8 * len(cifar100))
    val_send = int(0.1 * len(cifar100))
    val_receive = int(0.1 * len(cifar100))
    train_dataset, val_send_dataset, val_receive_dataset = \
                    random_split(cifar100, [train_size, val_send, val_receive])

    # DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, 
                              batch_size=config.t_batch, 
                              shuffle=True, num_workers=config.num_workers)
    val_send_loader = DataLoader(val_send_dataset, 
                            batch_size=config.v_batch, 
                            shuffle=False, num_workers=config.num_workers)
    
    val_receive_loader = DataLoader(val_receive_dataset, 
                            batch_size=config.v_batch, 
                            shuffle=False, num_workers=config.num_workers)
    
    # SendDataset for testing
    send_dataset = SendDataset(root_path + 'KBSC/data/send')
    test_loader = DataLoader(send_dataset, batch_size=config.v_batch, shuffle=False)

    return train_loader, val_send_loader, val_receive_loader, test_loader


def get_feature_loader(config):
    train_dataset = TrainDataset(root_path + 'KBSC/data/train_feature.pth')
    val_send_dataset = ValDataset(root_path + 'KBSC/data/val_send_feature.pth')
    val_receive_loader = ValDataset(root_path + 'KBSC/data/val_receive_feature.pth')
    send_dataset = TestDataset(root_path + 'KBSC/data/test_feature.pth')

    # DataLoaders
    train_loader = DataLoader(train_dataset, 
                              batch_size=config.t_batch, 
                              shuffle=True, num_workers=config.num_workers)
    val_send_loader = DataLoader(val_send_dataset, 
                            batch_size=config.v_batch, 
                            shuffle=False, num_workers=config.num_workers)
    val_receive_loader = DataLoader(val_receive_loader, 
                            batch_size=config.v_batch, 
                            shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(send_dataset, batch_size=config.v_batch, shuffle=False)

    return train_loader, val_send_loader, val_receive_loader, test_loader


if __name__ == '__main__':
    # **创建 DataLoader**
    send_dir = "./send"
    send_dataset = SendDataset(send_dir)
    send_loader = DataLoader(send_dataset, batch_size=32, shuffle=False)

    # **测试 DataLoader**
    for images, labels, filenames in send_loader:
        print(f"Batch 图片 shape: {images.shape}")  # (batch_size, 3, H, W)
        print(f"类别示例: {labels[:5]}")
        print(f"文件名示例: {filenames[:5]}")
        break  # 只查看第一个 batch