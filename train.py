import clip
import torch
from data.dataset import get_feature_loader
import faiss
from utils.utils import *
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
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    logger = get_logger('train', log_dir)
    log_config_info(config, logger)
    # copy model file to work_dir
    shutil.copy('model/net.py', config.work_dir)
    shutil.copy('utils/config.py', config.work_dir)

    print('#----------GPU init----------#')
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    train_loader, val_loader, _, test_loader = get_feature_loader(config)

    print('#----------Prepareing Model----------#')
    model = FeatureTranser(config).to(device)
    model.cuda()
    cal_params_flops(model, config.original_feature, logger)

    criterion = config.criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    print('#----------Training----------#')
    best_acc = 0
    no_imporve = 0
    for epoch in range(0, config.epoch):
        torch.cuda.empty_cache()
        train_one_epoch(config, train_loader, model, criterion, optimizer, logger, epoch)
        val_acc = validate_semantic_one_epoch(config, val_loader, model, criterion, logger)
        if best_acc < val_acc:
            best_acc = val_acc
            no_imporve = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            print('----------Best Model Saved----------')
            logger.info('----------Best Model Saved----------')
            # test(config, test_loader, model, criterion, logger)
        else:
            no_imporve += 1
            if no_imporve > config.early_stop:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                logger.info(f"Early stopping triggered after {epoch + 1} epochs.")
                break
    print('----------Training Finished----------')
    # loading the best parameters
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best.pth')))
    test(config, test_loader, model, criterion, logger)


if __name__ == '__main__':
    config = config()
    main(config)