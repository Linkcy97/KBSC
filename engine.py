import clip
import torch
import faiss
from utils.utils import *
import numpy as np
import random
import time

clip_net, preprocess = clip.load("./ViT-B-32.pt", device='cuda')

rece_KB = faiss.read_index("./data/receive_index.faiss")
rece_img_paths = np.load("./data/receive_image_paths.npy", allow_pickle=True)
receive_labels = np.load("./data/receive_labels.npy")

val_rece_KB = faiss.read_index("./data/val_receive_index.faiss")
val_receive_labels = np.load("./data/val_receive_labels.npy")

def train_one_epoch(config, train_loader, model, criterion, optimizer, logger, epoch):
    model.train()
    total_loss = 0

    for iter, data in enumerate(train_loader):
        image_features, labels = data
        image_features, labels = image_features.to(config.device), labels.to(config.device)

        # Pass through FeatureTranser
        choice = random.randint(0, len(config.train_snr) - 1)
        snr = config.train_snr[choice]
        outputs = model(image_features, snr)
        loss = criterion(outputs, image_features)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        total_loss += loss.item()
        
        if iter % config.print_interval == 0:
            log_info =(' | '.join([
                    f'Epoch {epoch}',
                    f'iter {iter}',
                    f'Loss {loss:.5f}',
                    f'SNR {snr:.1f} ',
                ]))
            print(log_info)
            logger.info(log_info)
    avg_loss = total_loss / len(train_loader)
    log_info = f"Epoch {epoch}: Train Loss: {avg_loss:.4f}"
    logger.info(log_info)
    print(log_info)



def validate_one_epoch(config, val_loader, model, criterion, logger):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for iter, data in enumerate(val_loader):
            image_features, labels = data
            image_features, labels = image_features.to(config.device), labels.to(config.device)

            choice = random.randint(0, len(config.train_snr) - 1)
            snr = config.train_snr[choice]
            outputs = model(image_features, snr)
            loss = criterion(outputs, image_features)
            total_loss += loss.item()
        avg_loss = total_loss / len(val_loader)
        log_info = f"Val Loss: {avg_loss:.4f}"
        logger.info(log_info)
        print(log_info)

    return avg_loss

def validate_semantic_one_epoch(config, val_loader, model, criterion, logger):
    model.eval()
    correct = loss_x_y = cos_x_y = loss_y_kb = cos_y_kb = snr_a = 0
    with torch.no_grad():
        for iter, data in enumerate(val_loader):
            image_features, labels = data
            image_features, labels = image_features.to(config.device), labels.to(config.device)

            choice = random.randint(0, len(config.train_snr) - 1)
            snr = config.train_snr[choice]
            snr_a += snr
            f_noise = model(image_features, snr)
            loss = criterion(f_noise, image_features)
            loss_x_y += loss.item()*len(labels)
            cos_x_y += F.cosine_similarity(image_features, f_noise).sum().item()

            D, I = val_rece_KB.search(f_noise.cpu().numpy().astype("float32"), 1)
            nearest_y = np.array([val_rece_KB.reconstruct(int(idx)) for idx in I[:, 0]])
            nearest_y = torch.from_numpy(nearest_y).to(config.device)

            # mse and cos compare to KB
            loss_y_kb += criterion(f_noise, nearest_y).item()*len(labels)
            cos_y_kb += F.cosine_similarity(f_noise, nearest_y).sum().item()  

            matched_labels = val_receive_labels[I[:, 0]]

            correct += (labels.cpu().numpy() == matched_labels).sum()

    # final metrics
    mse_avg_x_y = loss_x_y / len(val_loader.dataset)
    cos_avg_x_y = cos_x_y / len(val_loader.dataset)
    mse_avg_y_kb = loss_y_kb / len(val_loader.dataset)
    cos_avg_y_kb = cos_y_kb / len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset) * 100
    snr_a = snr_a / len(val_loader.dataset)

    log_info =(f"SNR:{snr_a:<6} x<--->y MSE:{mse_avg_x_y:<10.4f} Cos:{cos_avg_x_y:<10.4f}" 
            f"y<--->kb MSE:{mse_avg_y_kb:<10.4f} Cos:{cos_avg_y_kb:<10.4f}"
            f"Semantic Matching Accuracy: {accuracy:.2f}%")
    logger.info(log_info)
    print(log_info)

    return accuracy

def test_once(config, test_loader, model, criterion, logger, snr):
    correct = loss_x_y = cos_x_y = loss_y_kb = cos_y_kb = 0
    for iter, data in enumerate(test_loader):
        image_features, labels, filenames = data
        image_features = image_features.to(config.device)
        with torch.no_grad():
            f_noise = model(image_features, snr)
        
        loss = criterion(image_features, f_noise)
        loss_x_y += loss.item()*len(labels)
        cos_x_y += F.cosine_similarity(image_features, f_noise).sum().item()

        D, I = rece_KB.search(f_noise.cpu().numpy().astype("float32"), 1)
        nearest_y = np.array([rece_KB.reconstruct(int(idx)) for idx in I[:, 0]])
        nearest_y = torch.from_numpy(nearest_y).to(config.device)

        # mse and cos compare to KB
        loss_y_kb += criterion(f_noise, nearest_y).item()*len(labels)
        cos_y_kb += F.cosine_similarity(f_noise, nearest_y).sum().item()  

        matched_labels = receive_labels[I[:, 0]]
        matched_paths = rece_img_paths[I[:, 0]]

        correct += (labels.numpy() == matched_labels).sum()

    # final metrics
    mse_avg_x_y = loss_x_y / len(test_loader.dataset)
    cos_avg_x_y = cos_x_y / len(test_loader.dataset)
    mse_avg_y_kb = loss_y_kb / len(test_loader.dataset)
    cos_avg_y_kb = cos_y_kb / len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset) * 100

    if snr == 'no':
        log_info =(f"SNR:{snr:<6} x<--->y MSE:{mse_avg_x_y:<10.4f} Cos:{cos_avg_x_y:<10.4f}" 
              f"x=y<->kb:MSE:{mse_avg_y_kb:<10.4f} Cos:{cos_avg_y_kb:<10.4f}"
              f"Semantic Matching Accuracy: {accuracy:.2f}% ")
        logger.info(log_info)
        print(log_info)
    else:
        log_info =(f"SNR:{snr:<6} x<--->y MSE:{mse_avg_x_y:<10.4f} Cos:{cos_avg_x_y:<10.4f}" 
              f"y<--->kb MSE:{mse_avg_y_kb:<10.4f} Cos:{cos_avg_y_kb:<10.4f}"
              f"Semantic Matching Accuracy: {accuracy:.2f}%")
        logger.info(log_info)
        print(log_info)

    return accuracy / 100

# Testing loop
def test(config, test_loader, model, criterion, logger):
    model.eval()
    acc_list = []
    for snr in config.test_snr:
        acc = test_once(config, test_loader, model, criterion, logger, snr)
        acc_list.append(round(acc, 4))
    acc = test_once(config, test_loader, model, criterion, logger, 'no')
    acc_list.append(round(acc, 4))
    logger.info(acc_list)
    print(acc_list) 


def test_bpg(config, test_loader, criterion):
    correct = loss_x_y = cos_x_y = loss_y_kb = cos_y_kb = 0
    for iter, data in enumerate(test_loader):
        image_features, labels, filenames = data
        image_features = image_features.to(config.device)
        with torch.no_grad():
            f_noise = clip_net.encode_image(image_features).float()
        
        D, I = rece_KB.search(f_noise.cpu().numpy().astype("float32"), 1)
        nearest_y = np.array([rece_KB.reconstruct(int(idx)) for idx in I[:, 0]])
        nearest_y = torch.from_numpy(nearest_y).to(config.device)

        # mse and cos compare to KB
        loss_y_kb += criterion(f_noise, nearest_y).item()*len(labels)
        cos_y_kb += F.cosine_similarity(f_noise, nearest_y).sum().item()  

        matched_labels = receive_labels[I[:, 0]]
        matched_paths = rece_img_paths[I[:, 0]]

        correct += (labels.numpy() == matched_labels).sum()

    # final metrics
    mse_avg_x_y = 'nan'
    cos_avg_x_y = 'nan'
    mse_avg_y_kb = loss_y_kb / len(test_loader.dataset)
    cos_avg_y_kb = cos_y_kb / len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset) * 100

    snr = 'no'
    log_info =(f"SNR:{snr:<6} x<--->y MSE:{mse_avg_x_y:<4} Cos:{cos_avg_x_y:<4}" 
            f"y<--->kb MSE:{mse_avg_y_kb:<10.4f} Cos:{cos_avg_y_kb:<10.4f}"
            f"Semantic Matching Accuracy: {accuracy:.2f}%")
    print(log_info)
    return accuracy / 100


def test_time(config, test_loader, model, criterion, logger, snr):
    net_time = kb_time = 0
    for iter, data in enumerate(test_loader):
        image_features, labels, filenames = data
        image_features = image_features.to(config.device)
        with torch.no_grad():
            t0 = time.time()
            f_noise = model(image_features, snr)
            net_time += time.time() - t0 
        t1 = time.time()
        D, I = rece_KB.search(f_noise.cpu().numpy().astype("float32"), 1)
        kb_time += time.time() - t1
        if iter == 999:
            break

    net_time = net_time / 1000
    kb_time = kb_time / 1000
    print(f"Net time: {net_time:.4f} s, KB time: {kb_time:.4f} s")
    return