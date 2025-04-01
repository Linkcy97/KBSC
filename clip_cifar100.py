import os
import clip
import torch
from torchvision.datasets import CIFAR100
from data.dataset import get_loader
from utils.config import config
from utils.utils import set_seed
import time

config = config
set_seed(config.seed)

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('./ViT-B-32.pt', device)

# Load dataset
if_train = True

cifar100 = CIFAR100(root=os.path.expanduser("../datasets/CIFAR100"), download=False)
train_loader, val_send_loader, val_receive_loader, test_loader = get_loader(config)

# Prepare text inputs (for all classes)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

# Store correct predictions
correct_predictions = 0
total_predictions = 0

# Store all features and labels
all_features = []
all_labels = []
all_time = 0
# Evaluate the model on all images in CIFAR100
with torch.no_grad():
    for batch_idx, (images, class_ids) in enumerate(train_loader):
        # Move images and class_ids to the device
        images = images.to(device)
        class_ids = class_ids.to(device)

        # Calculate image features for the batch
        # t0 = time.time()
        image_features = model.encode_image(images)
        # all_time += time.time() - t0
        # if batch_idx == 999:
        #     break
        # Calculate text features for all classes (this is done only once per batch)
        text_features = model.encode_text(text_inputs)

        # Normalize the features  make L2 distance equal 1 
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)

        if config.l2_norm:
            image_features = image_features_norm
            text_features = text_features_norm

        # Store the features and labels
        all_features.append(image_features)
        all_labels.append(class_ids)

        # Calculate similarity between image features and text features for all classes
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # Get the predicted classes for the batch
        predicted_class_idx = similarity.argmax(dim=-1)

        # Count correct predictions
        correct_predictions += (predicted_class_idx == class_ids).sum().item()
        total_predictions += class_ids.size(0)
    # all_time /= 1000
    # print("Average time per batch: ", all_time)
# Calculate and print accuracy
accuracy = correct_predictions / total_predictions * 100
print(f"Accuracy on CIFAR100: {accuracy:.2f}%")

# Save train feature
if if_train:
    torch.save({"features": torch.cat(all_features, dim=0),
                "labels": torch.cat(all_labels, dim=0)}, "./data/train_feature.pth")

all_features = []
all_labels = []
with torch.no_grad():
    for batch_idx, (images, class_ids) in enumerate(val_send_loader):
        # Move images and class_ids to the device
        images = images.to(device)
        class_ids = class_ids.to(device)

        image_features = model.encode_image(images)
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)

        if config.l2_norm:
            image_features = image_features_norm
            text_features = text_features_norm

        all_features.append(image_features)
        all_labels.append(class_ids)

if if_train:
    torch.save({"features": torch.cat(all_features, dim=0),
                "labels": torch.cat(all_labels, dim=0)}, "./data/val_send_feature.pth")

all_features = []
all_labels = []
with torch.no_grad():
    for batch_idx, (images, class_ids) in enumerate(val_receive_loader):
        # Move images and class_ids to the device
        images = images.to(device)
        class_ids = class_ids.to(device)

        image_features = model.encode_image(images)
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)

        if config.l2_norm:
            image_features = image_features_norm

        all_features.append(image_features)
        all_labels.append(class_ids)

if if_train:
    torch.save({"features": torch.cat(all_features, dim=0),
                "labels": torch.cat(all_labels, dim=0)}, "./data/val_receive_feature.pth")
