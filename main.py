import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from backbones_unet.model.unet import Unet
from backbones_unet.utils.dataset import SemanticSegmentationDataset
from backbones_unet.model.losses import DiceLoss  # Use backbone DiceLoss
from torch.optim import AdamW
import numpy as np
from PIL import Image
from torch.nn.functional import one_hot

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define augmentation pipeline
augmentation_pipeline = A.Compose([
    A.RandomRotate90(),               # Random rotations by 90 degrees
    A.HorizontalFlip(p=0.5),          # 50% chance to horizontally flip the image
    A.VerticalFlip(p=0.5),            # 50% chance to vertically flip the image
    A.RandomResizedCrop(256, 256, scale=(0.8, 1.0)), # Random cropping
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.2), # Adjust brightness/contrast
    A.Normalize(mean=(0.5,), std=(0.5,)),  # Normalize the grayscale image
    ToTensorV2()
])

# Define custom dataset
class MedicalImageDataset(SemanticSegmentationDataset):
    def __init__(self, image_dir, label_dir=None, transform=None, augmentations=None, num_classes=13):
        super().__init__(image_dir, label_dir, transform)
        self.img_paths = self._get_all_images(image_dir)
        self.label_paths = self._get_all_images(label_dir) if label_dir else None
        self.num_classes = num_classes
        # Set augmentations
        # get augmentations from the augmentation pipeline
        augmentations = A.Compose([
            A.RandomRotate90(),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomResizedCrop(256, 256, scale=(0.8, 1.0)),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2()
        ])
        self.augmentations = augmentations

    def _get_all_images(self, root_dir):
        image_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if not file.startswith('.') and file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    image_paths.append(os.path.join(root, file))
        return sorted(image_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image
        image = np.array(Image.open(self.img_paths[idx]).convert("L"), dtype=np.float32) / 255.0
        label = None
        if self.label_paths:
            label = np.array(Image.open(self.label_paths[idx]), dtype=np.int64)

        # Apply augmentations if provided
        if True or self.augmentations:
            augmented = self.augmentations(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']
        else:
            # Default transformation without augmentation
            image = transforms.ToTensor()(image).float()
            if label is not None:
                label = torch.tensor(label, dtype=torch.long)

        if label is not None:
            label = one_hot(torch.tensor(label, dtype=torch.long), num_classes=self.num_classes)
            label = label.permute(2, 0, 1).float()  # Convert to [C, H, W]

        return {'images': image, 'labels': label}

# Paths to images and masks
train_image_dir = '../Public_leaderboard_data/train_images'
train_label_dir = '../Public_leaderboard_data/train_labels'
val_image_dir = '../Public_leaderboard_data/val_images'
val_label_dir = '../Public_leaderboard_data/val_labels'

# Initialize datasets and dataloaders
train_dataset = MedicalImageDataset(image_dir=train_image_dir, label_dir=train_label_dir, num_classes=13)
val_dataset = MedicalImageDataset(image_dir=val_image_dir, label_dir=val_label_dir, num_classes=13)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Initialize the model with the pretrained backbone
model = Unet(
    backbone='convnext_base',
    in_channels=1,
    num_classes=13
).to(device)

# # load model from checkpoints/best_model_epoch26_dice0.8468.pth
# model_path = 'checkpoints/best_model_epoch7_dice0.8488.pth'
# model.load_state_dict(torch.load(model_path, map_location=device
# ))

# Define combined loss with DiceLoss and weighted Cross-Entropy Loss
import torch
import torch.nn as nn
from backbones_unet.model.losses import DiceLoss  # Use backbone DiceLoss

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()  # Using DiceLoss from backbone
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        # Apply softmax to predictions for multi-class probabilities
        pred_softmax = torch.softmax(pred, dim=1)
        
        # If target is not one-hot encoded, encode it
        if target.shape[1] != pred.shape[1]:  # Check if target is not one-hot
            target = torch.nn.functional.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()

        # Compute Dice and Cross-Entropy Loss
        dice = self.dice_loss(pred_softmax, target)
        ce = self.ce_loss(pred, torch.argmax(target, dim=1))
        
        # Return weighted combination
        return 0.3 * ce + 0.7 * dice


# Define class weights for Cross-Entropy Loss only
criterion = CombinedLoss()

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=1e-4)

# Training and Validation Functions
def train_model(model, train_loader, val_loader, epochs=10, checkpoint_dir='checkpoints'):
    best_dice = 0.0
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss, avg_dice = validate_model(model, val_loader)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Dice: {avg_dice:.4f}")

        if avg_dice > best_dice:
            best_dice = avg_dice
            best_model_path = os.path.join(checkpoint_dir, f'best_model_epoch{epoch+1}_dice{best_dice:.4f}.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with Dice: {best_dice:.4f}")

    final_model_path = os.path.join(checkpoint_dir, 'final_model_adam1.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

def validate_model(model, val_loader, num_classes=13):
    model.eval()
    val_loss = 0.0
    per_class_intersection = torch.zeros(num_classes, device=device)
    per_class_union = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for batch in val_loader:
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            targets = torch.argmax(labels, dim=1)

            for cls in range(num_classes):
                pred_class = (preds == cls).float()
                target_class = (targets == cls).float()

                intersection = (pred_class * target_class).sum()
                union = pred_class.sum() + target_class.sum()

                per_class_intersection[cls] += intersection
                per_class_union[cls] += union

    avg_val_loss = val_loss / len(val_loader)
    per_class_dice_scores = 2 * per_class_intersection / torch.clamp(per_class_union, min=1e-6)
    avg_dice_score = per_class_dice_scores.mean().item()

    for cls in range(num_classes):
        print(f"Class {cls} Dice Score: {per_class_dice_scores[cls]:.4f}")

    print(f"Average Dice Score: {avg_dice_score:.4f}")
    return avg_val_loss, avg_dice_score

# Start training
train_model(model, train_loader, val_loader, epochs=30)
