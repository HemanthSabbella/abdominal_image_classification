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

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define custom dataset
class MedicalImageDataset(SemanticSegmentationDataset):
    def __init__(self, image_dir, label_dir=None, transform=None, num_classes=13):
        super().__init__(image_dir, label_dir, transform)
        self.img_paths = self._get_all_images(image_dir)
        self.label_paths = self._get_all_images(label_dir) if label_dir else None
        self.transform = transform
        self.num_classes = num_classes

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
        image = Image.open(self.img_paths[idx]).convert("L")
        image = np.array(image, dtype=np.float32) / 255.0

        if self.transform:
            image = self.transform(image)

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        label = None
        if self.label_paths:
            label = Image.open(self.label_paths[idx])
            label = np.array(label, dtype=np.int64)
            label = torch.tensor(label, dtype=torch.long)
            label = one_hot(label, num_classes=self.num_classes).permute(2, 0, 1).float()

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
        return 0.35 * ce + 0.65 * dice


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

    final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')
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
