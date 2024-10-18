import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from scipy import ndimage
from model import UNet
from dataloader import MedicalImageDataset, custom_transform

train_image_dir = 'Public_leaderboard_data/train_images'
train_label_dir = 'Public_leaderboard_data/train_labels'
voxel_spacing_file = 'dataset/spacing_mm.txt'

train_dataset = MedicalImageDataset(image_dir=train_image_dir, label_dir=train_label_dir, voxel_spacing_file=voxel_spacing_file, transform=custom_transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=1, out_channels=13).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

def dice_score(pred, target, smooth=1e-6):
    pred = (pred > 0).float()
    target = (target > 0).float()
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()

def compute_surface_distances(pred, target, spacing):
    pred_surface = np.logical_xor(pred, ndimage.binary_erosion(pred))
    target_surface = np.logical_xor(target, ndimage.binary_erosion(target))
    distances_target_to_pred = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    distances_pred_to_target = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    return distances_target_to_pred[pred_surface], distances_target_to_pred[target_surface]

def normalized_surface_dice(pred, target, spacing, tolerance=1.0):
    distances_pred_to_target, distances_target_to_pred = compute_surface_distances(pred, target, spacing)
    num_surface_points_pred = distances_pred_to_target.size
    num_surface_points_target = distances_target_to_pred.size
    num_within_tolerance_pred = np.sum(distances_pred_to_target <= tolerance)
    num_within_tolerance_target = np.sum(distances_target_to_pred <= tolerance)
    nsd = (num_within_tolerance_pred + num_within_tolerance_target) / (num_surface_points_pred + num_surface_points_target)
    return nsd

def train_model(model, train_loader, val_loader, epochs, model_path='unet_model.pth'):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        val_loss = validate_model(model, val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss:.4f}, Val Loss: {val_loss:.4f}")
        checkpoint_path = f'unet_model_epoch_{epoch+1}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def validate_model(model, val_loader):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(val_loader)

def evaluate_model(model, test_loader, voxel_spacing, tolerance=1.0):
    model.eval()
    dice_scores = []
    nsd_scores = []
    with torch.no_grad():
        for images, labels, spacing in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            for i in range(preds.shape[0]):
                dice = dice_score(preds[i], labels[i])
                dice_scores.append(dice)
                nsd = normalized_surface_dice(preds[i].cpu().numpy(), labels[i].cpu().numpy(), spacing[i], tolerance)
                nsd_scores.append(nsd)
    avg_dice = np.mean(dice_scores)
    avg_nsd = np.mean(nsd_scores)
    return avg_dice, avg_nsd

train_model(model, train_loader, None, epochs=50, model_path='final_unet_model.pth')
