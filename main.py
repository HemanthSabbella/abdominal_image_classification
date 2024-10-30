import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from scipy import ndimage
import os
from model import UNet
from dataloader import MedicalImageDataset, custom_transform
from PIL import Image

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Convert target to one-hot encoding
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        
        # Apply softmax to the predictions if not already applied
        pred = F.softmax(pred, dim=1)
        
        # Compute intersection and union for each class
        intersection = torch.sum(pred * target_one_hot, dim=(2, 3))
        union = torch.sum(pred, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
        
        # Calculate Dice Score for each class and average over batch and classes
        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score.mean()

        return dice_loss

train_image_dir = '../Public_leaderboard_data/train_images'
train_label_dir = '../Public_leaderboard_data/train_labels'
val_image_dir = '../Public_leaderboard_data/val_images'
val_label_dir = '../Public_leaderboard_data/val_labels'
voxel_spacing_file = '../Public_leaderboard_data/spacing_mm.txt'

train_dataset = MedicalImageDataset(
    image_dir=train_image_dir, 
    label_dir=train_label_dir, 
    voxel_spacing_file=voxel_spacing_file, 
    transform=custom_transform
)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

val_dataset = MedicalImageDataset(
    image_dir=val_image_dir, 
    label_dir=val_label_dir, 
    voxel_spacing_file=voxel_spacing_file, 
    transform=custom_transform
)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=1, out_channels=13).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
# criterion = torch.nn.CrossEntropyLoss()
criterion_main = torch.nn.CrossEntropyLoss()  # For main output
criterion_aux = torch.nn.CrossEntropyLoss()   # For auxiliary output
# criterion = DiceLoss()

checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

def compute_dice(pred, target, smooth=1e-6):
    # pred_flat = pred.flatten()
    # target_flat = target.flatten()
    pred_flat = (pred > 0.5).astype(np.uint8).flatten()  # Ensure binary mask
    target_flat = (target > 0.5).astype(np.uint8).flatten()  
    intersection = np.sum(pred_flat * target_flat)
    dice = (2. * intersection + smooth) / (np.sum(pred_flat) + np.sum(target_flat) + smooth)
    return dice

def compute_nsd(pred, target, spacing, tolerance=1.0):
    pred_surface = np.logical_xor(pred, ndimage.binary_erosion(pred))
    target_surface = np.logical_xor(target, ndimage.binary_erosion(target))

    # Compute distance transforms
    pred_distances = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    target_distances = ndimage.distance_transform_edt(~target_surface, sampling=spacing)

    pred_to_target_distance = pred_distances[target_surface]
    target_to_pred_distance = target_distances[pred_surface]

    within_tolerance_pred = np.sum(pred_to_target_distance <= tolerance)
    within_tolerance_target = np.sum(target_to_pred_distance <= tolerance)

    total_surface_points = np.sum(pred_surface) + np.sum(target_surface)
    nsd = (within_tolerance_pred + within_tolerance_target) / total_surface_points

    return nsd

# def compute_nsd(pred, target, spacing, tolerance=1.0):
#     pred_surface = np.logical_xor(pred, ndimage.binary_erosion(pred))
#     target_surface = np.logical_xor(target, ndimage.binary_erosion(target))
    
#     spacing_2d = spacing[:2] if len(spacing) > 2 else spacing

#     distances_target_to_pred = ndimage.distance_transform_edt(~target_surface, sampling=spacing_2d)
#     distances_pred_to_target = ndimage.distance_transform_edt(~pred_surface, sampling=spacing_2d)

#     num_surface_points_pred = distances_pred_to_target.size
#     num_surface_points_target = distances_target_to_pred.size
#     num_within_tolerance_pred = np.sum(distances_pred_to_target <= tolerance)
#     num_within_tolerance_target = np.sum(distances_target_to_pred <= tolerance)

#     nsd = (num_within_tolerance_pred + num_within_tolerance_target) / (num_surface_points_pred + num_surface_points_target)
#     return nsd

def train_model(model, train_loader, val_loader, epochs, model_path='final_unet_model.pth'):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_aux_loss = 0.0  # To keep track of auxiliary loss

        for batch in train_loader:
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            # outputs = model(images)
            outputs, aux_output = model(images)
            # loss = criterion(outputs, labels)

            # Calculate main loss
            loss_main = criterion_main(outputs, labels)
            
            # Calculate Auxiliary losses
            # upsample aux_output to ensure it is same dim as labels
            aux_output_upsampled = F.interpolate(aux_output, size=(512, 512), mode='bilinear', align_corners=False)
            loss_aux = criterion_aux(aux_output_upsampled, labels)  # Using the same labels for auxiliary
            
            loss = loss_main + 0.3 * loss_aux  # Combine losses (you can adjust the weight)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # running_loss += loss.item()
            running_loss += loss_main.item()
            running_aux_loss += loss_aux.item() 

        avg_train_loss = running_loss / len(train_loader)
        avg_aux_loss = running_aux_loss / len(train_loader) # Average auxiliary loss
        avg_combined_loss = (running_loss + running_aux_loss) / (len(train_loader) * 2)


        avg_val_loss, avg_dsc, avg_nsd = validate_model(model, val_loader)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, DSC: {avg_dsc:.4f}, NSD: {avg_nsd:.4f}, Auxiliary Loss: {avg_aux_loss:.4f}, Combined Loss: {avg_combined_loss:.4f}")

        # checkpoint_path = os.path.join(checkpoint_dir, f'unet_model_epoch_{epoch+1}.pth')
        # torch.save(model.state_dict(), checkpoint_path)
        # print(f"Checkpoint saved to {checkpoint_path}")

    torch.save(model.state_dict(), model_path)
    print(f"Final model saved to {model_path}")

def validate_model(model, val_loader, save_images=True, save_dir='validation_samples'):
    model.eval()
    val_loss = 0.0
    dsc_scores = []
    nsd_scores = []

    print("here")
    if save_images:
        # print("Save images folder")
        # print(save_images)
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)
            outputs, aux_output = model(images)

            # loss = criterion(outputs, labels)
            # val_loss += loss.item()
            loss_main = criterion_main(outputs, labels)
            aux_output_upsampled = F.interpolate(aux_output, size=(512, 512), mode='bilinear', align_corners=False)
            loss_aux = criterion_aux(aux_output, labels)  # Calculate auxiliary loss
            val_loss += (loss_main + loss_aux).item()  # Combine validation loss

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()

            for i in range(preds.shape[0]):
                pred = preds[i]
                target = labels[i]

                if 'voxel_spacing' in batch and len(batch['voxel_spacing']) > i:
                    spacing = batch['voxel_spacing'][i][:2]
                else:
                    spacing = (1.0, 1.0)

                dsc = compute_dice(pred, target)
                dsc_scores.append(dsc)

                nsd = compute_nsd(pred, target, spacing, tolerance=2.0)  # Adjusted tolerance
                nsd_scores.append(nsd)

                # if save_images and batch_idx < 5 and i < 5:
                #     # print(save_images)
                #     # print(batch_idx)
                #     # print(i)
                #     # pred_img = (pred * 255 / (pred.max() if pred.max() > 0 else 1)).astype(np.uint8)
                #     # target_img = (target * 255 / (target.max() if target.max() > 0 else 1)).astype(np.uint8)

                #     pred_img = (pred * 255).astype(np.uint8)
                #     target_img = (target * 255).astype(np.uint8)

                #     Image.fromarray(pred_img).save(os.path.join(save_dir, f'pred_batch{batch_idx}_img{i}.png'))
                #     Image.fromarray(target_img).save(os.path.join(save_dir, f'gt_batch{batch_idx}_img{i}.png'))

    avg_val_loss = val_loss / len(val_loader)
    avg_dsc = np.mean(dsc_scores) if dsc_scores else 0
    avg_nsd = np.mean(nsd_scores) if nsd_scores else 0

    return avg_val_loss, avg_dsc, avg_nsd



train_model(model, train_loader, val_loader, epochs=10, model_path='final_unet_model_ep10.pth')
