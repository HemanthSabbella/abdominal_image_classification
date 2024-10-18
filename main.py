import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from scipy import ndimage
import os
from PIL import Image
import shutil
from model import UNet
from dataloader import MedicalImageDataset, custom_transform

train_image_dir = 'Public_leaderboard_data/train_images'
train_label_dir = 'Public_leaderboard_data/train_labels'
val_image_dir = 'Public_leaderboard_data/val_images'
val_label_dir = 'Public_leaderboard_data/val_labels'
test_image_dir = 'Public_leaderboard_data/test_images'
voxel_spacing_file = 'Public_leaderboard_data/spacing_mm.txt'
bbox_file = 'Public_leaderboard_data/test1_bbox.txt'

train_dataset = MedicalImageDataset(
    image_dir=train_image_dir, 
    label_dir=train_label_dir, 
    voxel_spacing_file=voxel_spacing_file, 
    transform=custom_transform
)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

val_dataset = MedicalImageDataset(
    image_dir=val_image_dir, 
    label_dir=val_label_dir, 
    voxel_spacing_file=voxel_spacing_file, 
    transform=custom_transform
)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

test_dataset = MedicalImageDataset(
    image_dir=test_image_dir, 
    label_dir=None, 
    voxel_spacing_file=None, 
    transform=custom_transform
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

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

def generate_test_predictions_with_bbox(model, test_loader, bbox_file, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    bbox_data = {}
    with open(bbox_file, 'r') as f:
        for line in f:
            parts = line.strip().split(': ')
            key = tuple(map(int, parts[0].strip('<>').split(', ')))
            bbox = list(map(int, parts[1].strip('[]').split(', ')))
            bbox_data[key] = bbox

    with torch.no_grad():
        for batch in test_loader:
            images = batch['images'].to(device)
            ct_id = batch['ct_id']
            num_slices = images.shape[-1]

            for slice_idx in range(num_slices):
                for organ in range(1, 13):
                    bbox_key = (int(ct_id[0]), slice_idx + 1, organ)
                    if bbox_key in bbox_data:
                        xmin, ymin, xmax, ymax = bbox_data[bbox_key]
                        cropped_image = images[0, 0, ymin:ymax, xmin:xmax].unsqueeze(0).unsqueeze(0)
                        cropped_output = model(cropped_image)
                        pred_crop = torch.argmax(cropped_output, dim=1)[0].cpu().numpy()

                        pred_slice = np.zeros((512, 512), dtype=np.uint8)
                        pred_slice[ymin:ymax, xmin:xmax] = pred_crop

                        ct_folder = os.path.join(output_dir, f'{ct_id[0]:02d}')
                        os.makedirs(ct_folder, exist_ok=True)
                        slice_filename = os.path.join(ct_folder, f'{slice_idx+1}.png')
                        Image.fromarray(pred_slice).save(slice_filename)

def zip_results(output_dir, zip_filename):
    shutil.make_archive(zip_filename, 'zip', output_dir)

train_model(model, train_loader, val_loader, epochs=50, model_path='final_unet_model.pth')

output_dir = 'test_labels'
generate_test_predictions_with_bbox(model, test_loader, bbox_file, output_dir)

zip_filename = 'test_labels'
zip_results(output_dir, zip_filename)
print(f"Results zipped to {zip_filename}.zip")
