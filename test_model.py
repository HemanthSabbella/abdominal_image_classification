import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from dataloader import MedicalImageDataset, custom_transform
from model import UNet
from scipy import ndimage

test_image_dir = '../Public_leaderboard_data/test1_images'
voxel_spacing_file = '../Public_leaderboard_data/spacing_mm.txt'
bbox_file = '../Public_leaderboard_data/test1_bbox.txt'
output_dir = 'test_labels'

test_dataset = MedicalImageDataset(
    image_dir=test_image_dir, 
    label_dir=None, 
    voxel_spacing_file=voxel_spacing_file, 
    transform=custom_transform
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=1, out_channels=13).to(device)
model.load_state_dict(torch.load('final_unet_model.pth'))
model.eval()

bbox_data = {}
with open(bbox_file, 'r') as f:
    for line in f:
        parts = line.strip().split(': ')
        key = tuple(map(int, parts[0].strip('<>').split(', ')))
        bbox = list(map(int, parts[1].strip('[]').split(', ')))
        bbox_data[key] = bbox

def compute_dice(pred, target, smooth=1e-6):
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = np.sum(pred_flat * target_flat)
    dice = (2. * intersection + smooth) / (np.sum(pred_flat) + np.sum(target_flat) + smooth)
    return dice

def compute_nsd(pred, target, spacing, tolerance=1.0):
    pred_surface = np.logical_xor(pred, ndimage.binary_erosion(pred))
    target_surface = np.logical_xor(target, ndimage.binary_erosion(target))
    distances_target_to_pred = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    distances_pred_to_target = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    num_surface_points_pred = distances_pred_to_target.size
    num_surface_points_target = distances_target_to_pred.size
    num_within_tolerance_pred = np.sum(distances_pred_to_target <= tolerance)
    num_within_tolerance_target = np.sum(distances_target_to_pred <= tolerance)
    nsd = (num_within_tolerance_pred + num_within_tolerance_target) / (num_surface_points_pred + num_surface_points_target)
    return nsd

os.makedirs(output_dir, exist_ok=True)
dsc_scores = []
nsd_scores = []

with torch.no_grad():
    for batch in test_loader:
        images = batch['images'].to(device)
        ct_scan_id = os.path.basename(os.path.dirname(test_dataset.image_slices[test_loader.dataset.image_slices.index(batch['images'].item())]))
        slice_idx = int(os.path.basename(test_dataset.image_slices[test_loader.dataset.image_slices.index(batch['images'].item())]).split('.')[0])

        for organ in range(1, 13):
            bbox_key = (int(ct_scan_id), slice_idx, organ)
            if bbox_key in bbox_data:
                xmin, ymin, xmax, ymax = bbox_data[bbox_key]
                cropped_image = images[:, :, ymin:ymax, xmin:xmax]

                outputs = model(cropped_image)
                pred_crop = torch.argmax(outputs, dim=1).cpu().numpy()

                pred_slice = np.zeros((512, 512), dtype=np.uint8)
                pred_slice[ymin:ymax, xmin:xmax] = pred_crop[0]

                label_path = test_dataset.label_slices[test_loader.dataset.image_slices.index(batch['images'].item())] if test_dataset.label_slices else None
                if label_path:
                    target = np.array(Image.open(label_path))
                    spacing = batch['voxel_spacing'][0] if 'voxel_spacing' in batch else (1.0, 1.0, 1.0)

                    dsc = compute_dice(pred_slice, target)
                    dsc_scores.append(dsc)

                    nsd = compute_nsd(pred_slice, target, spacing, tolerance=1.0)
                    nsd_scores.append(nsd)

                ct_folder = os.path.join(output_dir, f'{ct_scan_id:02d}')
                os.makedirs(ct_folder, exist_ok=True)
                slice_filename = os.path.join(ct_folder, f'{slice_idx}.png')
                Image.fromarray(pred_slice).save(slice_filename)

if dsc_scores:
    avg_dsc = np.mean(dsc_scores)
    print(f'Average DSC: {avg_dsc:.4f}')
if nsd_scores:
    avg_nsd = np.mean(nsd_scores)
    print(f'Average NSD: {avg_nsd:.4f}')

import shutil
zip_filename = 'test_labels'
shutil.make_archive(zip_filename, 'zip', output_dir)
print(f"Results zipped to {zip_filename}.zip")
