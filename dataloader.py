import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class MedicalImageDataset(Dataset):
    def __init__(self, image_dir, label_dir, voxel_spacing_file, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.voxel_spacings = self.load_voxel_spacings(voxel_spacing_file)
        self.ct_scans = os.listdir(self.image_dir)

    def load_voxel_spacings(self, voxel_spacing_file):
        voxel_spacings = {}
        with open(voxel_spacing_file, 'r') as f:
            for line in f:
                ct_id, x_spacing, y_spacing, z_spacing = line.strip().split(',')
                voxel_spacings[ct_id] = (float(x_spacing), float(y_spacing), float(z_spacing))
        return voxel_spacings

    def __len__(self):
        return len(self.ct_scans)

    def __getitem__(self, idx):
        ct_scan_id = self.ct_scans[idx]
        image_folder = os.path.join(self.image_dir, ct_scan_id)
        label_folder = os.path.join(self.label_dir, ct_scan_id)
        image_slices = sorted(os.listdir(image_folder))
        label_slices = sorted(os.listdir(label_folder))
        images = []
        labels = []
        for image_slice, label_slice in zip(image_slices, label_slices):
            image_path = os.path.join(image_folder, image_slice)
            label_path = os.path.join(label_folder, label_slice)
            image = Image.open(image_path).convert("L")
            label = Image.open(label_path)
            images.append(np.array(image))
            labels.append(np.array(label))
        images = np.stack(images, axis=-1)
        labels = np.stack(labels, axis=-1)
        voxel_spacing = self.voxel_spacings[ct_scan_id]
        if self.transform:
            images = self.transform(images)
            labels = self.transform(labels)
        images = torch.tensor(images, dtype=torch.float32).unsqueeze(0)
        labels = torch.tensor(labels, dtype=torch.long)
        sample = {'images': images, 'labels': labels, 'voxel_spacing': voxel_spacing}
        return sample

def custom_transform(image):
    return image / 255.0
