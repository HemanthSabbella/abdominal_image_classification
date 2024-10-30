import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class MedicalImageDataset(Dataset):
    def __init__(self, image_dir, label_dir=None, voxel_spacing_file=None, bbox_file=None, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.voxel_spacings = self.load_voxel_spacings(voxel_spacing_file) if voxel_spacing_file else None

        # Collect all 2D slices from image directories
        self.image_slices = []
        self.label_slices = []
        
        for ct_scan_id in sorted(os.listdir(self.image_dir)):
            image_folder = os.path.join(self.image_dir, ct_scan_id)
            if not os.path.isdir(image_folder):
                continue
            images = [os.path.join(image_folder, f) for f in sorted(os.listdir(image_folder)) if not f.startswith('.')]
            self.image_slices.extend(images)

            if self.label_dir:
                label_folder = os.path.join(self.label_dir, ct_scan_id)
                labels = [os.path.join(label_folder, f) for f in sorted(os.listdir(label_folder)) if not f.startswith('.')]
                self.label_slices.extend(labels)

    def load_voxel_spacings(self, voxel_spacing_file):
        voxel_spacings = {}
        with open(voxel_spacing_file, 'r') as f:
            for line in f:
                ct_id, spacing_str = line.strip().split(': ')
                x_spacing, y_spacing, z_spacing = map(float, spacing_str.strip('[]').split(', '))
                voxel_spacings[ct_id] = (x_spacing, y_spacing, z_spacing)
        return voxel_spacings

    def __len__(self):
        return len(self.image_slices)

    def __getitem__(self, idx):
        image_path = self.image_slices[idx]
        image = Image.open(image_path).convert("L")
        #image = np.array(image)
        image = np.array(image) / 255.0  # Normalize image to 0-1 range
        
        label = None
        if self.label_dir:
            label_path = self.label_slices[idx]
            label = Image.open(label_path)
            label = np.array(label)

        if self.transform:
            image = self.transform(image)
            if label is not None:
                label = self.transform(label)

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # [1, height, width]
        if label is not None:
            label = torch.tensor(label, dtype=torch.long)

        ct_scan_id = os.path.basename(os.path.dirname(image_path))
        voxel_spacing = self.voxel_spacings[ct_scan_id] if self.voxel_spacings else None
        bbox = self.bboxes.get(f"{ct_scan_id}, {idx}", None) if self.bboxes else None

        sample = {'images': image, 'labels': label, 'voxel_spacing': voxel_spacing}
        return sample
    
    def load_bboxes(self, bbox_file):
        bboxes = {}
        with open(bbox_file, 'r') as f:
            for line in f:
                key, bbox_str = line.strip().split(': ')
                bbox = list(map(int, bbox_str.strip('[]').split(', ')))
                bboxes[key] = bbox
        return bboxes


def custom_transform(image):
    #return image / 255.0
    return image
