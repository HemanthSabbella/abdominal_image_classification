import os
import torch
import numpy as np
from PIL import Image
from model import UNet
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MedicalTestDataset(Dataset):
    def __init__(self, image_dir, bbox_file=None):
        self.image_dir = image_dir
        self.image_slices = []
        self.bboxes = self.load_bboxes(bbox_file) if bbox_file else None
        
        for ct_scan_id in sorted(os.listdir(self.image_dir)):
            image_folder = os.path.join(self.image_dir, ct_scan_id)
            if not os.path.isdir(image_folder):
                continue
            images = [os.path.join(image_folder, f) for f in sorted(os.listdir(image_folder)) if not f.startswith('.')]
            self.image_slices.extend(images)

    def load_bboxes(self, bbox_file):
        bboxes = {}
        with open(bbox_file, 'r') as f:
            for line in f:
                key, bbox_str = line.strip().split(': ')
                bbox = list(map(int, bbox_str.strip('[]').split(', ')))
                bboxes[key] = bbox
        return bboxes

    def __len__(self):
        return len(self.image_slices)

    def __getitem__(self, idx):
        image_path = self.image_slices[idx]
        image = Image.open(image_path).convert("L")
        image = np.array(image) / 255.0  # Normalize image to 0-1 range
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # [1, height, width]

        ct_scan_id = os.path.basename(os.path.dirname(image_path))
        bbox = self.bboxes.get(f"{ct_scan_id}, {idx}", None) if self.bboxes else None

        sample = {'images': image, 'bbox': bbox}
        return sample


def crop_to_bbox(image, bbox):
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        return image[:, y_min:y_max, x_min:x_max]
    return image


# test_dataset = MedicalTestDataset('Public_leaderboard_data/test1_images')
test_dataset = MedicalTestDataset('Public_leaderboard_data/test1_images', bbox_file='../Public_leaderboard_data/test1_bbox.txt')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=1, out_channels=13).to(device)
model.load_state_dict(torch.load('final_unet_model_ep10.pth', map_location=device))
model.eval()

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        images = batch['images'].to(device)
        bboxes = batch['bbox']
        
        # Crop images to bounding boxes
        images = torch.stack([crop_to_bbox(img, bbox) for img, bbox in zip(images, bboxes)])

        # Get both outputs but only use the main output for predictions
        # outputs = model(images)
        outputs, aux_output = model(images)  # Get both main and auxiliary outputs
        preds = torch.argmax(outputs, dim=1).cpu().numpy()[0]
        pred_mask = preds.astype(np.uint8)

        ct_scan_id = os.path.basename(os.path.dirname(test_dataset.image_slices[batch_idx]))
        slice_idx = int(os.path.basename(test_dataset.image_slices[batch_idx]).split('.')[0])

        ct_folder = os.path.join('test_labels', f'{ct_scan_id}')
        os.makedirs(ct_folder, exist_ok=True)
        slice_filename = os.path.join(ct_folder, f'{slice_idx}.png')
        Image.fromarray(pred_mask).save(slice_filename)
