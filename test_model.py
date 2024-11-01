import os
import torch
import numpy as np
from PIL import Image
from backbones_unet.model.unet import Unet
from backbones_unet.utils.dataset import SemanticSegmentationDataset
from backbones_unet.model.losses import DiceLoss 
from torch.utils.data import Dataset, DataLoader

# Define the dataset for test images
class MedicalTestDataset(Dataset):
    def __init__(self, image_dir):
        self.image_slices = self._get_all_images(image_dir)

    def _get_all_images(self, root_dir):
        """Recursively collect all image paths from the directory and subdirectories."""
        image_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if not file.startswith('.') and file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    image_paths.append(os.path.join(root, file))
        return sorted(image_paths)

    def __len__(self):
        return len(self.image_slices)

    def __getitem__(self, idx):
        image_path = self.image_slices[idx]
        image = Image.open(image_path).convert("L")
        image = np.array(image, dtype=np.float32) / 255.0  # Normalize image to [0, 1]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        return {'images': image, 'image_path': image_path}

# Define the test dataset and dataloader
test_image_dir = '../Public_leaderboard_data/test1_images'
test_dataset = MedicalTestDataset(test_image_dir)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load U-Net model from MONAI
model = Unet(
    backbone='convnext_base',
    in_channels=1,
    num_classes=13
).to(device)

# Load the pre-trained model weights
model_path = 'checkpoints/best_model_epoch7_dice0.8488.pth'  # Adjust the path as needed
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Directory to save the predicted masks
output_dir = 'test_predictions'
os.makedirs(output_dir, exist_ok=True)

# Perform inference on the test data
with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        images = batch['images'].to(device)
        image_path = batch['image_path'][0]

        # Generate predictions
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()[0]
        pred_mask = preds.astype(np.uint8)

        # Extract info to organize saved masks
        ct_scan_id = os.path.basename(os.path.dirname(image_path))
        slice_idx = os.path.splitext(os.path.basename(image_path))[0]

        # Save the predicted mask
        ct_folder = os.path.join(output_dir, ct_scan_id)
        os.makedirs(ct_folder, exist_ok=True)
        slice_filename = os.path.join(ct_folder, f'{slice_idx}.png')
        Image.fromarray(pred_mask).save(slice_filename)  # Scale mask to [0, 255]

        # print(f"Saved predicted mask to: {slice_filename}")

print("Test predictions completed.")
