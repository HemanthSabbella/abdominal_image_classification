import os
import numpy as np
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt

train_labels_path = '../Public_leaderboard_data/train_labels'
all_labels = []

for root, dirs, files in os.walk(train_labels_path):
    for mask_file in files:
        if mask_file.endswith('.png'):
            mask = np.array(Image.open(os.path.join(root, mask_file)))
            all_labels.extend(mask.flatten())

label_distribution = Counter(all_labels)

if 0 in label_distribution:
    del label_distribution[0]

if label_distribution:
    labels, counts = zip(*label_distribution.items())
    
    plt.figure(figsize=(12, 6))
    plt.bar(labels, counts, tick_label=labels)
    plt.xlabel('Class Labels')
    plt.ylabel('Frequency')
    plt.title('Distribution of Labels in Segmentation Masks (Excluding Background)')
    plt.xticks(range(min(labels), max(labels)+1))
    
    plot_save_path = 'label_distribution_plot_train.png'
    plt.savefig(plot_save_path)
    plt.close()

