import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T

class VOCSegmentationDataset(Dataset):
    def __init__(self, root, split='train', config=None):
        super().__init__()
        self.root = root
        self.split = split
        self.config = config
        
        self.image_dir = os.path.join(self.root, 'JPEGImages')
        self.mask_dir = os.path.join(self.root, 'SegmentationClass')
        
        # Get all mask files
        all_masks = [f for f in os.listdir(self.mask_dir) if f.endswith('.png')]
        all_ids = [os.path.splitext(f)[0] for f in all_masks]
        all_ids.sort()
        
        # Split with setting seed for reproducibility
        random.seed(42)
        random.shuffle(all_ids)
        
        num_total = len(all_ids)
        num_train = int(num_total * 0.7)
        num_val = int(num_total * 0.1)
        
        if split == 'train':
            self.ids = all_ids[:num_train]
        elif split == 'val':
            self.ids = all_ids[num_train:num_train + num_val]
        elif split == 'test':
            self.ids = all_ids[num_train + num_val:]
        else:
            raise ValueError(f"Unknown split: {split}")
            
        print(f"Loaded {len(self.ids)} images for {split} split.")
        
    def __len__(self):
        return len(self.ids)
        
    def __getitem__(self, index):
        img_id = self.ids[index]
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{img_id}.png")
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        
        # Apply transforms based on configuration
        if self.config:
            input_size = self.config['preprocess']['input_size']
            
            # Resize
            image = image.resize(input_size, Image.BILINEAR)
            mask = mask.resize(input_size, Image.NEAREST)
            
            # Convert to tensor
            image = TF.to_tensor(image)
            mask = torch.as_tensor(import_mask(mask), dtype=torch.long)
            
            # Normalize
            mean = self.config['preprocess']['mean']
            std = self.config['preprocess']['std']
            image = TF.normalize(image, mean=mean, std=std)
        else:
            image = TF.to_tensor(image)
            mask = torch.as_tensor(import_mask(mask), dtype=torch.long)
            
        return image, mask

def import_mask(mask_img):
    import numpy as np
    mask = np.array(mask_img)
    # Background is usually 0, classes 1-20, border is 255
    return mask
