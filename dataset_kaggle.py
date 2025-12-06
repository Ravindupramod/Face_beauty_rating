"""
Kaggle SCUT-FBP5500 Dataset Loader
Adapted for Kaggle format with labels.txt
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class KaggleSCUTDataset(Dataset):
    """
    Dataset loader for Kaggle SCUT-FBP5500 format
    
    Structure:
    - Images/ folder with .jpg files
    - labels.txt with format: filename rating
    """
    
    def __init__(
        self, 
        root_dir, 
        split='train',
        train_ratio=0.6,
        val_ratio=0.2,
        transform=None,
        img_size=224
    ):
        """
        Args:
            root_dir (str): Root directory containing Images/ and labels.txt
            split (str): 'train', 'val', or 'test'
            train_ratio (float): Training set ratio
            val_ratio (float): Validation set ratio
            transform (callable, optional): Additional transforms
            img_size (int): Target image size
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.img_size = img_size
        
        # Check for nested Images/Images structure AND if it has files
        nested_dir = os.path.join(root_dir, 'Images', 'Images')
        if os.path.exists(nested_dir) and len(os.listdir(nested_dir)) > 0:
            self.images_dir = nested_dir
            print(f"Detected nested structure with files: {self.images_dir}")
        else:
            self.images_dir = os.path.join(root_dir, 'Images')
            print(f"Using standard structure: {self.images_dir}")
        
        # Load labels
        self._load_labels()
        
        # Split data
        self._split_data(train_ratio, val_ratio)
        
        # Define transforms
        self.transform = self._get_transforms(transform)
        
        print(f"Loaded {len(self.filenames)} images for {split} split")
        
    def _load_labels(self):
        """Load labels from labels.txt"""
        labels_file = os.path.join(self.root_dir, 'labels.txt')
        
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"labels.txt not found at {labels_file}")
        
        self.filenames = []
        self.ratings = []
        
        with open(labels_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        filename = parts[0]
                        rating = float(parts[1])
                        self.filenames.append(filename)
                        self.ratings.append(rating)
        
        self.ratings = np.array(self.ratings)
        print(f"Loaded {len(self.filenames)} total samples")
        print(f"Rating range: [{self.ratings.min():.2f}, {self.ratings.max():.2f}]")
    
    def _split_data(self, train_ratio, val_ratio):
        """Split data into train/val/test"""
        n = len(self.filenames)
        indices = np.arange(n)
        
        # Shuffle with fixed seed for reproducibility
        np.random.seed(42)
        np.random.shuffle(indices)
        
        # Calculate split points
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        if self.split == 'train':
            split_indices = indices[:train_end]
        elif self.split == 'val':
            split_indices = indices[train_end:val_end]
        else:  # test
            split_indices = indices[val_end:]
        
        self.filenames = [self.filenames[i] for i in split_indices]
        self.ratings = self.ratings[split_indices]
    
    def _get_transforms(self, additional_transform=None):
        """Get transform pipeline based on split"""
        # ImageNet normalization
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        if self.split == 'train':
            # Training augmentations
            transform_list = [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                normalize
            ]
        else:
            # Validation/Test
            transform_list = [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                normalize
            ]
        
        if additional_transform:
            transform_list.insert(-2, additional_transform)
        
        return transforms.Compose(transform_list)
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        """
        Returns:
            image (Tensor): Preprocessed image [3, 224, 224]
            rating (float): Beauty rating
        """
        #Retry up to 3 times if image fails
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Load image
                img_name = self.filenames[idx]
                img_path = os.path.join(self.images_dir, img_name)
                
                if not os.path.exists(img_path):
                    if attempt < max_retries - 1:
                        idx = (idx + 1) % len(self)
                        continue
                    raise FileNotFoundError(f"Image not found: {img_path}")
                
                img = Image.open(img_path).convert('RGB')
                
                # Apply transforms
                if self.transform:
                    img = self.transform(img)
                
                rating = float(self.ratings[idx])
                
                return img, rating
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Warning: Failed to load {img_name}: {e}. Trying next image...")
                    idx = (idx + 1) % len(self)
                    continue
                else:
                    # Last attempt - return a blank image
                    print(f"Error: Skipping corrupted image after {max_retries} attempts")
                    blank_img = torch.zeros(3, self.img_size, self.img_size)
                    return blank_img, 3.0  # Return neutral rating

        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        rating = float(self.ratings[idx])
        
        return img, rating


def get_kaggle_dataloaders(root_dir, batch_size=32, num_workers=4):
    """
    Create dataloaders for Kaggle SCUT-FBP5500 format
    
    Args:
        root_dir (str): Root directory containing Dataset/
        batch_size (int): Batch size
        num_workers (int): Number of workers
    
    Returns:
        dict: Dictionary with 'train', 'val', 'test' DataLoaders
    """
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        try:
            dataset = KaggleSCUTDataset(root_dir, split=split)
            dataloaders[split] = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=num_workers,
                pin_memory=True,
                drop_last=(split == 'train')
            )
        except Exception as e:
            print(f"Warning: Could not create {split} dataloader: {e}")
            dataloaders[split] = None
    
    return dataloaders


if __name__ == '__main__':
    # Test
    print("Testing Kaggle SCUT Dataset Loader...")
    
    dataset_path = './Dataset'
    
    if os.path.exists(dataset_path):
        try:
            train_dataset = KaggleSCUTDataset(dataset_path, split='train')
            print(f"\n✓ Successfully loaded {len(train_dataset)} training samples")
            
            img, rating = train_dataset[0]
            print(f"✓ Sample image shape: {img.shape}")
            print(f"✓ Sample rating: {rating:.2f}")
            
            dataloaders = get_kaggle_dataloaders(dataset_path, batch_size=8, num_workers=0)
            if dataloaders['train']:
                batch_imgs, batch_ratings = next(iter(dataloaders['train']))
                print(f"\n✓ Batch images shape: {batch_imgs.shape}")
                print(f"✓ Batch ratings shape: {batch_ratings.shape}")
            
            print("\n✅ Kaggle dataset loader test passed!")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
    else:
        print(f"\n⚠ Dataset not found at {dataset_path}")
