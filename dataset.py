"""
SCUT-FBP5500 Dataset Loader with Face Cropping and Context Padding
Author: Senior Computer Vision Engineer
Purpose: Load and preprocess facial beauty dataset with 20% context padding
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SCUTDataset(Dataset):
    """
    Custom Dataset for SCUT-FBP5500 Facial Beauty Prediction
    
    Features:
    - 20% context padding around face bounding boxes (captures forehead & chin)
    - ImageNet normalization for MobileNetV3 compatibility
    - Configurable augmentations for training/validation
    """
    
    def __init__(
        self, 
        root_dir, 
        split='train', 
        transform=None,
        face_padding=0.2,
        img_size=224
    ):
        """
        Args:
            root_dir (str): Root directory of SCUT-FBP5500 dataset
            split (str): 'train', 'val', or 'test'
            transform (callable, optional): Additional transforms
            face_padding (float): Padding ratio around face bbox (default: 0.2 for 20%)
            img_size (int): Target image size (default: 224 for MobileNetV3)
        """
        self.root_dir = root_dir
        self.split = split
        self.face_padding = face_padding
        self.img_size = img_size
        self.images_dir = os.path.join(root_dir, 'Images')
        
        # Load ratings and split information
        self._load_annotations()
        
        # Define transforms
        self.transform = self._get_transforms(transform)
        
    def _load_annotations(self):
        """Load beauty ratings and train/test split"""
        # Load all ratings from Excel file
        ratings_file = os.path.join(
            self.root_dir, 
            'train_test_files', 
            'All_Ratings.xlsx'
        )
        
        if not os.path.exists(ratings_file):
            raise FileNotFoundError(
                f"Ratings file not found: {ratings_file}\n"
                "Please ensure SCUT-FBP5500 dataset is properly downloaded."
            )
        
        # Read ratings (typically has columns: Filename, Rating1, Rating2, ..., Average)
        df = pd.read_excel(ratings_file)
        
        # Extract filename and average rating
        # Adjust column names based on actual dataset structure
        if 'Filename' in df.columns:
            self.filenames = df['Filename'].tolist()
        elif 'Image' in df.columns:
            self.filenames = df['Image'].tolist()
        else:
            # Assume first column is filename
            self.filenames = df.iloc[:, 0].tolist()
        
        # Get average ratings (usually last column or specifically named)
        if 'Average' in df.columns:
            self.ratings = df['Average'].values
        elif 'Rating' in df.columns:
            self.ratings = df['Rating'].values
        else:
            # Calculate average from all rating columns (skip first column which is filename)
            rating_cols = [col for col in df.columns if 'Rating' in col or col.startswith('R')]
            if rating_cols:
                self.ratings = df[rating_cols].mean(axis=1).values
            else:
                # Fallback: use last column
                self.ratings = df.iloc[:, -1].values
        
        # Load train/test split
        split_file = os.path.join(
            self.root_dir,
            'train_test_files',
            'split_of_60%training and 40%testing',
            f'{self.split}.txt'
        )
        
        # If split file doesn't exist, create a default split
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                split_filenames = [line.strip() for line in f.readlines()]
            
            # Filter data based on split
            indices = [i for i, fname in enumerate(self.filenames) if fname in split_filenames]
            self.filenames = [self.filenames[i] for i in indices]
            self.ratings = self.ratings[indices]
        else:
            # Create default 60/20/20 split
            print(f"Split file not found. Creating default {self.split} split...")
            n = len(self.filenames)
            if self.split == 'train':
                indices = list(range(int(0.6 * n)))
            elif self.split == 'val':
                indices = list(range(int(0.6 * n), int(0.8 * n)))
            else:  # test
                indices = list(range(int(0.8 * n), n))
            
            self.filenames = [self.filenames[i] for i in indices]
            self.ratings = self.ratings[indices]
        
        print(f"Loaded {len(self.filenames)} images for {self.split} split")
        
    def _get_transforms(self, additional_transform=None):
        """
        Get transform pipeline based on split
        Training: Augmentation + Normalization
        Val/Test: Only resize + Normalization
        """
        # ImageNet normalization (required for MobileNetV3)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        if self.split == 'train':
            # Training augmentations
            transform_list = [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),  # ±10° rotation
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                normalize
            ]
        else:
            # Validation/Test: No augmentation
            transform_list = [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                normalize
            ]
        
        if additional_transform:
            transform_list.insert(-2, additional_transform)  # Before ToTensor
        
        return transforms.Compose(transform_list)
    
    def _crop_face_with_padding(self, img):
        """
        Crop face with 20% context padding
        
        For beauty assessment, we need:
        - Forehead (hairline, eyebrows)
        - Chin and jawline
        - Overall facial harmony
        
        A tight crop (like in face recognition) removes these critical features.
        """
        width, height = img.size
        
        # Simple center crop with padding as baseline
        # In production, you would use actual face detection here
        # For training, we assume faces are already roughly centered
        
        # Detect face region (for this implementation, assume centered face)
        # In practice, you'd use MediaPipe or similar during data preprocessing
        face_width = int(width * 0.6)
        face_height = int(height * 0.7)
        
        # Calculate center
        center_x = width // 2
        center_y = int(height * 0.45)  # Slightly above center
        
        # Apply 20% padding
        padding_w = int(face_width * self.face_padding)
        padding_h = int(face_height * self.face_padding)
        
        # Calculate crop box
        left = max(0, center_x - face_width // 2 - padding_w)
        top = max(0, center_y - face_height // 2 - padding_h)
        right = min(width, center_x + face_width // 2 + padding_w)
        bottom = min(height, center_y + face_height // 2 + padding_h)
        
        # Crop with padding
        cropped = img.crop((left, top, right, bottom))
        
        return cropped
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        """
        Returns:
            image (Tensor): Preprocessed image tensor [3, 224, 224]
            rating (float): Beauty rating (typically 1.0 - 5.0)
        """
        # Load image
        img_name = self.filenames[idx]
        
        # Handle different filename formats
        if not img_name.endswith(('.jpg', '.png', '.jpeg')):
            img_name = img_name + '.jpg'
        
        img_path = os.path.join(self.images_dir, img_name)
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")
        
        # Apply face cropping with 20% padding
        img = self._crop_face_with_padding(img)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        # Get rating
        rating = float(self.ratings[idx])
        
        return img, rating


def get_dataloaders(root_dir, batch_size=32, num_workers=4):
    """
    Convenience function to create train/val/test dataloaders
    
    Args:
        root_dir (str): Root directory of SCUT-FBP5500 dataset
        batch_size (int): Batch size for training
        num_workers (int): Number of worker processes for data loading
    
    Returns:
        dict: Dictionary with 'train', 'val', 'test' DataLoaders
    """
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        try:
            dataset = SCUTDataset(root_dir, split=split)
            dataloaders[split] = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=num_workers,
                pin_memory=True,
                drop_last=(split == 'train')  # Drop last incomplete batch for training
            )
        except Exception as e:
            print(f"Warning: Could not create {split} dataloader: {e}")
            dataloaders[split] = None
    
    return dataloaders


if __name__ == '__main__':
    # Test the dataset loader
    print("Testing SCUT-FBP5500 Dataset Loader...")
    
    # Update this path to your dataset location
    dataset_path = './SCUT-FBP5500'
    
    if os.path.exists(dataset_path):
        try:
            # Create dataset
            train_dataset = SCUTDataset(dataset_path, split='train')
            print(f"\n✓ Successfully loaded {len(train_dataset)} training samples")
            
            # Test __getitem__
            img, rating = train_dataset[0]
            print(f"✓ Sample image shape: {img.shape}")
            print(f"✓ Sample rating: {rating:.2f}")
            print(f"✓ Rating range: [{train_dataset.ratings.min():.2f}, {train_dataset.ratings.max():.2f}]")
            
            # Test dataloader
            dataloaders = get_dataloaders(dataset_path, batch_size=8, num_workers=0)
            if dataloaders['train']:
                batch_imgs, batch_ratings = next(iter(dataloaders['train']))
                print(f"\n✓ Batch images shape: {batch_imgs.shape}")
                print(f"✓ Batch ratings shape: {batch_ratings.shape}")
            
            print("\n✅ Dataset loader test passed!")
            
        except Exception as e:
            print(f"\n❌ Error testing dataset: {e}")
    else:
        print(f"\n⚠ Dataset not found at {dataset_path}")
        print("Please download SCUT-FBP5500 and update the path.")
