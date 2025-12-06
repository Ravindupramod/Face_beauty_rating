"""
Simplified Training Script for Kaggle SCUT-FBP5500
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from pathlib import Path

from model import BeautyPredictor
from loss import HybridBeautyLoss, compute_metrics
from dataset_kaggle import get_kaggle_dataloaders

def train_model(
    data_dir='./Dataset',
    epochs=50,
    batch_size=32,
    lr=1e-4,
    device='cpu'
):
    """Train beauty prediction model"""
    
    print(f"\n{'='*70}")
    print("BEAUTY PREDICTION MODEL TRAINING")
    print(f"{'='*70}\n")
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load datasets
    print(f"\nLoading dataset from {data_dir}...")
    dataloaders = get_kaggle_dataloaders(data_dir, batch_size=batch_size, num_workers=0)
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    if train_loader is None or val_loader is None:
        raise ValueError("Failed to load dataset")
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    # Create model
    print("\nInitializing MobileNetV3-Large model...")
    model = BeautyPredictor(pretrained=True)
    model.to(device)
    print("✓ Model loaded")
    
    # Setup training
    criterion = HybridBeautyLoss(mse_weight=0.7, pearson_weight=0.3)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    best_pearson = -1.0
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Starting training for {epochs} epochs")
    print(f"{'='*70}\n")
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 50)
        
        # Train
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        start_time = time.time()
        
        for batch_idx, (images, ratings) in enumerate(train_loader):
            images = images.to(device)
            ratings = ratings.to(device).unsqueeze(1).float()
            
            # Forward+Backward
            predictions = model(images)
            loss = criterion(predictions, ratings)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.append(predictions.detach())
            train_targets.append(ratings.detach())
            
            #print progress every 20 batches
            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}] Loss: {train_loss/(batch_idx+1):.4f}")
        
        # Train metrics
        train_preds = torch.cat(train_preds, dim=0)
        train_targets = torch.cat(train_targets, dim=0)
        train_metrics = compute_metrics(train_preds, train_targets)
        
        train_time = time.time() - start_time
        avg_train_loss = train_loss / len(train_loader)
        
        print(f"\n  Training:")
        print(f"    Loss: {avg_train_loss:.4f}")
        print(f"    MAE: {train_metrics['mae']:.4f}")
        print(f"    Pearson: {train_metrics['pearson']:.4f}")
        print(f"    Time: {train_time:.1f}s")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for images, ratings in val_loader:
                images = images.to(device)
                ratings = ratings.to(device).unsqueeze(1).float()
                
                predictions = model(images)
                loss = criterion(predictions, ratings)
                
                val_loss += loss.item()
                val_preds.append(predictions)
                val_targets.append(ratings)
        
        val_preds = torch.cat(val_preds, dim=0)
        val_targets = torch.cat(val_targets, dim=0)
        val_metrics = compute_metrics(val_preds, val_targets)
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"  Validation:")
        print(f"    Loss: {avg_val_loss:.4f}")
        print(f"    MAE: {val_metrics['mae']:.4f}")
        print(f"    Pearson: {val_metrics['pearson']:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if val_metrics['pearson'] > best_pearson:
            best_pearson = val_metrics['pearson']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'metrics': val_metrics
            }
            torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
            print(f"    ⭐ New best Pearson: {best_pearson:.4f} - Model saved!")
    
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"Best Pearson Correlation: {best_pearson:.4f}")
    print(f"Model saved to: checkpoints/best_model.pth")
    print(f"{'='*70}\n")
    
    return model, best_pearson


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./Dataset')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )
