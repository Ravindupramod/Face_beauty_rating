"""
Training Script with ONNX Export and Int8 Quantization
Author: Senior Computer Vision Engineer
Purpose: Train MobileNetV3 beauty predictor and export to optimized ONNX format
"""

import os
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

from model import BeautyPredictor
from loss import HybridBeautyLoss, compute_metrics
from dataset import SCUTDataset, get_dataloaders


class Trainer:
    """Training manager for beauty prediction model"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        checkpoint_dir='checkpoints'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.best_pearson = -1.0
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        start_time = time.time()
        
        for batch_idx, (images, ratings) in enumerate(self.train_loader):
            # Move to device
            images = images.to(self.device)
            ratings = ratings.to(self.device).unsqueeze(1).float()
            
            # Forward pass
            predictions = self.model(images)
            loss = self.criterion(predictions, ratings)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            all_predictions.append(predictions.detach())
            all_targets.append(ratings.detach())
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f'  Batch [{batch_idx + 1}/{len(self.train_loader)}] '
                      f'Loss: {avg_loss:.4f}')
        
        # Compute epoch metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = compute_metrics(all_predictions, all_targets)
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(self.train_loader)
        
        print(f'\nEpoch {epoch} Training Summary:')
        print(f'  Time: {epoch_time:.2f}s')
        print(f'  Avg Loss: {avg_loss:.4f}')
        print(f'  MAE: {metrics["mae"]:.4f}')
        print(f'  RMSE: {metrics["rmse"]:.4f}')
        print(f'  Pearson: {metrics["pearson"]:.4f}')
        
        return avg_loss, metrics
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, ratings in self.val_loader:
                images = images.to(self.device)
                ratings = ratings.to(self.device).unsqueeze(1).float()
                
                predictions = self.model(images)
                loss = self.criterion(predictions, ratings)
                
                total_loss += loss.item()
                all_predictions.append(predictions)
                all_targets.append(ratings)
        
        # Compute metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = compute_metrics(all_predictions, all_targets)
        
        avg_loss = total_loss / len(self.val_loader)
        
        print(f'\nEpoch {epoch} Validation Summary:')
        print(f'  Avg Loss: {avg_loss:.4f}')
        print(f'  MAE: {metrics["mae"]:.4f}')
        print(f'  RMSE: {metrics["rmse"]:.4f}')
        print(f'  Pearson: {metrics["pearson"]:.4f}')
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch, val_loss, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f'  ✓ Saved best model to {best_path}')
    
    def train(self, num_epochs):
        """Full training loop"""
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_metrics = self.validate(epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'\n  Current LR: {current_lr:.6f}')
            
            # Save checkpoint
            is_best = val_metrics['pearson'] > self.best_pearson
            if is_best:
                self.best_pearson = val_metrics['pearson']
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, val_metrics, is_best)
            
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        print(f"Best Pearson Correlation: {self.best_pearson:.4f}")
        print(f"{'='*60}\n")


def export_to_onnx(
    checkpoint_path,
    onnx_path='beauty_model.onnx',
    input_shape=(1, 3, 224, 224),
    opset_version=12
):
    """
    Export trained PyTorch model to ONNX format
    
    Args:
        checkpoint_path (str): Path to trained model checkpoint
        onnx_path (str): Output path for ONNX model
        input_shape (tuple): Input tensor shape
        opset_version (int): ONNX opset version
    """
    print(f"\n{'='*60}")
    print("Exporting model to ONNX format")
    print(f"{'='*60}\n")
    
    # Load trained model
    device = torch.device('cpu')  # Export on CPU for compatibility
    model = BeautyPredictor(pretrained=False)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded checkpoint from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Validation Loss: {checkpoint['val_loss']:.4f}")
    print(f"  Pearson: {checkpoint['metrics']['pearson']:.4f}")
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"\n✓ Exported ONNX model to {onnx_path}")
    
    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"✓ ONNX model verification passed")
    
    # Get file size
    file_size_mb = os.path.getsize(onnx_path) / (1024 ** 2)
    print(f"✓ Model size: {file_size_mb:.2f} MB")
    
    return onnx_path


def quantize_model(
    onnx_path,
    quantized_path='beauty_model_int8.onnx'
):
    """
    Apply dynamic Int8 quantization to ONNX model
    
    This reduces model size to <5MB and improves CPU inference speed
    
    Args:
        onnx_path (str): Path to original ONNX model
        quantized_path (str): Output path for quantized model
    """
    print(f"\n{'='*60}")
    print("Applying Int8 Dynamic Quantization")
    print(f"{'='*60}\n")
    
    # Apply dynamic quantization
    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_path,
        weight_type=QuantType.QInt8,
        optimize_model=True
    )
    
    # Compare sizes
    original_size = os.path.getsize(onnx_path) / (1024 ** 2)
    quantized_size = os.path.getsize(quantized_path) / (1024 ** 2)
    reduction = (1 - quantized_size / original_size) * 100
    
    print(f"✓ Original model size: {original_size:.2f} MB")
    print(f"✓ Quantized model size: {quantized_size:.2f} MB")
    print(f"✓ Size reduction: {reduction:.1f}%")
    
    if quantized_size < 5.0:
        print(f"✓ Model size is under 5MB target!")
    else:
        print(f"⚠ Warning: Model size exceeds 5MB target")
    
    # Test inference
    print(f"\n✓ Testing quantized model inference...")
    session = ort.InferenceSession(quantized_path)
    dummy_input = torch.randn(1, 3, 224, 224).numpy()
    
    start_time = time.time()
    outputs = session.run(None, {'input': dummy_input})
    inference_time = (time.time() - start_time) * 1000
    
    print(f"✓ Sample output: {outputs[0][0][0]:.3f}")
    print(f"✓ Inference time: {inference_time:.2f}ms")
    
    if inference_time < 150:
        print(f"✓ Inference time is under 150ms target!")
    else:
        print(f"⚠ Warning: Inference time exceeds 150ms target")
    
    return quantized_path


def main():
    parser = argparse.ArgumentParser(description='Train Beauty Prediction Model')
    parser.add_argument('--data_dir', type=str, default='./SCUT-FBP5500',
                        help='Path to SCUT-FBP5500 dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--export_only', action='store_true',
                        help='Only export existing model to ONNX')
    parser.add_argument('--quantize_only', action='store_true',
                        help='Only quantize existing ONNX model')
    
    args = parser.parse_args()
    
    # Export only mode
    if args.export_only:
        checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
        onnx_path = export_to_onnx(checkpoint_path)
        quantize_model(onnx_path)
        return
    
    # Quantize only mode
    if args.quantize_only:
        quantize_model('beauty_model.onnx')
        return
    
    # Full training mode
    print("Initializing training...")
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    print(f"\nLoading dataset from {args.data_dir}...")
    dataloaders = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    if train_loader is None or val_loader is None:
        raise ValueError("Failed to load dataloaders. Check dataset path.")
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    # Create model
    print("\nInitializing model...")
    model = BeautyPredictor(pretrained=True)
    model = model.to(device)
    print(f"✓ Model created and moved to {device}")
    
    # Setup training components
    criterion = HybridBeautyLoss(mse_weight=0.7, pearson_weight=0.3)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Train
    trainer.train(args.epochs)
    
    # Export to ONNX
    print("\nExporting trained model...")
    checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
    onnx_path = export_to_onnx(checkpoint_path)
    
    # Quantize
    quantize_model(onnx_path)
    
    print("\n✅ Training and export completed successfully!")


if __name__ == '__main__':
    main()
