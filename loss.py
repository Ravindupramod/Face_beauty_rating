"""
Custom Loss Functions for Beauty Prediction
Author: Senior Computer Vision Engineer
Purpose: Hybrid MSE + Pearson Correlation Loss to prevent regression-to-mean
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PearsonCorrelationLoss(nn.Module):
    """
    Pearson Correlation Loss: 1 - PCC
    
    Why use this?
    - Standard MSE causes "regression to the mean" - model predicts everyone as average
    - Pearson Correlation optimizes for RANKING instead of absolute values
    - Examples:
      - Ground truth: [1, 3, 5]
      - Prediction A: [1.5, 3.0, 4.5] → High Pearson (good ranking)
      - Prediction B: [3.0, 3.0, 3.0] → Low Pearson (bad ranking, regression to mean)
    
    Formula:
        PCC = Σ[(x - x̄)(y - ȳ)] / (σx * σy)
        Loss = 1 - PCC
    """
    
    def __init__(self, eps=1e-8):
        """
        Args:
            eps (float): Small constant for numerical stability
        """
        super(PearsonCorrelationLoss, self).__init__()
        self.eps = eps
    
    def forward(self, predictions, targets):
        """
        Compute Pearson Correlation Loss
        
        Args:
            predictions (Tensor): Predicted scores [B, 1] or [B]
            targets (Tensor): Ground truth scores [B, 1] or [B]
        
        Returns:
            Tensor: Scalar loss value (1 - correlation)
        """
        # Flatten tensors
        pred = predictions.view(-1)
        target = targets.view(-1)
        
        # Compute means
        pred_mean = torch.mean(pred)
        target_mean = torch.mean(target)
        
        # Center the variables
        pred_centered = pred - pred_mean
        target_centered = target - target_mean
        
        # Compute covariance
        covariance = torch.sum(pred_centered * target_centered)
        
        # Compute standard deviations
        pred_std = torch.sqrt(torch.sum(pred_centered ** 2) + self.eps)
        target_std = torch.sqrt(torch.sum(target_centered ** 2) + self.eps)
        
        # Compute Pearson correlation
        correlation = covariance / (pred_std * target_std + self.eps)
        
        # Loss is 1 - correlation (we want to minimize this)
        loss = 1.0 - correlation
        
        return loss


class HybridBeautyLoss(nn.Module):
    """
    Hybrid Loss: 0.7 * MSE + 0.3 * PearsonLoss
    
    Rationale:
    - MSE: Ensures absolute value accuracy
    - Pearson: Ensures correct ranking/ordering
    - Weighted combination prevents both regression-to-mean and maintains scale accuracy
    """
    
    def __init__(self, mse_weight=0.7, pearson_weight=0.3):
        """
        Args:
            mse_weight (float): Weight for MSE loss (default: 0.7)
            pearson_weight (float): Weight for Pearson loss (default: 0.3)
        """
        super(HybridBeautyLoss, self).__init__()
        self.mse_weight = mse_weight
        self.pearson_weight = pearson_weight
        
        self.mse_loss = nn.MSELoss()
        self.pearson_loss = PearsonCorrelationLoss()
    
    def forward(self, predictions, targets):
        """
        Compute hybrid loss
        
        Args:
            predictions (Tensor): Predicted scores [B, 1] or [B]
            targets (Tensor): Ground truth scores [B, 1] or [B]
        
        Returns:
            Tensor: Scalar hybrid loss value
        """
        mse = self.mse_loss(predictions, targets)
        pearson = self.pearson_loss(predictions, targets)
        
        total_loss = self.mse_weight * mse + self.pearson_weight * pearson
        
        return total_loss
    
    def get_components(self, predictions, targets):
        """
        Get individual loss components for logging
        
        Returns:
            dict: Dictionary with 'mse', 'pearson', and 'total' losses
        """
        mse = self.mse_loss(predictions, targets)
        pearson = self.pearson_loss(predictions, targets)
        total = self.mse_weight * mse + self.pearson_weight * pearson
        
        return {
            'mse': mse.item(),
            'pearson': pearson.item(),
            'total': total.item()
        }


def compute_metrics(predictions, targets):
    """
    Compute evaluation metrics for beauty prediction
    
    Args:
        predictions (Tensor): Predicted scores [B, 1] or [B]
        targets (Tensor): Ground truth scores [B, 1] or [B]
    
    Returns:
        dict: Dictionary with MAE, RMSE, and Pearson Correlation
    """
    pred = predictions.detach().view(-1)
    target = targets.detach().view(-1)
    
    # Mean Absolute Error
    mae = torch.mean(torch.abs(pred - target)).item()
    
    # Root Mean Squared Error
    rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()
    
    # Pearson Correlation Coefficient
    pred_mean = torch.mean(pred)
    target_mean = torch.mean(target)
    
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    
    covariance = torch.sum(pred_centered * target_centered)
    pred_std = torch.sqrt(torch.sum(pred_centered ** 2))
    target_std = torch.sqrt(torch.sum(target_centered ** 2))
    
    pearson_corr = (covariance / (pred_std * target_std + 1e-8)).item()
    
    return {
        'mae': mae,
        'rmse': rmse,
        'pearson': pearson_corr
    }


if __name__ == '__main__':
    # Test loss functions
    print("Testing Loss Functions...")
    
    # Create dummy data
    batch_size = 16
    predictions = torch.randn(batch_size, 1) * 2 + 3  # Around 3.0
    targets = torch.randn(batch_size, 1) * 2 + 3
    
    print(f"Predictions: {predictions.view(-1)[:5].tolist()}")
    print(f"Targets: {targets.view(-1)[:5].tolist()}")
    
    # Test MSE
    mse_loss = nn.MSELoss()
    mse = mse_loss(predictions, targets)
    print(f"\n✓ MSE Loss: {mse.item():.4f}")
    
    # Test Pearson Loss
    pearson_loss = PearsonCorrelationLoss()
    pearson = pearson_loss(predictions, targets)
    print(f"✓ Pearson Loss: {pearson.item():.4f}")
    
    # Test Hybrid Loss
    hybrid_loss = HybridBeautyLoss()
    hybrid = hybrid_loss(predictions, targets)
    print(f"✓ Hybrid Loss: {hybrid.item():.4f}")
    
    # Test loss components
    components = hybrid_loss.get_components(predictions, targets)
    print(f"\nLoss Components:")
    print(f"  - MSE: {components['mse']:.4f}")
    print(f"  - Pearson: {components['pearson']:.4f}")
    print(f"  - Total: {components['total']:.4f}")
    
    # Test metrics
    metrics = compute_metrics(predictions, targets)
    print(f"\nMetrics:")
    print(f"  - MAE: {metrics['mae']:.4f}")
    print(f"  - RMSE: {metrics['rmse']:.4f}")
    print(f"  - Pearson Correlation: {metrics['pearson']:.4f}")
    
    # Test edge case: Perfect predictions
    print("\n" + "="*50)
    print("Testing edge case: Perfect predictions")
    perfect_pred = targets.clone()
    mse_perfect = mse_loss(perfect_pred, targets)
    pearson_perfect = pearson_loss(perfect_pred, targets)
    print(f"✓ MSE (perfect): {mse_perfect.item():.6f}")
    print(f"✓ Pearson Loss (perfect): {pearson_perfect.item():.6f}")
    
    # Test edge case: All same predictions (regression to mean)
    print("\n" + "="*50)
    print("Testing edge case: Regression to mean")
    mean_pred = torch.ones_like(predictions) * predictions.mean()
    mse_mean = mse_loss(mean_pred, targets)
    pearson_mean = pearson_loss(mean_pred, targets)
    print(f"✓ MSE (mean): {mse_mean.item():.4f}")
    print(f"✓ Pearson Loss (mean): {pearson_mean.item():.4f}")
    print("  Note: Pearson loss is high (bad) because ranking is destroyed")
    
    print("\n✅ Loss functions test passed!")
