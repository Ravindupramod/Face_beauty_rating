"""
MobileNetV3-Large Beauty Prediction Model
Author: Senior Computer Vision Engineer
Purpose: CPU-optimized facial beauty regression using MobileNetV3 backbone
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


class BeautyPredictor(nn.Module):
    """
    Facial Beauty Prediction Model
    
    Architecture:
    - Backbone: MobileNetV3-Large (pre-trained on ImageNet)
    - Head: Linear(1280→128) → Hardswish → Dropout → Linear(128→1)
    
    Why MobileNetV3?
    - Research shows ~0.90 correlation (comparable to ResNet-50)
    - Much faster on CPU: ~12ms vs 65ms (ResNet-50)
    - Hardswish activations are optimized for CPU vectorization
    """
    
    def __init__(self, pretrained=True, dropout_rate=0.2):
        """
        Args:
            pretrained (bool): Load ImageNet pre-trained weights
            dropout_rate (float): Dropout probability in head (default: 0.2)
        """
        super(BeautyPredictor, self).__init__()
        
        # Load MobileNetV3-Large backbone
        if pretrained:
            weights = MobileNet_V3_Large_Weights.DEFAULT
            self.backbone = mobilenet_v3_large(weights=weights)
        else:
            self.backbone = mobilenet_v3_large(weights=None)
        
        # Remove the original classifier
        # MobileNetV3 structure: features -> avgpool -> classifier
        # We'll keep features + avgpool, replace classifier
        
        # Get the number of input features to classifier
        in_features = self.backbone.classifier[0].in_features  # Should be 1280
        
        # Replace classifier with custom regression head
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Hardswish(inplace=True),  # CPU-optimized activation
            nn.Dropout(p=dropout_rate, inplace=False),
            nn.Linear(128, 1)  # Single regression output
        )
        
        # Initialize new layers with proper weights
        self._init_head()
    
    def _init_head(self):
        """Initialize the new regression head with Xavier/Kaiming initialization"""
        for m in self.backbone.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (Tensor): Input images [B, 3, 224, 224]
        
        Returns:
            Tensor: Predicted beauty scores [B, 1]
        """
        return self.backbone(x)
    
    def predict(self, x):
        """
        Prediction with score clamping to valid range
        
        Args:
            x (Tensor): Input images [B, 3, 224, 224]
        
        Returns:
            Tensor: Clamped beauty scores [B, 1] in range [1.0, 5.0]
        """
        with torch.no_grad():
            scores = self.forward(x)
            # Clamp to valid rating range
            scores = torch.clamp(scores, min=1.0, max=5.0)
        return scores


class BeautyPredictorLite(nn.Module):
    """
    Lighter version using MobileNetV3-Small for even faster inference
    Use this if you need <10ms latency but can accept slightly lower accuracy
    """
    
    def __init__(self, pretrained=True, dropout_rate=0.2):
        super(BeautyPredictorLite, self).__init__()
        
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        
        if pretrained:
            weights = MobileNet_V3_Small_Weights.DEFAULT
            self.backbone = mobilenet_v3_small(weights=weights)
        else:
            self.backbone = mobilenet_v3_small(weights=None)
        
        # MobileNetV3-Small has 576 features
        in_features = self.backbone.classifier[0].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout_rate, inplace=False),
            nn.Linear(64, 1)
        )
        
        self._init_head()
    
    def _init_head(self):
        for m in self.backbone.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.backbone(x)
    
    def predict(self, x):
        with torch.no_grad():
            scores = self.forward(x)
            scores = torch.clamp(scores, min=1.0, max=5.0)
        return scores


def count_parameters(model):
    """Count trainable and total parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_model_size_mb(model):
    """Estimate model size in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb


if __name__ == '__main__':
    # Test the model
    print("Testing BeautyPredictor Model...")
    
    # Create model
    model = BeautyPredictor(pretrained=True)
    model.eval()
    
    # Model info
    trainable, total = count_parameters(model)
    size_mb = get_model_size_mb(model)
    
    print(f"\n✓ Model created successfully")
    print(f"✓ Trainable parameters: {trainable:,}")
    print(f"✓ Total parameters: {total:,}")
    print(f"✓ Estimated model size: {size_mb:.2f} MB")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\n✓ Input shape: {dummy_input.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Sample output: {output.item():.3f}")
    
    # Test prediction method
    pred = model.predict(dummy_input)
    print(f"✓ Clamped prediction: {pred.item():.3f}")
    
    # Test lite version
    print("\n" + "="*50)
    print("Testing BeautyPredictorLite Model...")
    
    model_lite = BeautyPredictorLite(pretrained=True)
    model_lite.eval()
    
    trainable_lite, total_lite = count_parameters(model_lite)
    size_mb_lite = get_model_size_mb(model_lite)
    
    print(f"\n✓ Lite model created successfully")
    print(f"✓ Trainable parameters: {trainable_lite:,}")
    print(f"✓ Total parameters: {total_lite:,}")
    print(f"✓ Estimated model size: {size_mb_lite:.2f} MB")
    
    with torch.no_grad():
        output_lite = model_lite(dummy_input)
    
    print(f"✓ Lite output: {output_lite.item():.3f}")
    
    print("\n✅ Model test passed!")
