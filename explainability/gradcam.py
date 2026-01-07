import torch
import torch.nn.functional as F
import numpy as np
import cv2
from utils.visualization import normalize_heatmap


class GradCAM:
    """
    Grad-CAM implementation for CNN visualization
    """
    def __init__(self, model, target_layer):
        """
        Args:
            model: The trained model
            target_layer: The target layer to compute gradients (e.g., last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.register_hooks()
    
    def register_hooks(self):
        """
        Register forward and backward hooks on target layer
        """
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Register hooks
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input image tensor (1, 1, 224, 224)
            target_class: Target class index (if None, uses predicted class)
        
        Returns:
            Grad-CAM heatmap (normalized to [0, 1])
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # If target class not specified, use predicted class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        class_score = output[0, target_class]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)
        
        # Global average pooling on gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (1, 1, H, W)
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize to [0, 1]
        cam = normalize_heatmap(cam)
        
        return cam, target_class
    
    def __call__(self, input_tensor, target_class=None):
        """
        Convenience method to generate Grad-CAM
        """
        return self.generate_cam(input_tensor, target_class)


def get_gradcam_heatmap(model, input_tensor, target_class=None):
    """
    High-level function to generate Grad-CAM heatmap
    
    Args:
        model: Trained HybridCNNViT model
        input_tensor: Input image tensor
        target_class: Target class for visualization
    
    Returns:
        Tuple of (heatmap, predicted_class)
    """
    # Get the last convolutional layer of CNN backbone
    # This is features[10] - Conv2d(128, 256)
    target_layer = model.cnn_backbone.features[10]
    
    # Create Grad-CAM object
    gradcam = GradCAM(model, target_layer)
    
    # Generate heatmap
    heatmap, predicted_class = gradcam(input_tensor, target_class)
    
    return heatmap, predicted_class


def resize_heatmap(heatmap, target_size):
    """
    Resize heatmap to target size
    
    Args:
        heatmap: Original heatmap
        target_size: Tuple of (height, width)
    
    Returns:
        Resized heatmap
    """
    return cv2.resize(heatmap, (target_size[1], target_size[0]))


def generate_gradcam_explanation(model, input_tensor, original_image_size):
    """
    Complete Grad-CAM explanation pipeline
    
    Args:
        model: Trained model
        input_tensor: Preprocessed input tensor
        original_image_size: Size of original image for resizing heatmap
    
    Returns:
        Dictionary with heatmap and prediction info
    """
    # Generate Grad-CAM heatmap
    heatmap, predicted_class = get_gradcam_heatmap(model, input_tensor)
    
    # Get prediction probabilities
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence = probabilities[0, predicted_class].item() * 100
    
    # Resize heatmap to original image size
    heatmap_resized = resize_heatmap(heatmap, original_image_size)
    
    return {
        'heatmap': heatmap_resized,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'all_probabilities': probabilities[0].cpu().numpy()
    }