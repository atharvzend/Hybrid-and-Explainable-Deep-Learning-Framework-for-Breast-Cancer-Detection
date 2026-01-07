import torch
import torch.nn.functional as F
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image
import cv2


class LIMEExplainer:
    """
    LIME explainer for image classification
    """
    def __init__(self, model, device='cpu'):
        """
        Args:
            model: Trained model
            device: Device to run model on
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        # Create LIME explainer
        self.explainer = lime_image.LimeImageExplainer()
    
    def predict_fn(self, images):
        """
        Prediction function for LIME
        
        Args:
            images: Batch of images (N, H, W, C) in RGB format [0, 255]
        
        Returns:
            Probabilities for each class (N, num_classes)
        """
        batch_predictions = []
        
        for img in images:
            # Convert to grayscale
            if len(img.shape) == 3 and img.shape[2] == 3:
                gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            
            # Resize to 224x224
            resized = cv2.resize(gray, (224, 224))
            
            # Convert to tensor and normalize to [0, 1]
            tensor = torch.FloatTensor(resized).unsqueeze(0).unsqueeze(0) / 255.0
            tensor = tensor.to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(tensor)
                probs = F.softmax(output, dim=1)
                batch_predictions.append(probs.cpu().numpy()[0])
        
        return np.array(batch_predictions)
    
    def explain(self, image, num_samples=1000, num_features=10, top_labels=3):
        """
        Generate LIME explanation
        
        Args:
            image: PIL Image or numpy array (RGB)
            num_samples: Number of samples for LIME
            num_features: Number of top features to show
            top_labels: Number of top labels to explain
        
        Returns:
            Dictionary with explanation results
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        
        # Ensure RGB format
        if len(image_array.shape) == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            image_array,
            self.predict_fn,
            top_labels=top_labels,
            hide_color=0,
            num_samples=num_samples
        )
        
        # Get the predicted label
        predicted_label = explanation.top_labels[0]
        
        # Get explanation mask and image
        temp, mask = explanation.get_image_and_mask(
            predicted_label,
            positive_only=False,
            num_features=num_features,
            hide_rest=False
        )
        
        return {
            'explanation': explanation,
            'predicted_label': predicted_label,
            'mask': mask,
            'image_with_mask': temp,
            'segments': explanation.segments
        }
    
    def visualize_explanation(self, image, explanation_result):
        """
        Create visualization of LIME explanation
        
        Args:
            image: Original image
            explanation_result: Result from explain() method
        
        Returns:
            Visualization image
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        
        # Get mask
        mask = explanation_result['mask']
        
        # Create visualization with boundaries
        visualization = mark_boundaries(image_array / 255.0, mask)
        
        # Convert to uint8
        visualization = (visualization * 255).astype(np.uint8)
        
        return visualization


def generate_lime_explanation(model, image, device='cpu', num_samples=1000, num_features=10):
    """
    High-level function to generate LIME explanation
    
    Args:
        model: Trained model
        image: PIL Image or numpy array
        device: Device to run on
        num_samples: Number of samples for LIME
        num_features: Number of features to show
    
    Returns:
        Dictionary with explanation and visualization
    """
    # Create LIME explainer
    lime_explainer = LIMEExplainer(model, device)
    
    # Generate explanation
    explanation_result = lime_explainer.explain(
        image,
        num_samples=num_samples,
        num_features=num_features
    )
    
    # Create visualization
    visualization = lime_explainer.visualize_explanation(image, explanation_result)
    
    # Get prediction probabilities
    image_array = np.array(image) if isinstance(image, Image.Image) else image
    probs = lime_explainer.predict_fn([image_array])[0]
    
    return {
        'mask': explanation_result['mask'],
        'visualization': visualization,
        'predicted_class': explanation_result['predicted_label'],
        'confidence': probs[explanation_result['predicted_label']] * 100,
        'all_probabilities': probs,
        'segments': explanation_result['segments']
    }


def get_lime_heatmap(mask):
    """
    Convert LIME mask to heatmap format
    
    Args:
        mask: LIME explanation mask
    
    Returns:
        Heatmap normalized to [0, 1]
    """
    # Normalize mask values
    mask_normalized = mask.astype(float)
    mask_normalized = (mask_normalized - mask_normalized.min()) / (mask_normalized.max() - mask_normalized.min() + 1e-8)
    
    return mask_normalized