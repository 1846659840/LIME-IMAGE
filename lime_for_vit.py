import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import random
from sklearn.linear_model import Lasso
from skimage.segmentation import slic, mark_boundaries
from skimage import color, measure

class LIME:
    """
    Implementation of LIME (Local Interpretable Model-agnostic Explanations)
    Based on the original algorithm from 'Why Should I Trust You?: Explaining the Predictions of Any Classifier'
    Specifically designed for explaining ViT (Vision Transformer) models
    """
    def __init__(self, model, preprocess, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize LIME explainer
        
        Args:
            model: Pretrained ViT model
            preprocess: Input image preprocessing function
            device: Running device ('cuda' or 'cpu')
        """
        self.model = model
        self.preprocess = preprocess
        self.device = device
        self.model.to(self.device)
        self.model.eval()
    
    def segment_image(self, image, n_segments=50, compactness=10):
        """
        Segment image into superpixels using skimage's SLIC algorithm
        
        Args:
            image: Input PIL image
            n_segments: Number of segments to create
            compactness: Balances color proximity and space proximity.
                         Higher values give more weight to space proximity,
                         making superpixels more compact.
            
        Returns:
            segments: Superpixel segmentation mask
        """
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Apply SLIC algorithm from skimage
        segments = slic(img_array, n_segments=n_segments, compactness=compactness, start_label=0)
        
        return segments
    
    def get_image_segments(self, image, segments):
        """
        Get all segmented regions of the image
        
        Args:
            image: Input PIL image
            segments: Superpixel segmentation mask
            
        Returns:
            segment_imgs: List of segmented images
        """
        n_segments = np.max(segments) + 1
        segment_imgs = []
        img_array = np.array(image).copy()
        
        for segment_idx in range(n_segments):
            mask = (segments == segment_idx)
            segment_img = img_array.copy()
            # Set pixels not in the segment to gray
            segment_img[~mask] = 128
            segment_imgs.append(Image.fromarray(segment_img))
            
        return segment_imgs
    
    def perturb_image(self, image, segments, n_samples=1000):
        """
        Generate perturbed image samples
        
        Args:
            image: Input PIL image
            segments: Superpixel segmentation mask
            n_samples: Number of samples to generate
            
        Returns:
            perturbations: List of perturbed images
            binary_labels: Binary labels indicating which segments are kept (1) or grayed out (0)
        """
        n_segments = np.max(segments) + 1
        perturbations = []
        binary_labels = np.zeros((n_samples, n_segments), dtype=np.int8)
        img_array = np.array(image).copy()
        
        for i in range(n_samples):
            # Randomly select segments to keep
            # Sample "around" the interpretable version, as per Algorithm 1
            active_segments = np.random.randint(0, 2, size=n_segments)
            binary_labels[i] = active_segments
            
            # Create perturbed image
            perturbed_img = img_array.copy()
            for segment_idx in range(n_segments):
                if active_segments[segment_idx] == 0:  # If not keeping this segment
                    mask = (segments == segment_idx)
                    perturbed_img[mask] = 128  # Set to gray
            
            perturbations.append(Image.fromarray(perturbed_img))
                    
        return perturbations, binary_labels
    
    def predict_batch(self, images, target_class=None):
        """
        Make predictions with ViT model for a batch of images
        
        Args:
            images: List of input PIL images
            target_class: Target class index (optional)
            
        Returns:
            predictions: Prediction probabilities or probabilities for target class
        """
        batch_tensor = torch.stack([self.preprocess(img).to(self.device) for img in images])
        
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            if isinstance(outputs, tuple):  # Some ViT models return multiple outputs
                outputs = outputs[0]
            
            probs = F.softmax(outputs, dim=1)
            
            if target_class is not None:
                return probs[:, target_class].cpu().numpy()
            else:
                return probs.cpu().numpy()
    
    def compute_distances(self, binary_labels):
        """
        Compute distance between original instance (all features enabled)
        and perturbed instances for similarity kernel
        
        Args:
            binary_labels: Binary labels for each perturbed sample
            
        Returns:
            distances: Euclidean distances
        """
        # Original instance has all features enabled (all 1s)
        original = np.ones(binary_labels.shape[1])
        
        # Compute cosine similarity
        distances = np.sqrt(np.sum((binary_labels - original)**2, axis=1))
        return distances
    
    def similarity_kernel(self, binary_labels, kernel_width=0.25):
        """
        Compute similarity between original and perturbed instances
        using an exponential kernel (RBF)
        
        Args:
            binary_labels: Binary labels for each perturbed sample
            kernel_width: Kernel width parameter
            
        Returns:
            similarities: Similarity scores
        """
        distances = self.compute_distances(binary_labels)
        return np.sqrt(np.exp(-(distances**2) / kernel_width**2))
    
    def fit_lasso_model(self, binary_labels, predictions, similarities, K=5):
        """
        Fit sparse linear model (Lasso) using perturbed samples
        as per Algorithm 1 in the LIME paper
        
        Args:
            binary_labels: Binary labels for each perturbed sample (z'i)
            predictions: Model predictions for perturbed images (f(z))
            similarities: Sample weights from similarity kernel (πx(z))
            K: Number of features to include in explanation
            
        Returns:
            weights: Linear model weights
            intercept: Linear model intercept
        """
        # Use weighted Lasso regression with sample_weight as similarity
        alpha = 0.01  # Regularization strength
        lasso = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000)
        
        # Fit model with sample weights from similarity kernel
        lasso.fit(binary_labels, predictions, sample_weight=similarities)
        
        # Get coefficients
        weights = lasso.coef_
        intercept = lasso.intercept_
        
        # Enforce sparsity by keeping only K features with highest absolute weights
        if K < len(weights) and np.sum(weights != 0) > K:
            # Sort features by absolute weight
            sorted_indices = np.argsort(-np.abs(weights))
            # Keep only top K features
            mask = np.zeros_like(weights, dtype=bool)
            mask[sorted_indices[:K]] = True
            sparse_weights = np.zeros_like(weights)
            sparse_weights[mask] = weights[mask]
            return sparse_weights, intercept
        
        return weights, intercept
    
    def explain(self, image, target_class=None, n_segments=50, n_samples=1000, K=5):
        """
        Explain ViT model prediction for a given image
        Implementation of Algorithm 1 from the LIME paper
        
        Args:
            image: Input PIL image (instance x)
            target_class: Target class index (optional)
            n_segments: Number of segments to create
            n_samples: Number of perturbed samples (N in Algorithm 1)
            K: Maximum number of features for explanation
            
        Returns:
            explanation: Feature importance weights
            segments: Superpixel segmentation mask
        """
        # 1. Segment image to create interpretable representation x'
        segments = self.segment_image(image, n_segments)
        
        # 2. Get original prediction
        if target_class is None:
            original_pred = self.predict_batch([image])[0]
            target_class = np.argmax(original_pred)
            print(f"Target class: {target_class}, Prediction probability: {original_pred[target_class]:.4f}")
        
        # 3. Generate perturbed samples z'i by sampling around x'
        perturbations, binary_labels = self.perturb_image(image, segments, n_samples)
        
        # 4. Predict perturbed images to get f(z)
        predictions = self.predict_batch(perturbations, target_class)
        
        # 5. Compute similarity kernel πx(z)
        similarities = self.similarity_kernel(binary_labels)
        
        # 6. Fit sparse linear model using K-Lasso
        #    with z'i as features, f(z) as target, and πx(z) as sample weights
        weights, intercept = self.fit_lasso_model(binary_labels, predictions, similarities, K)
        
        return weights, segments, intercept
    
    def visualize_explanation(self, image, weights, segments, K=5):
        """
        Visualize LIME explanation
        
        Args:
            image: Input PIL image
            weights: Feature importance weights
            segments: Superpixel segmentation mask
            K: Number of top features to highlight
            
        Returns:
            fig: Matplotlib figure object
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        
        # Original image with heatmap overlay
        plt.subplot(2, 2, 1)
        
        # Normalize weights to [0,1] range for visualization
        abs_weights = np.abs(weights)
        max_weight = np.max(abs_weights)
        if max_weight > 0:
            norm_weights = abs_weights / max_weight
        else:
            norm_weights = abs_weights
            
        # Create heatmap
        heatmap = np.zeros(segments.shape)
        for i in range(len(weights)):
            heatmap[segments == i] = norm_weights[i]
        
        # Show original image
        plt.imshow(image)
        
        # Overlay heatmap with jet colormap
        plt.imshow(heatmap, alpha=0.5, cmap='jet')
        plt.title("LIME Explanation Heatmap")
        plt.axis('off')
        
        # Classic LIME visualization - important regions in original image, others grayed out
        plt.subplot(2, 2, 2)
        
        # Get top K features by absolute weight
        if K is not None and K < len(weights):
            top_indices = np.argsort(-np.abs(weights))[:K]
        else:
            top_indices = np.argsort(-np.abs(weights))
        
        # Create mask for important regions
        important_mask = np.zeros_like(segments, dtype=bool)
        for idx in top_indices:
            important_mask = important_mask | (segments == idx)
        
        # Create visualization - keep important regions, gray out others
        masked_img = np.array(image).copy()
        # Gray out unimportant regions
        masked_img[~important_mask] = 128
        
        # Add segment boundaries
        masked_img = mark_boundaries(masked_img, segments, color=(1,0,0))
        
        plt.imshow(masked_img)
        plt.title("Important Regions (Others Masked)")
        plt.axis('off')
        
        # Show positive and negative contributions separately
        plt.subplot(2, 2, 3)
        
        # Create separate masks for positive and negative influence
        positive_mask = np.zeros_like(segments, dtype=bool)
        negative_mask = np.zeros_like(segments, dtype=bool)
        
        # Create masks for positive and negative contributions
        for idx in top_indices:
            mask = (segments == idx)
            if weights[idx] > 0:
                positive_mask = positive_mask | mask
            elif weights[idx] < 0:
                negative_mask = negative_mask | mask
        
        # Create visualizations for positive and negative regions
        pos_neg_img = np.array(image).copy()
        
        # Gray out non-important regions
        pos_neg_img[~(positive_mask | negative_mask)] = 128
        
        # Mark positive regions with green boundaries
        pos_neg_img = mark_boundaries(pos_neg_img, segments, 
                                    outline_color=(0,1,0), 
                                    color=(0,1,0),
                                    mode='thick')
        
        plt.imshow(pos_neg_img)
        plt.title("Positive (Green) and Negative (Red) Influence")
        plt.axis('off')
        
        # Add segment boundaries
        plt.subplot(2, 2, 4)
        boundaries_img = mark_boundaries(np.array(image), segments)
        plt.imshow(boundaries_img)
        plt.title("Superpixel Segmentation")
        plt.axis('off')
        
        # Add a custom legend
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        
        legend_elements = [
            Patch(facecolor='green', alpha=0.6, label='Positive influence'),
            Patch(facecolor='red', alpha=0.6, label='Negative influence')
        ]
        
        # Add legend to the last subplot
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        return fig


# Example usage
def load_example_vit_model():
    """
    Example function to load a pretrained ViT model
    (Replace with your own ViT model loading code)
    """
    try:
        # Try to load a ViT model from torchvision
        model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return model, preprocess
    except:
        print("Failed to load pretrained ViT model, please ensure dependencies are correctly installed or provide your own model")
        return None, None

def example_lime_explanation():
    """
    Example function for LIME explanation
    """
    model, preprocess = load_example_vit_model()
    if model is None:
        return
    
    # Load example image
    try:
        image_path = "Interpretable20250403/20241130test.jpg"  # Test image in repository
        image = Image.open(image_path).convert('RGB')
    except:
        print(f"Failed to load example image: {image_path}")
        return
    
    # Create LIME explainer
    lime = LIME(model, preprocess)
    
    # Generate explanation
    weights, segments, intercept = lime.explain(image, n_segments=50, n_samples=1000, K=5)
    
    # Visualize explanation
    fig = lime.visualize_explanation(image, weights, segments, K=5)
    
    # Save results
    output_path = "Interpretable20250403/results/lime_explanation.png"
    fig.savefig(output_path)
    print(f"LIME explanation saved to: {output_path}")


if __name__ == "__main__":
    example_lime_explanation() 