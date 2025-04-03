"""
LIME Explainer Example Script - for ViT Model Interpretability
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os
import sys

# Ensure the path is relative to where the script is run from
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our LIME implementation
from Interpretable20250403.lime_for_vit import LIME, load_example_vit_model

def main():
    """Run LIME explanation example"""
    # Ensure results folder exists
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Load pretrained ViT model
    print("Loading ViT model...")
    model, preprocess = load_example_vit_model()
    if model is None:
        print("Model loading failed, please check if necessary dependencies are installed")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Model loaded on {device} device")
    
    # Load test image
    try:
        # Try to find the test image in a few possible locations
        possible_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "20241130test.jpg"),
            "Interpretable20250403/20241130test.jpg",
            "20241130test.jpg"
        ]
        
        image_path = None
        for path in possible_paths:
            if os.path.exists(path):
                image_path = path
                break
                
        if image_path is None:
            print("Could not find test image. Please place '20241130test.jpg' in the working directory.")
            return
            
        print(f"Loading image: {image_path}")
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Failed to load image: {e}")
        return
    
    # Display original image
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title("Original Test Image")
    plt.axis('off')
    plt.savefig(os.path.join(results_dir, "original_image.png"))
    print("Original image saved to results/original_image.png")
    
    # Create LIME explainer
    lime = LIME(model, preprocess, device=device)
    
    # Get model predictions
    print("Getting model predictions...")
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    print("Top 5 predicted classes:")
    for i in range(5):
        print(f"Class {top5_catid[i].item()}: {top5_prob[i].item():.4f}")
    
    # Target class for explanation
    target_class = top5_catid[0].item()
    print(f"Using most probable class {target_class} for LIME explanation")
    
    # Generate LIME explanation with the sparse linear model approach (K-Lasso)
    print("Generating LIME explanation (this may take a few minutes)...")
    
    # Parameters for LIME
    n_segments = 50    # Number of superpixels to segment the image into
    n_samples = 200    # Number of perturbed samples to generate
    K = 5              # Number of features to include in the explanation (sparsity parameter)
    compactness = 20   # SLIC parameter: higher values make superpixels more compact
    
    # Run the LIME algorithm
    weights, segments, intercept = lime.explain(
        image, 
        target_class=target_class,
        n_segments=n_segments,
        n_samples=n_samples,
        K=K
    )
    print("LIME explanation generation complete")
    
    # Visualize segmentation result
    from skimage.segmentation import mark_boundaries
    
    # Create figure showing the segmentation boundaries
    plt.figure(figsize=(10, 5))
    img_array = np.array(image)
    boundaries = mark_boundaries(img_array, segments)
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(boundaries)
    plt.title('SLIC Superpixels')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "segmentation.png"))
    print("Segmentation result saved to results/segmentation.png")
    
    # Generate and save one perturbation example
    perturbations, _ = lime.perturb_image(image, segments, n_samples=1)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(perturbations[0])
    plt.title('Perturbed Sample')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "perturbation_example.png"))
    print("Perturbation example saved to results/perturbation_example.png")
    
    # Visualize LIME explanation results with top K features
    fig = lime.visualize_explanation(image, weights, segments, K=K)
    fig.savefig(os.path.join(results_dir, "lime_explanation.png"))
    print("LIME explanation saved to results/lime_explanation.png")
    
    # Check which features were selected by the sparse model
    nonzero_features = np.nonzero(weights)[0]
    if len(nonzero_features) > 0:
        print(f"\nSelected {len(nonzero_features)} features out of {len(weights)}")
        for idx in nonzero_features:
            print(f"Feature {idx}: weight = {weights[idx]:.4f}")
    else:
        print("No features were selected by the model.")
    
    print(f"Intercept (bias term): {intercept:.4f}")
    
    print("\nAll results saved to results/ folder")
    print("\nLIME explanation complete!")


if __name__ == "__main__":
    main() 