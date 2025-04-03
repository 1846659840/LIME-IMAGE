# LIME for ViT Model Interpretation

This project implements LIME (Local Interpretable Model-agnostic Explanations) algorithm specifically designed for explaining predictions from ViT (Vision Transformer) models. LIME is a model-agnostic explanation method that works by perturbing the input and observing changes in the model's output to understand its decision making.

## Algorithm

The implementation follows Algorithm 1 from the original LIME paper "Why Should I Trust You?: Explaining the Predictions of Any Classifier":

```
Algorithm 1: Sparse Linear Explanations using LIME
Require: Classifier f, Number of samples N
Require: Instance x, and its interpretable version x'
Require: Similarity kernel πx, Length of explanation K
Z ← {}
for i ∈ {1, 2, 3, ..., N} do
    z'i ← sample_around(x')
    Z ← Z ∪ {z'i, f(zi), πx(zi)}
end for
w ← K-Lasso(Z, K) # with z'i as features, f(z) as target
return w
```

## Key Components

1. **Superpixel Segmentation**: Transforms the input image into an interpretable representation (x') by segmenting it into superpixels using the SLIC algorithm from the scikit-image library.

2. **Perturbation Sampling**: Generates perturbed samples (z'i) by randomly turning on/off superpixels in the image.

3. **Similarity Kernel (πx)**: Computes the proximity between the original instance and each perturbed sample using an exponential kernel.

4. **K-Lasso**: Fits a sparse linear model with at most K non-zero weights to explain the model's predictions, using similarity as sample weights.

5. **Visualization**: Displays the explanation as a heatmap, highlights the top K most important superpixels, and shows superpixel boundaries.

## Usage

```python
from lime_for_vit import LIME
from PIL import Image

# Load your ViT model
model, preprocess = load_your_vit_model()

# Create LIME explainer
lime = LIME(model, preprocess)

# Load image to explain
image = Image.open("your_image.jpg").convert('RGB')

# Generate explanation with K=5 (at most 5 features in explanation)
weights, segments, intercept = lime.explain(image, n_segments=50, n_samples=1000, K=5)

# Visualize explanation result
fig = lime.visualize_explanation(image, weights, segments, K=5)
fig.savefig("lime_explanation.png")
```

## Parameter Tuning

- `n_segments`: Number of segments to divide the image into. Larger values provide more fine-grained explanations.
- `n_samples`: Number of perturbed samples to generate. Larger values provide more stable explanations but increase computation time.
- `K`: Maximum number of features (superpixels) to include in the explanation. Controls the sparsity of the explanation.
- `compactness`: Parameter for SLIC algorithm that balances color proximity and space proximity. Higher values make superpixels more compact.

## Example Results

Running the example code will generate explanation visualizations in the `results` folder, including:
- Original image
- Superpixel segmentation with boundaries
- Perturbation examples
- LIME explanation heatmap
- Top K most important features highlighted
- Colored visualization of superpixel weights

## Dependencies

- NumPy and matplotlib for computation and visualization
- PyTorch for model inference
- scikit-learn for Lasso regression
- scikit-image for SLIC superpixel segmentation
- PIL for image processing

## References

- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?: Explaining the predictions of any classifier". ACM SIGKDD.
- Achanta, R., Shaji, A., Smith, K., Lucchi, A., Fua, P., & Süsstrunk, S. (2012). "SLIC Superpixels Compared to State-of-the-Art Superpixel Methods". IEEE Transactions on Pattern Analysis and Machine Intelligence. 