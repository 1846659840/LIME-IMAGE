"""
Interactive LIME Explanation Comparison Tool

This script allows users to:
1. View superpixel segmentation of an image
2. Interactively select superpixels they believe are important for classification
3. Compare their selection with LIME's model explanation
4. Visualize the comparison results
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
from matplotlib.widgets import Button
from skimage.segmentation import mark_boundaries
import matplotlib.patches as mpatches
from collections import deque
import importlib.util
from scipy.stats import spearmanr

# Ensure the path is relative to where the script is run from
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our LIME implementation
from Interpretable20250403.lime_for_vit import LIME, load_example_vit_model

# Try to import vit_lime_example
try:
    vit_lime_module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vit_lime_example.py")
    spec = importlib.util.spec_from_file_location("vit_lime_example", vit_lime_module_path)
    vit_lime_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vit_lime_module)
    VIT_LIME_EXAMPLE_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not import vit_lime_example module: {e}")
    VIT_LIME_EXAMPLE_AVAILABLE = False


class HumanLimeComparison:
    """
    Interactive tool to compare human-selected superpixels with LIME explanations
    """
    def __init__(self, model, preprocess, image, segments, target_class, device='cpu'):
        """
        Initialize the comparison tool
        
        Args:
            model: The ViT model
            preprocess: Preprocessing function for the model
            image: The input image (PIL Image)
            segments: Superpixel segmentation of the image
            target_class: Target class for explanation
            device: Computing device
        """
        self.model = model
        self.preprocess = preprocess
        self.image = image
        self.segments = segments
        self.target_class = target_class
        self.device = device
        
        # Convert image to numpy array for display
        self.img_array = np.array(image)
        
        # Number of superpixels
        self.n_segments = np.max(segments) + 1
        
        # Initialize human selection
        self.human_selection = np.zeros(self.n_segments, dtype=bool)
        
        # History for undo functionality
        self.selection_history = deque(maxlen=20)  # Store up to 20 previous states
        self.selection_history.append(self.human_selection.copy())
        
        # Cache of segment information
        self.segment_areas = {}
        for i in range(self.n_segments):
            self.segment_areas[i] = np.sum(segments == i)
    
    def get_segment_at_position(self, x, y):
        """Get the segment index at the specified position"""
        if 0 <= y < self.segments.shape[0] and 0 <= x < self.segments.shape[1]:
            return self.segments[int(y), int(x)]
        return -1
    
    def run_interactive_selection(self, max_selections=5):
        """
        Run the interactive segment selection tool
        
        Args:
            max_selections: Maximum number of segments the user can select
        """
        self.max_selections = max_selections
        self.selected_count = 0
        
        # Create interactive figure
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.suptitle(f"Select up to {max_selections} superpixels you think are most important\n"
                         f"for classifying this image as class {self.target_class}")
        
        # Show the original image with segment boundaries
        boundaries_img = mark_boundaries(self.img_array, self.segments)
        self.img_display = self.ax.imshow(boundaries_img)
        
        # Add instructions text
        self.ax.set_title(f"Click to select segments (0/{max_selections} selected)")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Create a "Done" button
        ax_done = plt.axes([0.8, 0.05, 0.1, 0.075])
        self.done_button = Button(ax_done, 'Done')
        self.done_button.on_clicked(self.on_done_clicked)
        
        # Create a "Reset" button
        ax_reset = plt.axes([0.65, 0.05, 0.1, 0.075])
        self.reset_button = Button(ax_reset, 'Reset')
        self.reset_button.on_clicked(self.on_reset_clicked)
        
        # Create an "Undo" button
        ax_undo = plt.axes([0.5, 0.05, 0.1, 0.075])
        self.undo_button = Button(ax_undo, 'Undo')
        self.undo_button.on_clicked(self.on_undo_clicked)
        
        # Connect the click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        plt.tight_layout()
        plt.show()
    
    def on_click(self, event):
        """Handle click events on the image"""
        if event.inaxes != self.ax:
            return
        
        # Get the segment at the clicked position
        segment_idx = self.get_segment_at_position(event.xdata, event.ydata)
        
        if segment_idx < 0 or segment_idx >= self.n_segments:
            return
        
        # Save current state for undo
        self.selection_history.append(self.human_selection.copy())
        
        # Toggle the segment selection
        if self.human_selection[segment_idx]:
            # If already selected, deselect it
            self.human_selection[segment_idx] = False
            self.selected_count -= 1
        elif self.selected_count < self.max_selections:
            # If not selected and under the max limit, select it
            self.human_selection[segment_idx] = True
            self.selected_count += 1
        
        # Update the display
        self._update_display()
    
    def _update_display(self):
        """Update the display with the current selection"""
        # Create a mask for selected segments
        selected_mask = np.zeros_like(self.segments, dtype=bool)
        for i in range(self.n_segments):
            if self.human_selection[i]:
                selected_mask[self.segments == i] = True
        
        # Create a new image with boundaries and highlights
        boundaries_img = mark_boundaries(self.img_array, self.segments)
        highlighted_img = boundaries_img.copy()
        
        # Add a green overlay for selected segments on the original image
        for i in range(3):
            channel = highlighted_img[:, :, i].copy()
            # Add green tint to selected areas (reduce red and blue, increase green)
            if i == 1:  # Green channel
                channel[selected_mask] = np.clip(channel[selected_mask] * 1.5, 0, 1)
            else:  # Red and blue channels
                channel[selected_mask] = channel[selected_mask] * 0.7
            highlighted_img[:, :, i] = channel
        
        self.img_display.set_data(highlighted_img)
        self.ax.set_title(f"Click to select segments ({self.selected_count}/{self.max_selections} selected)")
        self.fig.canvas.draw_idle()
    
    def on_done_clicked(self, event):
        """Handle the Done button click"""
        plt.close(self.fig)
    
    def on_reset_clicked(self, event):
        """Handle the Reset button click"""
        # Save current state for undo
        self.selection_history.append(self.human_selection.copy())
        
        # Reset selections
        self.human_selection = np.zeros(self.n_segments, dtype=bool)
        self.selected_count = 0
        self._update_display()
    
    def on_undo_clicked(self, event):
        """Handle the Undo button click"""
        if len(self.selection_history) > 1:
            # Pop the current state
            self.selection_history.pop()
            
            # Restore the previous state
            if hasattr(self, 'selection_order'):
                # For advanced selection mode, restore both selection and order
                if isinstance(self.selection_history[-1], tuple):
                    self.human_selection, self.selection_order = self.selection_history[-1]
                    self.selected_count = len(self.selection_order)
                    self._update_ordered_display()
                else:
                    # Fallback for basic selection mode
                    self.human_selection = self.selection_history[-1].copy()
                    self.selected_count = np.sum(self.human_selection)
                    self._update_display()
            else:
                # For basic selection mode
                self.human_selection = self.selection_history[-1].copy()
                self.selected_count = np.sum(self.human_selection)
                self._update_display()
    
    def compute_lime_explanation(self, n_samples=1000, K=5):
        """
        Compute LIME explanation for the image using vit_lime_example.py if available,
        otherwise fall back to direct LIME implementation
        
        Args:
            n_samples: Number of perturbed samples
            K: Number of top features for explanation
            
        Returns:
            weights: Feature weights from LIME
            intercept: Intercept term from LIME
        """
        print("Computing LIME explanation (this may take a few minutes)...")
        
        # Use vit_lime_example.py implementation if available
        if VIT_LIME_EXAMPLE_AVAILABLE:
            try:
                # Create a new LIME explainer
                lime = LIME(self.model, self.preprocess, device=self.device)
                
                # Use the same parameters as in vit_lime_example.py
                return lime.explain(
                    self.image, 
                    target_class=self.target_class,
                    n_segments=len(np.unique(self.segments)), # Use the same number of segments
                    n_samples=n_samples,
                    K=K
                )
            except Exception as e:
                print(f"Failed to use vit_lime_example implementation: {e}")
                print("Falling back to direct LIME implementation")
        
        # Direct LIME implementation (fallback)
        # Create LIME explainer
        lime = LIME(self.model, self.preprocess, device=self.device)
        
        # We already have the segments, so we'll modify the process
        # to skip the segmentation step
        
        # Generate perturbed samples
        perturbations, binary_labels = lime.perturb_image(self.image, self.segments, n_samples)
        
        # Predict perturbed images
        predictions = lime.predict_batch(perturbations, self.target_class)
        
        # Compute similarity kernel
        similarities = lime.similarity_kernel(binary_labels)
        
        # Fit sparse linear model
        weights, intercept = lime.fit_lasso_model(binary_labels, predictions, similarities, K)
        
        return weights, self.segments, intercept
    
    def compare_and_visualize(self, top_k=5, plot_results=True, save_dir=None):
        """
        Compare human selection with LIME explanation and visualize results
        
        Args:
            top_k: Number of top segments to consider from LIME
            plot_results: Whether to display the visualization
            save_dir: Directory to save visualization results (default: None)
            
        Returns:
            dict: Dictionary with comparison results
        """
        # Create results directory if it doesn't exist and save_dir is not specified
        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
            os.makedirs(save_dir, exist_ok=True)
        
        # Get top K indices from LIME explanation
        lime_indices = self.get_top_indices(self.lime_weights, top_k)
        
        # Get human selected indices and their order
        human_indices = []
        human_ordered_indices = []
        
        # First check if we have a selection order (from advanced_selection_mode)
        if hasattr(self, 'selection_order') and len(self.selection_order) > 0:
            human_ordered_indices = self.selection_order[:top_k]
            human_indices = human_ordered_indices
        else:
            # If no selection order is available, get indices of selected segments
            human_indices = np.where(self.human_selection)[0].tolist()[:top_k]
        
        # Calculate accuracy metrics
        intersection = set(lime_indices) & set(human_indices)
        union = set(lime_indices) | set(human_indices)
        
        # If both sets are empty, set metrics to 1.0 (perfect agreement on no features)
        if len(union) == 0:
            precision = recall = f1_score = jaccard = 1.0
        else:
            precision = len(intersection) / len(human_indices) if len(human_indices) > 0 else 0.0
            recall = len(intersection) / len(lime_indices) if len(lime_indices) > 0 else 0.0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0
        
        # Calculate rank correlation if we have ordered human selections
        rank_correlation = None
        if len(human_ordered_indices) > 0:
            # Get the ranks in LIME for the human-selected indices
            lime_ranks = {}
            for idx, seg_idx in enumerate(np.argsort(np.abs(self.lime_weights))[::-1]):
                lime_ranks[seg_idx] = idx + 1
            
            # Get the ranks for human selections
            human_ranks = {seg_idx: idx + 1 for idx, seg_idx in enumerate(human_ordered_indices)}
            
            # Calculate rank correlation for indices that appear in both
            common_indices = set(human_ordered_indices) & set(lime_ranks.keys())
            if len(common_indices) >= 2:  # Need at least 2 points for correlation
                human_rank_values = [human_ranks[idx] for idx in common_indices]
                lime_rank_values = [lime_ranks[idx] for idx in common_indices]
                
                # Use Spearman's rank correlation
                rank_correlation, _ = spearmanr(human_rank_values, lime_rank_values)
            
        # Create results dictionary
        results = {
            'lime_indices': lime_indices,
            'human_indices': human_indices,
            'intersection': list(intersection),
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'jaccard': jaccard,
            'rank_correlation': rank_correlation
        }
        
        # Visualize the comparison if requested
        if plot_results:
            # Create a figure with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # 1. Human selection visualization
            human_overlay = self.img_array.copy()
            human_mask = np.zeros_like(self.segments, dtype=bool)
            
            for idx in human_indices:
                human_mask = human_mask | (self.segments == idx)
            
            # Gray out non-selected regions
            human_overlay_grayed = self.img_array.copy()
            human_overlay_grayed[~human_mask] = human_overlay_grayed[~human_mask] * 0.3
            
            # Apply boundaries for better visualization
            human_with_boundaries = mark_boundaries(human_overlay_grayed, self.segments, color=(1,0,0))
            axes[0].imshow(human_with_boundaries)
            
            # Add order numbers if we have them
            if len(human_ordered_indices) > 0:
                for order_idx, segment_idx in enumerate(human_ordered_indices):
                    mask = (self.segments == segment_idx)
                    y_indices, x_indices = np.where(mask)
                    if len(y_indices) > 0:
                        center_y = np.mean(y_indices)
                        center_x = np.mean(x_indices)
                        axes[0].text(center_x, center_y, str(order_idx+1), 
                                    color='white', fontsize=14, ha='center', va='center',
                                    bbox=dict(facecolor='blue', alpha=0.8, pad=3, boxstyle="round,pad=0.3"))
            
            axes[0].set_title(f"Human Selection (Top {len(human_indices)})")
            axes[0].axis('off')
            
            # 2. LIME explanation visualization
            # Create a heatmap based on segment weights
            lime_weights_normalized = {}
            positive_lime_weights = {}
            negative_lime_weights = {}
            
            # Normalize weights and separate positive/negative
            for seg_idx, weight in enumerate(self.lime_weights):
                lime_weights_normalized[seg_idx] = weight
                if weight > 0:
                    positive_lime_weights[seg_idx] = weight
                elif weight < 0:
                    negative_lime_weights[seg_idx] = weight
            
            # Sort by absolute value
            positive_sorted = sorted(positive_lime_weights.items(), key=lambda x: x[1], reverse=True)
            negative_sorted = sorted(negative_lime_weights.items(), key=lambda x: x[1])
            
            # Get top K positive and negative indices
            top_positive = [idx for idx, _ in positive_sorted[:min(top_k, len(positive_sorted))]]
            top_negative = [idx for idx, _ in negative_sorted[:min(top_k, len(negative_sorted))]]
            
            # Create visualization for LIME explanation
            lime_overlay = self.img_array.copy()
            lime_mask_pos = np.zeros_like(self.segments, dtype=bool)
            lime_mask_neg = np.zeros_like(self.segments, dtype=bool)
            
            for idx in top_positive:
                lime_mask_pos = lime_mask_pos | (self.segments == idx)
                
            for idx in top_negative:
                lime_mask_neg = lime_mask_neg | (self.segments == idx)
                
            # Combined mask
            lime_mask = lime_mask_pos | lime_mask_neg
            
            # Gray out unimportant regions
            lime_overlay[~lime_mask] = lime_overlay[~lime_mask] * 0.3
            
            # Create an overlay with boundaries
            lime_with_boundaries = mark_boundaries(lime_overlay, self.segments, color=(1,0,0))
            
            # Add rank numbers to LIME segments
            for rank, idx in enumerate(lime_indices):
                mask = (self.segments == idx)
                # Find centroid of the segment for placing the number
                y_indices, x_indices = np.where(mask)
                if len(y_indices) > 0:
                    center_y = np.mean(y_indices)
                    center_x = np.mean(x_indices)
                    
                    # Different color for positive vs negative
                    color = 'red' if idx in top_positive else 'blue'
                    
                    # Add order number with appropriate background
                    axes[1].text(center_x, center_y, str(rank+1), 
                             color='white', fontsize=14, ha='center', va='center',
                             bbox=dict(facecolor=color, alpha=0.8, pad=3, boxstyle="round,pad=0.3"))
            
            axes[1].imshow(lime_with_boundaries)
            axes[1].set_title(f"LIME Explanation (Top {len(lime_indices)})")
            axes[1].axis('off')
            
            # 3. Comparison visualization (overlay of both)
            comparison = self.img_array.copy()
            
            # Create masks for different categories
            human_only_mask = np.zeros_like(self.segments, dtype=bool)
            lime_only_mask = np.zeros_like(self.segments, dtype=bool)
            both_mask = np.zeros_like(self.segments, dtype=bool)
            
            for h_idx in human_indices:
                h_mask = (self.segments == h_idx)
                for l_idx in lime_indices:
                    l_mask = (self.segments == l_idx)
                    if h_idx == l_idx:
                        both_mask = both_mask | h_mask
                    else:
                        human_only_mask = human_only_mask | (h_mask & ~l_mask)
                        lime_only_mask = lime_only_mask | (l_mask & ~h_mask)
            
            # Gray out unimportant regions
            all_masks = human_only_mask | lime_only_mask | both_mask
            comparison[~all_masks] = comparison[~all_masks] * 0.3
            
            # Create a version with colored boundaries
            comparison_with_boundaries = mark_boundaries(comparison, self.segments, color=(1,1,1))
            axes[2].imshow(comparison_with_boundaries)
            
            # Add custom legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', alpha=0.7, label='Both'),
                Patch(facecolor='blue', alpha=0.7, label='Human Only'),
                Patch(facecolor='red', alpha=0.7, label='LIME Only')
            ]
            axes[2].legend(handles=legend_elements, loc='upper right')
            
            # Add human selection order numbers
            if len(human_ordered_indices) > 0:
                for order_idx, segment_idx in enumerate(human_ordered_indices):
                    if segment_idx in lime_indices:
                        color = 'green'  # Both human and LIME
                    else:
                        color = 'blue'   # Human only
                        
                    mask = (self.segments == segment_idx)
                    y_indices, x_indices = np.where(mask)
                    if len(y_indices) > 0:
                        center_y = np.mean(y_indices)
                        center_x = np.mean(x_indices)
                        axes[2].text(center_x, center_y, f"H{order_idx+1}", 
                                    color='white', fontsize=12, ha='center', va='center',
                                    bbox=dict(facecolor=color, alpha=0.8, pad=2, boxstyle="round,pad=0.3"))
            
            # Add LIME rank numbers
            for rank, idx in enumerate(lime_indices):
                if idx not in human_indices:
                    mask = (self.segments == idx)
                    y_indices, x_indices = np.where(mask)
                    if len(y_indices) > 0:
                        center_y = np.mean(y_indices)
                        center_x = np.mean(x_indices)
                        axes[2].text(center_x, center_y, f"L{rank+1}", 
                                    color='white', fontsize=12, ha='center', va='center',
                                    bbox=dict(facecolor='red', alpha=0.8, pad=2, boxstyle="round,pad=0.3"))
            
            # Set title with metrics
            metrics_text = f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1_score:.2f}, Jaccard: {jaccard:.2f}"
            if rank_correlation is not None:
                metrics_text += f", Rank Correlation: {rank_correlation:.2f}"
                
            axes[2].set_title(f"Comparison\n{metrics_text}")
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # Save comparison visualization
            comparison_path = os.path.join(save_dir, "human_lime_comparison.png")
            plt.savefig(comparison_path)
            print(f"Main comparison visualization saved to: {comparison_path}")
            
            plt.show()
            
            # Create additional visualizations and save them
            
            # 1. Side-by-side comparison with rankings
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(f"Human Selection vs LIME Explanation with Rankings", fontsize=16)
            
            # Human selection with rankings
            human_overlay_ranked = self.img_array.copy()
            human_overlay_ranked[~human_mask] = human_overlay_ranked[~human_mask] * 0.3
            human_overlay_ranked = mark_boundaries(human_overlay_ranked, self.segments)
            
            axes[0].imshow(human_overlay_ranked)
            axes[0].set_title("Human Selection with Rankings")
            axes[0].axis('off')
            
            # Add numbers to human selections
            if len(human_ordered_indices) > 0:
                for order_idx, segment_idx in enumerate(human_ordered_indices):
                    mask = (self.segments == segment_idx)
                    y_indices, x_indices = np.where(mask)
                    if len(y_indices) > 0:
                        center_y = np.mean(y_indices)
                        center_x = np.mean(x_indices)
                        axes[0].text(center_x, center_y, str(order_idx+1), 
                                 color='white', fontsize=14, ha='center', va='center',
                                 bbox=dict(facecolor='blue', alpha=0.8, pad=3, boxstyle="round,pad=0.3"))
            
            # LIME explanation with rankings
            lime_overlay_ranked = self.img_array.copy()
            lime_overlay_ranked[~lime_mask] = lime_overlay_ranked[~lime_mask] * 0.3
            lime_overlay_ranked = mark_boundaries(lime_overlay_ranked, self.segments)
            
            axes[1].imshow(lime_overlay_ranked)
            axes[1].set_title("LIME Explanation with Rankings")
            axes[1].axis('off')
            
            # Add numbers to LIME selections
            for rank, idx in enumerate(lime_indices):
                mask = (self.segments == idx)
                y_indices, x_indices = np.where(mask)
                if len(y_indices) > 0:
                    center_y = np.mean(y_indices)
                    center_x = np.mean(x_indices)
                    color = 'red' if idx in top_positive else 'blue'
                    axes[1].text(center_x, center_y, str(rank+1), 
                             color='white', fontsize=14, ha='center', va='center',
                             bbox=dict(facecolor=color, alpha=0.8, pad=3, boxstyle="round,pad=0.3"))
            
            plt.tight_layout()
            ranked_path = os.path.join(save_dir, "human_lime_ranked_comparison.png")
            plt.savefig(ranked_path)
            print(f"Ranked comparison visualization saved to: {ranked_path}")
            plt.show()
            
            # 2. Detailed LIME explanation with positive and negative influences
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f"Detailed LIME Explanation", fontsize=16)
            
            # Original image
            axes[0].imshow(self.img_array)
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            
            # Positive influence regions
            positive_img = self.img_array.copy()
            positive_mask = np.zeros_like(self.segments, dtype=bool)
            for idx in top_positive:
                positive_mask = positive_mask | (self.segments == idx)
                
            positive_img[~positive_mask] = positive_img[~positive_mask] * 0.3
            positive_img = mark_boundaries(positive_img, self.segments, color=(0,1,0))
            
            axes[1].imshow(positive_img)
            axes[1].set_title("Positive Influence Regions")
            axes[1].axis('off')
            
            # Negative influence regions
            negative_img = self.img_array.copy()
            negative_mask = np.zeros_like(self.segments, dtype=bool)
            for idx in top_negative:
                negative_mask = negative_mask | (self.segments == idx)
                
            negative_img[~negative_mask] = negative_img[~negative_mask] * 0.3
            negative_img = mark_boundaries(negative_img, self.segments, color=(1,0,0))
            
            axes[2].imshow(negative_img)
            axes[2].set_title("Negative Influence Regions")
            axes[2].axis('off')
            
            plt.tight_layout()
            influence_path = os.path.join(save_dir, "lime_positive_negative_regions.png")
            plt.savefig(influence_path)
            print(f"Influence regions visualization saved to: {influence_path}")
            plt.show()
            
            # 3. Top segments with weights
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(f"Top {top_k} Most Important Segments", fontsize=16)
            
            # Original image with numbered LIME segments
            lime_detailed = self.img_array.copy()
            lime_detailed[~lime_mask] = lime_detailed[~lime_mask] * 0.3
            marked_img = mark_boundaries(lime_detailed, self.segments)
            
            axes[0].imshow(marked_img)
            axes[0].set_title("Important Regions with Segment Numbers")
            axes[0].axis('off')
            
            # Add numbers to the LIME segments
            for rank, idx in enumerate(lime_indices):
                mask = (self.segments == idx)
                y_indices, x_indices = np.where(mask)
                if len(y_indices) > 0:
                    center_y = np.mean(y_indices)
                    center_x = np.mean(x_indices)
                    color = 'red' if idx in top_positive else 'blue'
                    axes[0].text(center_x, center_y, str(rank+1), 
                             color='white', fontsize=14, ha='center', va='center',
                             bbox=dict(facecolor=color, alpha=0.8, pad=3, boxstyle="round,pad=0.3"))
            
            # Barchart showing weights of top segments
            segment_indices = lime_indices.copy()
            segment_weights = [self.lime_weights[idx] for idx in segment_indices]
            
            # Create a horizontal bar chart that properly shows positive and negative values
            y_pos = np.arange(len(segment_indices))
            
            # Split positive and negative weights for different coloring
            positive_weights = [max(0, w) for w in segment_weights]
            negative_weights = [min(0, w) for w in segment_weights]
            
            # Plot positive weights in green
            pos_bars = axes[1].barh(y_pos, positive_weights, color='green', alpha=0.7, label='Positive influence')
            # Plot negative weights in red
            neg_bars = axes[1].barh(y_pos, negative_weights, color='red', alpha=0.7, label='Negative influence')
            
            # Add a vertical line at x=0 to better visualize the direction
            axes[1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
            
            # Add labels and titles
            axes[1].set_yticks(y_pos)
            axes[1].set_yticklabels([f"Segment {idx}" for idx in segment_indices])
            axes[1].set_xlabel('Weight')
            axes[1].set_title('LIME Weights for Top Segments')
            axes[1].legend()
            
            # Add segment size information and ensure it appears on the correct side
            for i, idx in enumerate(segment_indices):
                segment_size = np.sum(self.segments == idx)
                total_size = self.segments.size
                percentage = (segment_size / total_size) * 100
                weight = self.lime_weights[idx]
                
                # Position text based on weight direction - Remove percentage text
                # if weight > 0:
                #     text_pos = weight * 1.1
                #     ha_align = 'left'
                # else:
                #     text_pos = weight * 1.1
                #     ha_align = 'right'
                
                # axes[1].text(text_pos, i, 
                #          f"{percentage:.1f}% of image", 
                #          va='center',
                #          ha=ha_align,
                #          fontsize=10)
            
            plt.tight_layout()
            weights_path = os.path.join(save_dir, "lime_weights_analysis.png")
            plt.savefig(weights_path)
            print(f"Weights analysis visualization saved to: {weights_path}")
            plt.show()
            
            # Additional: Save human vs LIME selected segments as CSV for further analysis
            import pandas as pd
            
            # Create DataFrame with segment details
            segment_data = []
            all_segments = set(human_indices) | set(lime_indices)
            
            for idx in all_segments:
                segment_size = np.sum(self.segments == idx)
                total_size = self.segments.size
                percentage = (segment_size / total_size) * 100
                
                in_human = idx in human_indices
                in_lime = idx in lime_indices
                lime_weight = self.lime_weights[idx] if idx < len(self.lime_weights) else 0
                human_rank = human_ordered_indices.index(idx) + 1 if idx in human_ordered_indices else None
                lime_rank = lime_indices.index(idx) + 1 if idx in lime_indices else None
                
                segment_data.append({
                    'segment_id': idx,
                    'size_percentage': percentage,
                    'in_human_selection': in_human,
                    'in_lime_selection': in_lime, 
                    'lime_weight': lime_weight,
                    'human_rank': human_rank,
                    'lime_rank': lime_rank
                })
            
            df = pd.DataFrame(segment_data)
            csv_path = os.path.join(save_dir, "segment_analysis.csv")
            df.to_csv(csv_path, index=False)
            print(f"Segment analysis data saved to: {csv_path}")
        
        return results

    def _update_ordered_display(self):
        """Update the display with ordered selection indicators"""
        # Create a new image with boundaries
        boundaries_img = mark_boundaries(self.img_array, self.segments)
        highlighted_img = boundaries_img.copy()
        
        # Create a mask for all selected segments
        selected_mask = np.zeros_like(self.segments, dtype=bool)
        for segment_idx in self.selection_order:
            selected_mask = selected_mask | (self.segments == segment_idx)
        
        # Gray out non-selected areas to make selection more visible
        base_img = self.img_array.copy()
        base_img[~selected_mask] = 128
        highlighted_img = mark_boundaries(base_img, self.segments, color=(0,1,0), outline_color=(0,1,0))
        
        # Add numbers and highlighting for each selection in order
        for order_idx, segment_idx in enumerate(self.selection_order):
            mask = (self.segments == segment_idx)
            
            # Find centroid of the segment for placing the number
            y_indices, x_indices = np.where(mask)
            if len(y_indices) > 0:
                center_y = np.mean(y_indices)
                center_x = np.mean(x_indices)
                
                # Add order number with blue background (different from LIME colors)
                self.ax.text(center_x, center_y, str(order_idx+1), 
                         color='white', fontsize=14, ha='center', va='center',
                         bbox=dict(facecolor='blue', alpha=0.8, pad=3, boxstyle="round,pad=0.3"))
        
        self.img_display.set_data(highlighted_img)
        
        # Update title based on selection mode
        if self.max_selections is None:
            self.ax.set_title(f"Selected {self.selected_count} segments in order of importance")
        else:
            self.ax.set_title(f"Selected {self.selected_count}/{self.max_selections} segments in order of importance")
            
        self.fig.canvas.draw_idle()
    
    def advanced_selection_mode(self, max_selections=None):
        """
        Run an advanced selection mode that tracks selection order.
        If max_selections is None, users can select as many as they want.
        
        Args:
            max_selections: Maximum number of segments the user can select, or None for unlimited
        """
        self.max_selections = max_selections
        self.selected_count = 0
        self.selection_order = []  # Track the order of selections
        
        # Create interactive figure
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        if max_selections is None:
            self.fig.suptitle(f"Select superpixels in order of importance (most important first)\n"
                            f"for classifying this image as class {self.target_class}")
            instruction = "Click to select segments in order of importance.\nSelect as many segments as you think are important."
        else:
            self.fig.suptitle(f"Select up to {max_selections} superpixels in order of importance\n"
                            f"for classifying this image as class {self.target_class}")
            instruction = f"Click to select segments in order (0/{max_selections} selected)"
        
        # Show the original image with segment boundaries
        boundaries_img = mark_boundaries(self.img_array, self.segments, mode='thick')
        self.img_display = self.ax.imshow(boundaries_img)
        
        # Add instructions text
        self.ax.set_title(instruction)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Create button panel with better layout
        button_panel_height = 0.075
        button_width = 0.15
        button_spacing = 0.02
        button_base_x = 0.5 - (button_width * 3 + button_spacing * 2) / 2
        
        # Create a "Done" button
        ax_done = plt.axes([button_base_x + (button_width + button_spacing) * 2, 0.05, button_width, button_panel_height])
        self.done_button = Button(ax_done, 'Done', color='lightgreen')
        self.done_button.on_clicked(self.on_done_clicked)
        
        # Create a "Reset" button
        ax_reset = plt.axes([button_base_x + (button_width + button_spacing), 0.05, button_width, button_panel_height])
        self.reset_button = Button(ax_reset, 'Reset', color='lightcoral')
        self.reset_button.on_clicked(self.on_reset_clicked)
        
        # Create an "Undo" button
        ax_undo = plt.axes([button_base_x, 0.05, button_width, button_panel_height])
        self.undo_button = Button(ax_undo, 'Undo', color='lightskyblue')
        self.undo_button.on_clicked(self.on_undo_clicked)
        
        # Add help text
        plt.figtext(0.5, 0.01, 
                   "Instructions: Click on segments in order of importance.\nUse 'Undo' to remove last selection, 'Reset' to start over, 'Done' when finished.",
                   ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
        
        # Connect the click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click_ordered)
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust layout to make room for buttons and help text
        plt.show()
        
        return self.selection_order
    
    def on_click_ordered(self, event):
        """Handle click events for ordered selection"""
        if event.inaxes != self.ax:
            return
        
        # Get the segment at the clicked position
        segment_idx = self.get_segment_at_position(event.xdata, event.ydata)
        
        if segment_idx < 0 or segment_idx >= self.n_segments:
            return
        
        # Save current state for undo
        self.selection_history.append((self.human_selection.copy(), self.selection_order.copy() if hasattr(self, 'selection_order') else []))
        
        # If already selected, ignore and show a message
        if self.human_selection[segment_idx]:
            # Display a temporary message about already selected segment
            old_title = self.ax.get_title()
            self.ax.set_title(f"Segment {segment_idx} already selected (#{self.selection_order.index(segment_idx)+1})")
            self.fig.canvas.draw_idle()
            # Reset the title after a short delay
            import threading
            def reset_title():
                import time
                time.sleep(1.5)
                self.ax.set_title(old_title)
                self.fig.canvas.draw_idle()
            threading.Thread(target=reset_title).start()
            return
            
        # If under the max limit (or unlimited), select it in order
        if self.max_selections is None or self.selected_count < self.max_selections:
            self.human_selection[segment_idx] = True
            self.selection_order.append(segment_idx)
            self.selected_count += 1
            
            # Play a sound to indicate selection (if available)
            try:
                import winsound
                frequency = 800  # Hz
                duration = 100   # milliseconds
                winsound.Beep(frequency, duration)
            except:
                pass  # Silently fail if sound not available
        else:
            # Display a temporary message about max selections reached
            old_title = self.ax.get_title()
            self.ax.set_title(f"Maximum selections ({self.max_selections}) reached")
            self.fig.canvas.draw_idle()
            # Reset the title after a short delay
            import threading
            def reset_title():
                import time
                time.sleep(1.5)
                self.ax.set_title(old_title)
                self.fig.canvas.draw_idle()
            threading.Thread(target=reset_title).start()
            return
        
        # Update the display with order numbers
        self._update_ordered_display()

    def get_top_indices(self, weights, top_k):
        """
        Get indices of top K features based on absolute weight values
        
        Args:
            weights: Feature weights
            top_k: Number of top features to select
            
        Returns:
            List of indices of top K features
        """
        top_indices = np.argsort(-np.abs(weights))[:top_k].tolist()
        return top_indices


def main():
    """Main function to run the interactive comparison"""
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
    print(f"Using most probable class {target_class} for explanation")
    
    # First, segment the image
    lime = LIME(model, preprocess, device=device)
    segments = lime.segment_image(image, n_segments=50, compactness=20)
    
    # Display segmentation for reference
    plt.figure(figsize=(10, 8))
    boundaries_img = mark_boundaries(np.array(image), segments)
    plt.imshow(boundaries_img)
    plt.title("Superpixel Segmentation (Reference)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "segmentation_reference.png"))
    plt.show()
    
    # Initialize the interactive comparison tool
    comparator = HumanLimeComparison(model, preprocess, image, segments, target_class, device)
    
    # Step 1: Let user select important regions with unlimited selection
    print(f"Please select superpixels that you think are important for classifying this image as class {target_class}...")
    print("Select them in order of importance (most important first). You can select as many as you want.")
    user_selections = comparator.advanced_selection_mode(max_selections=None)
    
    # Save the human selection visualization
    if hasattr(comparator, 'selection_order') and len(comparator.selection_order) > 0:
        plt.figure(figsize=(10, 8))
        selected_mask = np.zeros_like(segments, dtype=bool)
        for idx in comparator.selection_order:
            selected_mask = selected_mask | (segments == idx)
        
        human_selected_img = image.copy()
        human_selected_array = np.array(human_selected_img)
        human_selected_array[~selected_mask] = human_selected_array[~selected_mask] * 0.3
        
        # Add boundaries and numbers for order
        human_selected_boundaries = mark_boundaries(human_selected_array, segments, color=(0,1,0))
        plt.imshow(human_selected_boundaries)
        
        # Add numbers for order
        for order_idx, segment_idx in enumerate(comparator.selection_order):
            mask = (segments == segment_idx)
            y_indices, x_indices = np.where(mask)
            if len(y_indices) > 0:
                center_y = np.mean(y_indices)
                center_x = np.mean(x_indices)
                plt.text(center_x, center_y, str(order_idx+1), 
                     color='white', fontsize=14, ha='center', va='center',
                     bbox=dict(facecolor='blue', alpha=0.8, pad=3, boxstyle="round,pad=0.3"))
        
        plt.title(f"Human Selected Regions (Total: {len(comparator.selection_order)})")
        plt.axis('off')
        plt.tight_layout()
        human_selection_path = os.path.join(results_dir, "human_selection.png")
        plt.savefig(human_selection_path)
        print(f"Human selection visualization saved to: {human_selection_path}")
        plt.show()
    
    # Get number of user selections (this will determine how many top-k features LIME will show)
    num_selections = len(user_selections)
    
    if num_selections == 0:
        print("You didn't select any regions. Please run the program again and select at least one region.")
        return
    
    print(f"\nYou selected {num_selections} superpixels.")
    
    # Step 2: Compute LIME explanation with same number of features as user selections
    print(f"Computing LIME explanation for top {num_selections} features (this may take a few minutes)...")
    lime_weights, lime_segments, lime_intercept = comparator.compute_lime_explanation(n_samples=1000, K=num_selections)
    # Store LIME results for later use
    comparator.lime_weights = lime_weights
    
    # Save LIME weights to file
    import pandas as pd
    lime_weights_df = pd.DataFrame({
        'segment_id': range(len(lime_weights)),
        'weight': lime_weights
    })
    lime_weights_df = lime_weights_df.sort_values(by='weight', ascending=False)
    lime_weights_path = os.path.join(results_dir, "lime_weights.csv")
    lime_weights_df.to_csv(lime_weights_path, index=False)
    print(f"LIME weights saved to: {lime_weights_path}")
    
    # Save LIME explanation visualization 
    plt.figure(figsize=(10, 8))
    lime_indices = comparator.get_top_indices(lime_weights, num_selections)
    
    # Separate positive and negative indices based on weights
    positive_indices = [idx for idx in lime_indices if lime_weights[idx] > 0]
    negative_indices = [idx for idx in lime_indices if lime_weights[idx] <= 0]
    
    # Create masks for visualization
    lime_mask = np.zeros_like(segments, dtype=bool)
    lime_mask_pos = np.zeros_like(segments, dtype=bool)
    lime_mask_neg = np.zeros_like(segments, dtype=bool)
    
    # Create combined mask for all important segments
    for idx in lime_indices:
        lime_mask = lime_mask | (segments == idx)
    
    # Create separate masks for positive and negative segments
    for idx in positive_indices:
        lime_mask_pos = lime_mask_pos | (segments == idx)
    
    for idx in negative_indices:
        lime_mask_neg = lime_mask_neg | (segments == idx)
    
    # Prepare visualization
    lime_viz = np.array(image.copy())
    lime_viz[~lime_mask] = lime_viz[~lime_mask] * 0.3
    lime_viz = mark_boundaries(lime_viz, segments, color=(1,0,0))
    plt.imshow(lime_viz)
    
    # Add numbers for ranking with appropriate colors
    for rank, idx in enumerate(lime_indices):
        mask = (segments == idx)
        y_indices, x_indices = np.where(mask)
        if len(y_indices) > 0:
            center_y = np.mean(y_indices)
            center_x = np.mean(x_indices)
            weight = lime_weights[idx]
            # Use green for positive weights, red for negative weights
            color = 'green' if weight > 0 else 'red'
            plt.text(center_x, center_y, str(rank+1), 
                 color='white', fontsize=14, ha='center', va='center',
                 bbox=dict(facecolor=color, alpha=0.8, pad=3, boxstyle="round,pad=0.3"))
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Positive influence'),
        Patch(facecolor='red', alpha=0.7, label='Negative influence')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title(f"LIME Explanation (Top {num_selections} features)")
    plt.axis('off')
    plt.tight_layout()
    lime_viz_path = os.path.join(results_dir, "lime_explanation.png")
    plt.savefig(lime_viz_path)
    print(f"LIME explanation visualization saved to: {lime_viz_path}")
    plt.show()
    
    # Also save a separate visualization showing weights as bar chart
    plt.figure(figsize=(12, 8))
    plt.suptitle(f"LIME Weights for Top {num_selections} Segments", fontsize=16)
    
    # Get indices and weights
    y_pos = np.arange(len(lime_indices))
    segment_weights = [lime_weights[idx] for idx in lime_indices]
    
    # Split positive and negative weights for different coloring
    positive_weights = [max(0, w) for w in segment_weights]
    negative_weights = [min(0, w) for w in segment_weights]
    
    # Plot positive weights in green
    plt.barh(y_pos, positive_weights, color='green', alpha=0.7, label='Positive influence')
    # Plot negative weights in red
    plt.barh(y_pos, negative_weights, color='red', alpha=0.7, label='Negative influence')
    
    # Add a vertical line at x=0 to better visualize the direction
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Add labels and title
    plt.yticks(y_pos, [f"Segment {idx}" for idx in lime_indices])
    plt.xlabel('Weight')
    plt.legend()
    
    # Add percentage information
    for i, idx in enumerate(lime_indices):
        segment_size = np.sum(segments == idx)
        total_size = segments.size
        percentage = (segment_size / total_size) * 100
        weight = lime_weights[idx]
        
        # Position text based on weight direction - Remove percentage text
        # if weight > 0:
        #     text_pos = weight * 1.1
        #     ha_align = 'left'
        # else:
        #     text_pos = weight * 1.1
        #     ha_align = 'right'
        
        # plt.text(text_pos, i, 
        #      f"{percentage:.1f}% of image", 
        #      va='center',
        #      ha=ha_align,
        #      fontsize=10)
    
    plt.tight_layout()
    weights_viz_path = os.path.join(results_dir, "lime_weights_barchart.png")
    plt.savefig(weights_viz_path)
    print(f"LIME weights bar chart saved to: {weights_viz_path}")
    plt.show()
    
    # Step 3: Compare human selection with LIME explanation and visualize
    print("Comparing human selection with LIME explanation...")
    comparison_result = comparator.compare_and_visualize(top_k=num_selections, plot_results=True, save_dir=results_dir)
    
    # Print comparison summary
    print("\nComparison Summary:")
    print(f"Precision: {comparison_result['precision']:.2f}")
    print(f"Recall: {comparison_result['recall']:.2f}")
    print(f"F1 Score: {comparison_result['f1_score']:.2f}")
    print(f"Jaccard Index: {comparison_result['jaccard']:.2f}")
    if comparison_result['rank_correlation'] is not None:
        print(f"Rank Correlation: {comparison_result['rank_correlation']:.2f}")
    print(f"Matched segments: {len(comparison_result['intersection'])} / {len(comparison_result['human_indices'])}")
    
    # Save detailed results to text file
    with open(os.path.join(results_dir, "comparison_results.txt"), 'w') as f:
        f.write(f"Image: {image_path}\n")
        f.write(f"Target Class: {target_class}\n\n")
        f.write("Comparison Summary:\n")
        f.write(f"Precision: {comparison_result['precision']:.2f}\n")
        f.write(f"Recall: {comparison_result['recall']:.2f}\n")
        f.write(f"F1 Score: {comparison_result['f1_score']:.2f}\n")
        f.write(f"Jaccard Index: {comparison_result['jaccard']:.2f}\n")
        if comparison_result['rank_correlation'] is not None:
            f.write(f"Rank Correlation: {comparison_result['rank_correlation']:.2f}\n")
        f.write(f"Matched segments: {len(comparison_result['intersection'])} / {len(comparison_result['human_indices'])}\n\n")
        f.write(f"Human selected segments: {comparison_result['human_indices']}\n")
        f.write(f"LIME selected segments: {comparison_result['lime_indices']}\n")
        f.write(f"Segments in both selections: {comparison_result['intersection']}\n")
    
    print(f"All results have been saved to: {results_dir}")
    print("\nExperiment complete! Thank you for your participation!")


if __name__ == "__main__":
    main() 