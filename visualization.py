"""
Visualization module for DICE-FER training and evaluation.
Generates publication-quality figures matching paper standards.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix
import torch


class DICEFERVisualizer:
    """
    Generates all figures referenced in the DICE-FER paper:
    - Training loss curves (Stage 1, Stage 2)
    - MI gap evolution
    - Confusion matrices
    - t-SNE visualizations of disentangled representations
    - Ablation study results
    """

    def __init__(self, save_dir='./results/figures'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        # Publication-quality settings
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10

    def plot_stage1_training_history(self, history, dataset_name='CK+48', fold_idx=0):
        """
        Plot Stage 1 (Expression Learning) training curves.
        
        Figure shows:
        - Total loss (MI - δL1)
        - MI loss component
        - L1 distance component
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        epochs = np.arange(1, len(history['total_loss']) + 1)

        # Total Loss
        axes[0].plot(epochs, history['total_loss'], 'b-', linewidth=2.5, label='Total Loss')
        axes[0].fill_between(epochs, history['total_loss'], alpha=0.3)
        axes[0].set_xlabel('Epoch', fontweight='bold')
        axes[0].set_ylabel('Loss', fontweight='bold')
        axes[0].set_title('Stage 1: Total Loss', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # MI Loss
        axes[1].plot(epochs, history['mi_loss'], 'g-', linewidth=2.5, label='MI Loss')
        axes[1].fill_between(epochs, history['mi_loss'], alpha=0.3, color='green')
        axes[1].set_xlabel('Epoch', fontweight='bold')
        axes[1].set_ylabel('MI Loss', fontweight='bold')
        axes[1].set_title('Stage 1: MI Maximization', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        # L1 Loss
        axes[2].plot(epochs, history['l1_loss'], 'r-', linewidth=2.5, label='L1 Loss')
        axes[2].fill_between(epochs, history['l1_loss'], alpha=0.3, color='red')
        axes[2].set_xlabel('Epoch', fontweight='bold')
        axes[2].set_ylabel('L1 Distance', fontweight='bold')
        axes[2].set_title('Stage 1: L1 Regularization', fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{dataset_name}_fold{fold_idx}_stage1_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()

    def plot_stage2_training_history(self, history, dataset_name='CK+48', fold_idx=0):
        """
        Plot Stage 2 (Identity Learning) training curves.
        
        Figure shows:
        - Total loss (MI - ζAdv)
        - MI loss component
        - Adversarial loss component
        - Discriminator loss component
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        epochs = np.arange(1, len(history['total_loss']) + 1)

        # Total Loss
        axes[0, 0].plot(epochs, history['total_loss'], 'b-', linewidth=2.5)
        axes[0, 0].fill_between(epochs, history['total_loss'], alpha=0.3)
        axes[0, 0].set_xlabel('Epoch', fontweight='bold')
        axes[0, 0].set_ylabel('Total Loss', fontweight='bold')
        axes[0, 0].set_title('Stage 2: Total Loss (L_MI - ζL_adv)', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)

        # MI Loss
        axes[0, 1].plot(epochs, history['mi_loss'], 'g-', linewidth=2.5)
        axes[0, 1].fill_between(epochs, history['mi_loss'], alpha=0.3, color='green')
        axes[0, 1].set_xlabel('Epoch', fontweight='bold')
        axes[0, 1].set_ylabel('MI Loss', fontweight='bold')
        axes[0, 1].set_title('Stage 2: MI Maximization', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        # Adversarial Loss
        axes[1, 0].plot(epochs, history['adv_loss'], 'orange', linewidth=2.5)
        axes[1, 0].fill_between(epochs, history['adv_loss'], alpha=0.3, color='orange')
        axes[1, 0].set_xlabel('Epoch', fontweight='bold')
        axes[1, 0].set_ylabel('Adversarial Loss', fontweight='bold')
        axes[1, 0].set_title('Stage 2: Adversarial MI Minimization', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        # Discriminator Loss
        axes[1, 1].plot(epochs, history['disc_loss'], 'purple', linewidth=2.5)
        axes[1, 1].fill_between(epochs, history['disc_loss'], alpha=0.3, color='purple')
        axes[1, 1].set_xlabel('Epoch', fontweight='bold')
        axes[1, 1].set_ylabel('Discriminator Loss', fontweight='bold')
        axes[1, 1].set_title('Stage 2: Discriminator Training', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{dataset_name}_fold{fold_idx}_stage2_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()

    def plot_confusion_matrix(self, cm, dataset_name='CK+48', fold_idx=0, num_classes=7):
        """
        Plot confusion matrix with proper formatting.
        
        Paper Figure: Confusion matrices showing per-expression accuracy
        """
        expression_names = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']
        expression_names = expression_names[:num_classes]

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=expression_names,
            yticklabels=expression_names,
            cbar_kws={'label': 'Count'},
            linewidths=0.5
        )
        plt.xlabel('Predicted Expression', fontweight='bold', fontsize=12)
        plt.ylabel('True Expression', fontweight='bold', fontsize=12)
        plt.title(f'Confusion Matrix - {dataset_name} Fold {fold_idx}', fontweight='bold', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, f'{dataset_name}_fold{fold_idx}_confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()

    def plot_representation_distribution(self, expr_features, id_features, 
                                        dataset_name='CK+48', fold_idx=0):
        """
        Plot 2D histogram of expression and identity representation spreads.
        Shows successful disentanglement.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Expression representation histogram
        axes[0].hist(expr_features.flatten(), bins=50, color='blue', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Feature Value', fontweight='bold')
        axes[0].set_ylabel('Frequency', fontweight='bold')
        axes[0].set_title('Expression Representation Distribution', fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Identity representation histogram
        axes[1].hist(id_features.flatten(), bins=50, color='green', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Feature Value', fontweight='bold')
        axes[1].set_ylabel('Frequency', fontweight='bold')
        axes[1].set_title('Identity Representation Distribution', fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{dataset_name}_fold{fold_idx}_representation_dist.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()

    def plot_ablation_study(self, zeta_values, accuracy_dict, dataset_name='CK+48'):
        """
        Plot ablation study results (effect of ζ^adv on accuracy).
        
        Replicates Figure 4 from the paper.
        """
        plt.figure(figsize=(10, 6))

        for dataset, accuracies in accuracy_dict.items():
            plt.plot(zeta_values, accuracies, 'o-', linewidth=2.5, markersize=8, label=dataset)

        plt.xlabel('ζ^adv (Adversarial Weight)', fontweight='bold', fontsize=12)
        plt.ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
        plt.title('Ablation Study: Effect of ζ^adv on Performance', fontweight='bold', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, 'ablation_zeta_adv.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()

    def plot_cross_dataset_performance(self, results_dict):
        """
        Plot cross-dataset generalization results.
        
        Shows DICE-FER performance across multiple datasets.
        """
        datasets = list(results_dict.keys())
        accuracies = [results_dict[d]['accuracy'] for d in datasets]
        f1_scores = [results_dict[d]['f1_score'] for d in datasets]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy bar chart
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        axes[0].bar(datasets, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        axes[0].set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
        axes[0].set_title('Cross-Dataset Accuracy', fontweight='bold', fontsize=14)
        axes[0].set_ylim([0, 105])
        axes[0].grid(True, alpha=0.3, axis='y')
        for i, (d, acc) in enumerate(zip(datasets, accuracies)):
            axes[0].text(i, acc + 2, f'{acc:.1f}%', ha='center', fontweight='bold')

        # F1-Score bar chart
        axes[1].bar(datasets, f1_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        axes[1].set_ylabel('F1-Score', fontweight='bold', fontsize=12)
        axes[1].set_title('Cross-Dataset F1-Score', fontweight='bold', fontsize=14)
        axes[1].set_ylim([0, 1.05])
        axes[1].grid(True, alpha=0.3, axis='y')
        for i, (d, f1) in enumerate(zip(datasets, f1_scores)):
            axes[1].text(i, f1 + 0.03, f'{f1:.3f}', ha='center', fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, 'cross_dataset_performance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()

    def plot_mi_evolution(self, expr_mi_history, id_mi_history, dataset_name='CK+48', fold_idx=0):
        """
        Plot evolution of Mutual Information Gap during training.
        
        Shows successful disentanglement over time.
        """
        epochs = np.arange(1, len(expr_mi_history) + 1)

        plt.figure(figsize=(12, 6))
        plt.plot(epochs, expr_mi_history, 'b-', linewidth=2.5, label='Expression MI', marker='o', markersize=4)
        plt.plot(epochs, id_mi_history, 'g-', linewidth=2.5, label='Identity MI', marker='s', markersize=4)

        # Fill between to show gap
        plt.fill_between(epochs, expr_mi_history, id_mi_history, alpha=0.2, label='MI Gap (Disentanglement)')

        plt.xlabel('Epoch', fontweight='bold', fontsize=12)
        plt.ylabel('Mutual Information', fontweight='bold', fontsize=12)
        plt.title('Mutual Information Evolution (Disentanglement Progress)', fontweight='bold', fontsize=14)
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, f'{dataset_name}_fold{fold_idx}_mi_evolution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()

    def plot_training_summary(self, stage1_hist, stage2_hist, classifier_acc, 
                             dataset_name='CK+48', fold_idx=0):
        """
        Create a comprehensive summary figure with all key metrics.
        """
        fig = GridSpec(3, 3, figure=plt.figure(figsize=(18, 12)))

        # Stage 1 curves
        ax1 = fig.subplots()[0, 0]
        epochs = np.arange(1, len(stage1_hist['total_loss']) + 1)
        ax1.plot(epochs, stage1_hist['total_loss'], 'b-', linewidth=2)
        ax1.set_title('Stage 1: Total Loss', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.grid(True, alpha=0.3)

        ax2 = fig.subplots()[0, 1]
        ax2.plot(epochs, stage1_hist['mi_loss'], 'g-', linewidth=2)
        ax2.set_title('Stage 1: MI Loss', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.grid(True, alpha=0.3)

        ax3 = fig.subplots()[0, 2]
        ax3.plot(epochs, stage1_hist['l1_loss'], 'r-', linewidth=2)
        ax3.set_title('Stage 1: L1 Loss', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.grid(True, alpha=0.3)

        # Stage 2 curves
        epochs2 = np.arange(1, len(stage2_hist['total_loss']) + 1)

        ax4 = fig.subplots()[1, 0]
        ax4.plot(epochs2, stage2_hist['total_loss'], 'b-', linewidth=2)
        ax4.set_title('Stage 2: Total Loss', fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.grid(True, alpha=0.3)

        ax5 = fig.subplots()[1, 1]
        ax5.plot(epochs2, stage2_hist['mi_loss'], 'g-', linewidth=2)
        ax5.set_title('Stage 2: MI Loss', fontweight='bold')
        ax5.set_xlabel('Epoch')
        ax5.grid(True, alpha=0.3)

        ax6 = fig.subplots()[1, 2]
        ax6.plot(epochs2, stage2_hist['disc_loss'], 'purple', linewidth=2)
        ax6.set_title('Stage 2: Discriminator Loss', fontweight='bold')
        ax6.set_xlabel('Epoch')
        ax6.grid(True, alpha=0.3)

        # Classifier accuracy
        epochs3 = np.arange(1, len(classifier_acc) + 1)
        ax7 = fig.subplots()[2, :]
        ax7.plot(epochs3, classifier_acc, 'o-', linewidth=2.5, markersize=6, color='darkblue')
        ax7.fill_between(epochs3, classifier_acc, alpha=0.3)
        ax7.set_title('Stage 3: Classifier Training Accuracy', fontweight='bold', fontsize=12)
        ax7.set_xlabel('Epoch', fontweight='bold')
        ax7.set_ylabel('Accuracy (%)', fontweight='bold')
        ax7.grid(True, alpha=0.3)

        plt.suptitle(f'DICE-FER Training Summary - {dataset_name} Fold {fold_idx}', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, f'{dataset_name}_fold{fold_idx}_training_summary.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()

    def plot_per_fold_comparison(self, all_metrics, dataset_name='CK+48'):
        """
        Plot per-fold performance comparison across all 10 folds.
        Shows mean, std, and individual fold results.
        """
        folds = np.arange(len(all_metrics['accuracy']))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy
        axes[0, 0].bar(folds, all_metrics['accuracy'], alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].axhline(y=np.mean(all_metrics['accuracy']), color='red', 
                           linestyle='--', linewidth=2, label=f"Mean: {np.mean(all_metrics['accuracy']):.2f}%")
        axes[0, 0].fill_between(
            [-0.5, len(folds)-0.5],
            np.mean(all_metrics['accuracy']) - np.std(all_metrics['accuracy']),
            np.mean(all_metrics['accuracy']) + np.std(all_metrics['accuracy']),
            alpha=0.2, color='red', label='±1 STD'
        )
        axes[0, 0].set_xlabel('Fold Index', fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy (%)', fontweight='bold')
        axes[0, 0].set_title('Per-Fold Accuracy', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        axes[0, 0].set_ylim([0, 105])

        # Precision
        axes[0, 1].bar(folds, all_metrics['precision'], alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].axhline(y=np.mean(all_metrics['precision']), color='red', 
                           linestyle='--', linewidth=2, label=f"Mean: {np.mean(all_metrics['precision']):.3f}")
        axes[0, 1].set_xlabel('Fold Index', fontweight='bold')
        axes[0, 1].set_ylabel('Precision', fontweight='bold')
        axes[0, 1].set_title('Per-Fold Precision', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        axes[0, 1].set_ylim([0, 1.05])

        # Recall
        axes[1, 0].bar(folds, all_metrics['recall'], alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].axhline(y=np.mean(all_metrics['recall']), color='red', 
                           linestyle='--', linewidth=2, label=f"Mean: {np.mean(all_metrics['recall']):.3f}")
        axes[1, 0].set_xlabel('Fold Index', fontweight='bold')
        axes[1, 0].set_ylabel('Recall', fontweight='bold')
        axes[1, 0].set_title('Per-Fold Recall', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].set_ylim([0, 1.05])

        # F1-Score
        axes[1, 1].bar(folds, all_metrics['f1_score'], alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].axhline(y=np.mean(all_metrics['f1_score']), color='red', 
                           linestyle='--', linewidth=2, label=f"Mean: {np.mean(all_metrics['f1_score']):.3f}")
        axes[1, 1].set_xlabel('Fold Index', fontweight='bold')
        axes[1, 1].set_ylabel('F1-Score', fontweight='bold')
        axes[1, 1].set_title('Per-Fold F1-Score', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].set_ylim([0, 1.05])

        plt.suptitle(f'10-Fold Cross-Validation Results - {dataset_name}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, f'{dataset_name}_10fold_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()