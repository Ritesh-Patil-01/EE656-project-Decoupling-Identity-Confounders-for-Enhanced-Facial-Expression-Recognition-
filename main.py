"""
DICE-FER: Decoupling Identity Confounders for Enhanced FER
Main execution script.

Usage:
    python main.py --dataset CK+ --epochs 100
    python main.py --dataset all --mode train
    python main.py --mode evaluate --checkpoint results/checkpoints/best.pt
"""

import argparse
import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime

from config import Config
from data.dataset import FERDataManager
from data.augmentation import FERAugmentation
from models.expression_encoder import ExpressionEncoder
from models.identity_encoder import IdentityEncoder
from models.classifier import ExpressionClassifier
from training.stage1_expression import Stage1Trainer
from training.stage2_identity import Stage2Trainer
from training.losses import ClassificationLoss
from evaluation.metrics import FERMetrics
from evaluation.visualization import DICEFERVisualizer
from evaluation.retrieval import ImageRetrieval


def parse_args():
    parser = argparse.ArgumentParser(
        description='DICE-FER Training and Evaluation'
    )
    parser.add_argument(
        '--dataset', type=str, default='CK+',
        choices=['CK+', 'Oulu-CASIA', 'RAF-DB',
                 'AffectNet', 'all', 'synthetic'],
        help='Dataset to use'
    )
    parser.add_argument(
        '--mode', type=str, default='train',
        choices=['train', 'evaluate', 'ablation', 'full'],
        help='Execution mode'
    )
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Number of training epochs per stage'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to checkpoint for evaluation'
    )
    parser.add_argument(
        '--fold', type=int, default=0,
        help='Cross-validation fold index'
    )
    return parser.parse_args()


def train_dice_fer(config, dataset_name, device):
    """
    Complete DICE-FER training pipeline.
    
    Stage 1: Expression Representation Learning
    Stage 2: Identity Representation Learning
    Stage 3: Classifier Training
    """
    print("\n" + "=" * 70)
    print(f"  DICE-FER Training on {dataset_name}")
    print("=" * 70)

    # ---- Data Setup ----
    augmenter = FERAugmentation(image_size=config.IMAGE_SIZE)
    
    num_expr = (6 if dataset_name == 'Oulu-CASIA' else 7)

    data_manager = FERDataManager(
        data_root=config.DATA_ROOT,
        dataset_name=dataset_name,
        num_expressions=num_expr,
        transform_train=augmenter.get_train_transform(),
        transform_test=augmenter.get_test_transform()
    )

    train_loader, test_loader = data_manager.get_kfold_loaders(
        n_splits=config.NUM_FOLDS,
        batch_size=config.BATCH_SIZE,
        fold_idx=0
    )

    # STAGE 1: Expression Learning
    print("\n--- Starting Stage 1 ---")
    stage1 = Stage1Trainer(config, device)
    expression_encoder, history1 = stage1.train(
        train_loader, num_epochs=config.NUM_EPOCHS
    )

    # Save Stage 1 checkpoint
    stage1_path = os.path.join(
        config.CHECKPOINT_DIR,
        f'{dataset_name}_stage1.pt'
    )
    stage1.save_checkpoint(stage1_path)

    # Visualize Stage 1
    DICEFERVisualizer.plot_training_history(
        history1, 'Stage 1: Expression Learning',
        save_path=os.path.join(
            config.FIGURES_DIR,
            f'{dataset_name}_stage1_loss.png'
        )
    )

    # STAGE 2: Identity Learning
    print("\n--- Starting Stage 2 ---")
    stage2 = Stage2Trainer(config, device, expression_encoder)
    identity_encoder, history2 = stage2.train(
        train_loader, num_epochs=config.NUM_EPOCHS
    )

    # Save Stage 2
    stage2_path = os.path.join(
        config.CHECKPOINT_DIR,
        f'{dataset_name}_stage2.pt'
    )
    stage2.save_checkpoint(stage2_path)

    DICEFERVisualizer.plot_training_history(
        history2, 'Stage 2: Identity Learning',
        save_path=os.path.join(
            config.FIGURES_DIR,
            f'{dataset_name}_stage2_loss.png'
        )
    )

    # STAGE 3: Classifier Training
    print("\n--- Training Classifier ---")
    classifier = ExpressionClassifier(
        input_dim=config.FEATURE_DIM,
        num_classes=num_expr
    ).to(device)

    cls_optimizer = optim.Adam(
        classifier.parameters(), lr=config.LEARNING_RATE
    )
    cls_loss_fn = ClassificationLoss()

    expression_encoder.eval()
    classifier.train()

    best_acc = 0
    for epoch in range(1, config.NUM_EPOCHS + 1):
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            images = batch['image_m'].to(device)
            labels = batch['label_m'].to(device)

            with torch.no_grad():
                expr_enc, _ = expression_encoder(images)

            logits = classifier(expr_enc)
            loss = cls_loss_fn.compute(logits, labels)

            cls_optimizer.zero_grad()
            loss.backward()
            cls_optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / max(total, 1) * 100

        if epoch % 10 == 0:
            print(
                f"  Classifier Epoch {epoch} | "
                f"Loss: {total_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}%"
            )

        if train_acc > best_acc:
            best_acc = train_acc
            torch.save(
                classifier.state_dict(),
                os.path.join(
                    config.CHECKPOINT_DIR,
                    f'{dataset_name}_classifier_best.pt'
                )
            )

    # EVALUATION
    print("\n--- Evaluation ---")
    metrics, cm, preds, labels = FERMetrics.evaluate_classifier(
        classifier, expression_encoder, test_loader, device
    )

    print(f"\n  Results on {dataset_name}:")
    print(f"  Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1-Score:  {metrics['f1_score']:.3f}")

    # Confusion matrix
    DICEFERVisualizer.plot_confusion_matrix(
        cm, dataset_name,
        save_path=os.path.join(
            config.FIGURES_DIR,
            f'{dataset_name}_confusion_matrix.png'
        ),
        num_classes=num_expr
    )

    # MIG score
    mig = FERMetrics.compute_mig(
        expression_encoder, identity_encoder,
        test_loader, device
    )
    print(f"  MIG Score: {mig:.3f}")

    # Save results
    results = {
        'dataset': dataset_name,
        'metrics': metrics,
        'mig': mig,
        'timestamp': datetime.now().isoformat()
    }

    results_path = os.path.join(
        config.RESULTS_DIR,
        f'{dataset_name}_results.json'
    )
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results, expression_encoder, identity_encoder, classifier


def run_ablation_study(config, dataset_name, device):
    """
    Ablation study: Effect of ζ^adv on performance.
    """
    print("\n" + "=" * 70)
    print("  Ablation Study: Impact of ζ^adv")
    print("=" * 70)

    zeta_values = [0.0, 0.005, 0.010, 0.025, 0.04, 0.05]
    results = {}

    for zeta in zeta_values:
        print(f"\n--- ζ^adv = {zeta} ---")
        config.ZETA_ADV = zeta

        result, _, _, _ = train_dice_fer(
            config, dataset_name, device
        )
        results[zeta] = result['metrics']['accuracy']

    # Plot ablation results
    DICEFERVisualizer.plot_zeta_ablation(
        zeta_values,
        {dataset_name: [results[z] for z in zeta_values]},
        save_path=os.path.join(
            config.FIGURES_DIR, 'ablation_zeta.png'
        )
    )

    return results


def main():
    args = parse_args()
    config = Config()
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.create_dirs()
    config.print_config()

    device = config.DEVICE
    print(f"\nUsing device: {device}")

    if args.mode == 'train' or args.mode == 'full':
        if args.dataset == 'all':
            datasets = ['CK+', 'Oulu-CASIA', 'RAF-DB', 'AffectNet']
        elif args.dataset == 'synthetic':
            datasets = ['synthetic']
        else:
            datasets = [args.dataset]

        all_results = {}
        for ds in datasets:
            results, _, _, _ = train_dice_fer(config, ds, device)
            all_results[ds] = results

        # Summary
        print("\n" + "=" * 70)
        print("  FINAL RESULTS SUMMARY")
        print("=" * 70)
        for ds, res in all_results.items():
            print(f"  {ds}: {res['metrics']['accuracy']:.2f}% "
                  f"(MIG: {res['mig']:.3f})")

    if args.mode == 'ablation' or args.mode == 'full':
        run_ablation_study(config, args.dataset, device)

    print("\n✅ DICE-FER execution complete!")


if __name__ == '__main__':
    main()