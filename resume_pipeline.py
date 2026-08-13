"""
DICE-FER: Resume 10-Fold Training from Last Completed Fold
Allows resuming training without losing progress on completed folds.
"""

import os
import json
import numpy as np
from datetime import datetime
from run_dice_fer import (
    Config, FERDataManager, Stage1Trainer, Stage2Trainer, 
    ExpressionClassifier, FERMetrics, DICEFERVisualizer,
    OptimizationPackage
)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T


class TrainingCheckpoint:
    """
    Manages training checkpoints to enable resuming from last completed fold.
    """
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.metadata_file = os.path.join(checkpoint_dir, 'training_metadata.json')
        self.load_metadata()

    def load_metadata(self):
        """Load metadata about completed folds."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'completed_folds': [],
                'current_fold': 0,
                'fold_results': [],
                'start_time': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            }

    def save_metadata(self):
        """Save metadata about completed folds."""
        self.metadata['last_updated'] = datetime.now().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def fold_completed(self, fold_idx, result):
        """Mark a fold as completed."""
        if fold_idx not in self.metadata['completed_folds']:
            self.metadata['completed_folds'].append(fold_idx)
            self.metadata['fold_results'].append(result)
            self.metadata['current_fold'] = fold_idx + 1
            self.save_metadata()
            print(f"✅ Marked Fold {fold_idx} as completed")

    def get_next_fold(self):
        """Get the next fold to train."""
        completed = sorted(self.metadata['completed_folds'])
        if len(completed) == 0:
            return 0
        # Find first missing fold
        for i in range(10):
            if i not in completed:
                return i
        return None  # All folds completed

    def fold_exists(self, fold_idx, dataset_name):
        """Check if fold checkpoint exists."""
        expr_ckpt = os.path.join(
            self.checkpoint_dir, 
            f'{dataset_name}_fold{fold_idx}_expr_encoder.pt'
        )
        id_ckpt = os.path.join(
            self.checkpoint_dir, 
            f'{dataset_name}_fold{fold_idx}_id_encoder.pt'
        )
        clf_ckpt = os.path.join(
            self.checkpoint_dir, 
            f'{dataset_name}_fold{fold_idx}_classifier_best.pt'
        )
        return os.path.exists(expr_ckpt) and os.path.exists(id_ckpt) and os.path.exists(clf_ckpt)

    def print_status(self):
        """Print training status."""
        print("\n" + "="*70)
        print(" TRAINING CHECKPOINT STATUS")
        print("="*70)
        print(f"Started: {self.metadata['start_time']}")
        print(f"Last updated: {self.metadata['last_updated']}")
        print(f"Completed folds: {sorted(self.metadata['completed_folds'])}")
        print(f"Next fold to train: {self.get_next_fold()}")
        print(f"Total folds completed: {len(self.metadata['completed_folds'])}/10")
        if self.metadata['fold_results']:
            accs = [r['accuracy'] for r in self.metadata['fold_results']]
            print(f"Mean accuracy so far: {np.mean(accs):.2f}% (±{np.std(accs):.2f}%)")
        print("="*70 + "\n")


def resume_pipeline():
    """
    Resume 10-fold training from last completed fold.
    """
    Config.create_dirs()
    device = Config.DEVICE
    visualizer = DICEFERVisualizer(save_dir=Config.FIGURES_DIR)
    checkpoint_manager = TrainingCheckpoint(Config.CHECKPOINT_DIR)

    # Print status
    checkpoint_manager.print_status()

    # GPU Check
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print(f"⚡ GPU Active: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  Running on CPU")

    # Data setup
    transform_train = T.Compose([
        T.ToPILImage(),
        T.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        T.RandomRotation(15),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    transform_test = T.Compose([
        T.ToPILImage(),
        T.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    data_manager = FERDataManager(
        data_root=Config.DATA_ROOT,
        dataset_name=Config.DATASET_NAME,
        transform_train=transform_train,
        transform_test=transform_test
    )

    # =====================================================================
    # RESUME 10-FOLD CROSS-VALIDATION
    # =====================================================================
    all_fold_results = checkpoint_manager.metadata['fold_results'].copy()
    all_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
    
    # Populate metrics from already completed folds
    for result in all_fold_results:
        all_metrics['accuracy'].append(result['accuracy'])
        all_metrics['precision'].append(result['precision'])
        all_metrics['recall'].append(result['recall'])
        all_metrics['f1_score'].append(result['f1_score'])
    
    num_folds = 10
    dataset_name = Config.DATASET_NAME

    print("\n" + "="*70)
    print(f" RESUMING 10-FOLD SUBJECT-INDEPENDENT CROSS-VALIDATION")
    print(f" Dataset: {dataset_name}")
    print("="*70)

    # Get starting fold
    start_fold = checkpoint_manager.get_next_fold()
    
    if start_fold is None:
        print("\n✅ All 10 folds already completed!")
        print("Skipping to results aggregation...\n")
    else:
        print(f"\nStarting from Fold {start_fold}...\n")

        # Train remaining folds
        for fold_idx in range(start_fold, num_folds):
            # Skip if already completed
            if fold_idx in checkpoint_manager.metadata['completed_folds']:
                print(f"\n⏭️  Fold {fold_idx} already completed, skipping...")
                continue

            print(f"\n{'='*70}")
            print(f" ▶️  FOLD {fold_idx + 1}/{num_folds}")
            print(f"{'='*70}")

            try:
                # Get train/test loaders
                train_loader, test_loader = data_manager.get_kfold_loaders(
                    n_splits=num_folds, batch_size=Config.BATCH_SIZE, fold_idx=fold_idx
                )

                # -----------------------------------------------------------------
                # STAGE 1: Expression Representation Learning
                # -----------------------------------------------------------------
                print(f"\n📍 [Fold {fold_idx}] Stage 1: Expression Learning...")
                stage1 = Stage1Trainer(Config, device)

                for epoch in range(1, Config.NUM_EPOCHS + 1):
                    losses = stage1.train_epoch(train_loader, epoch)
                    if epoch % 20 == 0 or epoch == 1:
                        print(f"   Epoch {epoch}/{Config.NUM_EPOCHS} | Loss: {losses['total']:.4f}")

                expr_encoder = stage1.encoder_m
                expr_checkpoint = os.path.join(
                    Config.CHECKPOINT_DIR, 
                    f'{dataset_name}_fold{fold_idx}_expr_encoder.pt'
                )
                torch.save(expr_encoder.state_dict(), expr_checkpoint)
                visualizer.plot_stage1_training_history(stage1.history, dataset_name, fold_idx)

                # -----------------------------------------------------------------
                # STAGE 2: Identity Representation Learning
                # -----------------------------------------------------------------
                print(f"\n📍 [Fold {fold_idx}] Stage 2: Identity Learning...")
                stage2 = Stage2Trainer(Config, device, expr_encoder)

                for epoch in range(1, Config.NUM_EPOCHS + 1):
                    losses = stage2.train_epoch(train_loader, epoch)
                    if epoch % 20 == 0 or epoch == 1:
                        print(f"   Epoch {epoch}/{Config.NUM_EPOCHS} | MI: {losses['mi']:.4f}")

                id_encoder = stage2.identity_encoder
                id_checkpoint = os.path.join(
                    Config.CHECKPOINT_DIR, 
                    f'{dataset_name}_fold{fold_idx}_id_encoder.pt'
                )
                torch.save(id_encoder.state_dict(), id_checkpoint)
                visualizer.plot_stage2_training_history(stage2.history, dataset_name, fold_idx)

                # -----------------------------------------------------------------
                # STAGE 3: Downstream Classifier Training
                # -----------------------------------------------------------------
                print(f"\n📍 [Fold {fold_idx}] Stage 3: Classifier Training...")
                classifier = ExpressionClassifier(
                    input_dim=Config.FEATURE_DIM, 
                    num_classes=Config.NUM_EXPRESSIONS
                ).to(device)
                
                criterion = OptimizationPackage.create_label_smoothing_loss(
                    num_classes=Config.NUM_EXPRESSIONS, smoothing=0.1
                )
                optimizer = OptimizationPackage.create_adamw_optimizer(
                    classifier.parameters(), learning_rate=Config.LEARNING_RATE, weight_decay=1e-5
                )

                expr_encoder.eval()
                classifier.train()

                best_acc = 0.0
                classifier_acc_history = []
                classifier_checkpoint = None
                
                for epoch in range(1, Config.NUM_EPOCHS + 1):
                    correct, total = 0, 0
                    
                    for batch in train_loader:
                        images = batch['image_m'].to(device)
                        labels = batch['label_m'].to(device)

                        with torch.no_grad():
                            expr_enc, _ = expr_encoder(images)

                        optimizer.zero_grad()
                        logits = classifier(expr_enc)
                        loss = criterion(logits, labels)
                        loss.backward()
                        optimizer.step()

                        preds = torch.argmax(logits, dim=1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)

                    acc = (correct / total) * 100
                    classifier_acc_history.append(acc)
                    
                    if acc > best_acc:
                        best_acc = acc
                        classifier_checkpoint = os.path.join(
                            Config.CHECKPOINT_DIR, 
                            f'{dataset_name}_fold{fold_idx}_classifier_best.pt'
                        )
                        torch.save(classifier.state_dict(), classifier_checkpoint)

                    if epoch % 20 == 0 or epoch == 1:
                        print(f"   Epoch {epoch}/{Config.NUM_EPOCHS} | Accuracy: {acc:.2f}%")

                # -----------------------------------------------------------------
                # FOLD EVALUATION
                # -----------------------------------------------------------------
                print(f"\n📊 [Fold {fold_idx}] Evaluating on Test Set...")
                
                if classifier_checkpoint is not None:
                    classifier.load_state_dict(torch.load(classifier_checkpoint))
                
                metrics, cm = FERMetrics.evaluate_classifier(
                    classifier, expr_encoder, test_loader, device
                )
                mig = FERMetrics.compute_mig(expr_encoder, id_encoder, test_loader, device)

                print(f"\n   📈 Fold {fold_idx} Results:")
                print(f"      Accuracy:  {metrics['accuracy']:.2f}%")
                print(f"      Precision: {metrics['precision']:.3f}")
                print(f"      Recall:    {metrics['recall']:.3f}")
                print(f"      F1-Score:  {metrics['f1_score']:.3f}")

                # Store and save results
                fold_result = {
                    'fold': fold_idx,
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'mig': mig
                }
                all_fold_results.append(fold_result)
                all_metrics['accuracy'].append(metrics['accuracy'])
                all_metrics['precision'].append(metrics['precision'])
                all_metrics['recall'].append(metrics['recall'])
                all_metrics['f1_score'].append(metrics['f1_score'])

                # Mark fold as completed
                checkpoint_manager.fold_completed(fold_idx, fold_result)

                # Visualizations
                visualizer.plot_confusion_matrix(cm, dataset_name, fold_idx, Config.NUM_EXPRESSIONS)
                visualizer.plot_training_summary(
                    stage1.history, stage2.history, classifier_acc_history,
                    dataset_name, fold_idx
                )

            except Exception as e:
                print(f"\n❌ ERROR in Fold {fold_idx}: {str(e)}")
                print(f"   To resume later, run: python resume_pipeline.py")
                print(f"   Already completed folds will be skipped.")
                raise

    # =====================================================================
    # AGGREGATE RESULTS ACROSS ALL COMPLETED FOLDS
    # =====================================================================
    print("\n" + "="*70)
    print(f" 📊 FINAL CROSS-VALIDATION RESULTS SUMMARY")
    print("="*70)

    if all_metrics['accuracy']:
        print(f"\n{'Fold':<6} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 54)
        for result in all_fold_results:
            print(f"{result['fold']:<6} {result['accuracy']:<12.2f} "
                  f"{result['precision']:<12.3f} {result['recall']:<12.3f} "
                  f"{result['f1_score']:<12.3f}")

        print("-" * 54)
        
        mean_accuracy = np.mean(all_metrics['accuracy'])
        std_accuracy = np.std(all_metrics['accuracy'])
        mean_precision = np.mean(all_metrics['precision'])
        std_precision = np.std(all_metrics['precision'])
        mean_recall = np.mean(all_metrics['recall'])
        std_recall = np.std(all_metrics['recall'])
        mean_f1 = np.mean(all_metrics['f1_score'])
        std_f1 = np.std(all_metrics['f1_score'])

        print(f"\n{'MEAN':<6} {mean_accuracy:<12.2f} {mean_precision:<12.3f} "
              f"{mean_recall:<12.3f} {mean_f1:<12.3f}")
        print(f"{'STD':<6} {std_accuracy:<12.2f} {std_precision:<12.3f} "
              f"{std_recall:<12.3f} {std_f1:<12.3f}")

        # Save final results
        results_summary = {
            'dataset': dataset_name,
            'num_folds': num_folds,
            'completed_folds': len(all_metrics['accuracy']),
            'per_fold_results': all_fold_results,
            'aggregate_metrics': {
                'accuracy': {
                    'mean': float(mean_accuracy),
                    'std': float(std_accuracy),
                    'all_folds': [float(x) for x in all_metrics['accuracy']]
                },
                'precision': {
                    'mean': float(mean_precision),
                    'std': float(std_precision),
                    'all_folds': [float(x) for x in all_metrics['precision']]
                },
                'recall': {
                    'mean': float(mean_recall),
                    'std': float(std_recall),
                    'all_folds': [float(x) for x in all_metrics['recall']]
                },
                'f1_score': {
                    'mean': float(mean_f1),
                    'std': float(std_f1),
                    'all_folds': [float(x) for x in all_metrics['f1_score']]
                }
            },
            'timestamp': datetime.now().isoformat()
        }

        results_path = os.path.join(Config.RESULTS_DIR, f'{dataset_name}_10fold_results.json')
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\n✅ Results saved to: {results_path}")

        # Generate final visualization
        visualizer.plot_per_fold_comparison(all_metrics, dataset_name)

    print(f"\n✅ All visualizations saved to {Config.FIGURES_DIR}")
    print(f"✅ All checkpoints saved to {Config.CHECKPOINT_DIR}")
    
    if len(all_metrics['accuracy']) == 10:
        print(f"\n🎯 DICE-FER 10-Fold Cross-Validation Complete!")
    else:
        print(f"\n⏳ DICE-FER Training in progress ({len(all_metrics['accuracy'])}/10 folds completed)")
        print(f"   Run 'python resume_pipeline.py' to continue training")


if __name__ == '__main__':
    resume_pipeline()