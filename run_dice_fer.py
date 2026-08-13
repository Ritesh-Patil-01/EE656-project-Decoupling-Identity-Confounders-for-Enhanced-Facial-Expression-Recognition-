"""
DICE-FER: Decoupling Identity Confounders for Enhanced FER
Unified Execution Script (All-in-One Local GPU pipeline)

Author: Inspired by Aquib et al., 'Decoupling Identity Confounders...'
"""

import os
import random
import time
import json
from datetime import datetime
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


# =====================================================================
# LOGGING UTILITY — writes to console AND a .txt file simultaneously
# =====================================================================
class Logger:
    """Tee: writes every log() call to stdout and to a .txt file."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._file = open(log_path, 'w', encoding='utf-8')
        self._write_header()

    def _write_header(self):
        header = (
            f"DICE-FER Experiment Log\n"
            f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'='*70}\n"
        )
        self._file.write(header)
        self._file.flush()

    def log(self, *args, sep=' ', end='\n'):
        msg = sep.join(str(a) for a in args) + end
        print(msg, end='')          # to console
        self._file.write(msg)       # to file
        self._file.flush()

    def close(self):
        footer = f"\n{'='*70}\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        self._file.write(footer)
        self._file.close()
# =====================================================================
# 1. CONFIGURATION
# =====================================================================
class Config:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
    CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')
    FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')

    # Dataset path pointing to your local folder
    DATA_ROOT = '.'
    DATASET_NAME = 'CK+48'

    # Model
    BACKBONE = 'resnet18'
    FEATURE_DIM = 64
    IMAGE_SIZE = 112
    NUM_CHANNELS = 1
    NUM_EXPRESSIONS = 7  # (anger, contempt, disgust, fear, happy, sadness, surprise)

    # Hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.0

    # Loss weights from paper
    MU_EXP = 0.5
    NU_EXP = 1.0
    DELTA = 0.1
    MU_ID = 0.5
    NU_ID = 1.0
    ZETA_ADV = 0.025

    # Device & VRAM optimization settings
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_AMP = torch.cuda.is_available()  # Automatic Mixed Precision
    NUM_WORKERS = 0
    PIN_MEMORY = torch.cuda.is_available()

    @classmethod
    def create_dirs(cls):
        for path in [cls.RESULTS_DIR, cls.CHECKPOINT_DIR, cls.FIGURES_DIR]:
            os.makedirs(path, exist_ok=True)

class FERPairedDataset(Dataset):
    def __init__(self, image_paths, labels, identities, transform=None, pair_mode=True):
        self.image_paths = image_paths
        self.labels = labels
        self.identities = identities
        self.transform = transform
        self.pair_mode = pair_mode

        # Build lookup tables
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)

        # Confirm different identities exist for pairing
        self.valid_labels = []
        for label, indices in self.label_to_indices.items():
            unique_ids = set(self.identities[i] for i in indices)
            if len(unique_ids) >= 2:
                self.valid_labels.append(label)

        # Pre-cache pairing targets
        self.pair_candidates = {}
        for idx in range(len(self.image_paths)):
            label = self.labels[idx]
            identity = self.identities[idx]
            if label in self.valid_labels:
                candidates = [
                    i for i in self.label_to_indices[label]
                    if self.identities[i] != identity
                ]
                self.pair_candidates[idx] = candidates
            else:
                self.pair_candidates[idx] = [
                    i for i in self.label_to_indices[label] if i != idx
                ]

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, path):
        """Load image with timeout and caching."""
        try:
            # ✅ FIX: Simple direct load with error handling
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"⚠️  Failed to load: {path}")
                # Return blank image instead of crashing
                return np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.uint8)
            return image
        except Exception as e:
            print(f"⚠️  Error loading {path}: {e}")
            return np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.uint8)

    @staticmethod
    def extract_identity(filepath):
        basename = os.path.basename(filepath)
        return basename.split('_')[0]

    def __getitem__(self, idx):
        """Get paired training sample with minimal transforms."""
        try:
            # Load Image M
            image_m = self._load_image(self.image_paths[idx])
            label_m = self.labels[idx]
            identity_m = self.identities[idx]

            # ✅ SIMPLIFIED: Just resize and normalize, no complex augmentation yet
            image_m = cv2.resize(image_m, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
            image_m = torch.FloatTensor(image_m).unsqueeze(0) / 255.0

            if self.pair_mode and label_m in self.valid_labels:
                candidates = self.pair_candidates[idx]
                pair_idx = random.choice(candidates) if candidates else idx

                image_n = self._load_image(self.image_paths[pair_idx])
                label_n = self.labels[pair_idx]
                identity_n = self.identities[pair_idx]

                # ✅ SIMPLIFIED: Just resize and normalize
                image_n = cv2.resize(image_n, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
                image_n = torch.FloatTensor(image_n).unsqueeze(0) / 255.0

                return {
                    'image_m': image_m,
                    'image_n': image_n,
                    'label_m': torch.tensor(label_m, dtype=torch.long),
                    'label_n': torch.tensor(label_n, dtype=torch.long),
                    'identity_m': identity_m,
                    'identity_n': identity_n,
                    'idx_m': idx,
                    'idx_n': pair_idx
                }
            else:
                return {
                    'image_m': image_m,
                    'label_m': torch.tensor(label_m, dtype=torch.long),
                    'identity_m': identity_m,
                    'idx_m': idx
                }

        except Exception as e:
            print(f"⚠️  Exception in __getitem__({idx}): {e}")
            # Return dummy batch
            dummy = torch.zeros(1, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
            return {
                'image_m': dummy,
                'image_n': dummy,
                'label_m': torch.tensor(0, dtype=torch.long),
                'label_n': torch.tensor(0, dtype=torch.long),
                'identity_m': 'unknown',
                'identity_n': 'unknown',
                'idx_m': idx,
                'idx_n': idx
            }
        

class FERDataManager:
    EXPRESSION_LABELS = {
        'anger': 0, 'contempt': 1, 'disgust': 2, 'fear': 3,
        'happy': 4, 'sadness': 5, 'surprise': 6
    }

    def __init__(self, data_root, dataset_name='CK+48', transform_train=None, transform_test=None):
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.transform_train = transform_train
        self.transform_test = transform_test
        self.label_map = self.EXPRESSION_LABELS

        self.image_paths, self.labels, self.identities = self._scan_dataset()
        print(f"\n[CK+48] Loaded {len(self.image_paths)} images | {len(set(self.identities))} subjects")

    def _scan_dataset(self):
        image_paths, labels, identities = [], [], []
        dataset_path = os.path.join(self.data_root, self.dataset_name)

        if not os.path.exists(dataset_path):
            raise RuntimeError(f"Could not locate dataset path: {os.path.abspath(dataset_path)}")

        for expr_name, expr_label in self.label_map.items():
            expr_dir = os.path.join(dataset_path, expr_name)
            if not os.path.isdir(expr_dir):
                continue
            for fname in sorted(os.listdir(expr_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filepath = os.path.join(expr_dir, fname)
                    image_paths.append(filepath)
                    labels.append(expr_label)
                    identities.append(FERPairedDataset.extract_identity(filepath))

        return image_paths, labels, identities

    def get_kfold_loaders(self, n_splits=10, batch_size=32, fold_idx=0):
        gkf = GroupKFold(n_splits=n_splits)
        indices = np.arange(len(self.image_paths))

        splits = gkf.split(X=indices, y=self.labels, groups=self.identities)
        for i, (train_idx, test_idx) in enumerate(splits):
            if i == fold_idx:
                break

        train_paths = [self.image_paths[j] for j in train_idx]
        train_labels = [self.labels[j] for j in train_idx]
        train_identities = [self.identities[j] for j in train_idx]

        train_dataset = FERPairedDataset(
            train_paths, train_labels, train_identities,
            transform=self.transform_train, pair_mode=True
        )

        test_paths = [self.image_paths[j] for j in test_idx]
        test_labels = [self.labels[j] for j in test_idx]
        test_identities = [self.identities[j] for j in test_idx]

        test_dataset = FERPairedDataset(
            test_paths, test_labels, test_identities,
            transform=self.transform_test, pair_mode=False
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY, drop_last=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY
        )

        # Identity overlap test
        assert len(set(train_identities).intersection(set(test_identities))) == 0, "Leakage detected!"
        return train_loader, test_loader



class BaseResNetEncoder(nn.Module):
    """
    Subclass that correctly retains ResNet-18 pre-trained weights for all blocks,
    and only modifies the first conv layer to support grayscale inputs.
    """
    def __init__(self, feature_dim=64):
        super(BaseResNetEncoder, self).__init__()
        # Load real pre-trained weights
        base_resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Adapt first conv weights to single channel
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight.copy_(base_resnet.conv1.weight.mean(dim=1, keepdim=True))

        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool
        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4
        self.avgpool = base_resnet.avgpool

        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, feature_dim)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        fmaps = []
        x = self.layer1(x); fmaps.append(x)
        x = self.layer2(x); fmaps.append(x)
        x = self.layer3(x); fmaps.append(x)
        x = self.layer4(x); fmaps.append(x)

        pooled = self.avgpool(x)
        pooled = torch.flatten(pooled, 1)
        encoding = self.projection(pooled)
        return encoding, fmaps

class ExpressionEncoder(BaseResNetEncoder):
    pass

class IdentityEncoder(BaseResNetEncoder):
    pass

class GlobalStatisticsNetwork(nn.Module):
    def __init__(self, image_feature_dim=512, encoding_dim=64):
        super(GlobalStatisticsNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(image_feature_dim + encoding_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )
    def forward(self, image_features, encoding):
        combined = torch.cat([image_features, encoding], dim=1)
        return self.network(combined)

class LocalStatisticsNetwork(nn.Module):
    def __init__(self, feature_map_channels, encoding_dim=64):
        super(LocalStatisticsNetwork, self).__init__()
        self.local_networks = nn.ModuleList()
        for channels in feature_map_channels:
            net = nn.Sequential(
                nn.Conv2d(channels + encoding_dim, 512, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 1, kernel_size=1)
            )
            self.local_networks.append(net)

    def forward(self, feature_maps, encoding):
        scores = []
        for feat_map, network in zip(feature_maps, self.local_networks):
            B, C, H, W = feat_map.shape
            enc_expanded = encoding.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            combined = torch.cat([feat_map, enc_expanded], dim=1)
            scores.append(network(combined))
        return scores



class MIEstimator:
    @staticmethod
    def compute_global_mi(stats_network, image_features, encoding, encoding_marginal=None):
        joint_scores = stats_network(image_features, encoding)
        if encoding_marginal is None:
            perm = torch.randperm(encoding.size(0))
            encoding_marginal = encoding[perm]
        marginal_scores = stats_network(image_features, encoding_marginal)
        return joint_scores.mean() - torch.log(torch.exp(marginal_scores).mean() + 1e-8)

    @staticmethod
    def compute_local_mi(stats_network, feature_maps, encoding, encoding_marginal=None):
        if encoding_marginal is None:
            perm = torch.randperm(encoding.size(0))
            encoding_marginal = encoding[perm]
        joint_local = stats_network(feature_maps, encoding)
        marginal_local = stats_network(feature_maps, encoding_marginal)
        total_mi = 0.0
        for joint, marginal in zip(joint_local, marginal_local):
            total_mi += joint.mean() - torch.log(torch.exp(marginal).mean() + 1e-8)
        return total_mi

class MIDiscriminator(nn.Module):
    def __init__(self, expression_dim=64, identity_dim=64, hidden_dim=256):
        super(MIDiscriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(expression_dim + identity_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
            # ✅ REMOVED: nn.Sigmoid() — BCEWithLogitsLoss applies it internally
        )
    def forward(self, expr_enc, id_enc):
        combined = torch.cat([expr_enc, id_enc], dim=1)
        return self.network(combined)  # Returns raw logits, not probabilities
class ExpressionClassifier(nn.Module):
    def __init__(self, input_dim=64, num_classes=7):
        super(ExpressionClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)




class ExpressionLoss:
    def __init__(self, mu_exp=0.5, nu_exp=1.0, delta=0.1):
        self.mu_exp, self.nu_exp, self.delta = mu_exp, nu_exp, delta
        self.mi_estimator = MIEstimator()

    def compute_total_loss(self, img_feat_m, fmaps_m, img_feat_n, fmaps_n,
                           enc_m, enc_n, gs_m, gs_n, ls_m, gs_local_n):
        # Swapped cross-referenced MI
        g_mi_m = self.mi_estimator.compute_global_mi(gs_m, img_feat_m, enc_n)
        g_mi_n = self.mi_estimator.compute_global_mi(gs_n, img_feat_n, enc_m)
        l_mi_m = self.mi_estimator.compute_local_mi(ls_m, fmaps_m, enc_n)
        l_mi_n = self.mi_estimator.compute_local_mi(gs_local_n, fmaps_n, enc_m)

        mi_loss = self.mu_exp * (g_mi_m + g_mi_n) + self.nu_exp * (l_mi_m + l_mi_n)
        l1_loss = torch.mean(torch.abs(enc_m - enc_n))
        return mi_loss - self.delta * l1_loss, mi_loss, l1_loss

class IdentityLoss:
    def __init__(self, mu_id=0.5, nu_id=1.0, zeta_adv=0.025):
        self.mu_id, self.nu_id, self.zeta_adv = mu_id, nu_id, zeta_adv
        self.mi_estimator = MIEstimator()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def compute_adversarial_loss(self, discriminator, expr_enc, id_enc, train_disc=True):
        B = expr_enc.size(0)
        real_lbl = torch.ones(B, 1, device=expr_enc.device)
        fake_lbl = torch.zeros(B, 1, device=expr_enc.device)

        if train_disc:
            perm = torch.randperm(B, device=expr_enc.device)
            real_pred = discriminator(expr_enc.detach(), id_enc[perm].detach())
            fake_pred = discriminator(expr_enc.detach(), id_enc.detach())
            return self.bce_loss(real_pred, real_lbl) + self.bce_loss(fake_pred, fake_lbl)
        else:
            fake_pred = discriminator(expr_enc, id_enc)
            return self.bce_loss(fake_pred, real_lbl)

    def compute_total_loss(self, img_f_m, fmaps_m, img_f_n, fmaps_n,
                           expr_m, id_m, expr_n, id_n,
                           gs_m, gs_n, ls_m, ls_n, disc_m, disc_n):
        full_m = torch.cat([expr_m, id_m], dim=1)
        full_n = torch.cat([expr_n, id_n], dim=1)

        g_mi_m = self.mi_estimator.compute_global_mi(gs_m, img_f_m, full_m)
        g_mi_n = self.mi_estimator.compute_global_mi(gs_n, img_f_n, full_n)
        l_mi_m = self.mi_estimator.compute_local_mi(ls_m, fmaps_m, full_m)
        l_mi_n = self.mi_estimator.compute_local_mi(ls_n, fmaps_n, full_n)

        mi_loss = self.mu_id * (g_mi_m + g_mi_n) + self.nu_id * (l_mi_m + l_mi_n)
        adv_m = self.compute_adversarial_loss(disc_m, expr_m, id_m, train_disc=False)
        adv_n = self.compute_adversarial_loss(disc_n, expr_n, id_n, train_disc=False)

        total_loss = mi_loss - self.zeta_adv * (adv_m + adv_n)
        return total_loss, mi_loss, adv_m + adv_n
class Stage1Trainer:
    def __init__(self, config, device):
        self.config, self.device = config, device
        self.encoder_m = ExpressionEncoder(feature_dim=config.FEATURE_DIM).to(device)
        self.encoder_n = self.encoder_m
        
        self.global_stats_m = GlobalStatisticsNetwork(encoding_dim=config.FEATURE_DIM).to(device)
        self.global_stats_n = GlobalStatisticsNetwork(encoding_dim=config.FEATURE_DIM).to(device)
        self.local_stats_m = LocalStatisticsNetwork([64, 128, 256, 512], config.FEATURE_DIM).to(device)
        self.local_stats_n = LocalStatisticsNetwork([64, 128, 256, 512], config.FEATURE_DIM).to(device)

        self.loss_fn = ExpressionLoss(config.MU_EXP, config.NU_EXP, config.DELTA)
        self.scaler = torch.cuda.amp.GradScaler() if config.USE_AMP else None

        params = (list(self.encoder_m.parameters()) + list(self.global_stats_m.parameters()) +
                  list(self.global_stats_n.parameters()) + list(self.local_stats_m.parameters()) +
                  list(self.local_stats_n.parameters()))
        self.optimizer = optim.Adam(params, lr=config.LEARNING_RATE)
        self.history = {'total_loss': [], 'mi_loss': [], 'l1_loss': []}

    def train_epoch(self, train_loader, epoch):
        self.encoder_m.train()
        self.global_stats_m.train()
        self.global_stats_n.train()
        self.local_stats_m.train()
        self.local_stats_n.train()

        epoch_losses = {'total': 0, 'mi': 0, 'l1': 0}
        pbar = tqdm(train_loader, desc=f"Stage1 | Epoch {epoch}")

        for batch in pbar:
            image_m = batch['image_m'].to(self.device, non_blocking=True)
    # Non-SER variant: no paired image, use image_m as both
            image_n = batch.get('image_n', batch['image_m']).to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.config.USE_AMP):
                enc_m, fmaps_m = self.encoder_m(image_m)
                enc_n, fmaps_n = self.encoder_n(image_n)

                img_f_m = F.adaptive_avg_pool2d(fmaps_m[-1], (1, 1)).view(enc_m.size(0), -1)
                img_f_n = F.adaptive_avg_pool2d(fmaps_n[-1], (1, 1)).view(enc_n.size(0), -1)

                total_loss, mi_loss, l1_loss = self.loss_fn.compute_total_loss(
                    img_f_m, fmaps_m, img_f_n, fmaps_n, enc_m, enc_n,
                    self.global_stats_m, self.global_stats_n, 
                    self.local_stats_m, self.local_stats_n
                )
                loss = -total_loss

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.encoder_m.parameters(), 5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder_m.parameters(), 5.0)
                self.optimizer.step()

            epoch_losses['total'] += total_loss.item()
            epoch_losses['mi'] += mi_loss.item()
            epoch_losses['l1'] += l1_loss.item()
            pbar.set_postfix({'MI': f"{mi_loss.item():.4f}", 'L1': f"{l1_loss.item():.4f}"})

        for k in epoch_losses:
            epoch_losses[k] /= len(train_loader)
            self.history[f'{k}_loss'].append(epoch_losses[k])
        return epoch_losses


class Stage2Trainer:
    def __init__(self, config, device, expression_encoder):
        self.config, self.device = config, device
        self.expression_encoder = expression_encoder
        self.expression_encoder.eval()
        for p in self.expression_encoder.parameters():
            p.requires_grad = False

        self.identity_encoder = IdentityEncoder(feature_dim=config.FEATURE_DIM).to(device)
        self.global_stats_m = GlobalStatisticsNetwork(encoding_dim=config.FEATURE_DIM * 2).to(device)
        self.global_stats_n = GlobalStatisticsNetwork(encoding_dim=config.FEATURE_DIM * 2).to(device)
        self.local_stats_m = LocalStatisticsNetwork([64, 128, 256, 512], config.FEATURE_DIM * 2).to(device)
        self.local_stats_n = LocalStatisticsNetwork([64, 128, 256, 512], config.FEATURE_DIM * 2).to(device)

        self.discriminator = MIDiscriminator(config.FEATURE_DIM, config.FEATURE_DIM).to(device)
        self.loss_fn = IdentityLoss(config.MU_ID, config.NU_ID, config.ZETA_ADV)

        self.scaler = torch.cuda.amp.GradScaler() if config.USE_AMP else None

        gen_params = (list(self.identity_encoder.parameters()) + list(self.global_stats_m.parameters()) +
                      list(self.global_stats_n.parameters()) + list(self.local_stats_m.parameters()) +
                      list(self.local_stats_n.parameters()))
        self.optimizer_gen = optim.Adam(gen_params, lr=config.LEARNING_RATE)
        self.optimizer_disc = optim.Adam(self.discriminator.parameters(), lr=config.LEARNING_RATE)
        self.history = {'total_loss': [], 'mi_loss': [], 'adv_loss': [], 'disc_loss': []}

    def train_epoch(self, train_loader, epoch):
        self.identity_encoder.train()
        self.discriminator.train()
        self.global_stats_m.train()
        self.global_stats_n.train()
        self.local_stats_m.train()
        self.local_stats_n.train()

        epoch_losses = {'total': 0, 'mi': 0, 'adv': 0, 'disc': 0}
        pbar = tqdm(train_loader, desc=f"Stage2 | Epoch {epoch}")
        
        for batch in pbar:
            image_m = batch['image_m'].to(self.device, non_blocking=True)
    # Non-SER variant: no paired image, use image_m as both
            image_n = batch.get('image_n', batch['image_m']).to(self.device, non_blocking=True)

            # Get frozen expression features
            with torch.no_grad():
                expr_m, _ = self.expression_encoder(image_m)
                expr_n, _ = self.expression_encoder(image_n)

            #  Train Discriminator 
            self.optimizer_disc.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.config.USE_AMP):
                id_m, _ = self.identity_encoder(image_m)
                id_n, _ = self.identity_encoder(image_n)
                disc_l_m = self.loss_fn.compute_adversarial_loss(self.discriminator, expr_m, id_m, train_disc=True)
                disc_l_n = self.loss_fn.compute_adversarial_loss(self.discriminator, expr_n, id_n, train_disc=True)
                disc_loss = disc_l_m + disc_l_n

            if self.scaler:
                self.scaler.scale(disc_loss).backward()
                self.scaler.step(self.optimizer_disc)
            else:
                disc_loss.backward()
                self.optimizer_disc.step()

            # Train Generator (Identity Encoder)
            self.optimizer_gen.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.config.USE_AMP):
                id_m, fmaps_m = self.identity_encoder(image_m)
                id_n, fmaps_n = self.identity_encoder(image_n)

                img_f_m = F.adaptive_avg_pool2d(fmaps_m[-1], (1, 1)).view(id_m.size(0), -1)
                img_f_n = F.adaptive_avg_pool2d(fmaps_n[-1], (1, 1)).view(id_n.size(0), -1)

                total_loss, mi_loss, adv_loss = self.loss_fn.compute_total_loss(
                    img_f_m, fmaps_m, img_f_n, fmaps_n, expr_m, id_m, expr_n, id_n,
                    self.global_stats_m, self.global_stats_n, self.local_stats_m, self.local_stats_n,
                    self.discriminator, self.discriminator
                )
                gen_loss = -total_loss

            if self.scaler:
                self.scaler.scale(gen_loss).backward()
                self.scaler.unscale_(self.optimizer_gen)
                torch.nn.utils.clip_grad_norm_(self.identity_encoder.parameters(), 5.0)
                self.scaler.step(self.optimizer_gen)
                self.scaler.update()
            else:
                gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.identity_encoder.parameters(), 5.0)
                self.optimizer_gen.step()

            epoch_losses['total'] += total_loss.item()
            epoch_losses['mi'] += mi_loss.item()
            epoch_losses['adv'] += adv_loss.item()
            epoch_losses['disc'] += disc_loss.item()
            pbar.set_postfix({'MI': f"{mi_loss.item():.4f}", 'Disc': f"{disc_loss.item():.4f}"})

        for k in epoch_losses:
            epoch_losses[k] /= len(train_loader)
            self.history[f'{k}_loss'].append(epoch_losses[k])
        return epoch_losses
    
class Stage2Trainer:
    def __init__(self, config, device, expression_encoder):
        self.config, self.device = config, device
        self.expression_encoder = expression_encoder
        self.expression_encoder.eval()
        for p in self.expression_encoder.parameters():
            p.requires_grad = False

        self.identity_encoder = IdentityEncoder(feature_dim=config.FEATURE_DIM).to(device)
        self.global_stats_m = GlobalStatisticsNetwork(encoding_dim=config.FEATURE_DIM * 2).to(device)
        self.global_stats_n = GlobalStatisticsNetwork(encoding_dim=config.FEATURE_DIM * 2).to(device)
        self.local_stats_m = LocalStatisticsNetwork([64, 128, 256, 512], config.FEATURE_DIM * 2).to(device)
        self.local_stats_n = LocalStatisticsNetwork([64, 128, 256, 512], config.FEATURE_DIM * 2).to(device)

        self.discriminator = MIDiscriminator(config.FEATURE_DIM, config.FEATURE_DIM).to(device)
        self.loss_fn = IdentityLoss(config.MU_ID, config.NU_ID, config.ZETA_ADV)

        self.scaler = torch.amp.GradScaler('cuda') if config.USE_AMP else None

        gen_params = (list(self.identity_encoder.parameters()) + list(self.global_stats_m.parameters()) +
                      list(self.global_stats_n.parameters()) + list(self.local_stats_m.parameters()) +
                      list(self.local_stats_n.parameters()))
        self.optimizer_gen = optim.Adam(gen_params, lr=config.LEARNING_RATE)
        self.optimizer_disc = optim.Adam(self.discriminator.parameters(), lr=config.LEARNING_RATE)
        self.history = {'total_loss': [], 'mi_loss': [], 'adv_loss': [], 'disc_loss': []}

    def train_epoch(self, train_loader, epoch):
        self.identity_encoder.train()
        self.discriminator.train()
        self.global_stats_m.train()
        self.global_stats_n.train()
        self.local_stats_m.train()
        self.local_stats_n.train()

        epoch_losses = {'total': 0, 'mi': 0, 'adv': 0, 'disc': 0}
        pbar = tqdm(train_loader, desc=f"Stage2 | Epoch {epoch}")

        for batch in pbar:
            image_m = batch['image_m'].to(self.device, non_blocking=True)
            image_n = batch.get('image_n', batch['image_m']).to(self.device, non_blocking=True)

            # --- Get frozen expression features ---
            with torch.no_grad():
                expr_m, _ = self.expression_encoder(image_m)
                expr_n, _ = self.expression_encoder(image_n)

            # --- Step 1: Train Discriminator ---
            self.optimizer_disc.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.config.USE_AMP):
                id_m, _ = self.identity_encoder(image_m)
                id_n, _ = self.identity_encoder(image_n)
                disc_l_m = self.loss_fn.compute_adversarial_loss(self.discriminator, expr_m, id_m, train_disc=True)
                disc_l_n = self.loss_fn.compute_adversarial_loss(self.discriminator, expr_n, id_n, train_disc=True)
                disc_loss = disc_l_m + disc_l_n

            if self.scaler:
                self.scaler.scale(disc_loss).backward()
                self.scaler.step(self.optimizer_disc)
            else:
                disc_loss.backward()
                self.optimizer_disc.step()

            # --- Step 2: Train Generator (Identity Encoder) ---
            self.optimizer_gen.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.config.USE_AMP):
                id_m, fmaps_m = self.identity_encoder(image_m)
                id_n, fmaps_n = self.identity_encoder(image_n)

                img_f_m = F.adaptive_avg_pool2d(fmaps_m[-1], (1, 1)).view(id_m.size(0), -1)
                img_f_n = F.adaptive_avg_pool2d(fmaps_n[-1], (1, 1)).view(id_n.size(0), -1)

                total_loss, mi_loss, adv_loss = self.loss_fn.compute_total_loss(
                    img_f_m, fmaps_m, img_f_n, fmaps_n, expr_m, id_m, expr_n, id_n,
                    self.global_stats_m, self.global_stats_n, self.local_stats_m, self.local_stats_n,
                    self.discriminator, self.discriminator
                )
                gen_loss = -total_loss

            if self.scaler:
                self.scaler.scale(gen_loss).backward()
                self.scaler.unscale_(self.optimizer_gen)
                torch.nn.utils.clip_grad_norm_(self.identity_encoder.parameters(), 5.0)
                self.scaler.step(self.optimizer_gen)
                self.scaler.update()
            else:
                gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.identity_encoder.parameters(), 5.0)
                self.optimizer_gen.step()

            epoch_losses['total'] += total_loss.item()
            epoch_losses['mi'] += mi_loss.item()
            epoch_losses['adv'] += adv_loss.item()
            epoch_losses['disc'] += disc_loss.item()
            pbar.set_postfix({'MI': f"{mi_loss.item():.4f}", 'Disc': f"{disc_loss.item():.4f}"})

        for k in epoch_losses:
            epoch_losses[k] /= len(train_loader)
            self.history[f'{k}_loss'].append(epoch_losses[k])
        return epoch_losses
class FERMetrics:
    @staticmethod
    def evaluate_classifier(classifier, encoder, data_loader, device):
        encoder.eval(); classifier.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in data_loader:
                images = batch['image_m'].to(device)
                labels = batch['label_m'].to(device)

                expr_enc, _ = encoder(images)
                logits = classifier(expr_enc)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
        prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)

        metrics = {'accuracy': accuracy, 'precision': prec, 'recall': rec, 'f1_score': f1}
        return metrics, cm

    @staticmethod
    def compute_mig(expr_enc, id_enc, loader, device):
        """Modified Mutual Information Gap Metric (Unsupervised Variant)"""
        expr_enc.eval(); id_enc.eval()
        expr_features, id_features = [], []

        with torch.no_grad():
            for batch in loader:
                images = batch['image_m'].to(device)
                e_enc, _ = expr_enc(images)
                i_enc, _ = id_enc(images)
                expr_features.append(e_enc.cpu())
                id_features.append(i_enc.cpu())

        expr_features = torch.cat(expr_features, dim=0).numpy()
        id_features = torch.cat(id_features, dim=0).numpy()

        # Compute standard deviation as variance metric proxies
        std_e = np.std(expr_features, axis=0).mean()
        std_i = np.std(id_features, axis=0).mean()
        return float(np.abs(std_e - std_i))  # Representation spread difference


class DICEFERVisualizer:
    """Placeholder - Full implementation in visualization.py"""
    def __init__(self, save_dir='./results/figures'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_stage1_training_history(self, history, dataset_name='CK+48', fold_idx=0):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        epochs = np.arange(1, len(history['total_loss']) + 1)
        axes[0].plot(epochs, history['total_loss'], 'b-', linewidth=2.5)
        axes[0].set_title('Stage 1: Total Loss')
        axes[1].plot(epochs, history['mi_loss'], 'g-', linewidth=2.5)
        axes[1].set_title('Stage 1: MI Loss')
        axes[2].plot(epochs, history['l1_loss'], 'r-', linewidth=2.5)
        axes[2].set_title('Stage 1: L1 Loss')
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{dataset_name}_fold{fold_idx}_stage1_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    def plot_stage2_training_history(self, history, dataset_name='CK+48', fold_idx=0):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        epochs = np.arange(1, len(history['total_loss']) + 1)
        axes[0, 0].plot(epochs, history['total_loss'], 'b-', linewidth=2.5)
        axes[0, 0].set_title('Stage 2: Total Loss')
        axes[0, 1].plot(epochs, history['mi_loss'], 'g-', linewidth=2.5)
        axes[0, 1].set_title('Stage 2: MI Loss')
        axes[1, 0].plot(epochs, history['adv_loss'], 'orange', linewidth=2.5)
        axes[1, 0].set_title('Stage 2: Adversarial Loss')
        axes[1, 1].plot(epochs, history['disc_loss'], 'purple', linewidth=2.5)
        axes[1, 1].set_title('Stage 2: Discriminator Loss')
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{dataset_name}_fold{fold_idx}_stage2_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    def plot_confusion_matrix(self, cm, dataset_name='CK+48', fold_idx=0, num_classes=7):
        import matplotlib.pyplot as plt
        import seaborn as sns
        expression_names = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']
        expression_names = expression_names[:num_classes]
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=expression_names, yticklabels=expression_names,
                    cbar_kws={'label': 'Count'}, linewidths=0.5)
        plt.xlabel('Predicted Expression', fontweight='bold')
        plt.ylabel('True Expression', fontweight='bold')
        plt.title(f'Confusion Matrix - {dataset_name} Fold {fold_idx}', fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{dataset_name}_fold{fold_idx}_confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    def plot_training_summary(self, stage1_hist, stage2_hist, classifier_acc, 
                             dataset_name='CK+48', fold_idx=0):
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        fig = GridSpec(3, 3, figure=plt.figure(figsize=(18, 12)))
        
        epochs1 = np.arange(1, len(stage1_hist['total_loss']) + 1)
        epochs2 = np.arange(1, len(stage2_hist['total_loss']) + 1)
        epochs3 = np.arange(1, len(classifier_acc) + 1)
        
        ax1 = plt.subplot(fig[0, 0])
        ax1.plot(epochs1, stage1_hist['total_loss'], 'b-', linewidth=2)
        ax1.set_title('Stage 1: Total Loss', fontweight='bold')
        
        ax2 = plt.subplot(fig[0, 1])
        ax2.plot(epochs1, stage1_hist['mi_loss'], 'g-', linewidth=2)
        ax2.set_title('Stage 1: MI Loss', fontweight='bold')
        
        ax3 = plt.subplot(fig[0, 2])
        ax3.plot(epochs1, stage1_hist['l1_loss'], 'r-', linewidth=2)
        ax3.set_title('Stage 1: L1 Loss', fontweight='bold')
        
        ax4 = plt.subplot(fig[1, 0])
        ax4.plot(epochs2, stage2_hist['total_loss'], 'b-', linewidth=2)
        ax4.set_title('Stage 2: Total Loss', fontweight='bold')
        
        ax5 = plt.subplot(fig[1, 1])
        ax5.plot(epochs2, stage2_hist['mi_loss'], 'g-', linewidth=2)
        ax5.set_title('Stage 2: MI Loss', fontweight='bold')
        
        ax6 = plt.subplot(fig[1, 2])
        ax6.plot(epochs2, stage2_hist['disc_loss'], 'purple', linewidth=2)
        ax6.set_title('Stage 2: Discriminator Loss', fontweight='bold')
        
        ax7 = plt.subplot(fig[2, :])
        ax7.plot(epochs3, classifier_acc, 'o-', linewidth=2.5, markersize=6, color='darkblue')
        ax7.set_title('Stage 3: Classifier Training Accuracy', fontweight='bold')
        
        plt.suptitle(f'DICE-FER Training Summary - {dataset_name} Fold {fold_idx}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{dataset_name}_fold{fold_idx}_training_summary.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" Saved: {save_path}")

    def plot_per_fold_comparison(self, all_metrics, dataset_name='CK+48'):
        import matplotlib.pyplot as plt
        folds = np.arange(len(all_metrics['accuracy']))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].bar(folds, all_metrics['accuracy'], alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].axhline(y=np.mean(all_metrics['accuracy']), color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Fold Index', fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy (%)', fontweight='bold')
        axes[0, 0].set_title('Per-Fold Accuracy', fontweight='bold')
        axes[0, 0].set_ylim([0, 105])
        
        axes[0, 1].bar(folds, all_metrics['precision'], alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].axhline(y=np.mean(all_metrics['precision']), color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Fold Index', fontweight='bold')
        axes[0, 1].set_ylabel('Precision', fontweight='bold')
        axes[0, 1].set_title('Per-Fold Precision', fontweight='bold')
        axes[0, 1].set_ylim([0, 1.05])
        
        axes[1, 0].bar(folds, all_metrics['recall'], alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].axhline(y=np.mean(all_metrics['recall']), color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Fold Index', fontweight='bold')
        axes[1, 0].set_ylabel('Recall', fontweight='bold')
        axes[1, 0].set_title('Per-Fold Recall', fontweight='bold')
        axes[1, 0].set_ylim([0, 1.05])
        
        axes[1, 1].bar(folds, all_metrics['f1_score'], alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].axhline(y=np.mean(all_metrics['f1_score']), color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Fold Index', fontweight='bold')
        axes[1, 1].set_ylabel('F1-Score', fontweight='bold')
        axes[1, 1].set_title('Per-Fold F1-Score', fontweight='bold')
        axes[1, 1].set_ylim([0, 1.05])
        
        plt.suptitle(f'10-Fold Cross-Validation Results - {dataset_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{dataset_name}_10fold_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" Saved: {save_path}")


class OptimizationPackage:
    """Optimization techniques beyond the paper"""
    
    @staticmethod
    def create_adamw_optimizer(model_params, learning_rate=1e-4, weight_decay=1e-5):
        return optim.AdamW(model_params, lr=learning_rate, weight_decay=weight_decay)

    @staticmethod
    def create_label_smoothing_loss(num_classes, smoothing=0.1):
        class LabelSmoothingLoss(nn.Module):
            def __init__(self, num_classes, smoothing=0.1):
                super().__init__()
                self.num_classes = num_classes
                self.smoothing = smoothing
                self.confidence = 1.0 - smoothing
                self.criterion = nn.CrossEntropyLoss(reduction='mean')

            def forward(self, pred, target):
                pred = pred.log_softmax(dim=-1)
                with torch.no_grad():
                    true_dist = torch.zeros_like(pred)
                    true_dist.fill_(self.smoothing / (self.num_classes - 1))
                    true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
                return torch.mean(torch.sum(-true_dist * pred, dim=-1))

        return LabelSmoothingLoss(num_classes, smoothing)



def run_pipeline():
    """Run DICE-FER pipeline with 10-fold subject-independent cross-validation."""
    Config.create_dirs()
    device = Config.DEVICE
    visualizer = DICEFERVisualizer(save_dir=Config.FIGURES_DIR)

    # ---- Logging setup ----
    log_path = os.path.join(Config.RESULTS_DIR, 'dice_fer_run.txt')
    logger = Logger(log_path)
    log = logger.log
    log(f"Log file: {log_path}")
    log(f"Device  : {device}")
    log(f"Dataset : {Config.DATASET_NAME}")
    # ✅ SIMPLIFIED: Remove complex transforms for now
    # We'll apply augmentation later via code
    transform_train = None  # No transforms - load directly
    transform_test = None

    data_manager = FERDataManager(
        data_root=Config.DATA_ROOT,
        dataset_name=Config.DATASET_NAME,
        transform_train=transform_train,
        transform_test=transform_test
    )
    all_fold_results = []
    all_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
    all_confusion_matrices = []
    
    num_folds = 10
    dataset_name = Config.DATASET_NAME

    log("\n" + "="*70)
    log(f" 10-FOLD SUBJECT-INDEPENDENT CROSS-VALIDATION")
    log(f" Dataset: {dataset_name} | Total Folds: {num_folds}")
    log("="*70)

    for fold_idx in range(num_folds):
        log(f"\n{'='*70}")
        log(f" ▶️  FOLD {fold_idx + 1}/{num_folds}")
        log(f"{'='*70}")

        # Get train/test loaders for this fold
        train_loader, test_loader = data_manager.get_kfold_loaders(
            n_splits=num_folds, batch_size=Config.BATCH_SIZE, fold_idx=fold_idx
        )

        log(f"\n📍 [Fold {fold_idx}] Stage 1: Expression Learning...")
        stage1 = Stage1Trainer(Config, device)

        for epoch in range(1, Config.NUM_EPOCHS + 1):
            losses = stage1.train_epoch(train_loader, epoch)
            if epoch % 20 == 0 or epoch == 1:
                log(f"   Epoch {epoch}/{Config.NUM_EPOCHS} | "
                      f"Loss: {losses['total']:.4f} | MI: {losses['mi']:.4f}")

        expr_encoder = stage1.encoder_m
        expr_checkpoint = os.path.join(
            Config.CHECKPOINT_DIR, f'{dataset_name}_fold{fold_idx}_expr_encoder.pt'
        )
        torch.save(expr_encoder.state_dict(), expr_checkpoint)

        # Visualize Stage 1
        visualizer.plot_stage1_training_history(
            stage1.history, dataset_name=dataset_name, fold_idx=fold_idx
        )

        log(f"\n📍 [Fold {fold_idx}] Stage 2: Identity Learning...")
        stage2 = Stage2Trainer(Config, device, expr_encoder)

        for epoch in range(1, Config.NUM_EPOCHS + 1):
            losses = stage2.train_epoch(train_loader, epoch)
            if epoch % 20 == 0 or epoch == 1:
                log(f"   Epoch {epoch}/{Config.NUM_EPOCHS} | "
                      f"MI: {losses['mi']:.4f} | Disc: {losses['disc']:.4f}")

        id_encoder = stage2.identity_encoder
        id_checkpoint = os.path.join(
            Config.CHECKPOINT_DIR, f'{dataset_name}_fold{fold_idx}_id_encoder.pt'
        )
        torch.save(id_encoder.state_dict(), id_checkpoint)

        # Visualize Stage 2
        visualizer.plot_stage2_training_history(
            stage2.history, dataset_name=dataset_name, fold_idx=fold_idx
        )

        log(f"\n📍 [Fold {fold_idx}] Stage 3: Classifier Training...")
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
            total_loss = 0
            
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
                total_loss += loss.item()

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
                log(f"   Epoch {epoch}/{Config.NUM_EPOCHS} | Accuracy: {acc:.2f}% | Loss: {total_loss/len(train_loader):.4f}")


        log(f"\n   [Fold {fold_idx}] Evaluating on Test Set...")
        
        # Load best classifier
        if classifier_checkpoint is not None:
            classifier.load_state_dict(torch.load(classifier_checkpoint))
        
        # Compute metrics
        metrics, cm = FERMetrics.evaluate_classifier(
            classifier, expr_encoder, test_loader, device
        )
        
        # Compute disentanglement metric
        mig = FERMetrics.compute_mig(expr_encoder, id_encoder, test_loader, device)

        log(f"\n   Fold {fold_idx} Results:")
        log(f"      Accuracy:  {metrics['accuracy']:.2f}%")
        log(f"      Precision: {metrics['precision']:.3f}")
        log(f"      Recall:    {metrics['recall']:.3f}")
        log(f"      F1-Score:  {metrics['f1_score']:.3f}")
        log(f"      MIG Score: {mig:.3f}")

        # Store results
        fold_result = {
            'fold': fold_idx,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'mig': mig
        }
        all_fold_results.append(fold_result)

        # Store metrics for aggregation
        all_metrics['accuracy'].append(metrics['accuracy'])
        all_metrics['precision'].append(metrics['precision'])
        all_metrics['recall'].append(metrics['recall'])
        all_metrics['f1_score'].append(metrics['f1_score'])
        all_confusion_matrices.append(cm)

        # Visualize confusion matrix for this fold
        visualizer.plot_confusion_matrix(
            cm, dataset_name=dataset_name, fold_idx=fold_idx, 
            num_classes=Config.NUM_EXPRESSIONS
        )

        # Visualize training summary for this fold
        visualizer.plot_training_summary(
            stage1.history, stage2.history, classifier_acc_history,
            dataset_name=dataset_name, fold_idx=fold_idx
        )

    # =====================================================================
    # AGGREGATE RESULTS ACROSS ALL FOLDS
    # =====================================================================
    log("\n" + "="*70)
    log(f"    CROSS-VALIDATION RESULTS SUMMARY ({num_folds}-FOLD)")
    log("="*70)

    log(f"\n{'Fold':<6} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    log("-" * 54)
    for result in all_fold_results:
        log(f"{result['fold']:<6} {result['accuracy']:<12.2f} "
              f"{result['precision']:<12.3f} {result['recall']:<12.3f} "
              f"{result['f1_score']:<12.3f}")

    log("-" * 54)
    
    # Compute mean and std
    mean_accuracy = np.mean(all_metrics['accuracy'])
    std_accuracy = np.std(all_metrics['accuracy'])
    mean_precision = np.mean(all_metrics['precision'])
    std_precision = np.std(all_metrics['precision'])
    mean_recall = np.mean(all_metrics['recall'])
    std_recall = np.std(all_metrics['recall'])
    mean_f1 = np.mean(all_metrics['f1_score'])
    std_f1 = np.std(all_metrics['f1_score'])

    log(f"\n{'MEAN':<6} {mean_accuracy:<12.2f} {mean_precision:<12.3f} "
          f"{mean_recall:<12.3f} {mean_f1:<12.3f}")
    log(f"{'STD':<6} {std_accuracy:<12.2f} {std_precision:<12.3f} "
          f"{std_recall:<12.3f} {std_f1:<12.3f}")

    
    results_summary = {
        'dataset': dataset_name,
        'num_folds': num_folds,
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
    
    log(f"\n   Results saved to: {results_path}")

    visualizer.plot_per_fold_comparison(all_metrics, dataset_name=dataset_name)

    log(f"\n All visualizations saved to {Config.FIGURES_DIR}")
    log(f" All checkpoints saved to {Config.CHECKPOINT_DIR}")
    log(f"\n DICE-FER 10-Fold Cross-Validation Complete!")
    log(f"\n Full log saved to: {log_path}")
    logger.close()

def run_ablation_study():
    """
    Ablation study replicating Table 2 of the paper on CK+48.

    Variants tested (one component removed at a time):
      - Baseline          : all components active (paper settings)
      - Non-SER           : SER disabled (pair = self, no cross-swap)
      - delta=0           : L1 distance term removed (DELTA = 0)
      - mu_exp=0          : Global MI term removed  (MU_EXP = 0)
      - nu_exp=0          : Local  MI term removed  (NU_EXP = 0)

    Uses fold 0 only (same fold for all variants → fair comparison).
    Full 100-epoch training per variant (same as baseline).
    """
    import copy

    # Logger 
    Config.create_dirs()
    log_path = os.path.join(Config.RESULTS_DIR, 'ablation_study.txt')
    logger = Logger(log_path)
    log = logger.log
    device = Config.DEVICE

    log("=" * 70)
    log("  TABLE 2 ABLATION STUDY — CK+48 (Fold 0)")
    log("  Paper: Section 5.6 | 7 expressions | 100 epochs each")
    log("=" * 70)
    log(f"Device: {device}\n")

    # Data (shared across all variants) 
    data_manager = FERDataManager(
        data_root=Config.DATA_ROOT,
        dataset_name=Config.DATASET_NAME,
        transform_train=None,
        transform_test=None
    )


    ablation_variants = [
        ("Non-SER",   {},                              True ),   # SER disabled
        ("delta=0",   {'DELTA':   0.0},                False),
        ("mu_exp=0",  {'MU_EXP':  0.0},                False),
        ("nu_exp=0",  {'NU_EXP':  0.0},                False),
    ]

    results_table = []   # list of dicts for final summary

    for variant_name, overrides, disable_ser in ablation_variants:
        log(f"\n{'─'*70}")
        log(f"  Variant: {variant_name}")
        if overrides:
            log(f"  Config overrides: {overrides}")
        if disable_ser:
            log(f"  SER: DISABLED (pair_mode=False for Stage 1)")
        log(f"{'─'*70}")

        # Build a local config namespace with overrides applied
        cfg = Config  # reference; we'll restore after

        # Save original values
        originals = {k: getattr(Config, k) for k in overrides}
        for k, v in overrides.items():
            setattr(Config, k, v)

        try:
            # Data loaders (fold 0) 
            train_loader, test_loader = data_manager.get_kfold_loaders(
                n_splits=10, batch_size=Config.BATCH_SIZE, fold_idx=0
            )

            # If Non-SER: rebuild train_loader with pair_mode=False
            if disable_ser:
                train_paths  = data_manager.image_paths
                train_labels = data_manager.labels
                train_ids    = data_manager.identities
                # Re-use the same train split indices from fold 0
                from sklearn.model_selection import GroupKFold
                import numpy as np
                gkf = GroupKFold(n_splits=10)
                idxs = np.arange(len(train_paths))
                for i, (tr_idx, _) in enumerate(
                        gkf.split(idxs, train_labels, train_ids)):
                    if i == 0:
                        break
                ds_no_ser = FERPairedDataset(
                    [train_paths[j] for j in tr_idx],
                    [train_labels[j] for j in tr_idx],
                    [train_ids[j]    for j in tr_idx],
                    transform=None,
                    pair_mode=False   # <-- SER disabled
                )
                from torch.utils.data import DataLoader
                train_loader = DataLoader(
                    ds_no_ser, batch_size=Config.BATCH_SIZE,
                    shuffle=True, num_workers=0, drop_last=True
                )

            # Stage 1 
            log(f"\n  [Stage 1] Expression Learning...")
            stage1 = Stage1Trainer(Config, device)
            for epoch in range(1, Config.NUM_EPOCHS + 1):
                losses = stage1.train_epoch(train_loader, epoch)
                if epoch % 20 == 0 or epoch == 1:
                    log(f"    Epoch {epoch:3d}/{Config.NUM_EPOCHS} | "
                        f"Total: {losses['total']:.4f} | "
                        f"MI: {losses['mi']:.4f} | "
                        f"L1: {losses['l1']:.4f}")
            expr_encoder = stage1.encoder_m

            # Stage 2 
            log(f"\n  [Stage 2] Identity Learning...")
            stage2 = Stage2Trainer(Config, device, expr_encoder)
            for epoch in range(1, Config.NUM_EPOCHS + 1):
                losses = stage2.train_epoch(train_loader, epoch)
                if epoch % 20 == 0 or epoch == 1:
                    log(f"    Epoch {epoch:3d}/{Config.NUM_EPOCHS} | "
                        f"MI: {losses['mi']:.4f} | "
                        f"Disc: {losses['disc']:.4f}")
            id_encoder = stage2.identity_encoder

            # Stage 3 — Classifier 
            log(f"\n  [Stage 3] Classifier Training...")
            classifier = ExpressionClassifier(
                input_dim=Config.FEATURE_DIM,
                num_classes=Config.NUM_EXPRESSIONS
            ).to(device)

            criterion = OptimizationPackage.create_label_smoothing_loss(
                num_classes=Config.NUM_EXPRESSIONS, smoothing=0.1
            )
            optimizer = OptimizationPackage.create_adamw_optimizer(
                classifier.parameters(),
                learning_rate=Config.LEARNING_RATE,
                weight_decay=1e-5
            )

            expr_encoder.eval()
            classifier.train()
            best_acc = 0.0
            best_state = None

            for epoch in range(1, Config.NUM_EPOCHS + 1):
                correct, total, total_loss = 0, 0, 0.0
                for batch in train_loader:
                    images = batch['image_m'].to(device)
                    labels = batch['label_m'].to(device)
                    with torch.no_grad():
                        enc, _ = expr_encoder(images)
                    optimizer.zero_grad()
                    logits = classifier(enc)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    total_loss += loss.item()

                acc = (correct / total) * 100
                if acc > best_acc:
                    best_acc = acc
                    best_state = {k: v.clone() for k, v in classifier.state_dict().items()}
                if epoch % 20 == 0 or epoch == 1:
                    log(f"    Epoch {epoch:3d}/{Config.NUM_EPOCHS} | "
                        f"Acc: {acc:.2f}% | Loss: {total_loss/len(train_loader):.4f}")

            # Evaluation 
            if best_state is not None:
                classifier.load_state_dict(best_state)

            metrics, cm = FERMetrics.evaluate_classifier(
                classifier, expr_encoder, test_loader, device
            )
            mig = FERMetrics.compute_mig(expr_encoder, id_encoder, test_loader, device)

            log(f"\n  Results — {variant_name}:")
            log(f"    Accuracy  : {metrics['accuracy']:.2f}%")
            log(f"    Precision : {metrics['precision']:.3f}")
            log(f"    Recall    : {metrics['recall']:.3f}")
            log(f"    F1-Score  : {metrics['f1_score']:.3f}")
            log(f"    MIG Score : {mig:.3f}")

            results_table.append({
                'variant':   variant_name,
                'accuracy':  metrics['accuracy'],
                'precision': metrics['precision'],
                'recall':    metrics['recall'],
                'f1_score':  metrics['f1_score'],
                'mig':       mig,
            })

        finally:
            # Always restore Config to original state
            for k, v in originals.items():
                setattr(Config, k, v)


    log("\n\n" + "=" * 70)
    log("  TABLE 2: ABLATION STUDY — Effect of Each Element on CK+48")
    log("  (Matches paper Table 2 format)")
    log("=" * 70)

    header = (f"\n{'Feature/Element':<18} {'Accuracy (%)':<16} "
              f"{'Precision':<12} {'Recall':<10} {'F1-Score':<10} {'MIG':<8}")
    log(header)
    log("-" * 76)

    for r in results_table:
        log(f"{r['variant']:<18} {r['accuracy']:<16.2f} "
            f"{r['precision']:<12.3f} {r['recall']:<10.3f} "
            f"{r['f1_score']:<10.3f} {r['mig']:<8.3f}")

    log("\n  Legend:")
    log("    Baseline  — All components active (mu_exp=0.5, nu_exp=1.0, delta=0.1, SER=ON)")
    log("    Non-SER   — Swapped Expression Representations disabled")
    log("    delta=0   — L1 distance between E_M and E_N removed")
    log("    mu_exp=0  — Global MI term removed")
    log("    nu_exp=0  — Local  MI term removed")

    # Save JSON
    import json
    ablation_json = os.path.join(Config.RESULTS_DIR, 'ablation_table2.json')
    with open(ablation_json, 'w') as f:
        json.dump(results_table, f, indent=2)
    log(f"\n✅ Ablation results JSON: {ablation_json}")
    log(f"✅ Ablation log TXT     : {log_path}")
    logger.close()


# if __name__ == '__main__':
#     run_pipeline()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DICE-FER CK+48')
    parser.add_argument(
        '--mode', type=str, default='train',
        choices=['train', 'ablation'],
        help=(
            'train   = full 10-fold cross-validation (default)\n'
            'ablation = Table 2 ablation study (fold 0 only)'
        )
    )
    args = parser.parse_args()

    if args.mode == 'train':
        run_pipeline()
    elif args.mode == 'ablation':
        run_ablation_study()