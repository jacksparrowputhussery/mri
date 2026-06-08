import os
import sys
import csv
import json
import time
import random
import logging
import argparse
import warnings
import platform
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import torchvision.transforms as T
from PIL import Image
import timm

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

# Optional: CLIP (transformers)
try:
    from transformers import CLIPVisionModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
REAL_DIR    = "ADNI_PNG"
FAKE_DIR    = "ADNI_Fake"
RESULTS_DIR = "results"
IMG_SIZE    = 224
SEED        = 42

CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

# Path where lr_search.py saves its results
BEST_LR_PATH = os.path.join(RESULTS_DIR, "best_lr.json")

# Fallback LRs — used when best_lr.json does not exist or a model is missing
DEFAULT_LR = {
    "EfficientNet-B0":   1e-4,
    "XceptionNet":       5e-5,
    "ViT-B/16":          5e-5,
    "FrequencyCNN":      1e-3,
    "EfficientNetV2-B2": 1e-4,
    "ConvNeXt-Tiny":     5e-5,
    "Swin-Tiny":         5e-5,
    "DeiT-Small":        5e-5,
    "MaxViT-Tiny":       5e-5,
    "GenConViT":         5e-5,
    "CLIP-Linear":       1e-3,
    "MedViT":            5e-5,
    "AFFETDS":           5e-5,
    "NPR-CNN":           1e-3,
    "LAA-Net":           5e-5,
    "FreqBlender":       1e-3,
    "WaveletEnsemble":   5e-5,
}
# ─────────────────────────────────────────────────────────────────────────────


def load_best_lrs(logger=None) -> dict:
    """
    Load per-model learning rates from results/best_lr.json (written by lr_search.py).
    Falls back to DEFAULT_LR for any model not present in the file.
    Returns a dict  {model_name: lr_float}.
    """
    if not os.path.exists(BEST_LR_PATH):
        msg = (f"  best_lr.json not found at {BEST_LR_PATH}.\n"
               "  Using built-in defaults. Run lr_search.py first for tuned LRs.")
        if logger:
            logger.info(msg)
        else:
            print(msg)
        return dict(DEFAULT_LR)

    with open(BEST_LR_PATH) as f:
        raw = json.load(f)

    lrs = dict(DEFAULT_LR)   # start from defaults, then override
    for name, val in raw.items():
        # lr_search.py stores either a plain float or {"best_lr": ..., ...}
        lrs[name] = val["best_lr"] if isinstance(val, dict) else float(val)

    if logger:
        logger.info(f"  Loaded per-model LRs from {BEST_LR_PATH}")
        for name, lr in lrs.items():
            src = "searched" if name in raw else "default"
            logger.debug(f"    {name:<24} LR={lr:.0e}  ({src})")
    return lrs

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════════════
def setup_logger(results_dir: str) -> logging.Logger:
    logs_dir = os.path.join(results_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, f"run_{run_id}.log")

    logger = logging.getLogger("deepfake_detector")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.log_path       = log_path
    logger.epoch_csv_path = os.path.join(logs_dir, "epoch_metrics.csv")
    logger.run_id         = run_id
    return logger


def init_epoch_csv(logger):
    with open(logger.epoch_csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "run_id", "model", "epoch", "lr",
            "train_loss", "val_loss",
            "accuracy", "precision", "recall", "f1", "auc_roc",
            "epoch_time_sec"
        ])


def log_epoch_csv(logger, model_name, epoch, lr,
                  train_loss, val_loss, metrics, elapsed):
    with open(logger.epoch_csv_path, "a", newline="") as f:
        csv.writer(f).writerow([
            logger.run_id, model_name, epoch, f"{lr:.2e}",
            f"{train_loss:.6f}", f"{val_loss:.6f}",
            metrics["Accuracy"], metrics["Precision"],
            metrics["Recall"],   metrics["F1"],
            metrics["AUC-ROC"],  f"{elapsed:.1f}"
        ])


def log_system_info(logger, device, epochs, batch_size, patience, min_delta):
    logger.info("=" * 65)
    logger.info("  Deepfake Brain MRI Detection — 17-Model Suite")
    logger.info(f"  Run ID       : {logger.run_id}")
    logger.info(f"  Log          : {logger.log_path}")
    logger.info("=" * 65)
    logger.debug(f"  Python   : {sys.version.split()[0]}")
    logger.debug(f"  PyTorch  : {torch.__version__}")
    logger.debug(f"  timm     : {timm.__version__}")
    logger.debug(f"  CLIP     : {'available' if CLIP_AVAILABLE else 'NOT installed — pip install transformers'}")
    logger.debug(f"  Platform : {platform.platform()}")
    logger.info(f"  Device        : {device}")
    logger.info(f"  Max Epochs    : {epochs}   Batch : {batch_size}   Seed : {SEED}")
    logger.info(f"  Early Stop    : patience={patience}  min_delta={min_delta}")
    logger.info(f"  CLIP model    : {'openai/clip-vit-base-patch32' if CLIP_AVAILABLE else 'SKIPPED (transformers not installed)'}")
    logger.info("")


# ══════════════════════════════════════════════════════════════════════════════
#  DEVICE
# ══════════════════════════════════════════════════════════════════════════════
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), f"CUDA — {torch.cuda.get_device_name(0)}"
    elif torch.backends.mps.is_available():
        return torch.device("mps"), "Apple MPS (M1/M2 GPU)"
    return torch.device("cpu"), "CPU"


# ══════════════════════════════════════════════════════════════════════════════
#  EARLY STOPPING
# ══════════════════════════════════════════════════════════════════════════════
class EarlyStopping:
    """
    Stops training when AUC-ROC hasn't improved by at least min_delta
    for `patience` consecutive epochs.
    Saves the best model checkpoint to results/checkpoints/<name>_best.pt
    """
    def __init__(self, patience=7, min_delta=0.001,
                 results_dir="results", model_name="model"):
        self.patience    = patience
        self.min_delta   = min_delta
        self.counter     = 0
        self.best_auc    = 0.0
        self.should_stop = False

        safe = model_name.replace(" ", "_").replace("/", "-")
        ckpt_dir = os.path.join(results_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        self.ckpt_path = os.path.join(ckpt_dir, f"{safe}_best.pt")

    def step(self, auc: float, model: nn.Module, logger) -> bool:
        if auc > self.best_auc + self.min_delta:
            self.best_auc = auc
            self.counter  = 0
            torch.save(model.state_dict(), self.ckpt_path)
            logger.debug(f"  [ES] New best AUC {auc:.4f} → checkpoint saved")
        else:
            self.counter += 1
            logger.debug(
                f"  [ES] No improvement {self.counter}/{self.patience} "
                f"(best={self.best_auc:.4f} cur={auc:.4f})"
            )
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"  ⏹  Early stop — no AUC gain >{self.min_delta} "
                    f"for {self.patience} epochs. Best: {self.best_auc:.4f}"
                )
        return self.should_stop


# ══════════════════════════════════════════════════════════════════════════════
#  DATASETS
# ══════════════════════════════════════════════════════════════════════════════
def load_paths_and_labels(real_dir, fake_dir):
    """Split dataset 70% train / 20% val / 10% test (stratified by class)."""
    real_paths = list(Path(real_dir).rglob("*.png"))
    fake_paths = list(Path(fake_dir).rglob("*.png"))

    def split_class(paths, seed=SEED):
        rng = random.Random(seed)
        p = paths[:]
        rng.shuffle(p)
        n = len(p)
        i_train = int(0.70 * n)
        i_val   = int(0.90 * n)   # 70 + 20 = 90%
        return p[:i_train], p[i_train:i_val], p[i_val:]

    r_tr, r_va, r_te = split_class(real_paths)
    f_tr, f_va, f_te = split_class(fake_paths)

    def combine_shuffle(a, b, label_a, label_b):
        combined = list(zip(a + b, [label_a]*len(a) + [label_b]*len(b)))
        random.shuffle(combined)
        paths, labels = zip(*combined)
        return list(paths), list(labels)

    tr_paths, tr_labels = combine_shuffle(r_tr, f_tr, 0, 1)
    va_paths, va_labels = combine_shuffle(r_va, f_va, 0, 1)
    te_paths, te_labels = combine_shuffle(r_te, f_te, 0, 1)

    return tr_paths, tr_labels, va_paths, va_labels, te_paths, te_labels


class MRIDataset(Dataset):
    """Standard RGB + ImageNet normalisation — used by most models."""
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), torch.tensor(self.labels[idx], dtype=torch.float32)


class CLIPDataset(Dataset):
    """CLIP-specific normalisation for CLIP-Linear model."""
    def __init__(self, paths, labels):
        self.paths, self.labels = paths, labels
        self.transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
        ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), torch.tensor(self.labels[idx], dtype=torch.float32)


class FreqMRIDataset(Dataset):
    """FFT log-magnitude spectrum — for FrequencyCNN & FreqBlender."""
    def __init__(self, paths, labels, size=IMG_SIZE):
        self.paths, self.labels, self.size = paths, labels, size

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L").resize((self.size, self.size))
        arr = np.array(img, dtype=np.float32) / 255.0
        fft       = np.fft.fft2(arr)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log1p(np.abs(fft_shift)).astype(np.float32)
        magnitude = (magnitude - magnitude.mean()) / (magnitude.std() + 1e-8)
        tensor = torch.tensor(magnitude).unsqueeze(0)
        return tensor, torch.tensor(self.labels[idx], dtype=torch.float32)


def make_sampler(labels):
    counts  = np.bincount(labels)
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def get_transforms():
    train_tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ══════════════════════════════════════════════════════════════════════════════
#  MODELS — PREVIOUS (4)
# ══════════════════════════════════════════════════════════════════════════════
def build_efficientnet_b0():
    return timm.create_model("efficientnet_b0", pretrained=True, num_classes=1)

def build_xception():
    return timm.create_model("xception", pretrained=True, num_classes=1)

def build_vit():
    return timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=1)

class FrequencyCNN(nn.Module):
    """Lightweight CNN on FFT log-magnitude spectrum."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),   nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),  nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4*4, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 1),
        )

    def forward(self, x): return self.classifier(self.features(x))


# ══════════════════════════════════════════════════════════════════════════════
#  MODELS — TIER 1 : Modern Architectures 2022-2025 (6)
# ══════════════════════════════════════════════════════════════════════════════
def build_efficientnetv2_b2():
    """
    EfficientNetV2-B2 (2022) — NAS-optimised with Fused-MBConv blocks.
    Faster & more accurate than EfficientNet-B0.
    Ref: ScienceDirect 2025 — 99.89% accuracy on deepfake binary task.
    """
    return timm.create_model("tf_efficientnetv2_b2", pretrained=True, num_classes=1)


def build_convnext_tiny():
    """
    ConvNeXt-Tiny (2022) — pure CNN redesigned with Transformer principles.
    Beats EfficientNet on most benchmarks. Best M1-MPS compatibility.
    Ref: Best lightweight CNN in modern deepfake literature 2022-2025.
    """
    return timm.create_model("convnext_tiny", pretrained=True, num_classes=1)


def build_swin_tiny():
    """
    Swin Transformer-Tiny (2022) — hierarchical shifted-window attention.
    Best cross-dataset generalisation of any model.
    Ref: Swin-Fake MDPI Electronics 2024 — 98% accuracy on FaceShifter.
    """
    return timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=1)


def build_deit_small():
    """
    DeiT-Small (2022) — ViT with knowledge distillation token.
    Works much better than raw ViT on limited data.
    Ref: DeiT ensemble arXiv 2502.10682 (2025).
    """
    return timm.create_model("deit_small_patch16_224", pretrained=True, num_classes=1)


def build_maxvit_tiny():
    """
    MaxViT-Tiny (2023) — multi-axis attention (local + global in every block).
    Newer than Swin, strong on fine-grained tasks.
    Ref: IEEE Deepfake Detection 2024-2025.
    """
    try:
        return timm.create_model("maxvit_tiny_tf_224", pretrained=True, num_classes=1)
    except Exception:
        # Fallback if MaxViT has MPS issues — use CoAtNet-Tiny instead
        return timm.create_model("coatnet_0_rw_224", pretrained=True, num_classes=1)


class GenConViT(nn.Module):
    """
    Generative Convolutional Vision Transformer (arXiv 2307.07036, 2023).
    Dual-path: ConvNeXt-Tiny (local conv features) + Swin-Tiny (global attention).
    Original paper: 95.8% accuracy, 99.3% AUC across DFDC, FF++, Celeb-DF.
    GitHub: https://github.com/erprogs/GenConViT
    """
    def __init__(self):
        super().__init__()
        self.conv_path = timm.create_model("convnext_tiny",
                                           pretrained=True, num_classes=0)
        self.attn_path = timm.create_model("swin_tiny_patch4_window7_224",
                                           pretrained=True, num_classes=0)
        c = self.conv_path.num_features   # 768
        s = self.attn_path.num_features   # 768
        self.head = nn.Sequential(
            nn.Linear(c + s, 512), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.head(torch.cat([self.conv_path(x), self.attn_path(x)], dim=1))


# ══════════════════════════════════════════════════════════════════════════════
#  MODELS — TIER 2 : Medical / Foundation Methods (3)
# ══════════════════════════════════════════════════════════════════════════════
class CLIPLinear(nn.Module):
    """
    CLIP ViT-B/32 (frozen) + single trainable linear head (2024).
    Best cross-dataset generalisation — foundation model features.
    Only the 768→1 linear layer is trained (0.03% of total params).
    Ref: arXiv 2503.19683 — SOTA cross-dataset AUROC on 14 benchmarks.
    Requires: pip install transformers
    """
    def __init__(self):
        super().__init__()
        assert CLIP_AVAILABLE, "transformers not installed. Run: pip install transformers"
        self.vision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        for p in self.vision.parameters():
            p.requires_grad = False
        self.classifier = nn.Linear(768, 1)

    def forward(self, x):
        with torch.no_grad():
            out = self.vision(pixel_values=x)
        return self.classifier(out.pooler_output)


class MedViT(nn.Module):
    """
    Medical Vision Transformer — simplified implementation (2023/2025).
    CNN pyramid (ECB-inspired: depthwise + pointwise) → Transformer encoder.
    Built for adversarially-robust medical image diagnosis.
    Ref: Computers in Biology and Medicine 2023; Applied Soft Computing 2025.
    GitHub: https://github.com/Omid-Nejati/MedViT
    """
    def __init__(self):
        super().__init__()
        # Stage 1 — ECB-inspired: depthwise + pointwise
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),           nn.BatchNorm2d(32),  nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32),         nn.BatchNorm2d(32),  nn.GELU(),
            nn.Conv2d(32, 64, 1),                               nn.BatchNorm2d(64),  nn.GELU(),
        )   # → [B, 64, 112, 112]
        # Stage 2
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1, groups=64), nn.BatchNorm2d(64),  nn.GELU(),
            nn.Conv2d(64, 128, 1),                               nn.BatchNorm2d(128), nn.GELU(),
        )   # → [B, 128, 56, 56]
        # Stage 3
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1, groups=128), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, 1),                                  nn.BatchNorm2d(256), nn.GELU(),
        )   # → [B, 256, 28, 28]
        # Stage 4 — compress to 7×7 patch grid
        self.stage4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2, padding=1, groups=256), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 512, 1),                                  nn.BatchNorm2d(512), nn.GELU(),
            nn.AdaptiveAvgPool2d(7),
        )   # → [B, 512, 7, 7]
        # Local Transformer Block (LTB-inspired)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=1024,
            dropout=0.1, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.stage4(self.stage3(self.stage2(self.stage1(x))))  # [B,512,7,7]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)   # [B, 49, 512]
        x = self.transformer(x)             # [B, 49, 512]
        x = x.mean(dim=1)                   # [B, 512]
        return self.head(x)


class AFFETDS(nn.Module):
    """
    Adversarial Feature Fusion Ensemble for Tumour Detection & Deepfake Screening.
    ResNet50 deep features + HOG-style gradient features (Sobel, no skimage) → fusion.
    Tested on ADNI + TCIA datasets — same dataset family as ours.
    Ref: Scientific Reports / Nature 2025 — 91.5% accuracy, 90.7% precision.
    """
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)
        # Fixed Sobel filters (gradient magnitude + direction, HOG-style)
        sx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]).reshape(1,1,3,3).repeat(3,1,1,1)
        sy = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]]).reshape(1,1,3,3).repeat(3,1,1,1)
        self.register_buffer("sobel_x", sx)
        self.register_buffer("sobel_y", sy)
        # HOG MLP: mag(588) + ang(588) = 1176 → 512 → 256
        self.hog_mlp = nn.Sequential(
            nn.Linear(1176, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),
        )
        # Fusion: ResNet50(2048) + HOG(256) = 2304 → 512 → 1
        self.head = nn.Sequential(
            nn.Linear(2048 + 256, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        deep = self.backbone(x)                                      # [B, 2048]
        gx   = F.conv2d(x, self.sobel_x, padding=1, groups=3)
        gy   = F.conv2d(x, self.sobel_y, padding=1, groups=3)
        mag  = torch.sqrt(gx**2 + gy**2 + 1e-8)
        ang  = torch.atan2(gy, gx)
        mag_p = F.adaptive_avg_pool2d(mag, 14).flatten(1)           # [B, 588]
        ang_p = F.adaptive_avg_pool2d(ang, 14).flatten(1)           # [B, 588]
        hog  = self.hog_mlp(torch.cat([mag_p, ang_p], dim=1))       # [B, 256]
        return self.head(torch.cat([deep, hog], dim=1))


# ══════════════════════════════════════════════════════════════════════════════
#  MODELS — TIER 3 : CVPR / NeurIPS / ICCV Papers (4)
# ══════════════════════════════════════════════════════════════════════════════
class NPRCNN(nn.Module):
    """
    Neighboring Pixel Relationships CNN — CVPR 2024.
    Detects upsampling artifacts: residual = img − (downsample then upsample).
    Also uses right/bottom pixel difference maps.
    Input: standard RGB → internally computes 9-channel NPR feature map.
    Ref: CVPR 2024 — +6.4% over LGrad, 92.5% mean accuracy.
    """
    def __init__(self):
        super().__init__()
        # 9-channel input: residual(3) + right_diff(3) + bottom_diff(3)
        self.features = nn.Sequential(
            nn.Conv2d(9, 32, 3, padding=1),  nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),nn.BatchNorm2d(256),nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4*4, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 1),
        )

    def _compute_npr(self, x):
        small    = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        up       = F.interpolate(small, size=(x.shape[2], x.shape[3]),
                                 mode="bilinear", align_corners=False)
        residual = x - up
        r_diff   = F.pad(x[:,:,:,:-1] - x[:,:,:,1:], (0,1,0,0))
        b_diff   = F.pad(x[:,:,:-1,:] - x[:,:,1:,:], (0,0,0,1))
        return torch.cat([residual, r_diff, b_diff], dim=1)   # [B,9,H,W]

    def forward(self, x):
        return self.classifier(self.features(self._compute_npr(x)))


class LAANet(nn.Module):
    """
    Localised Artifact Attention Network — CVPR 2024.
    ResNet-50 backbone + spatial attention map → attention-weighted pooling.
    Focuses on local regions where forgery artifacts appear.
    Ref: LAA-Net CVPR 2024 — quality-agnostic deepfake detection.
    """
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "resnet50", pretrained=True, num_classes=0, global_pool=""
        )   # → [B, 2048, 7, 7]
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 256, 1), nn.ReLU(),
            nn.Conv2d(256,  1,   1), nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        feat     = self.backbone(x)          # [B, 2048, 7, 7]
        attn     = self.attention(feat)      # [B, 1,    7, 7]
        weighted = feat * attn               # [B, 2048, 7, 7]
        return self.head(weighted)


class FreqBlender(nn.Module):
    """
    Multi-scale FrequencyCNN inspired by FreqBlender — NeurIPS 2024.
    Three parallel branches on the FFT log-magnitude spectrum:
      branch1 → full spectrum
      branch2 → low-frequency center crop (DC / smooth region)
      branch3 → high-frequency border (detail / noise region)
    Input: 1-channel FFT (FreqMRIDataset), same as FrequencyCNN.
    Ref: FreqBlender NeurIPS 2024 — frequency-domain knowledge blending.
    """
    def __init__(self):
        super().__init__()
        self.branch1 = nn.Sequential(          # full spectrum
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(4),
        )
        self.branch2 = nn.Sequential(          # low-frequency region
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.branch3 = nn.Sequential(          # high-frequency region
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        # 3 × (64×4×4=1024) = 3072
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512,  128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def _freq_regions(self, x):
        B, C, H, W = x.shape
        ch, cw = H // 4, W // 4
        low  = x[:, :, ch:H-ch, cw:W-cw]
        low  = F.interpolate(low, size=(H, W), mode="bilinear", align_corners=False)
        high = x.clone()
        high[:, :, ch:H-ch, cw:W-cw] = 0
        return low, high

    def forward(self, x):
        low, high = self._freq_regions(x)
        f1 = self.branch1(x).flatten(1)
        f2 = self.branch2(low).flatten(1)
        f3 = self.branch3(high).flatten(1)
        return self.classifier(torch.cat([f1, f2, f3], dim=1))


class WaveletEnsemble(nn.Module):
    """
    3-Branch Ensemble — DeiT-Small + ResNet-34 + Haar Wavelet CNN.
    Inspired by: arXiv 2502.10682 (2025) — DeiT + ResNet-34 + Wavelet Xception.

    Branch 1 — DeiT-Small   : global self-attention (384 features)
    Branch 2 — ResNet-34    : local spatial hierarchy (512 features)
    Branch 3 — WaveletCNN   : Haar DWT subbands, pure PyTorch (128 features)

    Haar DWT computed internally from RGB input — no extra dataset/library needed.
    """
    def __init__(self):
        super().__init__()
        self.deit   = timm.create_model("deit_small_patch16_224",
                                        pretrained=True, num_classes=0)   # 384
        self.resnet = timm.create_model("resnet34",
                                        pretrained=True, num_classes=0)   # 512
        # Wavelet branch: input 12 channels (LL,LH,HL,HH × 3 RGB channels)
        self.wavelet_cnn = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128),nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )   # → [B, 128]
        deit_dim    = self.deit.num_features    # 384
        resnet_dim  = self.resnet.num_features  # 512
        wavelet_dim = 128
        self.head = nn.Sequential(
            nn.Linear(deit_dim + resnet_dim + wavelet_dim, 512),
            nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 1),
        )

    def _haar_dwt(self, x):
        """
        Pure-PyTorch 2-D Haar DWT — no pywt needed.
        Input : [B, 3, H, W]
        Output: [B, 12, H/2, W/2]  (LL, LH, HL, HH per channel)
        """
        rows_lo = (x[:,:,0::2,:] + x[:,:,1::2,:]) * 0.5   # vertical avg  [B,3,H/2,W]
        rows_hi = (x[:,:,0::2,:] - x[:,:,1::2,:]) * 0.5   # vertical diff
        LL = (rows_lo[:,:,:,0::2] + rows_lo[:,:,:,1::2]) * 0.5
        HL = (rows_lo[:,:,:,0::2] - rows_lo[:,:,:,1::2]) * 0.5
        LH = (rows_hi[:,:,:,0::2] + rows_hi[:,:,:,1::2]) * 0.5
        HH = (rows_hi[:,:,:,0::2] - rows_hi[:,:,:,1::2]) * 0.5
        return torch.cat([LL, LH, HL, HH], dim=1)          # [B,12,H/2,W/2]

    def forward(self, x):
        d = self.deit(x)                           # [B, 384]
        r = self.resnet(x)                         # [B, 512]
        w = self.wavelet_cnn(self._haar_dwt(x))    # [B, 128]
        return self.head(torch.cat([d, r, w], dim=1))


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    for imgs, labels in loader:
        imgs   = imgs.to(device)
        logits = model(imgs).squeeze(1).cpu()
        probs  = torch.sigmoid(logits)
        preds  = (probs >= 0.5).long()
        all_probs.extend(probs.numpy())
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.long().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(labels, preds, probs):
    return {
        "Accuracy":  round(accuracy_score(labels, preds),              4),
        "Precision": round(precision_score(labels, preds, zero_division=0), 4),
        "Recall":    round(recall_score(labels, preds,    zero_division=0), 4),
        "F1":        round(f1_score(labels, preds,        zero_division=0), 4),
        "AUC-ROC":   round(roc_auc_score(labels, probs),               4),
    }


def save_confusion_matrix(labels, preds, model_name, results_dir):
    cm  = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real","Fake"], yticklabels=["Real","Fake"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    safe = model_name.replace(" ","_").replace("/","-")
    path = os.path.join(results_dir, f"cm_{safe}.png")
    plt.savefig(path, dpi=120); plt.close()
    return path


def save_loss_curve(train_losses, val_losses, model_name, results_dir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses,   label="Val Loss", linestyle="--")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"Loss Curve — {model_name}"); plt.legend(); plt.tight_layout()
    safe = model_name.replace(" ","_").replace("/","-")
    path = os.path.join(results_dir, f"loss_{safe}.png")
    plt.savefig(path, dpi=120); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_model(name, model, train_loader, val_loader, epochs, lr, device,
              results_dir, logger, patience=7, min_delta=0.001,
              test_loader=None):
    logger.info(f"\n{'='*65}")
    logger.info(f"  Training : {name}")
    logger.info(f"  LR:{lr:.1e}  Max epochs:{epochs}  "
                f"Patience:{patience}  min_delta:{min_delta}  Device:{device}")
    logger.info(f"{'='*65}")

    model     = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.debug(f"  Trainable params: {n_params:,}")

    early_stop    = EarlyStopping(patience, min_delta, results_dir, name)
    train_losses  = []
    val_losses    = []
    best_auc      = 0
    best_metrics  = {}
    best_labels   = best_preds = best_probs = None
    model_start   = time.time()
    stopped_epoch = epochs

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        current_lr  = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        labels, preds, probs = evaluate(model, val_loader, device)
        metrics = compute_metrics(labels, preds, probs)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device).unsqueeze(1)
                val_loss += criterion(model(imgs), lbls).item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()

        elapsed = time.time() - epoch_start

        if metrics["AUC-ROC"] > best_auc:
            best_auc     = metrics["AUC-ROC"]
            best_metrics = metrics
            best_labels  = labels
            best_preds   = preds
            best_probs   = probs

        is_best = "✓" if metrics["AUC-ROC"] == best_auc else " "
        logger.info(
            f"  [{name}] Ep {epoch:02d}/{epochs}{is_best} "
            f"lr={current_lr:.1e} "
            f"trn={train_loss:.4f} val={val_loss:.4f} "
            f"acc={metrics['Accuracy']:.4f} f1={metrics['F1']:.4f} "
            f"auc={metrics['AUC-ROC']:.4f} "
            f"p={early_stop.counter}/{patience} ({elapsed:.0f}s)"
        )
        logger.debug(
            f"  [{name}] Ep {epoch:02d} "
            f"prec={metrics['Precision']:.4f} rec={metrics['Recall']:.4f}"
        )
        log_epoch_csv(logger, name, epoch, current_lr,
                      train_loss, val_loss, metrics, elapsed)

        if early_stop.step(metrics["AUC-ROC"], model, logger):
            stopped_epoch = epoch
            break

    elapsed_total = time.time() - model_start
    save_loss_curve(train_losses, val_losses, name, results_dir)
    cm_path = save_confusion_matrix(best_labels, best_preds, name, results_dir)
    report  = classification_report(best_labels, best_preds,
                                    target_names=["Real","Fake"], digits=4)
    logger.debug(f"\n  Classification Report — {name}:\n{report}")

    note = (f"early stopped @ {stopped_epoch}/{epochs}"
            if stopped_epoch < epochs else f"all {epochs} epochs")
    logger.info(f"\n  ── Val   [{name}]  ({note}, {elapsed_total/60:.1f} min)")
    for k, v in best_metrics.items():
        logger.info(f"      {k:<12}: {v}")
    logger.info(f"      Checkpoint   → {early_stop.ckpt_path}")
    logger.info(f"      Confusion Mx → {cm_path}")

    # ── Test set evaluation (uses best checkpoint) ────────────────────────────
    test_metrics = None
    if test_loader is not None:
        saved_state = torch.load(early_stop.ckpt_path, map_location="cpu")
        model.load_state_dict(saved_state)
        model = model.to(device)
        te_labels, te_preds, te_probs = evaluate(model, test_loader, device)
        test_metrics = compute_metrics(te_labels, te_preds, te_probs)
        logger.info(f"\n  ── Test  [{name}]")
        for k, v in test_metrics.items():
            logger.info(f"      {k:<12}: {v}")
        save_confusion_matrix(te_labels, te_preds, f"{name}_test", results_dir)

    return {"val": best_metrics, "test": test_metrics}


def print_comparison_table(all_results, logger):
    metrics = ["Accuracy","Precision","Recall","F1","AUC-ROC"]
    col_w   = 13

    for split in ("val", "test"):
        # Only print test table if at least one model has test results
        if split == "test" and all(r["test"] is None for r in all_results.values()):
            continue

        lines = []
        lines.append(f"\n{'='*75}")
        label = "VALIDATION" if split == "val" else "TEST"
        lines.append(f"  FINAL MODEL COMPARISON — {label} SET")
        lines.append(f"{'='*75}")
        lines.append(f"  {'Model':<24}" + "".join(f"{m:>{col_w}}" for m in metrics))
        lines.append(f"  {'-'*24}" + "-"*(col_w*len(metrics)))

        valid = {n: r[split] for n, r in all_results.items() if r[split] is not None}
        for model_name, res in valid.items():
            lines.append(f"  {model_name:<24}" +
                         "".join(f"{res[m]:>{col_w}.4f}" for m in metrics))
        lines.append(f"{'='*75}")
        lines.append(f"\n  Best model per metric ({label}):")
        for m in metrics:
            best = max(valid, key=lambda n: valid[n][m])
            lines.append(f"    {m:<12} → {best}  ({valid[best][m]:.4f})")
        for line in lines:
            logger.info(line)


def save_results_csv(all_results, results_dir, logger):
    path    = os.path.join(results_dir, "model_comparison.csv")
    metrics = ["Accuracy","Precision","Recall","F1","AUC-ROC"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["Model"] + [f"val_{m}" for m in metrics] + [f"test_{m}" for m in metrics]
        w.writerow(header)
        for name, res in all_results.items():
            val_row  = [res["val"][m]  if res["val"]  else "" for m in metrics]
            test_row = [res["test"][m] if res["test"] else "" for m in metrics]
            w.writerow([name] + val_row + test_row)
    logger.info(f"\n  Model comparison CSV → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main(epochs, batch_size, seed, patience, min_delta):
    global SEED
    SEED = seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    logger = setup_logger(RESULTS_DIR)
    init_epoch_csv(logger)

    device, device_name = get_device()
    log_system_info(logger, device_name, epochs, batch_size, patience, min_delta)

    logger.info("Loading dataset (70 / 20 / 10 train/val/test split) ...")
    t0 = time.time()
    (tr_paths, tr_labels,
     va_paths, va_labels,
     te_paths, te_labels) = load_paths_and_labels(REAL_DIR, FAKE_DIR)
    logger.info(f"  Train : {len(tr_paths)}  (real={tr_labels.count(0)} fake={tr_labels.count(1)})")
    logger.info(f"  Val   : {len(va_paths)}  (real={va_labels.count(0)} fake={va_labels.count(1)})")
    logger.info(f"  Test  : {len(te_paths)}  (real={te_labels.count(0)} fake={te_labels.count(1)})")
    logger.info(f"  Loaded in {time.time()-t0:.1f}s\n")

    train_tf, val_tf = get_transforms()
    sampler = make_sampler(np.array(tr_labels))

    # ── Loaders ───────────────────────────────────────────────────────────────
    # Standard RGB (ImageNet norm) — used by most models
    train_rgb = DataLoader(MRIDataset(tr_paths, tr_labels, train_tf),
                           batch_size=batch_size, sampler=sampler,  num_workers=0)
    val_rgb   = DataLoader(MRIDataset(va_paths, va_labels, val_tf),
                           batch_size=batch_size, shuffle=False,    num_workers=0)
    test_rgb  = DataLoader(MRIDataset(te_paths, te_labels, val_tf),
                           batch_size=batch_size, shuffle=False,    num_workers=0)

    # FFT spectrum — FrequencyCNN, FreqBlender
    train_freq = DataLoader(FreqMRIDataset(tr_paths, tr_labels),
                            batch_size=batch_size, sampler=sampler, num_workers=0)
    val_freq   = DataLoader(FreqMRIDataset(va_paths, va_labels),
                            batch_size=batch_size, shuffle=False,   num_workers=0)
    test_freq  = DataLoader(FreqMRIDataset(te_paths, te_labels),
                            batch_size=batch_size, shuffle=False,   num_workers=0)

    # CLIP normalisation — CLIPLinear
    train_clip = DataLoader(CLIPDataset(tr_paths, tr_labels),
                            batch_size=batch_size, sampler=sampler, num_workers=0)
    val_clip   = DataLoader(CLIPDataset(va_paths, va_labels),
                            batch_size=batch_size, shuffle=False,   num_workers=0)
    test_clip  = DataLoader(CLIPDataset(te_paths, te_labels),
                            batch_size=batch_size, shuffle=False,   num_workers=0)

    # ── Load tuned LRs (from lr_search.py) or fall back to defaults ─────────
    lr = load_best_lrs(logger)
    logger.info(f"\n  LR source : {'results/best_lr.json' if os.path.exists(BEST_LR_PATH) else 'built-in defaults'}")
    logger.info("  Per-model LRs:")
    for n, v in lr.items():
        logger.info(f"    {n:<24} {v:.0e}")
    logger.info("")

    kw = dict(epochs=epochs, device=device, results_dir=RESULTS_DIR,
              logger=logger, patience=patience, min_delta=min_delta)

    all_results = {}
    run_start   = time.time()

    # ── Previous (4) ──────────────────────────────────────────────────────────
    logger.info("\n" + "─"*65)
    logger.info("  PREVIOUS MODELS (4)")
    logger.info("─"*65)

    all_results["EfficientNet-B0"] = run_model(
        "EfficientNet-B0", build_efficientnet_b0(),
        train_rgb, val_rgb, lr=lr["EfficientNet-B0"], test_loader=test_rgb, **kw)

    all_results["XceptionNet"] = run_model(
        "XceptionNet", build_xception(),
        train_rgb, val_rgb, lr=lr["XceptionNet"], test_loader=test_rgb, **kw)

    all_results["ViT-B/16"] = run_model(
        "ViT-B/16", build_vit(),
        train_rgb, val_rgb, lr=lr["ViT-B/16"], test_loader=test_rgb, **kw)

    all_results["FrequencyCNN"] = run_model(
        "FrequencyCNN", FrequencyCNN(),
        train_freq, val_freq, lr=lr["FrequencyCNN"], test_loader=test_freq, **kw)

    # ── Tier 1 (6) ────────────────────────────────────────────────────────────
    logger.info("\n" + "─"*65)
    logger.info("  TIER 1 — Modern Architectures 2022-2025 (6)")
    logger.info("─"*65)

    all_results["EfficientNetV2-B2"] = run_model(
        "EfficientNetV2-B2", build_efficientnetv2_b2(),
        train_rgb, val_rgb, lr=lr["EfficientNetV2-B2"], test_loader=test_rgb, **kw)

    all_results["ConvNeXt-Tiny"] = run_model(
        "ConvNeXt-Tiny", build_convnext_tiny(),
        train_rgb, val_rgb, lr=lr["ConvNeXt-Tiny"], test_loader=test_rgb, **kw)

    all_results["Swin-Tiny"] = run_model(
        "Swin-Tiny", build_swin_tiny(),
        train_rgb, val_rgb, lr=lr["Swin-Tiny"], test_loader=test_rgb, **kw)

    all_results["DeiT-Small"] = run_model(
        "DeiT-Small", build_deit_small(),
        train_rgb, val_rgb, lr=lr["DeiT-Small"], test_loader=test_rgb, **kw)

    all_results["MaxViT-Tiny"] = run_model(
        "MaxViT-Tiny", build_maxvit_tiny(),
        train_rgb, val_rgb, lr=lr["MaxViT-Tiny"], test_loader=test_rgb, **kw)

    all_results["GenConViT"] = run_model(
        "GenConViT", GenConViT(),
        train_rgb, val_rgb, lr=lr["GenConViT"], test_loader=test_rgb, **kw)

    # ── Tier 2 (3) ────────────────────────────────────────────────────────────
    logger.info("\n" + "─"*65)
    logger.info("  TIER 2 — Medical / Foundation Methods (3)")
    logger.info("─"*65)

    if CLIP_AVAILABLE:
        all_results["CLIP-Linear"] = run_model(
            "CLIP-Linear", CLIPLinear(),
            train_clip, val_clip, lr=lr["CLIP-Linear"], test_loader=test_clip, **kw)
    else:
        logger.info("  ⚠  CLIP-Linear SKIPPED — run: pip install transformers")

    all_results["MedViT"] = run_model(
        "MedViT", MedViT(),
        train_rgb, val_rgb, lr=lr["MedViT"], test_loader=test_rgb, **kw)

    all_results["AFFETDS"] = run_model(
        "AFFETDS", AFFETDS(),
        train_rgb, val_rgb, lr=lr["AFFETDS"], test_loader=test_rgb, **kw)

    # ── Tier 3 (4) ────────────────────────────────────────────────────────────
    logger.info("\n" + "─"*65)
    logger.info("  TIER 3 — CVPR / NeurIPS / arXiv Papers (4)")
    logger.info("─"*65)

    all_results["NPR-CNN"] = run_model(
        "NPR-CNN", NPRCNN(),
        train_rgb, val_rgb, lr=lr["NPR-CNN"], test_loader=test_rgb, **kw)

    all_results["LAA-Net"] = run_model(
        "LAA-Net", LAANet(),
        train_rgb, val_rgb, lr=lr["LAA-Net"], test_loader=test_rgb, **kw)

    all_results["FreqBlender"] = run_model(
        "FreqBlender", FreqBlender(),
        train_freq, val_freq, lr=lr["FreqBlender"], test_loader=test_freq, **kw)

    all_results["WaveletEnsemble"] = run_model(
        "WaveletEnsemble", WaveletEnsemble(),
        train_rgb, val_rgb, lr=lr["WaveletEnsemble"], test_loader=test_rgb, **kw)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_comparison_table(all_results, logger)
    save_results_csv(all_results, RESULTS_DIR, logger)

    total_mins = (time.time() - run_start) / 60
    logger.info(f"\n  Epoch metrics CSV → {logger.epoch_csv_path}")
    logger.info(f"  All plots         → ./{RESULTS_DIR}/")
    logger.info(f"  Full log          → {logger.log_path}")
    logger.info(f"\n  Total run time    : {total_mins:.1f} min")
    logger.info(f"  Completed at      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deepfake MRI Detection — 17-Model Comparison Suite")
    parser.add_argument("--epochs",     type=int,   default=50,
                        help="Max epochs per model (default 50, early stop will kick in)")
    parser.add_argument("--batch_size", type=int,   default=32,
                        help="Batch size (default 32; use 16 if OOM on 8GB M1)")
    parser.add_argument("--seed",       type=int,   default=42,
                        help="Random seed (default 42)")
    parser.add_argument("--patience",   type=int,   default=7,
                        help="Early-stop patience in epochs (default 7)")
    parser.add_argument("--min_delta",  type=float, default=0.001,
                        help="Min AUC improvement to reset patience (default 0.001)")
    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.seed, args.patience, args.min_delta)
