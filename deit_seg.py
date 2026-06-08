"""
deit_seg.py — DeiT-Small Segmentation Head: Training & Evaluation

Trains a lightweight U-Net decoder on top of frozen DeiT-Small patch tokens
to produce pixel-level forgery masks, directly addressing reviewer Q5.

Architecture
────────────
  DeiT-Small backbone  (frozen — loads results/checkpoints/DeiT-Small_best.pt)
      ↓  patch tokens  [B, 196, 384]  (14×14 grid)
      ↓  reshape       [B, 384, 14, 14]
      ↓
  U-Net decoder  (only trained component, ~1.5 M params):
      14×14  →  28×28   ConvT(384→256) + Conv(256→256) + BN + ReLU
      28×28  →  56×56   ConvT(256→128) + Conv(128→128) + BN + ReLU
      56×56  → 112×112  ConvT(128→64)  + Conv(64→64)   + BN + ReLU
     112×112 → 224×224  ConvT(64→32)   + Conv(32→32)   + BN + ReLU
     224×224 →   1      Conv(32→1)     [logit output]
      ↓
  Binary mask  [B, 1, 224, 224]

Loss: BCE + Dice (handles class imbalance in sparse forgery masks)

Data split: 70% train / 20% val / 10% test on matched real-fake pairs
            (same seed as train_detector.py → no data leakage)

Usage
─────
    # Train decoder then evaluate on test set (recommended)
    python deit_seg.py --mode both --epochs 30

    # Train only
    python deit_seg.py --mode train --epochs 30 --lr 1e-3 --batch_size 8

    # Evaluate a saved checkpoint on the test set
    python deit_seg.py --mode eval

    # Use a different GT threshold
    python deit_seg.py --mode both --gt_threshold 6

Outputs
───────
    results/checkpoints/DeiT-Seg_best.pt        ← best decoder weights
    results/segmentation/DeiT-Seg/
        ├── {id}_vis.png                         ← 6-panel visualisation
        ├── {id}_pred_mask.png                   ← binary predicted mask
        └── logs/seg_deit_*.log
    results/segmentation/deit_seg_metrics.csv    ← per-image metrics (test set)
"""

import os
import sys
import csv
import time
import random
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import timm

from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

# ── Import shared helpers ──────────────────────────────────────────────────────
from train_detector import get_device, REAL_DIR, FAKE_DIR, RESULTS_DIR, SEED
from segmentation_localization import (
    make_gt_mask, compute_metrics, save_vis, find_image_pairs,
    IMG_SIZE, GT_THRESHOLD,
)

# ── Config ─────────────────────────────────────────────────────────────────────
DEIT_CKPT   = os.path.join(RESULTS_DIR, "checkpoints", "DeiT-Small_best.pt")
SEG_CKPT    = os.path.join(RESULTS_DIR, "checkpoints", "DeiT-Seg_best.pt")
OUT_DIR     = os.path.join(RESULTS_DIR, "segmentation", "DeiT-Seg")
METRICS_CSV = os.path.join(RESULTS_DIR, "segmentation", "deit_seg_metrics.csv")
N_PAIRS     = 2952   # use all matched pairs for split
# ───────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════════════
def setup_logger(out_dir: str) -> logging.Logger:
    logs_dir = os.path.join(out_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, f"seg_deit_{run_id}.log")

    log = logging.getLogger("deit_seg")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    log.addHandler(ch)
    log.addHandler(fh)
    return log


# ══════════════════════════════════════════════════════════════════════════════
#  DECODER  (U-Net style: 14×14 → 224×224)
# ══════════════════════════════════════════════════════════════════════════════
class _UpBlock(nn.Module):
    """ConvTranspose2d ×2 + Conv3×3 + BN + ReLU."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(self.up(x))


class DeiTSegDecoder(nn.Module):
    """
    Learnable U-Net decoder: [B, embed_dim, 14, 14] → [B, 1, 224, 224].
    Four ×2 upsampling stages bring 14 → 28 → 56 → 112 → 224.
    Only ~1.5 M trainable parameters.
    """
    def __init__(self, embed_dim: int = 384):
        super().__init__()
        self.up1  = _UpBlock(embed_dim, 256)   # 14  → 28
        self.up2  = _UpBlock(256, 128)          # 28  → 56
        self.up3  = _UpBlock(128, 64)           # 56  → 112
        self.up4  = _UpBlock(64, 32)            # 112 → 224
        self.head = nn.Conv2d(32, 1, kernel_size=1)   # logit output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.head(x)   # [B, 1, 224, 224]


# ══════════════════════════════════════════════════════════════════════════════
#  FULL MODEL  (frozen DeiT backbone + trainable decoder)
# ══════════════════════════════════════════════════════════════════════════════
class DeiTSegModel(nn.Module):
    """
    DeiT-Small backbone (frozen) + DeiTSegDecoder (trained).

    The backbone's `norm` layer output is captured via a forward hook,
    giving tokens [B, N+1, 384].  We slice off the CLS token and reshape
    the 196 patch tokens to a 14×14 spatial grid for the decoder.
    """
    N_PATCHES = (IMG_SIZE // 16) ** 2   # 196 for 224×224 with patch_size=16
    EMBED_DIM = 384                      # DeiT-Small embedding dimension

    def __init__(self, backbone_ckpt: str | None = None):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            "deit_small_patch16_224", pretrained=True, num_classes=1
        )

        if backbone_ckpt and os.path.exists(backbone_ckpt):
            state = torch.load(backbone_ckpt, map_location="cpu")
            self.backbone.load_state_dict(state, strict=True)
            print(f"  DeiT backbone  : loaded from {backbone_ckpt}")
        else:
            print(f"  DeiT backbone  : ImageNet pretrained (no checkpoint found at {backbone_ckpt})")

        # Freeze every backbone parameter — only the decoder is trained
        for p in self.backbone.parameters():
            p.requires_grad = False

        # ── Hook: capture normalised token sequence after last block ──────────
        self._token_cache: torch.Tensor | None = None
        self.backbone.norm.register_forward_hook(
            lambda _m, _inp, out: setattr(self, "_token_cache", out)
        )

        # ── Decoder ───────────────────────────────────────────────────────────
        self.decoder = DeiTSegDecoder(self.EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run backbone (hook fires and fills self._token_cache)
        # no_grad on backbone — it is fully frozen, saves memory and compute
        with torch.no_grad():
            self.backbone(x)

        # _token_cache: [B, N+1, 384]  (CLS at index 0, patch tokens at 1..196)
        tokens = self._token_cache
        patch  = tokens[:, 1:1 + self.N_PATCHES, :]   # [B, 196, 384]
        B, N, C = patch.shape
        h = w = int(N ** 0.5)                          # 14
        feat = patch.transpose(1, 2).reshape(B, C, h, w)   # [B, 384, 14, 14]
        return self.decoder(feat)                           # [B, 1, 224, 224]

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════
IMG_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class SegPairDataset(Dataset):
    """
    Each sample: (fake_image_tensor [3,H,W],  gt_mask_tensor [1,H,W] in {0,1}).
    Input to the model is the FAKE image; output is the binary forgery mask.
    """
    def __init__(self, pairs: list, gt_threshold: int = GT_THRESHOLD):
        self.pairs        = pairs
        self.gt_threshold = gt_threshold

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        real_path, fake_path = self.pairs[idx]
        real_pil = Image.open(real_path)
        fake_pil = Image.open(fake_path)

        # Ground-truth forgery mask: uint8 [224,224]  (0 or 255)
        gt_arr = make_gt_mask(real_pil, fake_pil, self.gt_threshold)
        gt     = torch.tensor(gt_arr / 255.0, dtype=torch.float32).unsqueeze(0)   # [1,224,224]

        img = IMG_TRANSFORM(fake_pil.convert("RGB"))   # [3,224,224]
        return img, gt, str(real_path), str(fake_path)


def split_pairs(pairs: list, seed: int = SEED):
    """Deterministic 70/20/10 split on matched pairs."""
    rng = random.Random(seed)
    p   = pairs[:]
    rng.shuffle(p)
    n   = len(p)
    i1  = int(0.70 * n)
    i2  = int(0.90 * n)
    return p[:i1], p[i1:i2], p[i2:]


# ══════════════════════════════════════════════════════════════════════════════
#  LOSS  (BCE + Dice — handles sparse forgery masks)
# ══════════════════════════════════════════════════════════════════════════════
def dice_loss(logits: torch.Tensor, target: torch.Tensor,
              smooth: float = 1.0) -> torch.Tensor:
    prob  = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(2, 3))
    denom = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return (1.0 - (2.0 * inter + smooth) / (denom + smooth)).mean()


def seg_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce  = F.binary_cross_entropy_with_logits(logits, target)
    dice = dice_loss(logits, target)
    return bce + dice


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for imgs, masks, *_ in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        loss = seg_loss(model(imgs), masks)
        loss.backward()
        optimizer.step()
        total += loss.item() * imgs.size(0)
    return total / max(len(loader.dataset), 1)


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    total = 0.0
    for imgs, masks, *_ in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        total += seg_loss(model(imgs), masks).item() * imgs.size(0)
    return total / max(len(loader.dataset), 1)


def train(model, train_loader, val_loader, epochs, lr, device, log):
    optimizer = optim.AdamW(model.trainable_params(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_params = sum(p.numel() for p in model.trainable_params())
    log.info(f"  Decoder trainable params : {n_params:,}")
    log.info(f"  Epochs: {epochs}   LR: {lr:.0e}   Device: {device}\n")

    os.makedirs(os.path.dirname(SEG_CKPT), exist_ok=True)
    best_val  = float("inf")
    run_start = time.time()

    for ep in range(1, epochs + 1):
        t0       = time.time()
        tr_loss  = train_epoch(model, train_loader, optimizer, device)
        va_loss  = val_epoch(model, val_loader, device)
        scheduler.step()

        improved = va_loss < best_val
        if improved:
            best_val = va_loss
            torch.save(model.state_dict(), SEG_CKPT)

        log.info(
            f"  Ep {ep:03d}/{epochs}  "
            f"train={tr_loss:.4f}  val={va_loss:.4f}  "
            f"{'✓ saved' if improved else '':8}  "
            f"({time.time()-t0:.0f}s)"
        )

    total = (time.time() - run_start) / 60
    log.info(f"\n  Training complete — {total:.1f} min")
    log.info(f"  Best val loss : {best_val:.4f}")
    log.info(f"  Checkpoint    : {SEG_CKPT}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_seg(model, pairs, device, out_dir, gt_threshold, log):
    """
    Runs the model over all pairs, computes segmentation metrics per image,
    saves visualisations (every 5th image) and a metrics CSV.
    Returns mean metrics dict.
    """
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    all_metrics = []
    csv_rows    = []

    for idx, (real_path, fake_path) in enumerate(pairs):
        real_pil = Image.open(real_path)
        fake_pil = Image.open(fake_path)
        gt_mask  = make_gt_mask(real_pil, fake_pil, gt_threshold)   # [224,224] uint8

        img_tensor = IMG_TRANSFORM(fake_pil.convert("RGB")).unsqueeze(0).to(device)

        logit   = model(img_tensor)                           # [1,1,224,224]
        heatmap = torch.sigmoid(logit).squeeze().cpu().numpy()  # [224,224] float

        m = compute_metrics(gt_mask, heatmap)
        all_metrics.append(m)
        csv_rows.append({"img_idx": idx, "real_path": str(real_path), **m})

        # Save binary predicted mask
        mask_path = os.path.join(out_dir, f"{idx:04d}_pred_mask.png")
        cv2.imwrite(mask_path, (heatmap * 255).astype(np.uint8))

        # Save 6-panel visualisation every 5th image
        if idx % 5 == 0:
            vis_path = os.path.join(out_dir, f"{idx:04d}_vis.png")
            save_vis(real_pil, fake_pil, gt_mask, heatmap,
                     m, f"DeiT-Seg | img {idx}", vis_path)

        log.debug(
            f"  [{idx:04d}] IoU={m['IoU']}  Dice={m['Dice']}  "
            f"F1={m['F1']}  AUC={m['AUC_ROC']}"
        )

    # ── Save metrics CSV ──────────────────────────────────────────────────────
    fieldnames = ["img_idx", "real_path"] + list(all_metrics[0].keys())
    with open(METRICS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)
    log.info(f"  Per-image metrics → {METRICS_CSV}")

    # ── Compute mean metrics (skip NaN AUC/AP) ────────────────────────────────
    def nanmean(key):
        vals = [float(r[key]) for r in all_metrics if r[key] != "nan"]
        return round(np.mean(vals), 4) if vals else float("nan")

    mean_m = {k: nanmean(k) for k in all_metrics[0]}
    return mean_m


# ══════════════════════════════════════════════════════════════════════════════
#  COMPARISON HELPER
# ══════════════════════════════════════════════════════════════════════════════
BASELINE = {
    "EigenCAM + DeiT-Small (old)": {"IoU": 0.2838, "Dice": None,
                                     "F1": None, "AUC_ROC": 0.6228},
    "EigenCAM + ViT-B/16   (old)": {"IoU": 0.2683, "Dice": None,
                                     "F1": None, "AUC_ROC": 0.6364},
}


def print_comparison(mean_m, log):
    log.info(f"\n{'='*70}")
    log.info("  SEGMENTATION COMPARISON — DeiT-Seg vs EigenCAM baselines")
    log.info(f"{'='*70}")
    log.info(f"  {'Method':<38} {'IoU':>7} {'Dice':>7} {'F1':>7} {'AUC':>7}")
    log.info(f"  {'-'*38} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    def fmt(v): return f"{v:.4f}" if isinstance(v, float) else "   —  "
    log.info(
        f"  {'DeiT-Seg (decoder trained — this run)':<38} "
        f"{fmt(mean_m['IoU'])} {fmt(mean_m['Dice'])} "
        f"{fmt(mean_m['F1'])} {fmt(mean_m['AUC_ROC'])}"
    )
    for name, vals in BASELINE.items():
        log.info(
            f"  {name:<38} "
            f"{fmt(vals['IoU'])} {fmt(vals['Dice'])} "
            f"{fmt(vals['F1'])} {fmt(vals['AUC_ROC'])}"
        )
    log.info(f"{'='*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="DeiT-Small segmentation decoder — train & evaluate")
    parser.add_argument("--mode",         choices=["train","eval","both"],
                        default="both",
                        help="train=train decoder only, eval=load & evaluate, "
                             "both=train then evaluate (default)")
    parser.add_argument("--epochs",       type=int,   default=30,
                        help="Training epochs (default 30)")
    parser.add_argument("--lr",           type=float, default=1e-3,
                        help="Decoder learning rate (default 1e-3)")
    parser.add_argument("--batch_size",   type=int,   default=8,
                        help="Batch size (default 8; reduce to 4 if OOM)")
    parser.add_argument("--gt_threshold", type=int,   default=GT_THRESHOLD,
                        help=f"Pixel-diff threshold for GT mask (default {GT_THRESHOLD})")
    parser.add_argument("--n_pairs",      type=int,   default=N_PAIRS,
                        help="Max matched pairs to use (default 2952 = all)")
    parser.add_argument("--seed",         type=int,   default=SEED)
    args = parser.parse_args()

    # ── Seed ──────────────────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(OUT_DIR, exist_ok=True)
    log = setup_logger(OUT_DIR)
    device, dev_name = get_device()

    log.info("=" * 65)
    log.info("  DeiT-Small Segmentation Head")
    log.info("=" * 65)
    log.info(f"  Mode       : {args.mode}")
    log.info(f"  Device     : {dev_name}")
    log.info(f"  GT thresh  : {args.gt_threshold}")
    log.info(f"  Seed       : {args.seed}")
    log.info("")

    # ── Find matched real/fake pairs ──────────────────────────────────────────
    log.info("Finding matched image pairs ...")
    all_pairs = find_image_pairs(REAL_DIR, FAKE_DIR, n=args.n_pairs)
    log.info(f"  Found {len(all_pairs)} pairs")

    if not all_pairs:
        log.info("  No pairs found — check REAL_DIR and FAKE_DIR. Exiting.")
        return

    train_pairs, val_pairs, test_pairs = split_pairs(all_pairs, seed=args.seed)
    log.info(f"  Train: {len(train_pairs)}   Val: {len(val_pairs)}   Test: {len(test_pairs)}\n")

    # ── Build model ───────────────────────────────────────────────────────────
    log.info("Building DeiTSegModel ...")
    model = DeiTSegModel(backbone_ckpt=DEIT_CKPT).to(device)
    log.info(f"  Decoder params : {sum(p.numel() for p in model.trainable_params()):,}")
    log.info("")

    # ══════════════════════════════════════════════════════════════════════════
    #  TRAIN
    # ══════════════════════════════════════════════════════════════════════════
    if args.mode in ("train", "both"):
        log.info("─" * 65)
        log.info("  TRAINING PHASE")
        log.info("─" * 65)

        train_ds = SegPairDataset(train_pairs, args.gt_threshold)
        val_ds   = SegPairDataset(val_pairs,   args.gt_threshold)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True,  num_workers=0, pin_memory=False)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                                  shuffle=False, num_workers=0, pin_memory=False)

        log.info(f"  Train batches : {len(train_loader)}   Val batches : {len(val_loader)}")
        train(model, train_loader, val_loader, args.epochs, args.lr, device, log)

    # ══════════════════════════════════════════════════════════════════════════
    #  EVAL
    # ══════════════════════════════════════════════════════════════════════════
    if args.mode in ("eval", "both"):
        log.info("─" * 65)
        log.info("  EVALUATION PHASE  (test set)")
        log.info("─" * 65)

        if not os.path.exists(SEG_CKPT):
            log.info(f"  Checkpoint not found at {SEG_CKPT} — run with --mode train first.")
            return

        state = torch.load(SEG_CKPT, map_location="cpu")
        model.load_state_dict(state)
        model.to(device)
        log.info(f"  Loaded checkpoint : {SEG_CKPT}")
        log.info(f"  Evaluating on {len(test_pairs)} test pairs ...\n")

        t0     = time.time()
        mean_m = evaluate_seg(model, test_pairs, device,
                               OUT_DIR, args.gt_threshold, log)
        elapsed = (time.time() - t0) / 60

        log.info(f"\n  Test set results ({elapsed:.1f} min):")
        for k, v in mean_m.items():
            log.info(f"    {k:<12}: {v}")

        print_comparison(mean_m, log)

        log.info(f"  Output images → {OUT_DIR}/")
        log.info(f"  Metrics CSV   → {METRICS_CSV}")


if __name__ == "__main__":
    main()
