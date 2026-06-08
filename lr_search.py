"""
lr_search.py — Learning-rate grid search for every model in the detection suite.

Runs a coarse grid search over LR_GRID = {1e-3, 5e-4, 1e-4, 5e-5} for each model
using SEARCH_EPOCHS-epoch training runs.  The LR with the highest peak val AUC-ROC
is saved to  results/best_lr.json.

train_detector.py reads that file automatically when it exists, overriding the
built-in defaults.  Run this script once before the full training run:

    python lr_search.py
    python train_detector.py --epochs 50

Resuming: if results/best_lr.json already contains results for some models, those
models are skipped — only missing ones are searched.

Usage:
    # Full search — all models, 10 epochs per LR candidate
    python lr_search.py

    # Custom epochs / batch size
    python lr_search.py --epochs 15 --batch_size 16

    # Search only a subset of models
    python lr_search.py --models Swin-Tiny DeiT-Small XceptionNet

    # Re-run a model even if it already has a saved result
    python lr_search.py --models Swin-Tiny --force
"""

import os
import sys
import json
import time
import random
import argparse
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

# ── Import shared code from the main training script ─────────────────────────
from train_detector import (
    REAL_DIR, FAKE_DIR, RESULTS_DIR, SEED,
    CLIP_AVAILABLE,
    get_device,
    load_paths_and_labels,
    MRIDataset, FreqMRIDataset, CLIPDataset,
    get_transforms, make_sampler,
    train_one_epoch, evaluate, compute_metrics,
    # Model builders
    build_efficientnet_b0, build_xception, build_vit,
    build_efficientnetv2_b2, build_convnext_tiny,
    build_swin_tiny, build_deit_small, build_maxvit_tiny,
    FrequencyCNN, GenConViT, MedViT, AFFETDS,
    NPRCNN, LAANet, FreqBlender, WaveletEnsemble,
)

if CLIP_AVAILABLE:
    from train_detector import CLIPLinear

# ── Constants ─────────────────────────────────────────────────────────────────
LR_GRID      = [1e-3, 5e-4, 1e-4, 5e-5]
BEST_LR_PATH = os.path.join(RESULTS_DIR, "best_lr.json")

# Hardcoded fallback LRs — used when a model was not searched
# (matches the original per-model defaults in train_detector.py)
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


# ── Model registry ─────────────────────────────────────────────────────────────
def build_model_specs():
    """
    Maps model name → {factory callable, data loader type}.
    loader: "rgb"  — standard ImageNet-normalised RGB
            "freq" — FFT log-magnitude spectrum (1 channel)
            "clip" — CLIP normalisation (RGB)
    """
    specs = {
        "EfficientNet-B0":   {"factory": build_efficientnet_b0,   "loader": "rgb"},
        "XceptionNet":       {"factory": build_xception,           "loader": "rgb"},
        "ViT-B/16":          {"factory": build_vit,                "loader": "rgb"},
        "FrequencyCNN":      {"factory": FrequencyCNN,             "loader": "freq"},
        "EfficientNetV2-B2": {"factory": build_efficientnetv2_b2,  "loader": "rgb"},
        "ConvNeXt-Tiny":     {"factory": build_convnext_tiny,      "loader": "rgb"},
        "Swin-Tiny":         {"factory": build_swin_tiny,          "loader": "rgb"},
        "DeiT-Small":        {"factory": build_deit_small,         "loader": "rgb"},
        "MaxViT-Tiny":       {"factory": build_maxvit_tiny,        "loader": "rgb"},
        "GenConViT":         {"factory": GenConViT,                "loader": "rgb"},
        "MedViT":            {"factory": MedViT,                   "loader": "rgb"},
        "AFFETDS":           {"factory": AFFETDS,                  "loader": "rgb"},
        "NPR-CNN":           {"factory": NPRCNN,                   "loader": "rgb"},
        "LAA-Net":           {"factory": LAANet,                   "loader": "rgb"},
        "FreqBlender":       {"factory": FreqBlender,              "loader": "freq"},
        "WaveletEnsemble":   {"factory": WaveletEnsemble,          "loader": "rgb"},
    }
    if CLIP_AVAILABLE:
        specs["CLIP-Linear"] = {"factory": CLIPLinear, "loader": "clip"}
    return specs


# ── Core search loop ───────────────────────────────────────────────────────────
def search_one_model(name, factory, loader_type,
                     train_loaders, val_loaders,
                     epochs, device):
    """
    Tries every LR in LR_GRID for `epochs` epochs.
    Picks the LR whose run achieved the highest peak val AUC-ROC.

    Returns:
        best_lr   : float  — LR to use for the full training run
        best_auc  : float  — peak AUC-ROC achieved with best_lr
        all_aucs  : dict   — {lr: peak_auc} for every candidate
    """
    criterion = nn.BCEWithLogitsLoss()
    best_lr  = None
    best_auc = -1.0
    all_aucs = {}

    train_ldr = train_loaders[loader_type]
    val_ldr   = val_loaders[loader_type]

    for lr in LR_GRID:
        print(f"    LR={lr:.0e} ", end="", flush=True)
        t0 = time.time()

        model     = factory().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        peak_auc = 0.0
        for ep in range(1, epochs + 1):
            train_one_epoch(model, train_ldr, criterion, optimizer, device)
            labels, preds, probs = evaluate(model, val_ldr, device)
            m = compute_metrics(labels, preds, probs)
            if m["AUC-ROC"] > peak_auc:
                peak_auc = m["AUC-ROC"]
            scheduler.step()
            print(".", end="", flush=True)

        elapsed = time.time() - t0
        print(f"  peak_AUC={peak_auc:.4f}  ({elapsed:.0f}s)")

        all_aucs[lr] = round(peak_auc, 4)
        if peak_auc > best_auc:
            best_auc = peak_auc
            best_lr  = lr

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return best_lr, round(best_auc, 4), all_aucs


# ── Save / load helpers ────────────────────────────────────────────────────────
def load_saved(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_results(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="LR grid search — saves results/best_lr.json")
    parser.add_argument("--epochs",     type=int,   default=10,
                        help="Epochs per LR candidate (default 10)")
    parser.add_argument("--batch_size", type=int,   default=32,
                        help="Batch size (default 32)")
    parser.add_argument("--seed",       type=int,   default=SEED,
                        help="Random seed (default 42)")
    parser.add_argument("--models",     nargs="+",  default=None,
                        help="Subset of models to search (default: all)")
    parser.add_argument("--force",      action="store_true",
                        help="Re-search models that already have a saved LR")
    args = parser.parse_args()

    # ── Seed ──────────────────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device, dev_name = get_device()

    print(f"\n{'='*65}")
    print("  LR Grid Search")
    print(f"  Grid    : {[f'{lr:.0e}' for lr in LR_GRID]}")
    print(f"  Epochs  : {args.epochs} per candidate")
    print(f"  Device  : {dev_name}")
    print(f"  Seed    : {args.seed}")
    print(f"  Output  : {BEST_LR_PATH}")
    print(f"{'='*65}\n")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("Loading dataset ...")
    (tr_paths, tr_labels,
     va_paths, va_labels,
     _te_paths, _te_labels) = load_paths_and_labels(REAL_DIR, FAKE_DIR)
    print(f"  Train: {len(tr_paths)}   Val: {len(va_paths)}\n")

    train_tf, val_tf = get_transforms()
    sampler = make_sampler(np.array(tr_labels))

    train_loaders = {
        "rgb":  DataLoader(MRIDataset(tr_paths, tr_labels, train_tf),
                           batch_size=args.batch_size, sampler=sampler, num_workers=0),
        "freq": DataLoader(FreqMRIDataset(tr_paths, tr_labels),
                           batch_size=args.batch_size, sampler=sampler, num_workers=0),
    }
    val_loaders = {
        "rgb":  DataLoader(MRIDataset(va_paths, va_labels, val_tf),
                           batch_size=args.batch_size, shuffle=False, num_workers=0),
        "freq": DataLoader(FreqMRIDataset(va_paths, va_labels),
                           batch_size=args.batch_size, shuffle=False, num_workers=0),
    }
    if CLIP_AVAILABLE:
        train_loaders["clip"] = DataLoader(
            CLIPDataset(tr_paths, tr_labels),
            batch_size=args.batch_size, sampler=sampler, num_workers=0)
        val_loaders["clip"] = DataLoader(
            CLIPDataset(va_paths, va_labels),
            batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Load any existing results (allow resuming after crash) ────────────────
    # Format stored: { "ModelName": { "best_lr": 0.0001, "best_auc": 0.982,
    #                                  "all_aucs": {"0.001": 0.91, ...} } }
    saved = load_saved(BEST_LR_PATH)
    print(f"  Existing results: {len(saved)} model(s) already searched")
    if saved:
        for n, v in saved.items():
            info = v if isinstance(v, dict) else {"best_lr": v}
            print(f"    {n:<24} best_lr={info['best_lr']:.0e}"
                  + (f"  AUC={info.get('best_auc','?')}" if isinstance(v, dict) else ""))
    print()

    # ── Build model specs and decide which to run ─────────────────────────────
    model_specs   = build_model_specs()
    target_models = args.models if args.models else list(model_specs.keys())

    run_start = time.time()
    for name in target_models:
        if name not in model_specs:
            print(f"  [WARN] '{name}' not recognised — valid names: {list(model_specs)}")
            continue

        already_done = name in saved and not args.force
        if already_done:
            info = saved[name]
            lr   = info["best_lr"] if isinstance(info, dict) else info
            print(f"  [{name}] already in {BEST_LR_PATH} → LR={lr:.0e}  (use --force to redo)")
            continue

        spec = model_specs[name]
        print(f"\n  ── Searching {name}  (loader={spec['loader']}) ──")

        try:
            best_lr, best_auc, all_aucs = search_one_model(
                name, spec["factory"], spec["loader"],
                train_loaders, val_loaders,
                args.epochs, device,
            )
            saved[name] = {
                "best_lr":  best_lr,
                "best_auc": best_auc,
                "all_aucs": {f"{lr:.0e}": auc for lr, auc in all_aucs.items()},
            }
            print(f"  → WINNER  {name}: LR={best_lr:.0e}  (AUC={best_auc:.4f})")

        except Exception as exc:
            fallback = DEFAULT_LR.get(name, 1e-4)
            print(f"  [ERROR] {name}: {exc}")
            print(f"  → Using fallback LR={fallback:.0e}")
            saved[name] = {
                "best_lr":  fallback,
                "best_auc": None,
                "all_aucs": {},
                "error":    str(exc),
            }

        # Save after every model so a crash doesn't lose earlier work
        save_results(BEST_LR_PATH, saved)
        print(f"  Saved → {BEST_LR_PATH}")

    total = (time.time() - run_start) / 60
    print(f"\n{'='*65}")
    print("  GRID SEARCH COMPLETE")
    print(f"{'='*65}")
    print(f"  {'Model':<24} {'Best LR':>10}  {'Best AUC':>10}  {'All AUCs'}")
    print(f"  {'-'*24} {'-'*10}  {'-'*10}  {'-'*30}")
    for name in model_specs:
        if name not in saved:
            continue
        info     = saved[name]
        best_lr  = info["best_lr"]  if isinstance(info, dict) else info
        best_auc = info.get("best_auc", "-") if isinstance(info, dict) else "-"
        aucs_str = "  ".join(
            f"{lr}:{auc}" for lr, auc in info.get("all_aucs", {}).items()
        ) if isinstance(info, dict) else ""
        auc_fmt  = f"{best_auc:.4f}" if isinstance(best_auc, float) else str(best_auc)
        print(f"  {name:<24} {best_lr:>10.0e}  {auc_fmt:>10}  {aucs_str}")

    print(f"\n  Total time : {total:.1f} min")
    print(f"  Saved to   : {BEST_LR_PATH}")
    print("\n  Next step:")
    print("    python train_detector.py --epochs 50 --batch_size 32\n")


if __name__ == "__main__":
    main()
