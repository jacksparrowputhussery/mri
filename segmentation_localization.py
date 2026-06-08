import os, sys, csv, time, random, logging, warnings, platform
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import timm

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, accuracy_score
)

# ── Optional libs ─────────────────────────────────────────────────────────────
try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
    from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
    GRADCAM_OK = True
except ImportError:
    GRADCAM_OK = False

try:
    from captum.attr import IntegratedGradients, NoiseTunnel
    CAPTUM_OK = True
except ImportError:
    CAPTUM_OK = False

try:
    from diffusers import AutoencoderKL
    DIFFUSERS_OK = True
except ImportError:
    DIFFUSERS_OK = False

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
REAL_DIR     = "ADNI_PNG"
FAKE_DIR     = "ADNI_Fake"
OUT_DIR      = "results/segmentation"
CKPT_DIR     = "results/checkpoints"
IMG_SIZE     = 224
N_SAMPLES    = 2952     # full dataset — all matched real/fake pairs
GT_THRESHOLD = 3        # abs pixel diff threshold for GT mask (0–255)
                        # real vs fake diffs: 90th%=6, 95th%=12 — use 3 for ~17% coverage
N_SAMPLES_IG = 500      # cap for slow Integrated Gradients / SmoothGrad (Fix 3)
PATCH_SIZE   = 16       # patch size for patch heatmap approach
SEED         = 42
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED);  np.random.seed(SEED);  torch.manual_seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════════════
def setup_logger(out_dir):
    logs_dir = os.path.join(out_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, f"seg_{run_id}.log")

    log = logging.getLogger("seg_suite")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    log.addHandler(ch);  log.addHandler(fh)
    log.log_path = log_path;  log.run_id = run_id
    return log


# ══════════════════════════════════════════════════════════════════════════════
#  DEVICE
# ══════════════════════════════════════════════════════════════════════════════
def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps"),  "Apple MPS"
    if torch.cuda.is_available():         return torch.device("cuda"), f"CUDA {torch.cuda.get_device_name(0)}"
    return torch.device("cpu"), "CPU"


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL BUILDERS  (mirrors train_detector.py)
# ══════════════════════════════════════════════════════════════════════════════
class FrequencyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4*4, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 1),
        )
    def forward(self, x): return self.classifier(self.features(x))


MODEL_REGISTRY = {
    "EfficientNet-B0":    lambda: timm.create_model("efficientnet_b0",               pretrained=False, num_classes=1),
    "XceptionNet":        lambda: timm.create_model("xception",                       pretrained=False, num_classes=1),
    "ViT-B-16":           lambda: timm.create_model("vit_base_patch16_224",           pretrained=False, num_classes=1),
    "FrequencyCNN":       FrequencyCNN,
    "EfficientNetV2-B2":  lambda: timm.create_model("tf_efficientnetv2_b2",           pretrained=False, num_classes=1),
    "ConvNeXt-Tiny":      lambda: timm.create_model("convnext_tiny",                  pretrained=False, num_classes=1),
    "Swin-Tiny":          lambda: timm.create_model("swin_tiny_patch4_window7_224",   pretrained=False, num_classes=1),
    "DeiT-Small":         lambda: timm.create_model("deit_small_patch16_224",         pretrained=False, num_classes=1),
}

# Which models have saved checkpoints
CKPT_MODELS = [
    n for n in MODEL_REGISTRY
    if os.path.exists(os.path.join(CKPT_DIR, f"{n}_best.pt"))
]


def load_model(name, device):
    model = MODEL_REGISTRY[name]()
    ckpt  = os.path.join(CKPT_DIR, f"{name}_best.pt")
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model.to(device)


# ══════════════════════════════════════════════════════════════════════════════
#  GRAD-CAM TARGET LAYERS
# ══════════════════════════════════════════════════════════════════════════════
def get_target_layer_and_reshape(name, model):
    """
    Returns (target_layers_list, reshape_transform_or_None).
    Reshape transform is needed for ViT/Swin/DeiT to convert
    sequence tokens back to a 2-D spatial map.
    """
    def vit_reshape(tensor, h=14, w=14):
        # Remove CLS token, reshape to [B, C, H, W]
        result = tensor[:, 1:, :].reshape(tensor.size(0), h, w, tensor.size(2))
        return result.permute(0, 3, 1, 2)

    def swin_reshape(tensor, h=7, w=7):
        result = tensor.reshape(tensor.size(0), h, w, tensor.size(2))
        return result.permute(0, 3, 1, 2)

    mapping = {
        "EfficientNet-B0":   (lambda m: [m.conv_head],                     None),
        "XceptionNet":       (lambda m: [m.act4],                           None),
        "ViT-B-16":          (lambda m: [m.blocks[-1].norm1],               vit_reshape),
        "FrequencyCNN":      (lambda m: [m.features[12]],                   None),   # last Conv2d before avgpool
        "EfficientNetV2-B2": (lambda m: [m.conv_head],                      None),
        # Fix 4: hook entire last stage — .blocks[-1].norm was LayerNorm output
        # which EigenCAM can't use properly; hooking the stage gives raw features
        "ConvNeXt-Tiny":     (lambda m: [m.stages[-1]],                     None),
        "Swin-Tiny":         (lambda m: [m.layers[-1].blocks[-1].norm2],    swin_reshape),
        "DeiT-Small":        (lambda m: [m.blocks[-1].norm1],               vit_reshape),
    }
    if name not in mapping:
        return None, None
    layer_fn, reshape = mapping[name]
    try:
        layers = layer_fn(model)
        return layers, reshape
    except Exception:
        return None, None


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════
RGB_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def img_to_tensor(pil_img, freq=False):
    """Convert a PIL grayscale image to model input tensor [1, C, H, W]."""
    if freq:
        arr = np.array(pil_img.convert("L").resize((IMG_SIZE, IMG_SIZE)),
                       dtype=np.float32) / 255.0
        fft = np.fft.fftshift(np.fft.fft2(arr))
        mag = ((np.log1p(np.abs(fft))) .astype(np.float32))
        mag = (mag - mag.mean()) / (mag.std() + 1e-8)
        return torch.tensor(mag).unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
    else:
        return RGB_TRANSFORM(pil_img.convert("RGB")).unsqueeze(0)  # [1,3,H,W]


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET PAIRS
# ══════════════════════════════════════════════════════════════════════════════
def find_image_pairs(real_dir, fake_dir, n=N_SAMPLES):
    """
    Finds matching (real_path, fake_path) pairs by walking ADNI_PNG/
    and checking the same relative path exists in ADNI_Fake/.
    Returns a random sample of n pairs.
    """
    real_root = Path(real_dir)
    fake_root = Path(fake_dir)
    pairs = []
    for rp in real_root.rglob("*.png"):
        rel  = rp.relative_to(real_root)
        fp   = fake_root / rel
        if fp.exists():
            pairs.append((rp, fp))
    random.shuffle(pairs)
    return pairs[:n]


# ══════════════════════════════════════════════════════════════════════════════
#  GROUND TRUTH MASK
# ══════════════════════════════════════════════════════════════════════════════
def make_gt_mask(real_pil, fake_pil, threshold=GT_THRESHOLD):
    """
    Binary GT mask: pixels where |real - fake| > threshold.
    Applies morphological close to remove tiny isolated pixels.
    Returns uint8 array [H, W] with values 0 / 255.
    """
    r = np.array(real_pil.convert("L").resize((IMG_SIZE, IMG_SIZE)),
                 dtype=np.int32)
    f = np.array(fake_pil.convert("L").resize((IMG_SIZE, IMG_SIZE)),
                 dtype=np.int32)
    diff = np.abs(r - f).astype(np.uint8)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask   # [H, W] uint8


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(gt_mask, pred_heatmap):
    """
    gt_mask     : uint8 [H, W] 0/255
    pred_heatmap: float [H, W] in [0, 1]
    Returns dict with: IoU, Dice, PixelAcc, Precision, Recall, F1, AUC_ROC, AP
    """
    gt_bin  = (gt_mask > 127).astype(np.uint8).ravel()
    ph_flat = pred_heatmap.ravel().astype(np.float32)

    # Handle edge case: GT all zeros or all ones → AUC undefined
    if gt_bin.sum() == 0 or gt_bin.sum() == len(gt_bin):
        auc = ap = float("nan")
    else:
        auc = roc_auc_score(gt_bin, ph_flat)
        ap  = average_precision_score(gt_bin, ph_flat)

    # Fix 2: Otsu adaptive threshold — much better than fixed 0.5
    # Fixed 0.5 caused IoU=0 on 11.9% of images where heatmap max < 0.5
    heatmap_2d   = (ph_flat * 255).reshape(IMG_SIZE, IMG_SIZE).astype(np.uint8)
    _, pred_bin_2d = cv2.threshold(
        heatmap_2d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    pred_bin = (pred_bin_2d.ravel() > 0).astype(np.uint8)

    inter = np.logical_and(gt_bin, pred_bin).sum()
    union = np.logical_or(gt_bin, pred_bin).sum()
    iou   = inter / (union + 1e-8)
    dice  = 2 * inter / (gt_bin.sum() + pred_bin.sum() + 1e-8)

    return {
        "IoU":        round(float(iou), 4),
        "Dice":       round(float(dice), 4),
        "PixelAcc":   round(float(accuracy_score(gt_bin, pred_bin)), 4),
        "Precision":  round(float(precision_score(gt_bin, pred_bin, zero_division=0)), 4),
        "Recall":     round(float(recall_score(gt_bin,    pred_bin, zero_division=0)), 4),
        "F1":         round(float(f1_score(gt_bin,        pred_bin, zero_division=0)), 4),
        "AUC_ROC":    round(float(auc), 4) if not np.isnan(auc) else "nan",
        "AP":         round(float(ap),  4) if not np.isnan(ap)  else "nan",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION  (6-panel)
# ══════════════════════════════════════════════════════════════════════════════
def save_vis(real_pil, fake_pil, gt_mask, pred_heatmap,
             metrics, title, save_path):
    """
    6 panels:
      Col 1: Real image
      Col 2: Fake image
      Col 3: Ground-truth mask
      Col 4: Predicted heatmap
      Col 5: Heatmap overlay on real
      Col 6: Heatmap overlay on fake
    """
    r_arr = np.array(real_pil.convert("L").resize((IMG_SIZE, IMG_SIZE)))
    f_arr = np.array(fake_pil.convert("L").resize((IMG_SIZE, IMG_SIZE)))

    # Heatmap colourised
    heat_colour = mplcm.jet(pred_heatmap)[:, :, :3]   # [H,W,3] float

    def overlay(grey, heat, alpha=0.5):
        base  = np.stack([grey/255.]*3, axis=2)
        return np.clip(base*(1-alpha) + heat*alpha, 0, 1)

    fig, axes = plt.subplots(1, 6, figsize=(22, 4))
    fig.suptitle(title, fontsize=9)

    panels = [
        (r_arr,                    "Real",           "gray"),
        (f_arr,                    "Fake",           "gray"),
        (gt_mask,                  "GT Mask",        "gray"),
        (pred_heatmap,             "Pred Heatmap",   "jet"),
        (overlay(r_arr, heat_colour), "Overlay Real", None),
        (overlay(f_arr, heat_colour), "Overlay Fake", None),
    ]
    for ax, (img, lbl, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1 if img.max() <= 1 else 255)
        ax.set_title(lbl, fontsize=8)
        ax.axis("off")

    # Metrics text box
    m_text = " | ".join(f"{k}:{v}" for k, v in metrics.items())
    fig.text(0.5, 0.01, m_text, ha="center", fontsize=7,
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  APPROACH 1-3: GradCAM / GradCAM++ / EigenCAM
# ══════════════════════════════════════════════════════════════════════════════
def run_gradcam_approach(cam_class, approach_name, pairs,
                         device, out_dir, log, all_rows):
    if not GRADCAM_OK:
        log.info(f"  ⚠ Skipping {approach_name} — grad-cam not installed")
        return

    log.info(f"\n{'─'*60}")
    log.info(f"  Approach : {approach_name}")
    log.info(f"{'─'*60}")

    for model_name in CKPT_MODELS:
        is_freq = (model_name == "FrequencyCNN")
        t0 = time.time()
        log.info(f"  [{approach_name}] Loading {model_name} ...")

        model = load_model(model_name, device)
        target_layers, reshape_fn = get_target_layer_and_reshape(model_name, model)

        if target_layers is None:
            log.info(f"  [{approach_name}] {model_name} — no target layer, skipping")
            continue

        tag      = f"{approach_name}_{model_name}"
        img_dir  = os.path.join(out_dir, tag)
        os.makedirs(img_dir, exist_ok=True)

        # Wrap model for grad-cam (expects batch output [B,1] → needs [B])
        class ModelWrap(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x): return self.m(x).squeeze(1)

        try:
            cam = cam_class(
                model=ModelWrap(model),
                target_layers=target_layers,
                reshape_transform=reshape_fn,
            )
        except Exception as e:
            log.info(f"  [{approach_name}] {model_name} CAM init failed: {e}")
            continue

        batch_metrics = []
        for idx, (rp, fp) in enumerate(pairs):
            real_pil = Image.open(rp)
            fake_pil = Image.open(fp)
            gt_mask  = make_gt_mask(real_pil, fake_pil)

            tensor = img_to_tensor(fake_pil, freq=is_freq).to(device)

            try:
                # targets=[BinaryClassifierOutputTarget(1)] → maximise "fake"
                grayscale = cam(
                    input_tensor=tensor,
                    targets=[BinaryClassifierOutputTarget(1)],
                )
                heatmap = grayscale[0]   # [H, W] float 0-1
            except Exception as e:
                log.debug(f"  [{tag}] img {idx} failed: {e}")
                continue

            # Resize heatmap to IMG_SIZE × IMG_SIZE
            heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
            heatmap = np.clip(heatmap, 0, 1)

            m = compute_metrics(gt_mask, heatmap)
            batch_metrics.append(m)

            # Save predicted mask
            mask_path = os.path.join(img_dir, f"{idx:04d}_pred_mask.png")
            cv2.imwrite(mask_path, (heatmap * 255).astype(np.uint8))

            # Save visualisation (every 5th image to save disk space)
            if idx % 5 == 0:
                vis_path = os.path.join(img_dir, f"{idx:04d}_vis.png")
                save_vis(real_pil, fake_pil, gt_mask, heatmap,
                         m, f"{tag} | img {idx}", vis_path)

            # CSV row
            all_rows.append({
                "run_id": log.run_id, "approach": approach_name,
                "model": model_name, "img_idx": idx,
                "real_path": str(rp), **m
            })
            log.debug(f"  [{tag}] img {idx:03d}  IoU={m['IoU']}  Dice={m['Dice']}  F1={m['F1']}  AUC={m['AUC_ROC']}")

        elapsed = time.time() - t0
        if batch_metrics:
            means = {k: round(np.nanmean([r[k] for r in batch_metrics
                                          if r[k] != "nan"]), 4)
                     for k in batch_metrics[0]}
            log.info(f"  [{approach_name}] {model_name}  "
                     f"IoU={means['IoU']}  Dice={means['Dice']}  "
                     f"F1={means['F1']}  AUC={means['AUC_ROC']}  "
                     f"({elapsed:.0f}s)")
        del model


# ══════════════════════════════════════════════════════════════════════════════
#  APPROACH 4-5: Integrated Gradients / SmoothGrad  (Captum)
# ══════════════════════════════════════════════════════════════════════════════
def run_captum_approach(approach_name, pairs, device, out_dir, log, all_rows):
    if not CAPTUM_OK:
        log.info(f"  ⚠ Skipping {approach_name} — captum not installed")
        return

    log.info(f"\n{'─'*60}")
    log.info(f"  Approach : {approach_name}")
    log.info(f"{'─'*60}")

    # Run on CPU — captum has MPS limitations
    cap_device = torch.device("cpu")

    # Use best CNN models only (captum is slow on large models)
    target_models = [m for m in ["EfficientNet-B0", "XceptionNet", "ConvNeXt-Tiny"]
                     if m in CKPT_MODELS]

    for model_name in target_models:
        t0 = time.time()
        log.info(f"  [{approach_name}] {model_name} (running on CPU) ...")

        model = load_model(model_name, cap_device)

        class ScalarOut(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x): return torch.sigmoid(self.m(x))   # [B,1]

        wrapped = ScalarOut(model)

        ig  = IntegratedGradients(wrapped)
        sg  = NoiseTunnel(ig) if approach_name == "SmoothGrad" else None

        tag     = f"{approach_name}_{model_name}"
        img_dir = os.path.join(out_dir, tag)
        os.makedirs(img_dir, exist_ok=True)

        # Fix 3: cap slow captum approaches to N_SAMPLES_IG images
        # ~16s/image × 2952 = 13+ hrs — capped at 500 = ~2.2 hrs
        ig_pairs = pairs[:N_SAMPLES_IG]
        log.info(f"  [{approach_name}] Capped at {len(ig_pairs)} images (Fix 3 — speed)")

        batch_metrics = []
        for idx, (rp, fp) in enumerate(ig_pairs):
            real_pil = Image.open(rp)
            fake_pil = Image.open(fp)
            gt_mask  = make_gt_mask(real_pil, fake_pil)

            tensor    = img_to_tensor(fake_pil).to(cap_device)
            baseline  = torch.zeros_like(tensor)

            try:
                if approach_name == "SmoothGrad":
                    attrs = sg.attribute(
                        tensor, nt_type="smoothgrad", nt_samples=10,
                        stdevs=0.1, baselines=baseline, target=None
                    )
                else:
                    # Fix 3: reduced n_steps 30 → 15 (halves time, minimal accuracy loss)
                    attrs = ig.attribute(tensor, baselines=baseline, target=None,
                                         n_steps=15, internal_batch_size=1)

                # Sum abs attributions across channels → [H, W]
                attr_map = attrs.squeeze(0).abs().sum(dim=0).detach().numpy()
                attr_map = cv2.resize(attr_map, (IMG_SIZE, IMG_SIZE))
                # Normalize to [0, 1]
                mn, mx = attr_map.min(), attr_map.max()
                heatmap = (attr_map - mn) / (mx - mn + 1e-8)

            except Exception as e:
                log.debug(f"  [{tag}] img {idx} failed: {e}")
                continue

            m = compute_metrics(gt_mask, heatmap)
            batch_metrics.append(m)

            mask_path = os.path.join(img_dir, f"{idx:04d}_pred_mask.png")
            cv2.imwrite(mask_path, (heatmap * 255).astype(np.uint8))

            if idx % 5 == 0:
                vis_path = os.path.join(img_dir, f"{idx:04d}_vis.png")
                save_vis(real_pil, fake_pil, gt_mask, heatmap,
                         m, f"{tag} | img {idx}", vis_path)

            all_rows.append({
                "run_id": log.run_id, "approach": approach_name,
                "model": model_name, "img_idx": idx,
                "real_path": str(rp), **m
            })
            log.debug(f"  [{tag}] img {idx:03d}  IoU={m['IoU']}  F1={m['F1']}  AUC={m['AUC_ROC']}")

        elapsed = time.time() - t0
        if batch_metrics:
            means = {k: round(np.nanmean([r[k] for r in batch_metrics
                                          if r[k] != "nan"]), 4)
                     for k in batch_metrics[0]}
            log.info(f"  [{approach_name}] {model_name}  "
                     f"IoU={means['IoU']}  Dice={means['Dice']}  "
                     f"F1={means['F1']}  AUC={means['AUC_ROC']}  "
                     f"({elapsed:.0f}s)")
        del model


# ══════════════════════════════════════════════════════════════════════════════
#  APPROACH 6: VAE Reconstruction Error
# ══════════════════════════════════════════════════════════════════════════════
def run_vae_recon(pairs, device, out_dir, log, all_rows):
    if not DIFFUSERS_OK:
        log.info("  ⚠ Skipping VAE Recon — diffusers not installed")
        return

    log.info(f"\n{'─'*60}")
    log.info("  Approach : VAE-ReconError")
    log.info(f"{'─'*60}")
    log.info("  Loading VAE (stabilityai/sd-vae-ft-mse) ...")

    dtype = torch.float16 if str(device) in ("cuda", "mps") else torch.float32
    try:
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse", torch_dtype=dtype
        ).to(device)
        vae.eval()
    except Exception as e:
        log.info(f"  ⚠ VAE load failed: {e}")
        return

    vae_tf = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    tag     = "VAEReconError_None"
    img_dir = os.path.join(out_dir, tag)
    os.makedirs(img_dir, exist_ok=True)
    t0 = time.time()

    batch_metrics = []
    for idx, (rp, fp) in enumerate(pairs):
        real_pil = Image.open(rp)
        fake_pil = Image.open(fp)
        gt_mask  = make_gt_mask(real_pil, fake_pil)

        def recon_error(pil_img):
            tensor = vae_tf(pil_img.convert("RGB")).unsqueeze(0).to(device, dtype)
            with torch.no_grad():
                lat = vae.encode(tensor).latent_dist.sample()
                out = vae.decode(lat).sample
            out   = out.squeeze(0).float().cpu()
            orig  = vae_tf(pil_img.convert("RGB"))
            err   = (out - orig).abs().mean(dim=0).numpy()   # [H, W]
            return err

        try:
            # Error map on FAKE image — high error = reconstructed differently = fake region
            err_fake = recon_error(fake_pil.resize((512,512)))
            err_fake = cv2.resize(err_fake, (IMG_SIZE, IMG_SIZE))
            mn, mx   = err_fake.min(), err_fake.max()
            heatmap  = (err_fake - mn) / (mx - mn + 1e-8)
        except Exception as e:
            log.debug(f"  [VAE] img {idx} failed: {e}")
            continue

        m = compute_metrics(gt_mask, heatmap)
        batch_metrics.append(m)

        mask_path = os.path.join(img_dir, f"{idx:04d}_pred_mask.png")
        cv2.imwrite(mask_path, (heatmap * 255).astype(np.uint8))

        if idx % 5 == 0:
            vis_path = os.path.join(img_dir, f"{idx:04d}_vis.png")
            save_vis(real_pil, fake_pil, gt_mask, heatmap,
                     m, f"VAEReconError | img {idx}", vis_path)

        all_rows.append({
            "run_id": log.run_id, "approach": "VAEReconError",
            "model": "None", "img_idx": idx, "real_path": str(rp), **m
        })
        log.debug(f"  [VAE] img {idx:03d}  IoU={m['IoU']}  F1={m['F1']}  AUC={m['AUC_ROC']}")

    elapsed = time.time() - t0
    if batch_metrics:
        means = {k: round(np.nanmean([r[k] for r in batch_metrics
                                      if r[k] != "nan"]), 4)
                 for k in batch_metrics[0]}
        log.info(f"  [VAEReconError]  "
                 f"IoU={means['IoU']}  Dice={means['Dice']}  "
                 f"F1={means['F1']}  AUC={means['AUC_ROC']}  ({elapsed:.0f}s)")
    del vae


# ══════════════════════════════════════════════════════════════════════════════
#  APPROACH 7: Patch Classification Heatmap
# ══════════════════════════════════════════════════════════════════════════════
def run_patch_heatmap(pairs, device, out_dir, log, all_rows):
    log.info(f"\n{'─'*60}")
    log.info("  Approach : PatchHeatmap")
    log.info(f"{'─'*60}")

    best_model_name = next(
        (m for m in ["XceptionNet", "EfficientNet-B0", "ConvNeXt-Tiny"]
         if m in CKPT_MODELS), None
    )
    if best_model_name is None:
        log.info("  ⚠ No checkpoint found for PatchHeatmap")
        return

    log.info(f"  Using model : {best_model_name}  |  Patch size: {PATCH_SIZE}×{PATCH_SIZE}")
    model = load_model(best_model_name, device)

    tag     = f"PatchHeatmap_{best_model_name}"
    img_dir = os.path.join(out_dir, tag)
    os.makedirs(img_dir, exist_ok=True)
    t0 = time.time()

    n_patches = IMG_SIZE // PATCH_SIZE   # e.g. 224//16 = 14
    batch_metrics = []

    for idx, (rp, fp) in enumerate(pairs):
        real_pil = Image.open(rp)
        fake_pil = Image.open(fp)
        gt_mask  = make_gt_mask(real_pil, fake_pil)

        fake_rgb  = fake_pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        fake_arr  = np.array(fake_rgb)
        heatmap   = np.zeros((n_patches, n_patches), dtype=np.float32)

        for pi in range(n_patches):
            for pj in range(n_patches):
                y0, y1 = pi*PATCH_SIZE, (pi+1)*PATCH_SIZE
                x0, x1 = pj*PATCH_SIZE, (pj+1)*PATCH_SIZE
                patch     = fake_arr[y0:y1, x0:x1]
                patch_pil = Image.fromarray(patch)
                tensor    = img_to_tensor(patch_pil).to(device)
                with torch.no_grad():
                    prob = torch.sigmoid(model(tensor)).item()
                heatmap[pi, pj] = prob

        # Upsample heatmap back to IMG_SIZE × IMG_SIZE
        heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_LINEAR)
        heatmap = np.clip(heatmap, 0, 1)

        m = compute_metrics(gt_mask, heatmap)
        batch_metrics.append(m)

        mask_path = os.path.join(img_dir, f"{idx:04d}_pred_mask.png")
        cv2.imwrite(mask_path, (heatmap * 255).astype(np.uint8))

        if idx % 5 == 0:
            vis_path = os.path.join(img_dir, f"{idx:04d}_vis.png")
            save_vis(real_pil, fake_pil, gt_mask, heatmap,
                     m, f"PatchHeatmap | img {idx}", vis_path)

        all_rows.append({
            "run_id": log.run_id, "approach": "PatchHeatmap",
            "model": best_model_name, "img_idx": idx,
            "real_path": str(rp), **m
        })
        log.debug(f"  [PatchHeatmap] img {idx:03d}  IoU={m['IoU']}  F1={m['F1']}  AUC={m['AUC_ROC']}")

    elapsed = time.time() - t0
    if batch_metrics:
        means = {k: round(np.nanmean([r[k] for r in batch_metrics
                                      if r[k] != "nan"]), 4)
                 for k in batch_metrics[0]}
        log.info(f"  [PatchHeatmap]  "
                 f"IoU={means['IoU']}  Dice={means['Dice']}  "
                 f"F1={means['F1']}  AUC={means['AUC_ROC']}  ({elapsed:.0f}s)")
    del model


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE CSVs  +  SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
METRIC_KEYS = ["IoU","Dice","PixelAcc","Precision","Recall","F1","AUC_ROC","AP"]

def save_all_csv(all_rows, out_dir, log):
    path = os.path.join(out_dir, "metrics_all.csv")
    fieldnames = ["run_id","approach","model","img_idx","real_path"] + METRIC_KEYS
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    log.info(f"\n  All metrics CSV    → {path}")
    return path


def save_summary_csv(all_rows, out_dir, log):
    # Group by (approach, model) → mean ± std
    from collections import defaultdict
    groups = defaultdict(list)
    for row in all_rows:
        key = (row["approach"], row["model"])
        groups[key].append(row)

    path = os.path.join(out_dir, "metrics_summary.csv")
    cols = ["approach", "model", "n_images"]
    for m in METRIC_KEYS:
        cols += [f"{m}_mean", f"{m}_std"]

    rows_out = []
    for (approach, model), records in sorted(groups.items()):
        r = {"approach": approach, "model": model, "n_images": len(records)}
        for m in METRIC_KEYS:
            vals = [float(rec[m]) for rec in records
                    if rec[m] not in ("nan", float("nan"))]
            r[f"{m}_mean"] = round(np.mean(vals), 4)  if vals else "nan"
            r[f"{m}_std"]  = round(np.std(vals),  4)  if vals else "nan"
        rows_out.append(r)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows_out)
    log.info(f"  Summary CSV        → {path}")

    # Print table to console
    log.info(f"\n{'='*80}")
    log.info("  SEGMENTATION LOCALIZATION — FINAL SUMMARY")
    log.info(f"{'='*80}")
    log.info(f"  {'Approach':<20} {'Model':<22} {'N':>4}  "
             f"{'IoU':>7} {'Dice':>7} {'F1':>7} {'AUC':>7} {'AP':>7}")
    log.info(f"  {'-'*20} {'-'*22} {'-'*4}  "
             f"{'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for r in rows_out:
        log.info(f"  {r['approach']:<20} {r['model']:<22} {r['n_images']:>4}  "
                 f"{r.get('IoU_mean','nan'):>7} {r.get('Dice_mean','nan'):>7} "
                 f"{r.get('F1_mean','nan'):>7} {r.get('AUC_ROC_mean','nan'):>7} "
                 f"{r.get('AP_mean','nan'):>7}")
    log.info(f"{'='*80}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main(n_samples, gt_threshold, skip_vae):
    global N_SAMPLES, GT_THRESHOLD
    N_SAMPLES, GT_THRESHOLD = n_samples, gt_threshold

    os.makedirs(OUT_DIR, exist_ok=True)
    log    = setup_logger(OUT_DIR)
    device, dev_name = get_device()

    log.info("="*65)
    log.info("  Forgery Localization & Segmentation Suite")
    log.info(f"  Run ID      : {log.run_id}")
    log.info(f"  Log         : {log.log_path}")
    log.info("="*65)
    log.info(f"  Device      : {dev_name}")
    log.info(f"  Samples     : {n_samples} image pairs")
    log.info(f"  GT threshold: {gt_threshold} px diff")
    log.info(f"  Checkpoints : {CKPT_MODELS}")
    log.info(f"  grad-cam    : {GRADCAM_OK}   captum: {CAPTUM_OK}   diffusers: {DIFFUSERS_OK}")

    log.info(f"\n  Finding image pairs ...")
    pairs = find_image_pairs(REAL_DIR, FAKE_DIR, n=n_samples)
    log.info(f"  Found {len(pairs)} matching real/fake pairs")

    if not pairs:
        log.info("  ❌ No pairs found — check REAL_DIR and FAKE_DIR paths")
        return

    # GT mask stats
    gt_sizes = []
    for rp, fp in pairs[:10]:
        m = make_gt_mask(Image.open(rp), Image.open(fp), gt_threshold)
        gt_sizes.append((m > 127).sum())
    log.info(f"  GT mask non-zero px (sample 10): mean={np.mean(gt_sizes):.0f}  "
             f"min={min(gt_sizes)}  max={max(gt_sizes)}")

    all_rows  = []
    run_start = time.time()

    # ── 1. GradCAM ────────────────────────────────────────────────────────────
    run_gradcam_approach(GradCAM,       "GradCAM",    pairs, device, OUT_DIR, log, all_rows)

    # ── 2. GradCAM++ ─────────────────────────────────────────────────────────
    run_gradcam_approach(GradCAMPlusPlus,"GradCAM++", pairs, device, OUT_DIR, log, all_rows)

    # ── 3. EigenCAM ───────────────────────────────────────────────────────────
    run_gradcam_approach(EigenCAM,      "EigenCAM",   pairs, device, OUT_DIR, log, all_rows)

    # ── 4. Integrated Gradients ───────────────────────────────────────────────
    run_captum_approach("IntegratedGradients", pairs, device, OUT_DIR, log, all_rows)

    # ── 5. SmoothGrad ─────────────────────────────────────────────────────────
    run_captum_approach("SmoothGrad",   pairs, device, OUT_DIR, log, all_rows)

    # ── 6. VAE Recon Error ────────────────────────────────────────────────────
    if not skip_vae:
        run_vae_recon(pairs, device, OUT_DIR, log, all_rows)
    else:
        log.info("\n  Skipping VAE Recon (--no_vae flag set)")

    # ── 7. Patch Heatmap ──────────────────────────────────────────────────────
    run_patch_heatmap(pairs, device, OUT_DIR, log, all_rows)

    # ── Save results ──────────────────────────────────────────────────────────
    save_all_csv(all_rows, OUT_DIR, log)
    save_summary_csv(all_rows, OUT_DIR, log)

    total = (time.time() - run_start) / 60
    log.info(f"\n  Output folder : ./{OUT_DIR}/")
    log.info(f"  Total time    : {total:.1f} min")
    log.info(f"  Completed at  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Forgery Localization Suite")
    parser.add_argument("--n_samples",    type=int,  default=2952,
                        help="Number of image pairs to evaluate (default 2952 = full dataset)")
    parser.add_argument("--gt_threshold", type=int,  default=3,
                        help="Pixel diff threshold for GT mask 0-255 (default 3)")
    parser.add_argument("--no_vae",       action="store_true",
                        help="Skip VAE Recon approach (saves ~300MB download)")
    args = parser.parse_args()
    main(args.n_samples, args.gt_threshold, args.no_vae)
