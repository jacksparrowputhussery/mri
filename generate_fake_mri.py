import os
import random
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates
import torch
import torchvision.transforms as T

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_BASE  = "ADNI_PNG"   # real images
OUTPUT_BASE = "ADNI_Fake"  # fakes — same structure, same filenames
VAE_RATIO   = 0.5          # fraction of images processed via VAE
# ─────────────────────────────────────────────────────────────────────────────


# ── Device ────────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        print("Apple MPS (M1/M2 GPU) detected")
        return "mps"
    elif torch.cuda.is_available():
        print("CUDA GPU detected")
        return "cuda"
    print("Using CPU")
    return "cpu"


# ── Method 1: Subtle image-level perturbations ────────────────────────────────

def perturb_intensity(arr: np.ndarray, strength: float = 0.02) -> np.ndarray:
    """Smooth localised brightness shift — mimics scanner calibration drift."""
    result = arr.astype(np.float32).copy()
    h, w = result.shape[:2]
    mask = np.random.uniform(0, 1, (h // 8, w // 8)).astype(np.float32)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_CUBIC)
    mask = gaussian_filter(mask, sigma=15)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    delta = np.random.uniform(-strength, strength) * 255
    result += mask * delta
    return np.clip(result, 0, 255).astype(np.uint8)


def perturb_texture(arr: np.ndarray, strength: float = 0.015) -> np.ndarray:
    """Subtle Gaussian noise — mimics MRI acquisition noise."""
    result = arr.astype(np.float32).copy()
    noise = np.random.normal(0, strength * 255, result.shape).astype(np.float32)
    noise = gaussian_filter(noise, sigma=0.8)
    result += noise
    return np.clip(result, 0, 255).astype(np.uint8)


def perturb_contrast(arr: np.ndarray, strength: float = 0.03) -> np.ndarray:
    """Tiny localised contrast shift — mimics different tissue weighting."""
    result = arr.astype(np.float32).copy()
    h, w = result.shape[:2]
    cmap = np.random.uniform(1.0 - strength, 1.0 + strength, (h // 8, w // 8))
    cmap = cv2.resize(cmap.astype(np.float32), (w, h))
    cmap = gaussian_filter(cmap, sigma=20)
    mean_val = result.mean()
    result = mean_val + (result - mean_val) * cmap
    return np.clip(result, 0, 255).astype(np.uint8)


def perturb_elastic(arr: np.ndarray, alpha: float = 8.0, sigma: float = 4.0) -> np.ndarray:
    """Tiny elastic deformation — mimics slight head movement."""
    h, w = arr.shape[:2]
    dx = gaussian_filter(np.random.randn(h, w) * alpha, sigma=sigma)
    dy = gaussian_filter(np.random.randn(h, w) * alpha, sigma=sigma)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    ix = np.clip(x + dx, 0, w - 1)
    iy = np.clip(y + dy, 0, h - 1)
    if arr.ndim == 3:
        result = np.stack([
            map_coordinates(arr[:, :, c], [iy.ravel(), ix.ravel()], order=1).reshape(h, w)
            for c in range(arr.shape[2])
        ], axis=2)
    else:
        result = map_coordinates(arr, [iy.ravel(), ix.ravel()], order=1).reshape(h, w)
    return result.astype(np.uint8)


def apply_subtle_perturbations(arr: np.ndarray) -> np.ndarray:
    """Randomly combine a subset of perturbations — keeps fakes diverse."""
    result = arr.copy()
    if random.random() > 0.3:
        result = perturb_intensity(result, strength=random.uniform(0.01, 0.025))
    if random.random() > 0.3:
        result = perturb_texture(result, strength=random.uniform(0.008, 0.018))
    if random.random() > 0.4:
        result = perturb_contrast(result, strength=random.uniform(0.02, 0.04))
    if random.random() > 0.5:
        result = perturb_elastic(result,
                                 alpha=random.uniform(3.0, 8.0),
                                 sigma=random.uniform(3.0, 5.0))
    return result


# ── Method 2: VAE latent perturbation ─────────────────────────────────────────

class VAEPerturber:
    def __init__(self, device: str):
        self.device = device
        self.vae = None

    def load(self):
        from diffusers import AutoencoderKL
        print("\nLoading VAE (stabilityai/sd-vae-ft-mse) ...")
        print("  (~300 MB download on first run — cached after that)\n")
        dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=dtype,
        ).to(self.device)
        self.vae.eval()
        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])
        print("VAE loaded.\n")

    def perturb(self, pil_image: Image.Image, noise_strength: float = 0.04) -> Image.Image:
        original_size = pil_image.size
        rgb = pil_image.convert("RGB")
        tensor = self.transform(rgb).unsqueeze(0)
        dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        tensor = tensor.to(self.device, dtype=dtype)

        with torch.no_grad():
            latent = self.vae.encode(tensor).latent_dist.sample()
            latent = latent + torch.randn_like(latent) * noise_strength
            decoded = self.vae.decode(latent).sample

        decoded = decoded.squeeze(0).float().cpu()
        decoded = (decoded * 0.5 + 0.5).clamp(0, 1)
        decoded_np = (decoded.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        result = Image.fromarray(decoded_np).convert("L").resize(original_size, Image.LANCZOS)
        return result


# ── Main pipeline ─────────────────────────────────────────────────────────────

def generate(input_base: str, output_base: str, use_vae: bool, vae_ratio: float):
    input_root  = Path(input_base)
    output_root = Path(output_base)

    png_files = sorted(input_root.rglob("*.png"))
    total = len(png_files)
    print(f"Found {total} real PNG images in '{input_base}'")
    print(f"Output base   : '{output_base}'")
    print(f"VAE enabled   : {use_vae}  (ratio={vae_ratio:.0%})")
    print()

    device = get_device()

    vae_perturber = None
    if use_vae:
        vae_perturber = VAEPerturber(device)
        vae_perturber.load()

    # Decide which indices use VAE
    vae_count = int(total * vae_ratio)
    indices = list(range(total))
    random.shuffle(indices)
    vae_indices = set(indices[:vae_count])

    skipped = 0

    for idx, png_path in enumerate(tqdm(png_files, desc="Generating fakes")):
        try:
            pil_image = Image.open(png_path).convert("L")
            arr = np.array(pil_image)

            use_vae_here = use_vae and (vae_perturber is not None) and (idx in vae_indices)

            if use_vae_here:
                # VAE latent perturbation
                noise = random.uniform(0.025, 0.055)
                result = vae_perturber.perturb(pil_image, noise_strength=noise)
                result_arr = np.array(result)
                # Light additional texture pass
                if random.random() > 0.5:
                    result_arr = perturb_texture(result_arr, strength=random.uniform(0.005, 0.012))
            else:
                # Image-level subtle perturbations
                result_arr = apply_subtle_perturbations(arr)

            # Mirror exact relative path under output base, keep filename
            rel = png_path.relative_to(input_root)
            out_path = output_root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(result_arr).save(str(out_path))

        except Exception as e:
            print(f"\n  [WARN] Skipped {png_path.name}: {e}")
            skipped += 1

    saved = total - skipped
    print(f"\n{'='*55}")
    print(f"  Total real images : {total}")
    print(f"  Fakes saved       : {saved}")
    print(f"  VAE-based         : {vae_count}")
    print(f"  Perturbation-based: {total - vae_count}")
    print(f"  Skipped / errors  : {skipped}")
    print(f"  Output folder     : {output_root.resolve()}")
    print(f"{'='*55}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subtle Fake Brain MRI Generator")
    parser.add_argument("--input_base",  default=INPUT_BASE,  help="Real images base folder")
    parser.add_argument("--output_base", default=OUTPUT_BASE, help="Fake images base folder")
    parser.add_argument("--vae_ratio",   type=float, default=VAE_RATIO,
                        help="Fraction of images processed via VAE (0.0–1.0)")
    parser.add_argument("--no_vae",      action="store_true",
                        help="Skip VAE entirely — use image perturbations only (no download)")
    args = parser.parse_args()

    generate(
        input_base=args.input_base,
        output_base=args.output_base,
        use_vae=not args.no_vae,
        vae_ratio=args.vae_ratio,
    )
