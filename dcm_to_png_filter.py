"""
Convert DCM files to PNG and filter out mostly-black brain MRI slices.
Images with >= 70% black pixels are discarded.
"""

import os
import numpy as np
import pydicom
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────
INPUT_FOLDER  = "ADNI_Mapped"   # contains .dcm files organised by diagnosis/PTID
OUTPUT_FOLDER = "ADNI_PNG"      # will mirror the same folder structure as PNG
BLACK_THRESHOLD = 70            # % of black pixels above which image is removed
PIXEL_BLACK_VALUE = 10          # pixel intensity <= this value is considered "black"
# ─────────────────────────────────────────────────────────────────────────────


def dcm_to_array(dcm_path: str) -> np.ndarray | None:
    """Read a DICOM file and return a normalised 8-bit grayscale numpy array."""
    try:
        ds = pydicom.dcmread(dcm_path)
        arr = ds.pixel_array.astype(np.float32)

        # Normalise to 0-255
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max == arr_min:          # flat image — all one value
            arr = np.zeros_like(arr, dtype=np.uint8)
        else:
            arr = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)

        return arr
    except Exception as e:
        print(f"  [WARN] Could not read {dcm_path}: {e}")
        return None


def is_mostly_black(arr: np.ndarray, threshold_pct: int, black_value: int) -> bool:
    """Return True if >= threshold_pct% of pixels are <= black_value."""
    total_pixels = arr.size
    black_pixels  = np.sum(arr <= black_value)
    pct_black = (black_pixels / total_pixels) * 100
    return pct_black >= threshold_pct


def process():
    input_root  = Path(INPUT_FOLDER)
    output_root = Path(OUTPUT_FOLDER)

    dcm_files = list(input_root.rglob("*.dcm"))
    print(f"Found {len(dcm_files)} DCM files in '{INPUT_FOLDER}'")
    print(f"Black-pixel threshold : {BLACK_THRESHOLD}%  (pixel value <= {PIXEL_BLACK_VALUE})\n")

    kept    = 0
    removed = 0
    errors  = 0

    for dcm_path in tqdm(dcm_files, desc="Processing"):
        arr = dcm_to_array(str(dcm_path))
        if arr is None:
            errors += 1
            continue

        if is_mostly_black(arr, BLACK_THRESHOLD, PIXEL_BLACK_VALUE):
            removed += 1
            continue

        # Build mirrored output path  (.dcm → .png)
        relative = dcm_path.relative_to(input_root)
        png_path = output_root / relative.with_suffix(".png")
        png_path.parent.mkdir(parents=True, exist_ok=True)

        Image.fromarray(arr).save(str(png_path))
        kept += 1

    total = kept + removed + errors
    print(f"\n{'='*55}")
    print(f"  Total DCM files   : {total}")
    print(f"  Saved as PNG      : {kept}")
    print(f"  Removed (≥{BLACK_THRESHOLD}% black): {removed}")
    print(f"  Errors / skipped  : {errors}")
    print(f"  Output folder     : {output_root.resolve()}")
    print(f"{'='*55}")


if __name__ == "__main__":
    process()
