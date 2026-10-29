__author__ = "Shirin Heidarikahkesh"
__copyright__ = "Copyright 2025-2026, Universitätsklinikum Erlangen"

from pathlib import Path
import argparse
import json
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import subprocess
import re

import numpy as np
import pydicom
import scipy.ndimage as ndi
import shutil

try:
    import nibabel as nib
    HAVE_NIB = True
except ImportError:
    HAVE_NIB = False

try:
    from nibabel.processing import resample_from_to
    HAVE_NIB_PROCESSING = True
except Exception:
    HAVE_NIB_PROCESSING = False

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
HMRGC_ROOT = THIS_DIR / "fieldmapping-hmrGC"
sys.path.insert(0, str(HMRGC_ROOT))

from hmrGC.dixon_imaging import MultiEcho

try:
    from hmrGC.constants import GYRO
except Exception:
    # Hz per ppm per Tesla for proton frequency shift.
    # 42.57747892 Hz/T/ppm gives ~298 Hz/ppm at 7T.
    GYRO = 42.57747892

from copy import deepcopy
import hmrGC.dixon_imaging.multi_echo as me_mod

# -------------------------------------------------------------------------
# hmrGC compatibility patch
# -------------------------------------------------------------------------

def _set_phi_matrix_with_frequency_reference(self):
    """
    hmrGC signal model with optional non-water frequency reference.

    Default behavior is unchanged:
      WaterFreq_ppm = 0.0 -> water-peak-referenced data

    For fat-peak-reconstructed data:
      WaterFreq_ppm = +3.30 when the main fat peak is used as 0 ppm.
    """
    options = getattr(self, "options", {}) or {}
    signal_model = options.get("signal_model", self.params.get("signal_model"))

    if signal_model == "WF":
        num_species = 2
    elif signal_model == "WFS":
        num_species = 3
    elif signal_model in {"W", "S"}:
        num_species = 1
    else:
        raise RuntimeError(f"Unsupported signal_model: {signal_model}")

    te_ = self.params["TE_s"]
    field_strength = self.params["fieldStrength_T"]

    water_freq_ppm = float(self.params.get("WaterFreq_ppm", 0.0))
    water_delta_f = GYRO * water_freq_ppm * field_strength

    phi = np.ones((len(te_), num_species), dtype=np.complex64)

    for ii, te in enumerate(te_):
        if signal_model in {"W", "WF", "WFS"}:
            phi[ii, 0] = np.exp(2.0j * np.pi * water_delta_f * te)

        if signal_model in {"WF", "WFS"}:
            freqs_ppm = np.asarray(self.params["FatModel"]["freqs_ppm"], dtype=float)
            rel_amps = np.asarray(self.params["FatModel"]["relAmps"], dtype=float)

            delta_f = GYRO * freqs_ppm * field_strength

            fatphasor = 0.0
            for freq_hz, amp in zip(delta_f, rel_amps):
                fatphasor += amp * np.exp(2.0j * np.pi * freq_hz * te)

            phi[ii, 1] = fatphasor

        if signal_model == "WFS":
            delta_f = GYRO * np.asarray(self.params["siliconePeak_ppm"], dtype=float) * field_strength
            phi[ii, 2] = np.exp(2.0j * np.pi * delta_f * te)

        if signal_model == "S":
            delta_f = GYRO * self.params["siliconePeak_ppm"] * field_strength
            phi[ii, 0] = np.exp(2.0j * np.pi * delta_f * te)

    self.phi = phi


MultiEcho.set_phi_matrix = _set_phi_matrix_with_frequency_reference

# -------------------------------------------------------------------------
# Defaults
# -------------------------------------------------------------------------

# Default fat model (your current values)
DEFAULT_FAT_FREQS_PPM = -np.array(
    [3.70, 3.30, 3.01, 2.57, 2.53, 1.83, -0.71],
    dtype=float,
)
DEFAULT_FAT_REL_AMPS = np.array(
    [0.087, 0.693, 0.128, 0.004, 0.039, 0.039, 0.010],
    dtype=float,
)
DEFAULT_MAIN_FAT_PEAK_PPM = -3.30

import scipy.ndimage as ndi

def keep_largest_cc(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, index=np.arange(1, n+1))
    largest = 1 + int(np.argmax(sizes))
    return lab == largest

# -------------------------------------------------------------------------
# Basic helpers
# -------------------------------------------------------------------------

def _resolve_case_path(path_value, *, root_path: Path | None = None, base_path: Path | None = None) -> Path:
    """Resolve a config path while keeping absolute paths unchanged."""
    p = Path(path_value)
    if p.is_absolute():
        return p
    if base_path is not None:
        return base_path / p
    if root_path is not None:
        return root_path / p
    return p

def dicom_phase_to_radians(phase_vol: np.ndarray) -> np.ndarray:
    """Siemens phase 0..4095 → radians (-π..π)."""
    phase_vol = phase_vol.astype(np.float32)
    return (phase_vol - 2048.0) / 2048.0 * np.pi

def dicom_phase_to_radians_slope2(phase_vol: np.ndarray) -> np.ndarray:
    """Siemens phase -4096..4094 → radians (-π..π)."""
    phase_vol = phase_vol.astype(np.float32)
    return phase_vol / 4096.0 * np.pi

# -------------------------------------------------------------------------
# Mask building
# -------------------------------------------------------------------------


def build_mask(
    mag0: np.ndarray,
    *,
    mode: str = "relative",
    rel_thr: float = 0.03,
    abs_thr: float | None = None,
    percentile: float | None = None,
    fill_holes: bool = False
) -> np.ndarray:
    """
    Build a binary mask from magnitude image mag0.

    mode:
      - "relative": thr = rel_thr * mag0.max()
      - "absolute": thr = abs_thr  (in image units)
      - "percentile": thr = np.percentile(mag0, percentile)
    """
    mag0 = np.asarray(mag0)

    print(
        "[MASK] mag0 stats:",
        "min=", float(mag0.min()),
        "max=", float(mag0.max()),
    )
    for p in (50, 75, 90, 95, 99):
        print(f"[MASK] percentile {p}:", float(np.percentile(mag0, p)))

    if mode == "relative":
        thr = rel_thr * float(mag0.max())
        print(f"[MASK] relative mode: rel_thr={rel_thr} → thr={thr:.4g}")
    elif mode == "absolute":
        if abs_thr is None:
            raise ValueError("abs_thr must be set for mode='absolute'")
        thr = float(abs_thr)
        print(f"[MASK] absolute mode: thr={thr:.4g}")
    elif mode == "percentile":
        if percentile is None:
            raise ValueError("percentile must be set for mode='percentile'")
        thr = float(np.percentile(mag0, percentile))
        print(f"[MASK] percentile mode: p={percentile} → thr={thr:.4g}")
    else:
        raise ValueError(f"Unknown mask mode: {mode}")

    mask = mag0 > thr
    print("[MASK] voxels:", np.count_nonzero(mask), "of", mask.size)

    mask = keep_largest_cc(mask)

    # seal tiny gaps only in x/y directions, not across slices
    mask = ndi.binary_closing(mask, structure=np.ones((5, 5, 1), dtype=bool), iterations=1)

    if fill_holes:
        mask = ndi.binary_fill_holes(mask)

    # ensure only main component remains
    mask = keep_largest_cc(mask)
    return mask

def get_mask_source_image(
    mag4d: np.ndarray,
    phase4d: np.ndarray | None = None,
    *,
    source: str = "magnitude_first_echo",
) -> np.ndarray:
    """
    Create a 3D image used for automatic mask generation.

    Supported sources:
      - magnitude_first_echo: first echo magnitude image
      - magnitude_echo_mip: maximum-intensity projection across echo dimension
      - magnitude_echo_mean: mean magnitude across echo dimension
      - magnitude_echo_median: median magnitude across echo dimension
      - phase_first_echo: first echo phase image, experimental
      - phase_echo_mip: max phase value across echo dimension, experimental
      - phase_abs_echo_mip: max absolute phase across echo dimension, experimental

    Note:
      The MIP here is calculated across the echo dimension, not across slices.
      Therefore, the output remains a 3D image with shape x/y/z.
    """
    source = str(source).lower()

    if source == "magnitude_first_echo":
        return mag4d[..., 0]

    if source == "magnitude_echo_mip":
        return np.max(mag4d, axis=-1)

    if source == "magnitude_echo_mean":
        return np.mean(mag4d, axis=-1)

    if source == "magnitude_echo_median":
        return np.median(mag4d, axis=-1)

    if phase4d is None:
        raise RuntimeError(f"Mask source '{source}' requires phase4d, but phase4d is None.")

    if source == "phase_first_echo":
        return phase4d[..., 0]

    if source == "phase_echo_mip":
        return np.max(phase4d, axis=-1)

    if source == "phase_abs_echo_mip":
        return np.max(np.abs(phase4d), axis=-1)

    raise ValueError(
        f"Unknown mask source: {source}. "
        "Use 'magnitude_first_echo', 'magnitude_echo_mip', "
        "'magnitude_echo_mean', 'magnitude_echo_median', "
        "'phase_first_echo', 'phase_echo_mip', or 'phase_abs_echo_mip'."
    )

def load_precomputed_mask_nifti(
    mask_path: Path,
    ref_img,
    *,
    threshold: float = 0.5,
    invert: bool = False,
    resample: str = "auto",  # "auto", "always", or "off"
) -> np.ndarray:
    """
    Load a precomputed mask NIfTI safely and align it to the reference image.

    Important:
      Same shape is NOT enough.
      The mask is only used directly if both shape AND affine match.

    If the affine differs, the mask is resampled/reoriented into the
    reference image space using nearest-neighbor interpolation.
    """
    if not HAVE_NIB:
        raise RuntimeError("nibabel is required to load a precomputed mask NIfTI")

    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Precomputed mask not found: {mask_path}")

    mask_img_raw = nib.load(str(mask_path))
    mask_data_raw = mask_img_raw.get_fdata()

    # If mask is 4D, take first volume but preserve affine.
    if mask_data_raw.ndim == 4:
        print("[MASK] Precomputed mask is 4D. Using first volume.")
        mask_data_raw = mask_data_raw[..., 0]

    if mask_data_raw.ndim != 3:
        raise RuntimeError(
            f"Mask must be 3D, or 4D with a usable first volume. "
            f"Got ndim={mask_data_raw.ndim}"
        )

    # Build a clean 3D mask image with the original mask affine.
    mask_img = nib.Nifti1Image(
        mask_data_raw.astype(np.float32, copy=False),
        mask_img_raw.affine,
    )

    ref_shape3 = tuple(ref_img.shape[:3])
    mask_shape3 = tuple(mask_img.shape[:3])

    same_shape = mask_shape3 == ref_shape3
    same_affine = np.allclose(mask_img.affine, ref_img.affine, atol=1e-4)

    print("[MASK] Precomputed mask geometry check:")
    print("       mask path:       ", mask_path)
    print("       mask shape:      ", mask_shape3)
    print("       reference shape: ", ref_shape3)
    print("       same shape:      ", same_shape)
    print("       same affine:     ", same_affine)
    print("       mask affine:\n", mask_img.affine)
    print("       reference affine:\n", ref_img.affine)

    resample = str(resample or "auto").lower()

    if same_shape and same_affine and resample != "always":
        print("[MASK] Precomputed mask already matches reference geometry. Using directly.")
        m = mask_img.get_fdata()

    else:
        if resample == "off":
            raise RuntimeError(
                "Precomputed mask geometry does not match the reference image, "
                "but mask.precomputed.resample='off'.\n"
                f"Mask shape:      {mask_shape3}\n"
                f"Reference shape: {ref_shape3}\n"
                f"Same affine:     {same_affine}\n"
                "Set mask.precomputed.resample='auto' to resample/reorient safely."
            )

        if not HAVE_NIB_PROCESSING:
            raise RuntimeError(
                "Precomputed mask needs resampling/reorientation, but "
                "nibabel.processing.resample_from_to is not available."
            )

        print("[MASK] Resampling/reorienting precomputed mask to reference image space.")
        print("[MASK] Interpolation: nearest-neighbor/order=0")

        mask_img_rs = resample_from_to(
            mask_img,
            (ref_shape3, ref_img.affine),
            order=0,
        )

        m = mask_img_rs.get_fdata()

        if m.ndim == 4:
            m = m[..., 0]

        if tuple(m.shape) != ref_shape3:
            raise RuntimeError(
                f"Resampled mask shape mismatch: got {m.shape}, expected {ref_shape3}"
            )

        if not np.allclose(mask_img_rs.affine, ref_img.affine, atol=1e-4):
            raise RuntimeError(
                "Internal error: resampled mask affine does not match reference affine."
            )

    mask = m > float(threshold)

    if invert:
        print("[MASK] Inverting mask values because invert=true.")
        mask = ~mask

    fg_frac = float(np.mean(mask))
    n_vox = int(np.count_nonzero(mask))

    print(
        "[MASK] precomputed stats:",
        "voxels=", n_vox,
        "of", int(mask.size),
        "foreground_fraction=", f"{fg_frac:.6f}",
        "threshold=", float(threshold),
        "invert=", bool(invert),
    )

    if fg_frac < 0.001:
        raise RuntimeError(
            f"Precomputed mask looks empty after loading/resampling. "
            f"Foreground fraction={fg_frac:.6f}. "
            "Check mask path, threshold, and orientation."
        )

    if fg_frac > 0.80:
        raise RuntimeError(
            f"Precomputed mask looks probably inverted after loading/resampling. "
            f"Foreground fraction={fg_frac:.6f}. "
            "Check mask.precomputed.invert and threshold."
        )

    return mask.astype(bool, copy=False)

def fill_enclosed_holes(mask: np.ndarray, max_hole_voxels: int | None = None) -> np.ndarray:
    """
    Fill enclosed background components ("holes") inside a binary mask.

    A hole is defined as a connected background component that does NOT touch the volume border.
    If max_hole_voxels is set, only holes with voxel count <= max_hole_voxels are filled.
    """
    mask = np.asarray(mask, dtype=bool)
    bg = ~mask

    lab, n = ndi.label(bg)
    if n == 0:
        return mask

    # Labels touching any border are NOT holes
    border = set()
    border |= set(np.unique(lab[0, :, :]))
    border |= set(np.unique(lab[-1, :, :]))
    border |= set(np.unique(lab[:, 0, :]))
    border |= set(np.unique(lab[:, -1, :]))
    border |= set(np.unique(lab[:, :, 0]))
    border |= set(np.unique(lab[:, :, -1]))
    border.discard(0)

    candidates = [i for i in range(1, n + 1) if i not in border]
    if not candidates:
        return mask

    if max_hole_voxels is None:
        fill_labels = candidates
    else:
        sizes = ndi.sum(bg, lab, index=np.array(candidates, dtype=int))
        fill_labels = [lbl for lbl, sz in zip(candidates, sizes) if int(sz) <= int(max_hole_voxels)]

    if not fill_labels:
        return mask

    filled = mask | np.isin(lab, np.array(fill_labels, dtype=int))
    return filled


# -------------------------------------------------------------------------
# NIfTI export + in-plane rotation
# -------------------------------------------------------------------------

def parse_slice_subset(cfg: dict | None, nz: int) -> tuple[int, int, str]:

    if not cfg:
        return 0, nz, "all"

    if not bool(cfg.get("enable", True)):
        return 0, nz, "all"

    indexing = str(cfg.get("indexing", "0based")).lower()
    z0 = int(cfg.get("z_start", 0))
    z1 = int(cfg.get("z_end", nz - 1))

    if indexing == "1based":
        z0 -= 1
        z1 -= 1
    elif indexing != "0based":
        raise ValueError(f"slice_subset.indexing must be '0based' or '1based', got: {indexing}")

    z0 = max(0, min(z0, nz - 1))
    z1 = max(0, min(z1, nz - 1))
    if z1 < z0:
        raise ValueError(f"slice_subset invalid: z_end({z1}) < z_start({z0})")

    return z0, z1 + 1, f"slices_{z0:04d}_{z1:04d}"


def export_nifti_like_ref(vol, ref_img, out_path, ds0=None):
    """
    Save `vol` using the affine (and basic header) of `ref_img`.
    `vol` must have shape ref_img.shape[:3] (3D).
    """
    if not HAVE_NIB:
        print("nibabel not available, skipping export for", out_path)
        return

    vol_np = np.asarray(vol)
    ref_shape = ref_img.shape[:3]
    if vol_np.shape != ref_shape:
        raise RuntimeError(f"Shape mismatch for {out_path}: vol {vol_np.shape} vs ref {ref_shape}")

    img = nib.Nifti1Image(vol_np, ref_img.affine)

    try:
        img.header.set_xyzt_units(*ref_img.header.get_xyzt_units())
        img.header.set_zooms(ref_img.header.get_zooms()[:3])
    except Exception:
        pass

    if ds0 is not None:
        try:
            meta = ds0.to_json_dict()
            payload = json.dumps(meta).encode("utf-8")
            ext = nib.nifti1.Nifti1Extension(40, payload)
            img.header.extensions.append(ext)
        except Exception as e:
            print(f"[WARN] Could not attach DICOM JSON extension: {e}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(out_path))
    print("Saved NIfTI:", out_path)


def resolve_4d_paths(outdir: Path, slice_subset_cfg: dict | None) -> tuple[Path, Path, Path | None]:

    base = "4D_slices"
    if slice_subset_cfg and slice_subset_cfg.get("out_base"):
        base = str(slice_subset_cfg["out_base"])

    if (not slice_subset_cfg) or (not bool(slice_subset_cfg.get("enable", True))):
        suffix = "all"
    else:
        indexing = str(slice_subset_cfg.get("indexing", "0based")).lower()
        z0 = int(slice_subset_cfg.get("z_start", 0))
        z1 = int(slice_subset_cfg.get("z_end", 0))
        if indexing == "1based":
            z0 -= 1
            z1 -= 1
        suffix = f"slices_{z0:04d}_{z1:04d}"

    mag_path   = outdir / f"{base}_{suffix}_mag.nii.gz"
    phase_path = outdir / f"{base}_{suffix}_phase_rad.nii.gz"
    te_path    = outdir / f"{base}_{suffix}_TE_seconds.npy"

    return mag_path, phase_path, (te_path if te_path.exists() else None)

# -------------------------------------------------------------------------
# Input preparation: DICOM via dcm2niix OR existing 4D NIfTI
# -------------------------------------------------------------------------

def _find_dicom_files(series_dir: Path) -> list[Path]:
    """Find DICOM files. Prefer *.dcm but allow extensionless DICOM files."""
    series_dir = Path(series_dir)
    files = sorted(series_dir.glob("*.dcm"))
    if files:
        return files
    return sorted([p for p in series_dir.iterdir() if p.is_file()])


def _read_te_s_from_dicom_header(ds: pydicom.dataset.Dataset) -> float:
    """Read TE in seconds from a classic or enhanced MR DICOM header."""
    te_ms = 0.0
    if hasattr(ds, "PerFrameFunctionalGroupsSequence"):
        try:
            fg0 = ds.PerFrameFunctionalGroupsSequence[0]
            if hasattr(fg0, "MREchoSequence"):
                echo_seq = fg0.MREchoSequence[0]
                te_ms = float(
                    getattr(
                        echo_seq,
                        "EffectiveEchoTime",
                        getattr(echo_seq, "EchoTime", 0.0),
                    )
                )
        except Exception:
            te_ms = 0.0
    if te_ms == 0.0:
        te_ms = float(getattr(ds, "EchoTime", 0.0))
    if te_ms == 0.0:
        raise RuntimeError("Could not read EchoTime/EffectiveEchoTime from DICOM header.")
    return te_ms / 1000.0


def _read_first_dicom_header(series_dir: Path) -> tuple[Path, pydicom.dataset.Dataset]:
    files = _find_dicom_files(series_dir)
    if not files:
        raise FileNotFoundError(f"No DICOM files found in {series_dir}")
    return files[0], pydicom.dcmread(files[0], stop_before_pixels=True, force=True)


def _phase_to_radians_by_units(phase_vol: np.ndarray, units: str = "radians") -> np.ndarray:
    """
    Convert phase data to radians.

    Supported units:
      - radians
      - degrees
      - siemens_0_4095
      - auto
    """
    units = str(units or "radians").lower()
    phase_vol = np.asarray(phase_vol, dtype=np.float32)

    if units in {"auto", "infer"}:
        finite = phase_vol[np.isfinite(phase_vol)]
        if finite.size == 0:
            raise RuntimeError("Cannot infer phase units because phase image contains no finite values.")

        pmin = float(finite.min())
        pmax = float(finite.max())

        if pmin >= -np.pi - 0.05 and pmax <= np.pi + 0.05:
            units = "radians"
        elif pmin >= -0.5 and pmax <= 4095.5:
            units = "siemens_0_4095"
        elif pmin >= -4096.5 and pmax <= 4094.5:
            units = "siemens_int16_slope2"
        elif pmin >= -180.5 and pmax <= 180.5:
            units = "degrees"
        else:
            raise RuntimeError(
                "Could not automatically infer phase units.\n"
                f"Phase range: min={pmin}, max={pmax}\n"
                "Set input.phase_units explicitly to 'radians', 'degrees', or 'siemens_0_4095'."
            )

        print(f"[PHASE] auto-detected phase_units='{units}' from range min={pmin}, max={pmax}")

    if units in {"radian", "radians", "rad"}:
        return phase_vol

    if units in {"degree", "degrees", "deg"}:
        return np.deg2rad(phase_vol).astype(np.float32)
    
    if units in {"siemens", "dicom_siemens", "siemens_int16_slope2"}:
        return dicom_phase_to_radians_slope2(phase_vol)

    if units in {"siemens_0_4095", "siemens", "dicom_siemens", "0_4095"}:
        return dicom_phase_to_radians(phase_vol)

    raise RuntimeError(
        f"Unknown phase_units='{units}'. "
        "Use 'radians', 'degrees', 'siemens_0_4095', or 'auto'."
    )


def _load_graphcut_header_metadata(path: Path) -> dict:
    """
    Load metadata from GraphCutHeader sidecar.

    Supported:
      - .mat  MATLAB GraphCutHeader files
      - .json
      - .yaml / .yml
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".mat":
        try:
            from scipy.io import loadmat
            data = loadmat(str(path), squeeze_me=True, struct_as_record=False)
            meta = {k: v for k, v in data.items() if not k.startswith("__")}
            print(f"[NIFTI] Loaded MATLAB GraphCutHeader: {path}")
            print("[NIFTI] Available metadata fields:", sorted(meta.keys()))
            return meta
        except NotImplementedError:
            # MATLAB v7.3 files are HDF5-based. Use h5py if available.
            try:
                import h5py
            except ImportError as e:
                raise RuntimeError(
                    f"Could not read MATLAB v7.3 file: {path}\n"
                    "Install h5py or save the .mat file as non-v7.3."
                ) from e

            meta = {}
            with h5py.File(str(path), "r") as f:
                for k in f.keys():
                    meta[k] = np.array(f[k]).squeeze()

            print(f"[NIFTI] Loaded MATLAB v7.3 GraphCutHeader via h5py: {path}")
            print("[NIFTI] Available metadata fields:", sorted(meta.keys()))
            return meta

    if suffix == ".json":
        print(f"[NIFTI] Loaded JSON metadata: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as e:
            raise RuntimeError("YAML metadata requires PyYAML. Use JSON/MAT instead or install pyyaml.") from e

        print(f"[NIFTI] Loaded YAML metadata: {path}")
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    raise RuntimeError(
        f"Unsupported metadata file type: {path}\n"
        "Use .mat, .json, .yaml, or .yml."
    )


def _load_nifti_float(path: Path) -> tuple[np.ndarray, object]:
    if not HAVE_NIB:
        raise RuntimeError("nibabel is required for NIfTI input handling")
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    return data.astype(np.float32, copy=False), img


def _ensure_same_grid(img_a, img_b, *, label_a: str, label_b: str, atol: float = 1e-3) -> None:
    if img_a.shape[:3] != img_b.shape[:3]:
        raise RuntimeError(
            f"Shape mismatch between {label_a} and {label_b}: "
            f"{img_a.shape[:3]} vs {img_b.shape[:3]}"
        )
    if not np.allclose(img_a.affine, img_b.affine, atol=atol):
        print(
            f"[WARN] Affine mismatch between {label_a} and {label_b}. "
            "No resampling is applied; please verify that both images are already in the same grid."
        )


def _export_4d_like_ref(vol4d: np.ndarray, ref_img, out_path: Path) -> None:
    """Export a 4D array using the dcm2niix/reference NIfTI affine and header basics."""
    if not HAVE_NIB:
        raise RuntimeError("nibabel is required for NIfTI export")
    vol4d = np.asarray(vol4d, dtype=np.float32)
    header = ref_img.header.copy()
    img = nib.Nifti1Image(vol4d, ref_img.affine, header=header)
    img.header.set_data_dtype(np.float32)
    try:
        zoom3 = tuple(ref_img.header.get_zooms()[:3])
        img.header.set_zooms(zoom3 + (1.0,))
        img.header.set_xyzt_units(*ref_img.header.get_xyzt_units())
    except Exception:
        pass
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(out_path))
    print("Saved NIfTI:", out_path)

def _matlab_path_string(path: Path | str) -> str:
    """
    Convert path to MATLAB-safe string.
    MATLAB accepts forward slashes on Windows.
    """
    return str(path).replace("\\", "/").replace("'", "''")


def _export_3d_like_ref(vol3d: np.ndarray, ref_img, out_path: Path) -> None:
    """
    Export one 3D image using a reference NIfTI affine/header.
    Used for SPM bias-field correction input.
    """
    if not HAVE_NIB:
        raise RuntimeError("nibabel is required for NIfTI export")

    vol3d = np.asarray(vol3d, dtype=np.float32)

    if vol3d.shape != ref_img.shape[:3]:
        raise RuntimeError(
            f"Cannot export 3D SPM input. Shape mismatch: "
            f"vol={vol3d.shape}, ref={ref_img.shape[:3]}"
        )

    header = ref_img.header.copy()
    img = nib.Nifti1Image(vol3d, ref_img.affine, header=header)
    img.header.set_data_dtype(np.float32)

    try:
        img.header.set_zooms(tuple(ref_img.header.get_zooms()[:3]))
        img.header.set_xyzt_units(*ref_img.header.get_xyzt_units())
    except Exception:
        pass

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(out_path))
    print("[BIAS] Saved SPM input:", out_path)

def _bias_method_cfg(bias_cfg: dict, method: str) -> dict:
    """
    Merge common bias_correction settings with method-specific settings.

    Example:
      bias_correction.debug_export
      + bias_correction.spm.*
      or bias_correction.sitk_n4.*
    """
    bias_cfg = bias_cfg or {}
    method = str(method).lower()

    aliases = {
        "simpleitk": "sitk_n4",
        "sitk": "sitk_n4",
        "n4": "sitk_n4",
        "matlab_spm": "spm",
    }
    method_key = aliases.get(method, method)

    common = {
        k: v
        for k, v in bias_cfg.items()
        if k not in {"spm", "sitk_n4", "simpleitk", "sitk", "n4"}
    }

    specific = bias_cfg.get(method_key, {}) or {}

    merged = dict(common)
    merged.update(specific)
    merged["method"] = method_key

    return merged

def apply_bias_correction_to_mask_source(
    mag0: np.ndarray,
    ref_img,
    outdir: Path,
    bias_cfg: dict | None,
) -> np.ndarray:
    """
    Apply optional bias-field correction to the 3D image used for mask generation.

    This does NOT change the 4D complex signal used by hmrGC.
    It only returns a corrected 3D mask source image for thresholding.

    Supported methods:
      - spm
      - sitk_n4
    """
    bias_cfg = bias_cfg or {}

    if not bool(bias_cfg.get("enable", False)):
        return mag0

    method = str(bias_cfg.get("method", "spm")).lower()
    method_cfg = _bias_method_cfg(bias_cfg, method)

    if method_cfg["method"] == "spm":
        return apply_spm_bias_correction_to_mask_source(
            mag0,
            ref_img,
            outdir,
            method_cfg,
        )

    if method_cfg["method"] == "sitk_n4":
        return apply_sitk_n4_bias_correction_to_mask_source(
            mag0,
            ref_img,
            outdir,
            method_cfg,
        )

    raise RuntimeError(
        f"Unknown mask.bias_correction.method='{method}'. "
        "Use 'spm' or 'sitk_n4'."
    )

def apply_sitk_n4_bias_correction_to_mask_source(
    mag0: np.ndarray,
    ref_img,
    outdir: Path,
    bias_cfg: dict | None,
) -> np.ndarray:
    """
    Pure Python / SimpleITK N4 bias-field correction.

    Used only for the mask source image before thresholding.
    The original 4D complex signal remains unchanged.
    """
    bias_cfg = bias_cfg or {}

    try:
        import SimpleITK as sitk
    except ImportError as e:
        raise RuntimeError(
            "SimpleITK is required for mask.bias_correction.method='sitk_n4'.\n"
            "Install it with:\n"
            "  pip install SimpleITK"
        ) from e

    img_np = np.asarray(mag0, dtype=np.float32)
    img_np = np.nan_to_num(img_np, nan=0.0, posinf=0.0, neginf=0.0)

    if not np.any(img_np > 0):
        raise RuntimeError("Cannot run SimpleITK N4: mask source image has no positive voxels.")

    # SimpleITK returns the same array shape after GetArrayFromImage.
    sitk_img = sitk.GetImageFromArray(img_np)

    try:
        zooms = tuple(float(z) for z in ref_img.header.get_zooms()[:3])
        if len(zooms) == 3:
            # SimpleITK image axes are reversed relative to numpy image display,
            # but spacing is only used for smoothness estimation here.
            sitk_img.SetSpacing(tuple(reversed(zooms)))
    except Exception:
        pass

    otsu_bins = int(bias_cfg.get("otsu_bins", 200))
    sitk_mask = sitk.OtsuThreshold(sitk_img, 0, 1, otsu_bins)
    sitk_mask = sitk.Cast(sitk_mask, sitk.sitkUInt8)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    max_iterations = bias_cfg.get("max_iterations", [50, 50, 30, 20])
    if isinstance(max_iterations, int):
        max_iterations = [int(max_iterations)]
    else:
        max_iterations = [int(x) for x in max_iterations]

    convergence_threshold = float(bias_cfg.get("convergence_threshold", 0.001))

    corrector.SetMaximumNumberOfIterations(max_iterations)
    corrector.SetConvergenceThreshold(convergence_threshold)

    print("[BIAS] Running SimpleITK N4 bias-field correction")
    print("[BIAS] max_iterations:", max_iterations)
    print("[BIAS] convergence_threshold:", convergence_threshold)
    print("[BIAS] otsu_bins:", otsu_bins)

    corrected = corrector.Execute(sitk_img, sitk_mask)

    corr = sitk.GetArrayFromImage(corrected).astype(np.float32)

    if corr.shape != img_np.shape:
        raise RuntimeError(
            f"SimpleITK N4 corrected image shape mismatch: corrected={corr.shape}, original={img_np.shape}"
        )

    print(
        "[BIAS] SimpleITK N4 corrected mask source stats:",
        "min=", float(np.nanmin(corr)),
        "max=", float(np.nanmax(corr)),
    )

    if bool(bias_cfg.get("debug_export", False)):
        debug_path = Path(outdir) / "_debug_mask_source_bias_corrected_sitk_n4.nii.gz"
        _export_3d_like_ref(corr, ref_img, debug_path)
        print("[BIAS] Saved SimpleITK N4 debug image:", debug_path)

    return corr


def apply_spm_bias_correction_to_mask_source(
    mag0: np.ndarray,
    ref_img,
    outdir: Path,
    bias_cfg: dict | None,
) -> np.ndarray:
    """
    SPM/MATLAB bias-field correction.

    Used only for the mask source image before thresholding.
    The original 4D complex signal remains unchanged.
    """
    bias_cfg = bias_cfg or {}

    matlab_cmd = str(bias_cfg.get("matlab_cmd", "matlab"))
    spm_path = bias_cfg.get("spm_path", None)

    if not spm_path:
        raise RuntimeError(
            "SPM bias correction was requested, but mask.bias_correction.spm_path is empty.\n"
            "Set it to the folder containing spm.m, or use method='sitk_n4'."
        )

    biasreg = float(bias_cfg.get("biasreg", 0.001))
    biasfwhm = float(bias_cfg.get("biasfwhm", 30))
    cleanup = bool(bias_cfg.get("cleanup", False))

    workdir_raw = bias_cfg.get("workdir", "_temp/spm_bias_mask")
    workdir = _resolve_case_path(workdir_raw, base_path=outdir)
    workdir.mkdir(parents=True, exist_ok=True)

    input_nii = workdir / "mask_source_first_echo.nii"
    job_file = workdir / "run_spm_bias_correction.m"

    _export_3d_like_ref(mag0, ref_img, input_nii)

    input_nii_matlab = _matlab_path_string(input_nii)
    job_file_matlab = _matlab_path_string(job_file)

    corrected_nii = workdir / f"m{input_nii.name}"

    lines = [
        f"addpath('{_matlab_path_string(spm_path)}');",
        "if isempty(which('spm'))",
        "    error('SPM not found on MATLAB path. Check mask.bias_correction.spm_path.');",
        "end",
        "spm('defaults', 'FMRI');",
        "spm_jobman('initcfg');",
        "matlabbatch = {};",
        f"matlabbatch{{1}}.spm.spatial.preproc.channel.vols = {{'{input_nii_matlab},1'}};",
        f"matlabbatch{{1}}.spm.spatial.preproc.channel.biasreg = {biasreg};",
        f"matlabbatch{{1}}.spm.spatial.preproc.channel.biasfwhm = {biasfwhm};",
        "matlabbatch{1}.spm.spatial.preproc.channel.write = [1 1];",
        "spm_jobman('run', matlabbatch);",
    ]

    job_file.write_text("\n".join(lines), encoding="utf-8")
    print("[BIAS] Wrote SPM job:", job_file)

    matlab_mode = str(bias_cfg.get("matlab_mode", "batch")).lower()

    if matlab_mode == "batch":
        cmd = [
            matlab_cmd,
            "-batch",
            f"run('{job_file_matlab}')",
        ]
    else:
        cmd = [
            matlab_cmd,
            "-nosplash",
            "-nodesktop",
            "-r",
            f"try, run('{job_file_matlab}'); catch ME, disp(getReport(ME)); exit(1); end; exit(0);",
        ]

    print("[BIAS] Running SPM bias-field correction:")
    print("       " + " ".join(cmd))

    subprocess.run(cmd, check=True)

    if not corrected_nii.exists():
        candidates = sorted(workdir.glob("m*.nii"))
        if not candidates:
            raise RuntimeError(
                "SPM finished, but no bias-corrected output was found.\n"
                f"Expected: {corrected_nii}\n"
                f"Folder:   {workdir}"
            )
        corrected_nii = candidates[0]

    print("[BIAS] Loading SPM corrected mask source:", corrected_nii)

    corr_img = nib.load(str(corrected_nii))
    corr = corr_img.get_fdata(dtype=np.float32)

    if corr.shape != mag0.shape:
        raise RuntimeError(
            f"SPM corrected image shape mismatch: corrected={corr.shape}, original={mag0.shape}"
        )

    print(
        "[BIAS] SPM corrected mask source stats:",
        "min=", float(np.nanmin(corr)),
        "max=", float(np.nanmax(corr)),
    )

    if bool(bias_cfg.get("debug_export", False)):
        debug_path = Path(outdir) / "_debug_mask_source_bias_corrected_spm.nii.gz"
        _export_3d_like_ref(corr, ref_img, debug_path)
        print("[BIAS] Saved SPM debug image:", debug_path)

    if cleanup:
        try:
            for p in workdir.glob("*"):
                p.unlink()
            workdir.rmdir()
            print("[BIAS] Cleaned temporary SPM folder:", workdir)
        except Exception as e:
            print(f"[WARN] Could not clean SPM temp folder {workdir}: {e}")

    return corr.astype(np.float32, copy=False)

def export_raw_4d_like_ref(
    signal: np.ndarray,
    ref_img,
    TE_s: np.ndarray,
    outdir: Path,
    *,
    slice_subset_cfg: dict | None = None,
) -> None:
    """
    Export 4D magnitude and phase-radian NIfTIs using an existing reference affine.

    This is the new path used for dcm2niix/NIfTI inputs. It avoids manual DICOM
    affine creation and preserves the affine produced by dcm2niix or supplied by
    the input NIfTI.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nz = int(signal.shape[2])
    z0, z1, tag = parse_slice_subset(slice_subset_cfg, nz)
    sig = np.zeros(signal.shape, dtype=signal.dtype)
    sig[:, :, z0:z1, :] = signal[:, :, z0:z1, :]

    base = "4D_slices"
    if slice_subset_cfg and slice_subset_cfg.get("out_base"):
        base = str(slice_subset_cfg["out_base"])
    suffix = tag if tag != "all" else "all"

    mag_path = outdir / f"{base}_{suffix}_mag.nii.gz"
    phase_path = outdir / f"{base}_{suffix}_phase_rad.nii.gz"
    te_path = outdir / f"{base}_{suffix}_TE_seconds.npy"

    if mag_path.exists() and phase_path.exists():
        print(f"Files already exist, skipping 4D export:\n  {mag_path}\n  {phase_path}")
        if not te_path.exists():
            np.save(te_path, np.asarray(TE_s, dtype=float))
            print("Saved TE_s to", te_path)
        return

    mag4d = np.abs(sig).astype(np.float32)
    phase4d = np.angle(sig).astype(np.float32)

    _export_4d_like_ref(mag4d, ref_img, mag_path)
    _export_4d_like_ref(phase4d, ref_img, phase_path)
    np.save(te_path, np.asarray(TE_s, dtype=float))
    print("Saved TE_s to", te_path)

    # Save version txt file
    version_txt_filename = 'FWS_hmrGC_version.txt'
    shutil.copy2(version_txt_filename, outdir / version_txt_filename)


def _strip_nii_ext(path: Path) -> str:
    name = Path(path).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(path).stem

def _safe_filename_part(value: str) -> str:
    """
    Make a string safe for filenames while keeping readable sequence names.
    """
    value = str(value).strip()
    value = value.replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def _infer_measurement_id(case_cfg: dict) -> str:
    """
    Infer measurement/case ID.

    Priority:
      1. output.measurement_id
      2. case_cfg.measurement_id
      3. first path component containing '_FOR'
    """
    output_cfg = case_cfg.get("output", {}) or {}

    explicit = output_cfg.get("measurement_id") or case_cfg.get("measurement_id")
    if explicit is None:
        return None
    if explicit:
        return _safe_filename_part(explicit)

    root_path = Path(case_cfg.get("root_path", "."))

    for part in reversed(root_path.parts):
        if "_FOR" in part.upper():
            return _safe_filename_part(part)

    raise RuntimeError(
        "Could not infer measurement_id from root_path.\n"
        f"root_path: {root_path}\n"
        "Expected one path component containing '_FOR', e.g. '20260319_FOR_054_7T'.\n"
        "Fix by adding this to the config:\n"
        '  "output": {"measurement_id": "20260319_FOR_054_7T"}'
    )


def _sequence_name_from_input_name(value: str) -> str:
    """
    Infer clean sequence name from DICOM series folder or NIfTI filename.

    Examples:
      fl3d_vibe_DIXON_RR_36.nii
        -> fl3d_vibe_DIXON

      qsm_vibe_1iso_CAIPI6_7TE_RR_7_MR
        -> qsm_vibe_1iso_CAIPI6_7TE

      qsm_vibe_1iso_CAIPI6_8TE_2Av_RR_16_MR
        -> qsm_vibe_1iso_CAIPI6_8TE_2Av
    """
    stem = _strip_nii_ext(Path(value).name)

    # remove image-role endings
    stem = re.sub(r"_(magnitude|phase|ph|mag)$", "", stem, flags=re.IGNORECASE)

    # remove Siemens measurement ending: _RR_36, _RR_36_MR, _RR_37_ph, etc.
    stem = re.sub(r"_RR_\d+(?:_MR)?$", "", stem, flags=re.IGNORECASE)

    # fallback: remove simple trailing measurement number
    stem = re.sub(r"_\d+(?:_MR)?$", "", stem, flags=re.IGNORECASE)

    # final cleanup
    stem = re.sub(r"_MR$", "", stem, flags=re.IGNORECASE)

    return _safe_filename_part(stem)


def _infer_sequence_name(case_cfg: dict) -> str:
    """
    Infer sequence name.

    Priority:
      1. output.sequence_name
      2. case_cfg.sequence_name
      3. input.magnitude_4d / mag_4d / magnitude / mag_path
      4. first mag_series entry
    """
    output_cfg = case_cfg.get("output", {}) or {}
    input_cfg = case_cfg.get("input", {}) or {}

    explicit = output_cfg.get("sequence_name") or case_cfg.get("sequence_name")
    if explicit:
        if explicit is None:
            print(f"Input sequence_name={explicit} is invalid. Falls back to default naming via sequence name from measurement series.")
        return _safe_filename_part(explicit)

    mag_raw = (
        input_cfg.get("magnitude_4d")
        or input_cfg.get("mag_4d")
        or input_cfg.get("magnitude")
        or input_cfg.get("mag_path")
    )

    if mag_raw:
        return _sequence_name_from_input_name(mag_raw)

    mag_series = case_cfg.get("mag_series") or input_cfg.get("mag_series")
    if mag_series:
        if isinstance(mag_series, (list, tuple)):
            return _sequence_name_from_input_name(mag_series[0])
        return _sequence_name_from_input_name(mag_series)

    raise RuntimeError(
        "Could not infer sequence_name.\n"
        "Fix by adding this to the config:\n"
        '  "output": {"sequence_name": "fl3d_vibe_DIXON"}'
    )


def build_final_output_paths(case_cfg: dict, outdir: Path) -> dict[str, Path]:
    """
    Build final output filenames in the requested project naming style.

    Example:
      20260319_FOR_054_7T_fl3d_vibe_DIXON_fatimg.nii.gz
    """
    output_cfg = case_cfg.get("output", {}) or {}

    measurement_id = _infer_measurement_id(case_cfg)
    sequence_name = _infer_sequence_name(case_cfg)

    ext = str(output_cfg.get("extension", ".nii.gz"))
    if not ext.startswith("."):
        ext = "." + ext

    if measurement_id is None:
        prefix = f"{sequence_name}"
    else:
        prefix = f"{measurement_id}_{sequence_name}"

    print("[OUTPUT] measurement_id:", measurement_id)
    print("[OUTPUT] sequence_name:", sequence_name)
    print("[OUTPUT] filename prefix:", prefix)

    return {
        "water": outdir / f"{prefix}_waterimg{ext}",
        "fat": outdir / f"{prefix}_fatimg{ext}",
        "inphase": outdir / f"{prefix}_inphaseimg{ext}",
        "fieldmap": outdir / f"{prefix}_smoothedB0map{ext}",
        "fieldmap_raw": outdir / f"{prefix}_rawB0map{ext}",
        "mask": outdir / f"{prefix}_mask{ext}",
        "r2star": outdir / f"{prefix}_r2starmap{ext}",
        "opposed": outdir / f"{prefix}_opposed_phase{ext}",
        "pdff": outdir / f"{prefix}_PDFFmap{ext}",
    }

def _dcm2niix_echo_number(path: Path) -> int | None:
    """
    Extract echo number from dcm2niix output names, e.g.
      mag_001_e1.nii.gz
      phase_001_e1_ph.nii.gz
    """
    stem = _strip_nii_ext(path)
    m = re.search(r"_e(\d+)(?:_|$)", stem)
    if not m:
        return None
    return int(m.group(1))


def _sidecar_json_for_nii(nii_path: Path) -> Path:
    return Path(nii_path).with_name(_strip_nii_ext(nii_path) + ".json")


def _read_te_s_from_dcm2niix_json(nii_path: Path) -> float:
    """
    Read EchoTime from the dcm2niix BIDS JSON sidecar.
    dcm2niix usually stores EchoTime in seconds.
    """
    json_path = _sidecar_json_for_nii(nii_path)
    if not json_path.exists():
        raise FileNotFoundError(
            f"Missing dcm2niix JSON sidecar for TE extraction:\n"
            f"  NIfTI: {nii_path}\n"
            f"  JSON:  {json_path}"
        )

    meta = json.loads(json_path.read_text(encoding="utf-8"))

    te = meta.get("EchoTime", None)
    if te is None:
        te = meta.get("TE", None)

    if te is None:
        raise RuntimeError(f"No EchoTime found in JSON sidecar: {json_path}")

    te = float(te)

    # Safety: BIDS EchoTime is seconds. If a file stores ms, convert.
    if te > 0.1:
        te = te / 1000.0

    return te


def _find_dcm2niix_outputs(outdir: Path, prefix: str) -> list[Path]:
    """
    Find all NIfTI files created by dcm2niix for a prefix.
    For multi-echo Siemens data this returns e1, e2, ..., e8 files.
    """
    outdir = Path(outdir)
    candidates = list(outdir.glob(f"{prefix}*.nii.gz")) + list(outdir.glob(f"{prefix}*.nii"))

    candidates = sorted(
        candidates,
        key=lambda p: (
            999999 if _dcm2niix_echo_number(p) is None else _dcm2niix_echo_number(p),
            p.name,
        ),
    )

    return candidates


def _delete_dcm2niix_outputs_for_prefix(outdir: Path, prefix: str) -> None:
 
    outdir = Path(outdir)

    patterns = [
        f"{prefix}*.nii",
        f"{prefix}*.nii.gz",
        f"{prefix}*.json",
        f"{prefix}*.bval",
        f"{prefix}*.bvec",
    ]

    deleted = 0
    for pattern in patterns:
        for p in outdir.glob(pattern):
            try:
                p.unlink()
                deleted += 1
            except Exception as e:
                print(f"[WARN] Could not delete old dcm2niix file {p}: {e}")

    if deleted:
        print(f"[dcm2niix] Deleted {deleted} old output file(s) for prefix '{prefix}'")

def run_dcm2niix_for_series(
    series_dir: Path,
    outdir: Path,
    prefix: str,
    *,
    executable: str = "dcm2niix",
    reuse_existing: bool = True,
    compress: bool = True,
) -> list[Path]:
    """
    Run dcm2niix for one DICOM series and return all produced NIfTI paths.

    For your 8-echo data, this should return 8 files:
      mag_001_e1.nii.gz ... mag_001_e8.nii.gz
    """
    series_dir = Path(series_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    existing = _find_dcm2niix_outputs(outdir, prefix)
    if reuse_existing and existing:
        print(f"[dcm2niix] Reusing {len(existing)} existing NIfTI file(s) for {series_dir.name}")
        for p in existing:
            print("  ", p.name)
        return existing
    
    if (not reuse_existing) and existing:
        print(f"[dcm2niix] reuse_existing=False, deleting old files for prefix '{prefix}'")
        _delete_dcm2niix_outputs_for_prefix(outdir, prefix)

    z_flag = "y" if compress else "n"
    cmd = [str(executable), "-z", z_flag, "-f", prefix, "-o", str(outdir), str(series_dir)]
    print("[dcm2niix]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    out_paths = _find_dcm2niix_outputs(outdir, prefix)
    if not out_paths:
        raise RuntimeError(
            f"dcm2niix finished but no NIfTI matching prefix '{prefix}' was found in {outdir}"
        )

    print(f"[dcm2niix] Found {len(out_paths)} NIfTI file(s) for prefix {prefix}:")
    for p in out_paths:
        print("  ", p.name)

    return out_paths

def prepare_4d_from_dicom_via_dcm2niix(
    case_cfg: dict,
    *,
    root_path: Path,
    outdir: Path,
    slice_subset_cfg: dict | None,
):
    """
    DICOM input path:
      DICOM mag/phase series
      -> dcm2niix echo-specific 3D NIfTIs
      -> complex NumPy signal
      -> internal 4D mag/phase-radian NIfTIs using dcm2niix affine.
    """
    input_cfg = case_cfg.get("input", {}) or {}
    dcm2niix_cfg = input_cfg.get("dcm2niix", {}) or case_cfg.get("dcm2niix", {}) or {}

    mag_series_names = case_cfg.get("mag_series", input_cfg.get("mag_series"))
    phs_series_names = case_cfg.get("phs_series", input_cfg.get("phs_series"))

    if not mag_series_names or not phs_series_names:
        raise RuntimeError("DICOM input requires mag_series and phs_series in the case config.")

    if len(mag_series_names) != len(phs_series_names):
        raise ValueError("mag_series and phs_series must have the same length")

    dcm2niix_exe = input_cfg.get("dcm2niix_exe") or dcm2niix_cfg.get("exe") or "dcm2niix"
    dcm2niix_rel = input_cfg.get("dcm2niix_workdir") or dcm2niix_cfg.get("workdir") or "_temp/dcm2niix"
    dcm2niix_outdir = _resolve_case_path(dcm2niix_rel, base_path=outdir)

    reuse_existing = bool(input_cfg.get("reuse_existing_niftis", dcm2niix_cfg.get("reuse_existing", True)))
    phase_units = input_cfg.get("phase_units", "auto")

    mag_series_paths = [_resolve_case_path(s, root_path=root_path) for s in mag_series_names]
    phs_series_paths = [_resolve_case_path(s, root_path=root_path) for s in phs_series_names]

    entries = []
    ref_img = None
    ds0 = None

    for idx, (mag_dir, phs_dir) in enumerate(zip(mag_series_paths, phs_series_paths), start=1):
        print(f"\n== dcm2niix series pair {idx}: mag={mag_dir.name} / phase={phs_dir.name} ==")

        _, ds_mag = _read_first_dicom_header(mag_dir)
        if ds0 is None:
            ds0 = ds_mag

        mag_niis = run_dcm2niix_for_series(
            mag_dir,
            dcm2niix_outdir,
            f"mag_{idx:03d}",
            executable=dcm2niix_exe,
            reuse_existing=reuse_existing,
            compress=True,
        )

        phs_niis = run_dcm2niix_for_series(
            phs_dir,
            dcm2niix_outdir,
            f"phase_{idx:03d}",
            executable=dcm2niix_exe,
            reuse_existing=reuse_existing,
            compress=True,
        )

        mag_by_echo = {}
        for p in mag_niis:
            echo_no = _dcm2niix_echo_number(p)
            if echo_no is None:
                if len(mag_niis) == 1:
                    echo_no = 1
                else:
                    raise RuntimeError(f"Could not determine echo number from magnitude NIfTI name: {p.name}")
            mag_by_echo[echo_no] = p

        phs_by_echo = {}
        for p in phs_niis:
            echo_no = _dcm2niix_echo_number(p)
            if echo_no is None:
                if len(phs_niis) == 1:
                    echo_no = 1
                else:
                    raise RuntimeError(f"Could not determine echo number from phase NIfTI name: {p.name}")
            phs_by_echo[echo_no] = p

        common_echoes = sorted(set(mag_by_echo.keys()) & set(phs_by_echo.keys()))

        if not common_echoes:
            raise RuntimeError(
                f"No matching echo numbers between magnitude and phase outputs for pair {idx}.\n"
                f"Magnitude echoes: {sorted(mag_by_echo.keys())}\n"
                f"Phase echoes:     {sorted(phs_by_echo.keys())}"
            )

        missing_phase = sorted(set(mag_by_echo.keys()) - set(phs_by_echo.keys()))
        missing_mag = sorted(set(phs_by_echo.keys()) - set(mag_by_echo.keys()))

        if missing_phase:
            print(f"[WARN] Missing phase echo(s) for pair {idx}: {missing_phase}")
        if missing_mag:
            print(f"[WARN] Missing magnitude echo(s) for pair {idx}: {missing_mag}")

        print(f"[dcm2niix] Pair {idx}: using echo numbers {common_echoes}")

        for echo_no in common_echoes:
            mag_nii = mag_by_echo[echo_no]
            phs_nii = phs_by_echo[echo_no]

            mag_data, mag_img = _load_nifti_float(mag_nii)
            phs_data, phs_img = _load_nifti_float(phs_nii)

            if mag_data.ndim != 3 or phs_data.ndim != 3:
                raise RuntimeError(
                    "DICOM/dcm2niix mode expects echo-specific 3D NIfTIs. "
                    f"Got {mag_nii.name} shape {mag_data.shape} and "
                    f"{phs_nii.name} shape {phs_data.shape}."
                )

            _ensure_same_grid(mag_img, phs_img, label_a=mag_nii.name, label_b=phs_nii.name)

            if ref_img is None:
                ref_img = mag_img
            else:
                _ensure_same_grid(ref_img, mag_img, label_a="reference magnitude", label_b=mag_nii.name)

            te_s = _read_te_s_from_dcm2niix_json(mag_nii)

            ph_rad = _phase_to_radians_by_units(phs_data, units=phase_units)
            cvol = mag_data * np.exp(1j * ph_rad)

            entries.append(
                {
                    "te_s": te_s,
                    "complex": cvol,
                    "pair_index": idx,
                    "echo_number": echo_no,
                }
            )

            print(f"Echo {echo_no}: TE = {te_s:.6f} s")

    if len(entries) < 3:
        raise RuntimeError(f"Only {len(entries)} echoes found, need at least 3.")

    groups = {}
    for e in entries:
        key = int(round(e["te_s"] * 1e6))
        groups.setdefault(key, []).append(e)

    complex_list = []
    unique_te_s = []

    print("\n=== Grouped dcm2niix echoes by TE ===")
    for key in sorted(groups.keys()):
        group = groups[key]
        te_val = group[0]["te_s"]
        c_mean = np.mean([g["complex"] for g in group], axis=0)

        complex_list.append(c_mean)
        unique_te_s.append(te_val)

        details = [
            f"pair {g['pair_index']} echo {g['echo_number']}"
            for g in group
        ]
        print(f"TE = {te_val:.6f} s -> {len(group)} volume(s): {details}")

    signal = np.stack(complex_list, axis=-1)
    TE_s = np.asarray(unique_te_s, dtype=float)

    fieldStrength_T = float(getattr(ds0, "MagneticFieldStrength", 7.0))

    try:
        centerFreq_Hz = get_center_frequency_hz(ds0)
    except Exception as e:
        print(f"[WARN] Could not read center frequency from DICOM ({e}). Approximating from B0.")
        gamma_MHz_per_T = 42.57747892
        centerFreq_Hz = gamma_MHz_per_T * fieldStrength_T * 1e6

    print("Final dcm2niix-based signal shape:", signal.shape)
    print("Final TE_s (s):", TE_s)
    print(f"Center frequency (Hz): {centerFreq_Hz}")
    print(f"Field strength (T): {fieldStrength_T}")

    export_raw_4d_like_ref(
        signal,
        ref_img,
        TE_s,
        outdir,
        slice_subset_cfg=slice_subset_cfg,
    )

    return TE_s, centerFreq_Hz, fieldStrength_T, ds0

def _first_existing(meta: dict, names: list[str]):
    for name in names:
        if name in meta and meta[name] is not None:
            return meta[name], name
    return None, None


def _as_float_scalar(value, field_name: str) -> float:
    try:
        return float(value)
    except Exception:
        pass

    arr = np.asarray(value)
    if arr.size < 1:
        raise RuntimeError(f"Metadata field '{field_name}' is empty.")

    try:
        return float(arr.reshape(-1)[0])
    except Exception as e:
        raise RuntimeError(f"Could not convert metadata field '{field_name}' to scalar float.") from e


def _as_float_vector(value, field_name: str) -> np.ndarray:
    arr = np.asarray(value)

    try:
        arr = arr.astype(float)
    except Exception as e:
        raise RuntimeError(f"Could not convert metadata field '{field_name}' to float vector.") from e

    arr = arr.reshape(-1)

    if arr.size < 1:
        raise RuntimeError(f"Metadata field '{field_name}' is empty.")

    return arr


def _extract_graphcut_header_params(meta: dict):
    """
    Extract hmrGC-required parameters from GraphCutHeader metadata.

    Supports MATLAB GraphCutHeader fields:
      - TE_mat
      - CF
      - B0_strength

    Also supports JSON/YAML aliases:
      - TE_s / TE_seconds / echo_times_s
      - TE_ms / EchoTime_ms / echo_times_ms
      - centerFreq_Hz
      - ImagingFrequency_MHz / TransmitterFrequency_MHz / CF
      - fieldStrength_T / MagneticFieldStrength / B0_strength / B0_T
    """
    # Echo times
    te_value, te_name = _first_existing(
        meta,
        [
            "TE_s",
            "TE_seconds",
            "echo_times_s",
            "TE_mat",
            "TE",
            "TE_ms",
            "EchoTime_ms",
            "echo_times_ms",
        ],
    )

    if te_value is None:
        raise RuntimeError(
            "Could not find echo times in metadata.\n"
            "Expected one of: TE_s, TE_seconds, echo_times_s, TE_mat, TE, TE_ms, EchoTime_ms, echo_times_ms."
        )

    TE_s = _as_float_vector(te_value, te_name)

    # Explicit millisecond fields
    if te_name in {"TE_ms", "EchoTime_ms", "echo_times_ms"}:
        TE_s = TE_s / 1000.0
    else:
        # Safety: if values look like milliseconds, convert to seconds.
        if float(np.nanmax(TE_s)) > 0.1:
            print(f"[NIFTI] Echo times in '{te_name}' look like us. Converting to seconds.")
            TE_s = TE_s / 1e6

    # no_TE validation if available
    no_te_value, no_te_name = _first_existing(meta, ["no_TE", "n_TE", "nTE", "number_of_echoes"])
    if no_te_value is not None:
        no_te = int(round(_as_float_scalar(no_te_value, no_te_name)))
        if no_te != int(TE_s.size):
            print(
                f"[WARN] Metadata {no_te_name}={no_te}, but extracted {TE_s.size} echo times from {te_name}."
            )

    if TE_s.size < 3:
        raise RuntimeError(f"Only {TE_s.size} echo times found in metadata, need at least 3.")

    # Center frequency
    cf_value, cf_name = _first_existing(
        meta,
        [
            "centerFreq_Hz",
            "CenterFrequency_Hz",
            "center_frequency_hz",
            "CF_Hz",
            "ImagingFrequency_MHz",
            "TransmitterFrequency_MHz",
            "centerFreq_MHz",
            "CF",
        ],
    )

    if cf_value is None:
        raise RuntimeError(
            "Could not find center frequency in metadata.\n"
            "Expected one of: centerFreq_Hz, CenterFrequency_Hz, CF_Hz, ImagingFrequency_MHz, "
            "TransmitterFrequency_MHz, centerFreq_MHz, CF."
        )

    centerFreq_Hz = _as_float_scalar(cf_value, cf_name)

    # In your MATLAB header, CF is around 297, meaning MHz.
    if cf_name in {"ImagingFrequency_MHz", "TransmitterFrequency_MHz", "centerFreq_MHz", "CF"}:
        if centerFreq_Hz < 1e6:
            centerFreq_Hz *= 1e6

    # Safety if someone provides MHz under another name
    if centerFreq_Hz < 1e6:
        print(f"[WARN] Center frequency {centerFreq_Hz} is very small for Hz. Assuming MHz and converting.")
        centerFreq_Hz *= 1e6

    # Field strength
    b0_value, b0_name = _first_existing(
        meta,
        [
            "fieldStrength_T",
            "MagneticFieldStrength",
            "B0_strength",
            "B0_T",
            "field_strength_T",
        ],
    )

    if b0_value is None:
        # Fallback from center frequency
        gamma_MHz_per_T = 42.57747892
        fieldStrength_T = centerFreq_Hz / 1e6 / gamma_MHz_per_T
        print(
            "[WARN] Could not find field strength in metadata. "
            f"Approximating from center frequency: {fieldStrength_T:.4f} T"
        )
    else:
        fieldStrength_T = _as_float_scalar(b0_value, b0_name)

    print("[NIFTI] Metadata parameters:")
    print("  TE_s:", TE_s)
    print("  centerFreq_Hz:", centerFreq_Hz)
    print("  fieldStrength_T:", fieldStrength_T)

    return TE_s.astype(float), float(centerFreq_Hz), float(fieldStrength_T)


def prepare_4d_from_nifti_input(
    case_cfg: dict,
    *,
    root_path: Path,
    outdir: Path,
    slice_subset_cfg: dict | None,
):
    """
    NIfTI input path:
      4D magnitude NIfTI
      + 4D phase NIfTI
      + GraphCutHeader .mat/.json/.yaml metadata
      -> internal 4D mag/phase-radian NIfTIs using the NIfTI affine.
    """
    input_cfg = case_cfg.get("input", {}) or {}

    mag_raw = (
        input_cfg.get("magnitude_4d")
        or input_cfg.get("mag_4d")
        or input_cfg.get("magnitude")
        or input_cfg.get("mag_path")
    )

    phs_raw = (
        input_cfg.get("phase_4d")
        or input_cfg.get("phase")
        or input_cfg.get("phase_path")
        or input_cfg.get("phase_rad_4d")
    )

    meta_raw = (
        input_cfg.get("metadata")
        or input_cfg.get("graphcut_header")
        or input_cfg.get("header")
        or input_cfg.get("header_mat")
    )

    if not mag_raw:
        raise RuntimeError(
            "NIfTI input requires input.magnitude_4d, input.mag_4d, input.magnitude, or input.mag_path."
        )

    if not phs_raw:
        raise RuntimeError(
            "NIfTI input requires input.phase_4d, input.phase, input.phase_path, or input.phase_rad_4d."
        )

    if not meta_raw:
        raise RuntimeError(
            "NIfTI input requires input.metadata / input.graphcut_header pointing to the GraphCutHeader .mat/.json/.yaml file."
        )

    mag_path = _resolve_case_path(mag_raw, root_path=root_path)
    phs_path = _resolve_case_path(phs_raw, root_path=root_path)
    meta_path = _resolve_case_path(meta_raw, root_path=root_path)

    print("[NIFTI] magnitude:", mag_path)
    print("[NIFTI] phase:    ", phs_path)
    print("[NIFTI] metadata: ", meta_path)

    mag_data, mag_img = _load_nifti_float(mag_path)
    phs_data, phs_img = _load_nifti_float(phs_path)

    if mag_data.shape != phs_data.shape:
        raise RuntimeError(
            f"NIfTI magnitude/phase shape mismatch:\n"
            f"  magnitude {mag_path.name}: {mag_data.shape}\n"
            f"  phase     {phs_path.name}: {phs_data.shape}"
        )

    if mag_data.ndim != 4:
        raise RuntimeError(
            f"NIfTI input expects 4D magnitude/phase images with echo dimension last.\n"
            f"Got magnitude shape: {mag_data.shape}\n"
            f"If this is a 3D file, it does not contain all echoes."
        )

    _ensure_same_grid(mag_img, phs_img, label_a=mag_path.name, label_b=phs_path.name)

    meta = _load_graphcut_header_metadata(meta_path)
    TE_s, centerFreq_Hz, fieldStrength_T = _extract_graphcut_header_params(meta)

    if int(TE_s.size) != int(mag_data.shape[-1]):
        raise RuntimeError(
            f"Echo count mismatch between NIfTI and GraphCutHeader metadata.\n"
            f"  NIfTI echo dimension: {mag_data.shape[-1]}\n"
            f"  Metadata TE count:    {TE_s.size}\n"
            f"  NIfTI shape:          {mag_data.shape}\n"
            "Expected echo dimension to be the last NIfTI dimension."
        )

    phase_units = input_cfg.get("phase_units", "auto")
    ph_rad = _phase_to_radians_by_units(phs_data, units=phase_units)

    signal = mag_data.astype(np.float32) * np.exp(1j * ph_rad)

    print("[NIFTI] Final signal shape:", signal.shape)
    print("[NIFTI] Final TE_s:", TE_s)

    export_raw_4d_like_ref(
        signal,
        mag_img,
        TE_s,
        outdir,
        slice_subset_cfg=slice_subset_cfg,
    )

    # No DICOM dataset exists in pure NIfTI mode.
    ds0 = None

    return TE_s, centerFreq_Hz, fieldStrength_T, ds0


def prepare_case_input_4d(
    case_cfg: dict,
    *,
    root_path: Path,
    outdir: Path,
    slice_subset_cfg: dict | None,
):
    """Prepare internal 4D magnitude/phase-radian NIfTIs for DICOM or NIfTI input."""
    input_cfg = case_cfg.get("input", {}) or {}
    input_type = str(input_cfg.get("type", case_cfg.get("input_type", "dicom"))).lower()

    if input_type in {"dicom", "dcm"}:
        print("[INPUT] type=dicom -> using dcm2niix for initial NIfTI affine/header creation")
        return prepare_4d_from_dicom_via_dcm2niix(
            case_cfg,
            root_path=root_path,
            outdir=outdir,
            slice_subset_cfg=slice_subset_cfg,
        )

    if input_type in {"nifti", "nii", "nii.gz"}:
        print("[INPUT] type=nifti -> using supplied NIfTI affine/header and GraphCutHeader metadata")
        return prepare_4d_from_nifti_input(
            case_cfg,
            root_path=root_path,
            outdir=outdir,
            slice_subset_cfg=slice_subset_cfg,
        )

    raise RuntimeError(f"Unknown input type: {input_type}. Use 'dicom' or 'nifti'.")




# -------------------------------------------------------------------------
# Rebuild complex signal FROM 4D NIfTI
# -------------------------------------------------------------------------


def build_complex_signal_from_4d(
    outdir: Path,
    *,
    slice_subset_cfg: dict | None = None,
    mask_source: str = "magnitude_first_echo",
    mask_mode: str = "relative",
    mask_rel_thr: float = 0.03,
    mask_abs_thr: float | None = None,
    mask_percentile: float | None = None,
    precomputed_mask: dict | None = None,
    precomputed_mask_base_dir: Path | None = None,
    fill_holes: bool = False,
    bias_correction: dict | None = None,
):
    """
    Load 4D mag/phase NIfTIs and build complex signal + mask + voxelSize_mm.

    If precomputed_mask is provided and enabled, load that mask instead of
    building one from magnitude.
    """
    if not HAVE_NIB:
        raise RuntimeError("nibabel is required to build signal from 4D NIfTI")

    mag_path, phase_path, te_path = resolve_4d_paths(outdir, slice_subset_cfg)

    if not mag_path.exists() or not phase_path.exists():
        raise FileNotFoundError(
            "4D mag/phase NIfTIs not found.\n"
            f"  mag:   {mag_path}\n"
            f"  phase: {phase_path}\n"
            "Check slice_subset settings and export step."
        )

    mag_img = nib.load(str(mag_path))
    phase_img = nib.load(str(phase_path))

    mag4d = mag_img.get_fdata().astype(np.float32)
    phase4d = phase_img.get_fdata().astype(np.float32)

    if mag4d.shape != phase4d.shape:
        raise RuntimeError(f"4D mag/phase shape mismatch: {mag4d.shape} vs {phase4d.shape}")
    if mag4d.ndim != 4:
        raise RuntimeError(f"Expected 4D volumes, got ndim={mag4d.ndim}")

    signal = mag4d * np.exp(1j * phase4d)

    # Mask (either precomputed or computed from first echo)
    use_pre = False
    if precomputed_mask and bool(precomputed_mask.get("enable", False)):
        use_pre = True

    if use_pre:
        if precomputed_mask_base_dir is None:
            precomputed_mask_base_dir = outdir

        raw_path = precomputed_mask.get("path")
        if not raw_path:
            raise RuntimeError("mask.precomputed.enable=true but mask.precomputed.path is missing")

        mp = Path(raw_path)
        if not mp.is_absolute():
            mp = precomputed_mask_base_dir / mp

        thr = float(precomputed_mask.get("threshold", 0.5))
        inv = bool(precomputed_mask.get("invert", False))
        rs = str(precomputed_mask.get("resample", "auto")).lower()

        print(f"[MASK] Using precomputed mask: {mp}")
        mask = load_precomputed_mask_nifti(mp, mag_img, threshold=thr, invert=inv, resample=rs)
    else:
        mag0 = get_mask_source_image(mag4d, phase4d, source=mask_source)

        print(f"[MASK] source: {mask_source}")

        if bias_correction and bool(bias_correction.get("enable", False)):
            print("[MASK] Applying bias-field correction before thresholding.")
            mag0 = apply_bias_correction_to_mask_source(
                    mag0,
                    mag_img,
                    outdir,
                    bias_correction,
                )
        else:
            print("[MASK] Bias-field correction before thresholding: disabled.")

        mask = build_mask(
            mag0,
            mode=mask_mode,
            rel_thr=mask_rel_thr,
            abs_thr=mask_abs_thr,
            percentile=mask_percentile,
            fill_holes=fill_holes,
        )

    zooms = mag_img.header.get_zooms()
    voxelSize_mm = np.array(zooms[:3], dtype=float)
    print(f"[4D] voxel size from NIfTI: {voxelSize_mm}")

    return signal, mask, voxelSize_mm


# -------------------------------------------------------------------------
# Cropping / echo subset
# -------------------------------------------------------------------------


def crop_to_mask_bbox(signal, mask, margin=4):
    """
    Crops signal (Ny, Nx, Nz, nTE) and mask (Ny, Nx, Nz) to the
    minimal bounding box around mask==True.
    """
    idx = np.argwhere(mask)
    if idx.size == 0:
        raise RuntimeError("Mask is empty, nothing to crop")

    mins = idx.min(axis=0)
    maxs = idx.max(axis=0) + 1

    mins = np.maximum(mins - margin, 0)
    maxs = np.minimum(maxs + margin, mask.shape)

    sy = slice(int(mins[0]), int(maxs[0]))
    sx = slice(int(mins[1]), int(maxs[1]))
    sz = slice(int(mins[2]), int(maxs[2]))

    sig_crop = signal[sy, sx, sz, :]
    mask_crop = mask[sy, sx, sz]

    print(
        "Cropping from",
        signal.shape,
        "to",
        sig_crop.shape,
        "using bbox:",
        (sy, sx, sz),
    )
    return sig_crop, mask_crop, (sy, sx, sz)

def _parse_echo_indices(n_echoes, n_available: int) -> np.ndarray | None:
    """
    Parse echo index selection.

    Supported formats:
      - None          -> no index-based selection
      - int 5         -> keep first 5 echoes: 0,1,2,3,4
      - "0-5"         -> keep echo indices 0..5 inclusive
      - "2-8"         -> keep echo indices 2..8 inclusive
      - [0, 5]        -> keep echo indices 0..5 inclusive
      - {"start": 0, "end": 5} -> keep echo indices 0..5 inclusive

    Echo indices are 0-based.
    """
    if n_echoes is None or n_echoes == 'all':
        return None

    # Old behavior: integer means "keep first N echoes"
    if isinstance(n_echoes, int):
        n = int(n_echoes)
        if n < 3:
            raise RuntimeError(f"n_echoes={n} is too small, need at least 3 echoes.")
        if n > n_available:
            print(f"[WARN] n_echoes={n} > available={n_available}. Clamping.")
            n = n_available
        return np.arange(n, dtype=int)

    # String range, e.g. "0-5" or "2:8"
    if isinstance(n_echoes, str):
        s = n_echoes.strip().replace(":", "-")
        if "-" not in s:
            # Allow numeric strings such as "5" with old behavior
            n = int(s)
            if n < 3:
                raise RuntimeError(f"n_echoes={n} is too small, need at least 3 echoes.")
            if n > n_available:
                print(f"[WARN] n_echoes={n} > available={n_available}. Clamping.")
                n = n_available
            return np.arange(n, dtype=int)

        start_s, end_s = s.split("-", 1)
        start = int(start_s)
        end = int(end_s)

    # List/tuple range, e.g. [0, 5]
    elif isinstance(n_echoes, (list, tuple)) and len(n_echoes) == 2:
        start = int(n_echoes[0])
        end = int(n_echoes[1])

    # Dict range, e.g. {"start": 0, "end": 5}
    elif isinstance(n_echoes, dict):
        start = int(n_echoes.get("start", 0))
        end = int(n_echoes.get("end", n_available - 1))

    else:
        raise RuntimeError(
            "Unsupported n_echoes format. Use None, 5, '0-5', [0, 5], "
            "or {'start': 0, 'end': 5}."
        )

    if start < 0:
        raise RuntimeError(f"Echo range start must be >= 0, got {start}.")
    if end < start:
        raise RuntimeError(f"Echo range end {end} is smaller than start {start}.")
    if start >= n_available:
        raise RuntimeError(
            f"Echo range start {start} is outside available echoes 0..{n_available - 1}."
        )
    if end >= n_available:
        print(f"[WARN] Echo range end {end} > available max {n_available - 1}. Clamping.")
        end = n_available - 1

    keep_idx = np.arange(start, end + 1, dtype=int)

    if keep_idx.size < 3:
        raise RuntimeError(
            f"Only {keep_idx.size} echoes selected by n_echoes={n_echoes}; "
            "need at least 3 echoes."
        )

    return keep_idx

def select_echo_subset(signal, TE_s, min_te_s=None, max_te_s=None, n_echoes=None):
    """
    Flexible echo selection.

    Priority:
      1. If n_echoes is set, select by echo index.
         Supported examples:
           n_echoes = 5       -> keep first 5 echoes: 0..4
           n_echoes = "0-5"   -> keep echo indices 0..5 inclusive
           n_echoes = "2-8"   -> keep echo indices 2..8 inclusive
           n_echoes = [0, 5]  -> keep echo indices 0..5 inclusive

      2. If n_echoes is null, select by TE range:
           min_te_s <= TE <= max_te_s

         min_te_s may be null.
         max_te_s may be null.

      3. If nothing is set, fallback to max_te_s = 0.010 s.

    TE values are in seconds.
    Echo indices are 0-based.
    """
    signal = np.asarray(signal)
    TE_s = np.asarray(TE_s, dtype=float)
    n_te_signal = signal.shape[3]

    if TE_s.shape[0] > n_te_signal:
        print(
            f"[WARN] TE_s has length {TE_s.shape[0]} but signal has {n_te_signal} echoes. "
            f"Truncating TE_s."
        )
        TE_s = TE_s[:n_te_signal]

    keep_idx = _parse_echo_indices(n_echoes, n_te_signal)

    if keep_idx is None:
        if min_te_s is None and max_te_s is None:
            # max_te_s = 0.010
            keep_idx = np.arange(n_te_signal, dtype=int)
            print("[ECHO] No echo cutoff set. Using all available echoes.")
        else:
            te_mask = np.ones_like(TE_s, dtype=bool)

            if min_te_s is not None:
                min_te_s = float(min_te_s)
                te_mask &= TE_s >= min_te_s

            if max_te_s is not None:
                max_te_s = float(max_te_s)
                te_mask &= TE_s <= max_te_s

            keep_idx = np.where(te_mask)[0]

        if keep_idx.size < 3:
            raise RuntimeError(
                f"Only {keep_idx.size} echoes selected with "
                f"min_te_s={min_te_s}, max_te_s={max_te_s}; need at least 3 echoes."
            )

    signal_sub = signal[..., keep_idx]
    TE_s_sub = TE_s[keep_idx]

    print("Selected echo indices:", keep_idx.tolist())
    print("Selected TE_s:", TE_s_sub)

    return signal_sub, TE_s_sub, keep_idx


# -------------------------------------------------------------------------
# hmrGC run (WF + R2*)
# -------------------------------------------------------------------------
def deep_merge(dst, src):
    if isinstance(dst, dict) and isinstance(src, dict):
        for k, v in src.items():
            if k in dst:
                dst[k] = deep_merge(dst[k], v)
            else:
                dst[k] = v
        return dst

    if isinstance(dst, list) and isinstance(src, list):
        def is_named_dict_list(x):
            return all(isinstance(e, dict) and "name" in e for e in x)

        if is_named_dict_list(dst) and is_named_dict_list(src):
            dst_by_name = {e["name"]: e for e in dst}
            dst_order = [e["name"] for e in dst]

            for e in src:
                n = e["name"]
                if n in dst_by_name:
                    dst_by_name[n] = deep_merge(dst_by_name[n], e)
                else:
                    dst_by_name[n] = e
                    dst_order.append(n)

            return [dst_by_name[n] for n in dst_order]

        return src

    return src



def load_default_multi_echo_config() -> dict:
    base = Path(me_mod.__file__).with_name("multi_echo.json")
    return json.loads(base.read_text(encoding="utf-8"))

def prepare_fat_model_from_config(fat_cfg: dict | None) -> dict:
    """
    Prepare a fat model for hmrGC.

    frequency_reference:
      - "water_peak": water is at 0 ppm, fat peaks are relative to water.
      - "fat_peak": main fat peak is at 0 ppm, water is shifted by
        -main_fat_peak_ppm, e.g. +3.30 ppm for main_fat_peak_ppm=-3.30.
    """
    fat_cfg = fat_cfg or {}

    if "freqs_ppm" in fat_cfg and "rel_amps" in fat_cfg:
        freqs_ppm = np.asarray(fat_cfg["freqs_ppm"], dtype=float)
        rel_amps = np.asarray(fat_cfg["rel_amps"], dtype=float)
        source = "JSON"
    else:
        freqs_ppm = DEFAULT_FAT_FREQS_PPM.copy()
        rel_amps = DEFAULT_FAT_REL_AMPS.copy()
        source = "default"

    if freqs_ppm.ndim != 1 or rel_amps.ndim != 1:
        raise RuntimeError("fat_model.freqs_ppm and fat_model.rel_amps must be 1D arrays.")

    if freqs_ppm.size != rel_amps.size:
        raise RuntimeError(
            "fat_model.freqs_ppm and fat_model.rel_amps must have the same length. "
            f"Got {freqs_ppm.size} frequencies and {rel_amps.size} amplitudes."
        )

    amp_sum = float(np.sum(rel_amps))
    if amp_sum <= 0:
        raise RuntimeError("fat_model.rel_amps must sum to a positive value.")

    if not np.isclose(amp_sum, 1.0, atol=1e-6):
        print(f"[FAT] Normalizing relative amplitudes. Sum before normalization: {amp_sum:.8f}")
        rel_amps = rel_amps / amp_sum

    frequency_reference = str(fat_cfg.get("frequency_reference", "water_peak")).lower()
    main_fat_peak_ppm = float(fat_cfg.get("main_fat_peak_ppm", DEFAULT_MAIN_FAT_PEAK_PPM))

    if frequency_reference in {"water", "water_peak", "water-referenced", "water_referenced"}:
        frequency_reference = "water_peak"
        water_freq_ppm = 0.0

    elif frequency_reference in {"fat", "fat_peak", "fat-referenced", "fat_referenced"}:
        frequency_reference = "fat_peak"
        water_freq_ppm = -main_fat_peak_ppm

    else:
        raise RuntimeError(
            f"Unknown fat_model.frequency_reference='{frequency_reference}'. "
            "Use 'water_peak' or 'fat_peak'."
        )

    print(f"[FAT] Using {source} fat model.")
    print(f"[FAT] frequency_reference: {frequency_reference}")
    print(f"[FAT] main_fat_peak_ppm: {main_fat_peak_ppm}")
    print(f"[FAT] WaterFreq_ppm used in hmrGC signal model: {water_freq_ppm}")
    print(f"[FAT] freqs_ppm: {freqs_ppm.tolist()}")
    print(f"[FAT] rel_amps: {rel_amps.tolist()}")

    return {
        "freqs_ppm": freqs_ppm,
        "rel_amps": rel_amps,
        "frequency_reference": frequency_reference,
        "main_fat_peak_ppm": main_fat_peak_ppm,
        "water_freq_ppm": water_freq_ppm,
    }

def get_qsm_preset() -> dict:
    """
    Default QSM-oriented settings.
    User config can still override them.
    """
    return {
        "hmrgc": {
            "r2star_correction": True,
            "sampling_stepsize_fm": 0.5,
            "sampling_stepsize_r2s": 4.0,
            "fieldmap_range": {
                "mode": "matlab_period_3half",
                "n_sampling_periods": 1.0
            },
            "multi_echo_override": {
                "default_layer_options": {
                    "smoothing": True,
                    "smoothing_sigma": 1.2,
                    "noise_weighting_intra_edges": True
                },
                "methods": {
                    "multi-res": [
                        {
                            "name": "Unwrapping",
                            "reg_param": 1e-5,
                            "voxelSize_mm": [6.0, 6.0, 6.0]
                        },
                        {
                            "name": "Mid-res",
                            "reg_param": 1e-4,
                            "prior": {
                                "neighborhood_for_range": [2.0, 2.0]
                            }
                        },
                        {
                            "name": "High-res",
                            "reg_param": 1e-3,
                            "prior": {
                                "neighborhood_for_range": [0.8, 0.8],
                                "neighborhood_for_insert": [0.4, 0.4]
                            }
                        }
                    ]
                },
                "default_method": {
                    "WF": "multi-res"
                }
            }
        },
        "qsm": {
            "enable": True,
            "fieldmap_postprocess": {
                "offset_correction": True,
                "adaptive_smoothing": True,
                "kernel_size": 3,
                "threshold_hz": 15.0,
                "save_raw_fieldmap": True
            }
        }
    }


def apply_qsm_preset(case_cfg: dict) -> dict:
    """
    Merge QSM defaults with the current case config.
    User-provided config values keep priority.
    """
    merged = deep_merge(deepcopy(get_qsm_preset()), deepcopy(case_cfg))
    merged.setdefault("qsm", {})
    merged["qsm"]["enable"] = True
    return merged


def _compute_matlab_period_range_ppm(
    TE_s,
    ppm2hz: float,
    sampling_stepsize_fm: float,
    n_sampling_periods: float = 1.0,
):
    """
    MATLAB-like field-map range.

    MATLAB logic:
      period = 1 / DeltaTE
      range approximately = +/- 1.5 * period
    """
    te = np.asarray(TE_s, dtype=float).reshape(-1)
    if te.size < 2:
        raise RuntimeError("Need at least 2 echoes to derive MATLAB-like field-map range.")

    dte = np.diff(te)
    dte = dte[np.abs(dte) > 0]

    if dte.size == 0:
        raise RuntimeError("Echo times are invalid: could not compute non-zero dTE.")

    dt_s = float(np.median(np.abs(dte)))
    period_hz = abs(1.0 / dt_s)

    half_width_hz = 0.5 * float(n_sampling_periods) * period_hz * 3.0

    range_hz = np.array(
        [
            -half_width_hz,
            half_width_hz + 2.0 * float(sampling_stepsize_fm)
        ],
        dtype=float,
    )

    range_ppm = range_hz / float(ppm2hz)

    return range_ppm, range_hz, dt_s, period_hz


def qsm_offset_correct_fieldmap(fieldmap_hz, mask, TE_s):
    """
    MATLAB-like offset correction:
    subtract an integer multiple of Delta_f = 1 / Delta_TE.
    """
    te = np.asarray(TE_s, dtype=float).reshape(-1)
    if te.size < 2:
        return fieldmap_hz

    dte = np.diff(te)
    dte = dte[np.abs(dte) > 0]
    if dte.size == 0:
        return fieldmap_hz

    delta_te = float(np.median(np.abs(dte)))
    delta_f = 1.0 / delta_te

    mask_bool = mask > 0

    mid_slice = int(round(fieldmap_hz.shape[2] / 2.0)) - 1
    mid_slice = max(0, min(mid_slice, fieldmap_hz.shape[2] - 1))

    mid_mask = mask_bool[:, :, mid_slice]

    if np.any(mid_mask):
        mean_val = float(np.mean(fieldmap_hz[:, :, mid_slice][mid_mask]))
    elif np.any(mask_bool):
        mean_val = float(np.mean(fieldmap_hz[mask_bool]))
    else:
        mean_val = 0.0

    offset = round(mean_val / delta_f) * delta_f

    corrected = (fieldmap_hz - offset) * mask_bool.astype(fieldmap_hz.dtype)

    if round(mean_val / delta_f) != 0:
        print(f"[QSM] Offset correction applied: subtracted {offset:.3f} Hz")
    else:
        print("[QSM] Offset correction: no offset change needed")

    return corrected


def qsm_adaptive_smooth_fieldmap(fieldmap_hz, mask, kernel_size=3, threshold_hz=15.0):
    """
    MATLAB-inspired adaptive field-map smoothing.
    Replaces only voxels that deviate strongly from the local mean.
    """
    mask_bool = mask > 0
    fieldmap_hz = fieldmap_hz.astype(np.float32, copy=False)

    local_mean_temp = ndi.uniform_filter(fieldmap_hz, size=kernel_size, mode="nearest")

    img_filled = fieldmap_hz.copy()
    outside = ~mask_bool
    img_filled[outside] = local_mean_temp[outside]

    local_mean = ndi.uniform_filter(img_filled, size=kernel_size, mode="nearest")
    local_mean_masked = local_mean * mask_bool.astype(local_mean.dtype)

    diff_mask = (np.abs(fieldmap_hz - local_mean_masked) > float(threshold_hz)) & mask_bool

    smoothed = fieldmap_hz.copy()
    smoothed[diff_mask] = local_mean_masked[diff_mask]

    print(
        f"[QSM] Adaptive smoothing replaced {int(diff_mask.sum())} voxel(s) "
        f"(kernel_size={kernel_size}, threshold_hz={threshold_hz})"
    )

    return smoothed


def postprocess_fieldmap_for_qsm(fieldmap_hz, mask, TE_s, qsm_cfg: dict):
    """
    Apply QSM-specific field-map postprocessing.
    """
    cfg = qsm_cfg or {}
    out = fieldmap_hz.copy()

    if bool(cfg.get("offset_correction", True)):
        out = qsm_offset_correct_fieldmap(out, mask, TE_s)

    if bool(cfg.get("adaptive_smoothing", True)):
        out = qsm_adaptive_smooth_fieldmap(
            out,
            mask,
            kernel_size=int(cfg.get("kernel_size", 3)),
            threshold_hz=float(cfg.get("threshold_hz", 15.0)),
        )

    return out

def run_hmrgc(
    signal,
    mask,
    TE_s,
    centerFreq_Hz,
    fieldStrength_T,
    voxelSize_mm,
    # fat_freqs_ppm=None,
    # fat_rel_amps=None,
    fat_model: dict | None = None,
    hmr_cfg: dict | None = None,
    
):
    fat_model = fat_model or prepare_fat_model_from_config(None)
    hmr_cfg = hmr_cfg or {}

    # if fat_freqs_ppm is None or fat_rel_amps is None:
    #     FAT_FREQS_PPM = DEFAULT_FAT_FREQS_PPM
    #     FAT_REL_AMPS = DEFAULT_FAT_REL_AMPS
    #     print("[FAT] Using default fat model.")
    # else:
    #     FAT_FREQS_PPM = np.asarray(fat_freqs_ppm, dtype=float)
    #     FAT_REL_AMPS = np.asarray(fat_rel_amps, dtype=float)
    #     print("[FAT] Using fat model from JSON.")

    params = {
        "TE_s": TE_s,
        "centerFreq_Hz": centerFreq_Hz,
        "fieldStrength_T": fieldStrength_T,
        "voxelSize_mm": voxelSize_mm,
        "signal_model": "WF",
        "WaterFreq_ppm": float(fat_model.get("water_freq_ppm", 0.0)),
        "FatModel": {
            "freqs_ppm": np.asarray(fat_model["freqs_ppm"], dtype=float),
            "relAmps": np.asarray(fat_model["rel_amps"], dtype=float),
        },
    }


    ppm2hz = float(hmr_cfg["PPM2HZ"]) if hmr_cfg.get("PPM2HZ") is not None else params["centerFreq_Hz"] * 1e-6

    params["PPM2HZ"] = ppm2hz

    range_fm_ppm = None
    range_fm_ppm_cfg = hmr_cfg.get("range_fm_ppm", None)
    fieldmap_range_cfg = hmr_cfg.get("fieldmap_range", {}) or {}

    if range_fm_ppm_cfg is not None:
        range_fm_ppm = np.asarray(range_fm_ppm_cfg, dtype=float).reshape(-1)

        if range_fm_ppm.size != 2:
            raise RuntimeError(
                "hmrgc.range_fm_ppm must contain exactly two values, "
                f"for example [-1.5, 1.5]. Got: {range_fm_ppm_cfg}"
            )

        if range_fm_ppm[1] <= range_fm_ppm[0]:
            raise RuntimeError(
                "hmrgc.range_fm_ppm must be increasing, "
                f"got {range_fm_ppm.tolist()}"
            )

        params["range_fm_ppm"] = range_fm_ppm

    else:
        range_mode = str(fieldmap_range_cfg.get("mode", "manual")).lower()

        if range_mode == "matlab_period_3half":
            n_sampling_periods = float(fieldmap_range_cfg.get("n_sampling_periods", 1.0))

            range_fm_ppm, range_fm_hz, dt_s, period_hz = _compute_matlab_period_range_ppm(
                TE_s=TE_s,
                ppm2hz=ppm2hz,
                sampling_stepsize_fm=float(hmr_cfg.get("sampling_stepsize_fm", 0.5)),
                n_sampling_periods=n_sampling_periods,
            )

            params["range_fm_ppm"] = range_fm_ppm

            print(
                "[QSM] MATLAB-like field-map range active:\n"
                f"       dTE = {dt_s:.6f} s\n"
                f"       period = {period_hz:.3f} Hz\n"
                f"       range_hz = {range_fm_hz.tolist()}\n"
                f"       range_ppm = {range_fm_ppm.tolist()}"
            )


    override = hmr_cfg.get("multi_echo_override", None)
    if override:
        cfg0 = load_default_multi_echo_config()
        cfg = deep_merge(deepcopy(cfg0), override)
        g = MultiEcho(signal, mask, params, config=cfg)
    else:
        g = MultiEcho(signal, mask, params)  

    if range_fm_ppm is not None:
        g.range_fm_ppm = range_fm_ppm
        if hasattr(g, "params"):
            g.params["range_fm_ppm"] = range_fm_ppm
        print("[HMRGC] range_fm_ppm override:", range_fm_ppm.tolist())

    g.use_gpu = bool(hmr_cfg.get("use_gpu", True))
    print("use_gpu flag:", getattr(g, "use_gpu", None))

    g.r2star_correction = bool(hmr_cfg.get("r2star_correction", True))
    g.sampling_stepsize_fm = float(hmr_cfg.get("sampling_stepsize_fm", 10.0))
    g.sampling_stepsize_r2s = float(hmr_cfg.get("sampling_stepsize_r2s", 30.0))
    g.perform()
    print("hmrGC finished.")

    water = g.images["water"]
    fat = g.images["fat"]
    pdff = g.images.get("fatFraction_percent", None)
    fieldmap = g.fieldmap
    r2s = g.r2starmap

    return water, fat, pdff, fieldmap, r2s


# -------------------------------------------------------------------------
# Per-case pipeline
# -------------------------------------------------------------------------


def process_case(case_cfg: dict, config_path: Path):
    root_path = Path(case_cfg.get("root_path", "."))

    outdir = Path(case_cfg["outdir"])

    hmr_cfg = case_cfg.get("hmrgc", {})

    mask_cfg = case_cfg.get("mask", {})
    mask_bias_cfg = mask_cfg.get("bias_correction", {})
    fill_holes = bool(mask_cfg.get("fill_holes", False))
    mask_source = mask_cfg.get("source", "magnitude_first_echo")
    mask_mode = mask_cfg.get("mode", "relative")
    mask_rel_thr = float(mask_cfg.get("rel_thr", 0.03))
    mask_abs_thr = mask_cfg.get("abs_thr", None)
    mask_percentile = mask_cfg.get("percentile", None)
    if mask_percentile is not None:
        mask_percentile = float(mask_percentile)

    echo_cfg = case_cfg.get("echo_subset", {})
    min_te_s_cfg = echo_cfg.get("min_te_s", None)
    max_te_s_cfg = echo_cfg.get("max_te_s", None)
    n_echoes_cfg = echo_cfg.get("n_echoes", None)

    pre_mask_cfg = mask_cfg.get("precomputed", None)
    pre_mask_base = None
    if pre_mask_cfg and bool(pre_mask_cfg.get("enable", False)):
        base_mode = str(pre_mask_cfg.get("path_is_relative_to", "outdir")).lower()
        if base_mode == "root":
            pre_mask_base = root_path
        elif base_mode == "outdir":
            pre_mask_base = outdir
        elif base_mode == "absolute":
            pre_mask_base = Path("/")
        else:
            print(f"[WARN] Unknown path_is_relative_to={base_mode}, defaulting to outdir")
            pre_mask_base = outdir


    fat_model = prepare_fat_model_from_config(case_cfg.get("fat_model", {}))

    outdir.mkdir(parents=True, exist_ok=True)

    print("\n==============================")
    print("Processing case with root:", root_path)
    print("Output directory:", outdir)
    print("==============================")

    
    # Save config file
    shutil.copy2(config_path, outdir / config_path.name)

    slice_subset_cfg = case_cfg.get("slice_subset", None)

    (
        TE_s_full,
        centerFreq_Hz,
        fieldStrength_T,
        ds0,
    ) = prepare_case_input_4d(
        case_cfg,
        root_path=root_path,
        outdir=outdir,
        slice_subset_cfg=slice_subset_cfg,
    )

    signal_full, mask_full, voxelSize_mm = build_complex_signal_from_4d(
        outdir,
        slice_subset_cfg=slice_subset_cfg, 
        mask_source=mask_source,
        mask_mode=mask_mode,
        mask_rel_thr=mask_rel_thr,
        mask_abs_thr=mask_abs_thr,
        mask_percentile=mask_percentile,
        precomputed_mask=pre_mask_cfg,
        precomputed_mask_base_dir=pre_mask_base,
        fill_holes=fill_holes,
        bias_correction=mask_bias_cfg,
    )

    signal, TE_s, keep_idx = select_echo_subset(
        signal_full,
        TE_s_full,
        min_te_s=min_te_s_cfg,
        max_te_s=max_te_s_cfg,
        n_echoes=n_echoes_cfg,
    )
    signal, mask, bbox = crop_to_mask_bbox(signal, mask_full, margin=4)

    water, fat, pdff, fieldmap, r2s_crop = run_hmrgc(
        signal,
        mask,
        TE_s,
        centerFreq_Hz,
        fieldStrength_T,
        voxelSize_mm,
        # fat_freqs_ppm=fat_freqs_ppm,
        # fat_rel_amps=fat_rel_amps,
        fat_model=fat_model,
        hmr_cfg=case_cfg.get("hmrgc", {}),
    )

    qsm_cfg = case_cfg.get("qsm", {}) or {}
    fieldmap_raw = fieldmap.copy()

    if bool(qsm_cfg.get("enable", False)):
        print("[QSM] Applying QSM field-map postprocessing")
        fieldmap = postprocess_fieldmap_for_qsm(
            fieldmap_hz=fieldmap,
            mask=mask,
            TE_s=TE_s,
            qsm_cfg=qsm_cfg.get("fieldmap_postprocess", {}),
        )

    full_shape = mask_full.shape
    sy, sx, sz = bbox
    hmr_cfg = hmr_cfg or {}

    water_mag_crop = np.abs(water)
    fat_mag_crop = np.abs(fat)
    t1w_inphase_crop = water_mag_crop + fat_mag_crop

    water_full = np.zeros(full_shape, dtype=water_mag_crop.dtype)
    fat_full = np.zeros_like(water_full)
    t1w_full = np.zeros_like(water_full)
    fieldmap_full = np.zeros(full_shape, dtype=fieldmap.dtype)
    fieldmap_raw_full = None
    mask_out_full = np.zeros(full_shape, dtype=np.uint8)
    r2s_full = np.zeros(full_shape, dtype=r2s_crop.dtype)

    water_full[sy, sx, sz] = water_mag_crop
    fat_full[sy, sx, sz] = fat_mag_crop
    t1w_full[sy, sx, sz] = t1w_inphase_crop
    fieldmap_full[sy, sx, sz] = fieldmap
    if bool(qsm_cfg.get("enable", False)) and bool(
        qsm_cfg.get("fieldmap_postprocess", {}).get("save_raw_fieldmap", True)
    ):
        fieldmap_raw_full = np.zeros(full_shape, dtype=fieldmap_raw.dtype)
        fieldmap_raw_full[sy, sx, sz] = fieldmap_raw
    mask_out_full[sy, sx, sz] = mask.astype(np.uint8)
    r2s_full[sy, sx, sz] = r2s_crop
    oop_synth = np.abs(water_full - fat_full)



    mag_path, phase_path, te_path = resolve_4d_paths(outdir, slice_subset_cfg)
    ref_img = nib.load(str(mag_path))

    output_paths = build_final_output_paths(case_cfg, outdir)

    export_nifti_like_ref(water_full,    ref_img, output_paths["water"], ds0=ds0)
    export_nifti_like_ref(fat_full,      ref_img, output_paths["fat"], ds0=ds0)
    export_nifti_like_ref(t1w_full,      ref_img, output_paths["inphase"], ds0=ds0)

    export_nifti_like_ref(fieldmap_full, ref_img, output_paths["fieldmap"], ds0=ds0)
    if fieldmap_raw_full is not None:
        export_nifti_like_ref(fieldmap_raw_full, ref_img, output_paths["fieldmap_raw"], ds0=ds0)
    export_nifti_like_ref(mask_out_full, ref_img, output_paths["mask"], ds0=ds0)
    export_nifti_like_ref(r2s_full,      ref_img, output_paths["r2star"], ds0=ds0)

    export_nifti_like_ref(oop_synth,     ref_img, output_paths["opposed"], ds0=ds0)

    if pdff is not None:
        pdff_full = np.zeros(full_shape, dtype=pdff.dtype)
        pdff_full[sy, sx, sz] = pdff
        export_nifti_like_ref(pdff_full, ref_img, output_paths["pdff"], ds0=ds0)

    #TODO: add h5 output here
    import h5py
    with h5py.File(Path.joinpath(outdir, 'moba_ini.h5'), 'w') as f:
        f.create_dataset('field_map', data=fieldmap_full.swapaxes(0,1))
        f.create_dataset('r2star_map', data=r2s_full.swapaxes(0,1))
        f.create_dataset('fat_map', data=fat_full.swapaxes(0,1))
        f.create_dataset('wat_map', data=water_full.swapaxes(0,1))


    print("Finished case, outputs written to", outdir)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


def parse_args():
    ap = argparse.ArgumentParser(
        description="7T Dixon WF + R2* pipeline driven by JSON config (4D NIfTI-based hmrGC)"
    )
    ap.add_argument(
        "--config",
        required=True,
        help="Path to JSON config file (see example in script).",
    )
    ap.add_argument(
        "--input-type",
        choices=["dicom", "nifti"],
        default=None,
        help="Optional override for all cases: 'dicom' uses dcm2niix, 'nifti' uses supplied 4D NIfTIs plus metadata sidecar.",
    )
    ap.add_argument(
        "--qsm",
        action="store_true",
        help="Enable QSM preset: MATLAB-like field-map range, finer sampling, R2* on, and QSM field-map postprocessing.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    cases = cfg.get("cases", [])
    if not cases:
        raise RuntimeError("Config JSON has no 'cases' entries.")

    for i, case_cfg in enumerate(cases, start=1):
        if args.input_type is not None:
            case_cfg.setdefault("input", {})["type"] = args.input_type

        if args.qsm:
            case_cfg = apply_qsm_preset(case_cfg)
            print("[MODE] QSM preset enabled")

        print(f"\n##### CASE {i}/{len(cases)} #####")
        process_case(case_cfg, config_path=config_path)


if __name__ == "__main__":
    main()
