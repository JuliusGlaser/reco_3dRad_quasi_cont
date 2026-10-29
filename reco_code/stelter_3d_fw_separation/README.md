# Stelter_3D_fw_separation

Python pipeline for 3D water-fat separation of multi-echo 7T MRI data using the `fieldmapping-hmrGC` framework.

The pipeline supports two input modes:

1. **DICOM input**  
   Matching Siemens magnitude and phase DICOM series are converted to NIfTI with `dcm2niix`.

2. **NIfTI input**  
   Existing 4D magnitude and phase NIfTI files are used directly. In this mode, additional metadata is required via a GraphCutHeader sidecar file (`.mat`, `.json`, `.yaml`, or `.yml`) so that echo times, center frequency, and field strength can be read.

The script reconstructs a complex multi-echo signal, exports intermediate 4D magnitude/phase NIfTI files, creates or loads a processing mask, optionally applies bias-field correction to the mask-source image before thresholding, and runs hmrGC water-fat separation.

## Repository structure

```text
Stelter_3D_fw_separation/
├── fat_water.py
├── fat_water_config_breast_dicom.json
├── fat_water_config_breast_nifti.json
├── fat_water_config_calf_dicom.json
├── requirements.txt
├── fieldmapping-hmrGC/
├── README.md
└── docs/
    └── CONFIGURATION.md
```

`fieldmapping-hmrGC` is included as a Git submodule. It contains the external hmrGC implementation used for the actual water-fat separation. This repository provides the wrapper around it: input preparation, DICOM-to-NIfTI conversion, NIfTI handling, mask generation, configuration handling, and output export.

## Installation

### 1. Clone the repository with the submodule

```bash
git clone --recurse-submodules https://gitlab.rrze.fau.de/be17lika/stelter_3d_fw_separation.git
cd stelter_3d_fw_separation
```

If the repository was already cloned without submodules, run:

```bash
git submodule update --init --recursive
```

Check that the submodule exists:

```bash
ls fieldmapping-hmrGC
```

The folder must be named `fieldmapping-hmrGC`, because `fat_water.py` imports the hmrGC code from this local path.

### 2. Create the Python environment

```bash
conda create -n fat_water_sep python=3.11 -y
conda activate fat_water_sep
```

or with a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

For optional SimpleITK N4 bias-field correction:

```bash
pip install SimpleITK
```

## Optional external tools

### dcm2niix

DICOM input mode requires `dcm2niix`. The executable path is configured in the JSON file:

```json
"input": {
  "type": "dicom",
  "dcm2niix_exe": "tools/dcm2niix/dcm2niix.exe"
}
```

### MATLAB/SPM

SPM bias-field correction requires MATLAB and SPM. The config must point to the folder that directly contains `spm.m`:

```json
"spm_path": "W:/path/to/spm"
```

This is optional. SimpleITK N4 can be used instead.

## Running the pipeline

```bash
python fat_water.py --config fat_water_config_breast_nifti.json
```

or:

```bash
python fat_water.py --config fat_water_config_breast_dicom.json
```

or:

```bash
python fat_water.py --config fat_water_config_calf_dicom.json
```

The `--config` argument is required. All input paths, output paths, series names, mask settings, echo selection, fat model parameters, bias correction settings, and hmrGC settings are configured inside the JSON file.

## Input modes

### DICOM input

Example:

```json
"input": {
  "type": "dicom",
  "dcm2niix_exe": "tools/dcm2niix/dcm2niix.exe",
  "dcm2niix_workdir": "_temp/dcm2niix",
  "phase_units": "siemens_0_4095",
  "reuse_existing_niftis": true
},
"mag_series": [
  "qsm_vibe_1iso_CAIPI6_8TE_2Av_RR_16_MR",
  "qsm2_vibe_1iso_CAIPI6_8TE_2Av_RR_18_MR"
],
"phs_series": [
  "qsm_vibe_1iso_CAIPI6_8TE_2Av_RR_17_MR",
  "qsm2_vibe_1iso_CAIPI6_8TE_2Av_RR_19_MR"
]
```

The order of `mag_series` and `phs_series` must match.

### NIfTI input

Example:

```json
"input": {
  "type": "nifti",
  "magnitude_4d": "fl3d_vibe_DIXON_RR_36.nii",
  "phase_4d": "fl3d_vibe_DIXON_RR_37_ph.nii",
  "metadata": "GraphCutHeader/GraphCutHeader_fl3d_vibe_DIXON_RR_36_MR.mat",
  "phase_units": "radians"
}
```

For NIfTI input, the metadata file is required. It must provide:

```text
TE information
center frequency
field strength
```

Supported metadata file types:

```text
.mat
.json
.yaml
.yml
```

The NIfTI magnitude and phase files must be 4D, with the echo dimension last:

```text
x/y/z/echo
```

## Processing workflow

```text
DICOM or 4D NIfTI input
→ echo-time and metadata extraction
→ magnitude/phase loading
→ phase conversion to radians
→ complex multi-echo signal reconstruction
→ intermediate 4D NIfTI export
→ 4D NIfTI reload
→ automatic or precomputed mask generation
→ optional bias-field correction of mask-source image
→ thresholding and mask post-processing
→ crop to mask bounding box
→ hmrGC 3D water-fat separation
→ full-size final NIfTI output export
```

The hmrGC algorithm does not run on a single ordinary 3D image. It requires a complex multi-echo signal and echo timing information.

## Intermediate files

The pipeline writes intermediate 4D files to `outdir`:

```text
4D_slices_all_mag.nii.gz
4D_slices_all_phase_rad.nii.gz
4D_slices_all_TE_seconds.npy
```

If a slice subset is used, the filenames include the selected slice range:

```text
4D_slices_slices_0010_0050_mag.nii.gz
4D_slices_slices_0010_0050_phase_rad.nii.gz
4D_slices_slices_0010_0050_TE_seconds.npy
```

These files contain:

```text
*_mag.nii.gz          4D magnitude image
*_phase_rad.nii.gz    4D phase image in radians
*_TE_seconds.npy      echo times in seconds
```

## Output folder and filename structure

Final output files are written to `outdir`.

The final filenames use this structure:

```text
<measurement_id>_<sequence_name>_<output_suffix>.nii.gz
```

Example:

```text
20260319_FOR_054_7T_fl3d_vibe_DIXON_waterimg.nii.gz
20260319_FOR_054_7T_fl3d_vibe_DIXON_fatimg.nii.gz
20260319_FOR_054_7T_fl3d_vibe_DIXON_inphaseimg.nii.gz
20260319_FOR_054_7T_fl3d_vibe_DIXON_opposed_phase.nii.gz
20260319_FOR_054_7T_fl3d_vibe_DIXON_smoothedB0map.nii.gz
20260319_FOR_054_7T_fl3d_vibe_DIXON_r2starmap.nii.gz
20260319_FOR_054_7T_fl3d_vibe_DIXON_mask.nii.gz
20260319_FOR_054_7T_fl3d_vibe_DIXON_PDFFmap.nii.gz
```

`measurement_id` can be set explicitly:

```json
"output": {
  "measurement_id": "20260319_FOR_054_7T",
  "sequence_name": "fl3d_vibe_DIXON",
  "extension": ".nii.gz"
}
```

If not set explicitly, the script tries to infer it from the path component containing `_FOR`.

`sequence_name` can also be set explicitly. If omitted, the script tries to infer it from the NIfTI filename or the first magnitude DICOM series name.

## Final output meanings

```text
*_waterimg.nii.gz
    Water-only image estimated by hmrGC.

*_fatimg.nii.gz
    Fat-only image estimated by hmrGC.

*_inphaseimg.nii.gz
    Synthetic in-phase image calculated as water + fat.

*_opposed_phase.nii.gz
    Synthetic opposed-phase image calculated as abs(water - fat).

*_smoothedB0map.nii.gz
    Estimated B0 field map in Hz.

*_r2starmap.nii.gz
    Estimated R2* map. If R2* correction is disabled, this output should be interpreted accordingly.

*_mask.nii.gz
    Processing mask used for the hmrGC calculation.

*_PDFFmap.nii.gz
    Proton-density fat-fraction image if available from hmrGC.
```

Acquired first-echo in-phase images and acquired opposed-phase images are no longer exported as final outputs. The exported in-phase and opposed-phase images are synthetic images derived from the estimated water and fat components.

## Mask generation

The automatic mask is a processing mask, not a final tissue segmentation. It restricts hmrGC to relevant foreground voxels and avoids unnecessary computation in background air.

The automatic mask workflow is:

```text
select 3D mask-source image
→ optional bias-field correction
→ threshold image
→ keep largest connected component
→ apply small in-plane binary closing
→ optionally fill holes
→ keep largest connected component again
→ crop signal to mask bounding box
```

Supported mask sources:

```text
magnitude_first_echo
magnitude_echo_mip
magnitude_echo_mean
magnitude_echo_median
phase_first_echo
phase_echo_mip
phase_abs_echo_mip
```

Recommended mask sources:

```text
magnitude_first_echo
magnitude_echo_mip
magnitude_echo_mean
magnitude_echo_median
```

Phase-based mask sources are experimental.

## Bias-field correction for mask generation

Bias-field correction can be applied to the 3D mask-source image before thresholding. This does not change the original multi-echo complex signal used by hmrGC.

Supported methods:

```text
spm       MATLAB/SPM bias-field correction
sitk_n4   SimpleITK N4 bias-field correction
```

Recommended config structure:

```json
"bias_correction": {
  "enable": true,
  "method": "spm",
  "debug_export": true,

  "spm": {
    "matlab_cmd": "matlab",
    "matlab_mode": "batch",
    "spm_path": "W:/path/to/spm",
    "biasreg": 0.001,
    "biasfwhm": 30,
    "workdir": "_temp/spm_bias_mask",
    "cleanup": false
  },

  "sitk_n4": {
    "max_iterations": [50, 50, 30, 20],
    "convergence_threshold": 0.001,
    "otsu_bins": 200
  }
}
```

To compare methods, run the same case once with:

```json
"method": "spm"
```

and once with:

```json
"method": "sitk_n4"
```

If `debug_export` is enabled, the corrected mask-source image is exported for visual comparison.

## hmrGC settings

Example:

```json
"hmrgc": {
  "use_gpu": true,
  "r2star_correction": false,
  "sampling_stepsize_fm": 10.0,
  "sampling_stepsize_r2s": 30.0,
  "PPM2HZ": null
}
```

Important fields:

```text
use_gpu                 enable GPU usage if the local CUDA/CuPy environment supports it
r2star_correction       enable or disable R2* correction
sampling_stepsize_fm    field-map sampling step size
sampling_stepsize_r2s   R2* sampling step size
PPM2HZ                  ppm-to-Hz conversion; if null, calculated from center frequency
multi_echo_override     optional override of the default hmrGC multi-echo configuration
```

R2* correction can be disabled with:

```json
"r2star_correction": false
```

## CUDA/GPU notes

GPU execution requires a CUDA Toolkit version that is compatible with the installed GPU and CuPy version.
For stable testing, set:

```json
"use_gpu": true
```

Before running the pipeline with GPU enabled, activate the Python environment and run the following commands in PowerShell:
```bash
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
$env:CUDA_HOME = $env:CUDA_PATH

$env:Path = "$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;" + (
  ($env:Path -split ';' | Where-Object { $_ -notmatch 'NVIDIA GPU Computing Toolkit\\CUDA\\v12\.4' }) -join ';'
)

Remove-Item -Recurse -Force "$env:USERPROFILE\.cupy\kernel_cache" -ErrorAction SilentlyContinue
```

Check that CUDA 12.9 is now used:

```bash
where.exe nvcc
nvcc --version
echo $env:CUDA_PATH
```

Expected output should point to CUDA v12.9, for example:

´´´
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe
Cuda compilation tools, release 12.9
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
´´´

For GPU testing, check that a basic CuPy smoke test works before running the pipeline.


## Local external tools

Large external tools and installers are not committed to this repository.

The following local files/folders are intentionally ignored by Git:

```text
spm/
tools/
dcm2niix_win.zip

On the Breast Reconstruction PC / group workstation, these files can be found in:

W:\radiologie\mr-physik-data\Mitarbeiter\_Forschungsgruppe\Breast\FAT_WATER_SEP_ALGORITHM\fat_water\stelter_3d_fw_separation

Expected local examples:

spm/spm.m
tools/dcm2niix/dcm2niix.exe
dcm2niix_win.zip

## Notes

- At least three echoes are required for water-fat separation.
- The automatic mask should be checked visually for new datasets.
- Bias-field correction is only applied to the mask-source image before thresholding.
- The original hmrGC input signal is not bias-corrected.
- NIfTI input requires a valid GraphCutHeader metadata file.
- DICOM input requires `dcm2niix`.
- The `fieldmapping-hmrGC` submodule must be initialized before running the script.
