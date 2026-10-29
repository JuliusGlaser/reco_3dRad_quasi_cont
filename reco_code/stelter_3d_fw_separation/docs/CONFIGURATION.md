# Configuration guide

This document describes the JSON configuration files used by `fat_water.py`.

The script is started with:

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

The `--config` argument is required and points to a JSON file. All input paths, output paths, DICOM series names, NIfTI paths, mask settings, bias-correction settings, echo selection settings, fat model parameters, and hmrGC settings are configured inside the JSON file.

## General structure

A configuration file contains a top-level `cases` list. Each entry in `cases` describes one dataset/session.

```json
{
  "cases": [
    {
      "root_path": "/path/to/case/root",
      "outdir": "/path/to/output",
      "input": {
        "type": "nifti"
      }
    }
  ]
}
```

Multiple cases can be processed sequentially by adding several entries to the `cases` list.

## Case-level fields

```text
root_path     root folder for the input data
outdir        output folder for intermediate and final files
mag_series    DICOM magnitude series folders, relative to root_path
phs_series    DICOM phase series folders, relative to root_path
input         input-mode specific settings
output        final filename naming settings
```

The `mag_series` and `phs_series` fields are required for DICOM input. They are not required for NIfTI input.

## Input block

The `input` block defines whether the case is processed from DICOM or from existing NIfTI files.

## DICOM input

Example:

```json
"input": {
  "type": "dicom",
  "dcm2niix_exe": "W:/path/to/dcm2niix.exe",
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

Fields:

```text
type                    must be "dicom"
dcm2niix_exe             path to dcm2niix executable
dcm2niix_workdir         temporary output folder for dcm2niix results
phase_units              phase scaling; usually "siemens_0_4095" for Siemens DICOM phase
reuse_existing_niftis    reuse converted dcm2niix files if already present
```

The order of `mag_series` and `phs_series` must match:

```text
mag_series[0] belongs to phs_series[0]
mag_series[1] belongs to phs_series[1]
```

Repeated echoes with identical TE values are grouped and averaged in the complex domain.

## NIfTI input

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

Fields:

```text
type            must be "nifti"
magnitude_4d    4D magnitude NIfTI, relative to root_path unless absolute
phase_4d        4D phase NIfTI, relative to root_path unless absolute
metadata        GraphCutHeader metadata file
phase_units     radians, degrees, siemens_0_4095, or auto
```

The magnitude and phase NIfTI files must have the same shape and affine. The echo dimension must be the last dimension:

```text
x/y/z/echo
```

The metadata file is required and must contain or allow extraction of:

```text
echo times
center frequency
field strength
```

Supported metadata formats:

```text
.mat
.json
.yaml
.yml
```

For MATLAB GraphCutHeader files, the expected fields are usually:

```text
TE_mat
CF
B0_strength
```

## Output block

The `output` block controls final output filenames.

Example:

```json
"output": {
  "measurement_id": "20260319_FOR_054_7T",
  "sequence_name": "fl3d_vibe_DIXON",
  "extension": ".nii.gz"
}
```

Fields:

```text
measurement_id    scan/session identifier used at the beginning of each final filename
sequence_name     sequence name used after measurement_id
extension         output extension, usually ".nii.gz"
```

If `measurement_id` is omitted, the script tries to infer it from the path component containing `_FOR`.

If `sequence_name` is omitted, the script tries to infer it from:

```text
input.magnitude_4d
or first mag_series entry
```

Example inference:

```text
fl3d_vibe_DIXON_RR_36.nii
→ fl3d_vibe_DIXON

qsm_vibe_1iso_CAIPI6_8TE_2Av_RR_16_MR
→ qsm_vibe_1iso_CAIPI6_8TE_2Av
```

Final filenames use:

```text
<measurement_id>_<sequence_name>_<output_suffix>.nii.gz
```

Example:

```text
20260319_FOR_054_7T_fl3d_vibe_DIXON_waterimg.nii.gz
```

## Intermediate NIfTI files

The script exports intermediate 4D NIfTI files:

```text
4D_slices_all_mag.nii.gz
4D_slices_all_phase_rad.nii.gz
4D_slices_all_TE_seconds.npy
```

If `slice_subset.enable` is true, the slice range is included:

```text
4D_slices_slices_0010_0050_mag.nii.gz
4D_slices_slices_0010_0050_phase_rad.nii.gz
4D_slices_slices_0010_0050_TE_seconds.npy
```

The intermediate 4D image shape is:

```text
x/y/z/echo
```

The script reloads these intermediate files and reconstructs the complex multi-echo signal:

```python
signal = mag4d * np.exp(1j * phase4d)
```

## Slice subset

Example:

```json
"slice_subset": {
  "enable": true,
  "mode": "range",
  "z_start": 10,
  "z_end": 50,
  "indexing": "0based",
  "out_base": "4D_slices"
}
```

Fields:

```text
enable      true or false
mode        currently used as range mode
z_start     first selected slice
z_end       last selected slice
indexing    0based or 1based
out_base    filename prefix for intermediate 4D files
```

With `indexing: "0based"`, the first slice is slice `0`.

With `indexing: "1based"`, the first slice is slice `1`.

## Echo selection

The `echo_subset` block controls which echoes are used for hmrGC.

Example using a TE range:

```json
"echo_subset": {
  "min_te_s": 0.0020,
  "max_te_s": 0.0068,
  "n_echoes": null
}
```

Example using the first five echoes:

```json
"echo_subset": {
  "min_te_s": null,
  "max_te_s": null,
  "n_echoes": 5
}
```

Example using an echo index range:

```json
"echo_subset": {
  "min_te_s": null,
  "max_te_s": null,
  "n_echoes": "0-5"
}
```

Rules:

```text
Echo indices are 0-based.
If n_echoes is set, echo-index selection takes priority over min_te_s and max_te_s.
If n_echoes is null, echoes are selected by TE range.
At least three echoes are required.
```

## Mask generation

Example:

```json
"mask": {
  "source": "magnitude_first_echo",
  "mode": "relative",
  "rel_thr": 0.4,
  "fill_holes": true,
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
  },
  "precomputed": {
    "enable": false,
    "path": "",
    "path_is_relative_to": "root",
    "threshold": 0.5,
    "invert": false,
    "resample": "auto"
  }
}
```

The mask is not the final segmentation. It is a processing mask used to restrict the hmrGC calculation to the relevant foreground region and to avoid unnecessary computation in background air.

Automatic mask workflow:

```text
select 3D mask-source image
→ optional bias-field correction
→ threshold image
→ keep largest connected component
→ apply small in-plane binary closing
→ optionally fill holes
→ keep largest connected component again
→ crop signal to mask bounding box
→ run hmrGC inside crop
→ export final processing mask
```

## Mask source

Supported sources:

```text
magnitude_first_echo
magnitude_echo_mip
magnitude_echo_mean
magnitude_echo_median
phase_first_echo
phase_echo_mip
phase_abs_echo_mip
```

Recommended sources:

```text
magnitude_first_echo
magnitude_echo_mip
magnitude_echo_mean
magnitude_echo_median
```

Phase-based mask sources are experimental and are not recommended for routine processing.

## Mask threshold modes

### Relative threshold

```json
"mode": "relative",
"rel_thr": 0.4
```

Threshold:

```text
threshold = rel_thr × maximum intensity of mask-source image
```

### Absolute threshold

```json
"mode": "absolute",
"abs_thr": 100
```

Threshold:

```text
threshold = 100
```

### Percentile threshold

```json
"mode": "percentile",
"percentile": 90
```

Threshold:

```text
threshold = 90th percentile of the mask-source image
```

## Bias-field correction

Bias-field correction is optional and is applied only to the 3D mask-source image before thresholding. It does not modify the original complex multi-echo signal used by hmrGC.

Use one shared `bias_correction` block with method-specific sub-blocks.

### SPM

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

The `spm_path` must point to the folder containing `spm.m`, not to the file itself.

### SimpleITK N4

```json
"bias_correction": {
  "enable": true,
  "method": "sitk_n4",
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

SimpleITK N4 requires:

```bash
pip install SimpleITK
```

To compare SPM and SimpleITK N4, run the same case twice and only change:

```json
"method": "spm"
```

versus:

```json
"method": "sitk_n4"
```

If `debug_export` is true, corrected mask-source images are exported for visual comparison.

## Precomputed NIfTI mask

Instead of automatic thresholding, a precomputed NIfTI mask can be used:

```json
"precomputed": {
  "enable": true,
  "path": "/masks/case_mask.nii.gz",
  "path_is_relative_to": "root",
  "threshold": 0.5,
  "invert": false,
  "resample": "auto"
}
```

Fields:

```text
enable               use precomputed mask if true
path                 path to the NIfTI mask
path_is_relative_to  root, outdir, or absolute
threshold            mask values greater than this value are foreground
invert               invert foreground/background if true
resample             auto or off
```

If `resample` is `auto`, the mask is resampled to the 4D magnitude image grid using nearest-neighbor interpolation when needed.

## Fat model

Example:

```json
"fat_model": {
  "freqs_ppm": [-3.70, -3.30, -3.01, -2.57, -2.35, -1.83, 0.71],
  "rel_amps":  [0.0847, 0.6257, 0.0707, 0.0952, 0.0662, 0.0159, 0.0418]
}
```

Fields:

```text
freqs_ppm   fat peak frequency offsets in ppm
rel_amps    relative amplitudes of the fat peaks
```

Both lists must have the same length. If no fat model is provided, the default fat model in `fat_water.py` is used.

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

Fields:

```text
use_gpu                 enable GPU usage if the environment supports it
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

## Final outputs

Final output files use:

```text
<measurement_id>_<sequence_name>_<output_suffix>.nii.gz
```

Output suffixes:

```text
waterimg                 water-only image
fatimg                   fat-only image
inphaseimg               synthetic in-phase image, water + fat
opposed_phase            synthetic opposed-phase image, abs(water - fat)
smoothedB0map            estimated B0 field map in Hz
r2starmap                estimated R2* map
mask                     processing mask
PDFFmap                  proton-density fat-fraction map, if available
```

Acquired first-echo in-phase images and acquired opposed-phase images are no longer exported as final outputs.

## GPU / CUDA troubleshooting

If GPU mode fails with:

```text
CUDA_ERROR_NO_BINARY_FOR_GPU: no kernel image is available for execution on the device
```

then the installed CuPy/CUDA toolchain is not compatible with the GPU. Test CuPy independently before debugging the pipeline.

For stable CPU testing:

```json
"hmrgc": {
  "use_gpu": false
}
```

For GPU testing:

```json
"hmrgc": {
  "use_gpu": true
}
```

## Known open item

PDFF calculation still needs final review. Some cases can produce negative values or values greater than 100%. PDFF maps should therefore be checked carefully until this is resolved.
