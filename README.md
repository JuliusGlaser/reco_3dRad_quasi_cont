# **⚠️ This repository is under construction! ⚠️**
# Reconstruction for a 3D radial sequence at 7T with quasi continuous echotimes

## Introduction

This repository implements:

- reconstruction of a 3D radial data with quasi-continuous echotimes [see sequence details](https://onlinelibrary.wiley.com/doi/10.1002/mrm.70323)

- plotting of several figures

<details>
  <summary> <span style="font-size: 25px; font-weight: bold;">Installation (using conda)</span></summary>

Get basic packages
```bash
conda env create -f environment.yml -n rsr_llr
```

Get the [sigpy_glaser](https://github.com/JuliusGlaser/sigpy_glaser) and [twixtools](https://github.com/pehses/twixtools) repos and execute the following in the repositories with the conda env activated:

```bash
pip install -e .
```

Additionally get torch and cupy for gpu utilization

```bash
pip3 install torch torchvision
pip install cupy-cuda12x

```
</details>

<details>
  <summary> <span style="font-size: 25px; font-weight: bold;">Reconstruction</span></summary>

Set the config.yaml in reconstruction_code as desired.

If a measured gradient trajectory is to be used, density compensation has to be calculated with voronoi computation.

and then run

```bash
python reconstruction.py
```

After reconstruction run some fat water separation algorithm to get the water and fat images.

To compensate chemical shift artifacts a bit, reconstruction can be run with a shift to the main fat peak, which also needs to be respected in the fat-water separation afterwards.
</details>


