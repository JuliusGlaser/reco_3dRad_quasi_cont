# **This repository provides code for the ISMRM abstract #03918**
# Fat water separation using a model-based reconstruction for a 3D radial sequence with quasi-continuous echotimes at 7 T

## Introduction

This repository implements reproduction of the figures used in abstract #03918 of the ISMRM 2027 in Vancouver

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

  To reproduce results there are several steps necessary:
  1. calculate density compensation for measured gradient trajectory (voronoi_dc_precomputation.py) (can be skipped if theoretical trajectory is used but will lead to artifacts)
  2. run initial reconstruction: adjoint NUFFT with Hamming filter (reconstruction.py)
  3. run the stelter fat-water separation to get an initial solution for the fieldmap and R2* map (fat_water.py)
  4. run the model-based reconstruction (reco_3d.py) 

  This can be done for phantom and invivo data, the individual configs can be found in the stelter_3d_fw_separation directory for fat water separation.

  For the reconstruction they can be found in the reconstruction_code/invivo/ or reconstruction_code/phantom/ directory.

  To run the scripts with a specific config, run
  ```
  python script_name.py --config=config_name.yaml
  ```
  only for fat water separation jsons are use so adjust the ending.

  Figures can be reproduced using the notebooks in figures/ directory

  To reproduce original results the configs to run the reconstruction.py are also given.

  To reproduce results from the original paper, run the density compensation with the config_org and the reconstruction.py for fat and and water peak. Fat water separation is then done using the matlab scripts in matlab_code/ directory

  For further information consult julius.glaser@uk-erlangen.de

</details>


