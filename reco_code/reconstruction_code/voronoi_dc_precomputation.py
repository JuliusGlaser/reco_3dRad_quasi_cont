import numpy as np
import h5py
import os
import numpy as np
import twixtools
import sigpy as sp
from sigpy.mri import app
from pathlib import Path

from pydicom.uid import generate_uid

import re
from typing import Any, Dict, List, Union
from time import time

import yaml
from yaml import Loader
from scipy.io import loadmat


Number = Union[int, float]

from scipy.spatial import Voronoi, ConvexHull
import numpy as np

from RadialSlidingRecon.reconstruction import *
from RadialSlidingRecon.utility import *
from multiprocessing import Pool, cpu_count,shared_memory

from concurrent.futures import ProcessPoolExecutor

# --- worker state (shared via fork/spawn pickle) ---
_V = None
_regions = None
_point_region = None

def _init_worker(V, regions, point_region):
    global _V, _regions, _point_region
    _V = V
    _regions = regions
    _point_region = point_region

def _cell_volume(i: int) -> float:
    region_idx = _point_region[i]
    region = _regions[region_idx]

    if (not region) or (-1 in region):
        return 0.0

    verts = _V[np.asarray(region, dtype=int)]
    if verts.shape[0] < 4:
        return 0.0

    try:
        # QJ helps with near-coplanar numerical issues; can also slightly speed failures
        hull = ConvexHull(verts, qhull_options="QJ")
        return float(hull.volume)
    except Exception:
        return 0.0

def mri_dcf_voronoi_fast(kspace: np.ndarray, max_workers: int | None = None) -> np.ndarray:
    kspace = np.asarray(kspace, dtype=np.float64)
    if kspace.ndim != 2 or kspace.shape[1] != 3:
        raise ValueError("kspace must be (N, 3)")

    vor = Voronoi(kspace)
    V = np.asarray(vor.vertices, dtype=np.float64)

    N = kspace.shape[0]
    weights = np.zeros(N, dtype=np.float64)

    # parallel over points (volume per region)
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(V, vor.regions, vor.point_region),
    ) as ex:
        for i, vol in enumerate(ex.map(_cell_volume, range(N), chunksize=512)):
            weights[i] = vol

    return weights

def mri_dcf_voronoi(kspace: np.ndarray) -> np.ndarray:
    """
    Compute Voronoi-cell volumes as DCF weights (3D).
    kspace: (N, 3) array of k-space coordinates.
    Returns: (N,) array of cell volumes (0 for unbounded/invalid cells).
    """
    kspace = np.asarray(kspace, dtype=float)
    if kspace.ndim != 2 or kspace.shape[1] != 3:
        raise ValueError("kspace must be an (N, 3) array")

    N = kspace.shape[0]
    weights = np.zeros(N, dtype=float)

    # SciPy Voronoi uses:
    # - vor.vertices: (M, 3)
    # - vor.regions: list of regions, each is list of vertex indices (or [-1,...] for infinity)
    # - vor.point_region: maps each input point -> index into vor.regions
    vor = Voronoi(kspace)
    verts_all = vor.vertices

    for i in range(N):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]

        # Unbounded or empty region
        if not region or (-1 in region):
            weights[i] = 0.0
            continue

        verts = verts_all[np.array(region, dtype=int)]
        if verts.shape[0] < 4:  # need at least 4 non-coplanar points for 3D volume
            weights[i] = 0.0
            continue

        try:
            hull = ConvexHull(verts)
            weights[i] = float(hull.volume)
        except Exception:
            weights[i] = 0.0

    return weights

def getDensityCompensation_Voronoi(kvec, params):
    
    nSamples = params['acquisition']['radialSamples']
    nProj = params['reconstruction']['selectiveProjections'][1] - params['reconstruction']['selectiveProjections'][0]

    assert kvec.shape == (3, nSamples * nProj), f"kvec must have shape (3, {nSamples * nProj})"

    kvec = kvec.T

    nPts = nSamples*nProj

    print("Computing Voronoi-based DCF weights...")

    zeroMask = np.sum(np.abs(kvec), axis=1) == 0

    weights = np.zeros((nPts, 1), dtype=float)

    if np.sum(zeroMask) > 0:
        #TODO: implement this
        raise NotImplementedError("Voronoi DCF does not yet handle zero k-space points. Please implement handling for these cases.")
        # print(f"Warning: {np.sum(zeroMask)} k-space points are exactly zero. Setting their DCF weights to zero.")
        # weights[zeroMask, 0] = 0.0
    else:
        # weights = mri_dcf_voronoi(kvec)
        num_cpus = max(cpu_count(), 1)
        print(f"Using {num_cpus} CPU cores for Voronoi DCF computation.")
        weights = mri_dcf_voronoi_fast(kvec, num_cpus)
        # weights = mri_dcf_voronoi(kvec)

    # Normalize and reshape weights
    weights = abs(weights)

    if np.sum(weights) > 0:
        pass
    else:
        print("Warning: All Voronoi DCF weights are zero. Returning uniform weights.")
        kAbs = np.sqrt(np.sum(kvec**2, axis=1))
        weights = kAbs**2

    weights = weights / np.median(weights[weights > 0])  # normalize by median of non-zero weights

    densityCompenstation = weights.reshape((nProj, nSamples))

    # Fix edges
    densityCompenstation[:,-1] = densityCompenstation[:,-2]

    # densityCompenstation = densityCompenstation[np.newaxis, ...]

    return densityCompenstation

def main():
    stream = open('config_voronoi_calc.yaml', 'r')                   #TODO: adapt config file for your purpose
    config = yaml.load(stream, Loader)
    stream.close()

    params = {}

    params.setdefault('acquisition', {})
    params.setdefault('reconstruction', {})

    params['slidingwindowstep'] = config['slidingwindowstep']
    params['slidingwindowpros'] = config['slidingwindowpros']
    params['acquisition']['goldenAngle'] = config['goldenAngle']
    params['acquisition']['coordinatesVersion'] = config['coordinatesVersion'] 
    params['reconstruction']['rotation'] = config['rotation']
    params['acquisition']['radialSamples'] = config['radialSamples']

    pros = config['N_projections'] * config['N_TE']  # [#]
    steps = int((pros - config['slidingwindowpros']) / config['slidingwindowstep'] + 1)

    if config['cropped_equal_echo_arr']:
        params['acquisition']["TEspacing"] = config['echospacing']
        params['acquisition']["echos"] = config['N_TE']
        params['acquisition']["projections"] = config['N_projections']
        params['acquisition']["TE"] = [0.1, 2.0, 3.9, 5.8, 7.7]
        params['acquisition']['dwelltime'] = [2.0, 2.0]


        steps, meanTimes, T_rounded = get_quasi_continuous_params(params)
            
        N_selectedProjections = config['slidingwindowpros']
        N_window_step = config['slidingwindowstep']
        N_samples = params['acquisition']['radialSamples']

        TE_rastered = np.zeros((meanTimes.shape[0], N_selectedProjections, N_samples), dtype=np.float32)
        curr_projection = 0
        # print(params['acquisition']['echospacing'], params['acquisition']['projections'], params['acquisition']['echos'], params['acquisition']['TE'][0])
        for te in range(meanTimes.shape[0]):
            TE_rastered[te, :, :] = getTEArray_Rastered_pyth(params['acquisition']['TEspacing'], params['acquisition']['projections'], params['acquisition']['echos'], params['acquisition']['TE'][0], [curr_projection, curr_projection+N_selectedProjections], N_samples, params['acquisition']['dwelltime'][0])
            curr_projection += N_window_step

        TE_rastered = TE_rastered.reshape((TE_rastered.shape[0]*TE_rastered.shape[1], -1))
        n_to_keep, new_arr, dropped_indices, window_start_indices = get_indices_to_drop(TE_rastered)

        N_total_proj = config['N_projections'] * config['N_TE']
        steps = new_arr.shape[0]//n_to_keep

        params['slidingwindowpros'] = n_to_keep

    density_compensation_vornoi_list = []

    full_t = time()
    for i in range(steps):

        if config['cropped_equal_echo_arr'] is False:
            projection_start = i* params['slidingwindowstep']
            projection_end = projection_start + params['slidingwindowpros']
            params['reconstruction']['selectiveProjections'] = [projection_start, projection_end]
            print(f"Processing step {i+1}/{steps} with projections {projection_start} to {projection_end}") 
        else:
            projection_start = window_start_indices[i]
            projection_end = projection_start + n_to_keep
            params['reconstruction']['selectiveProjections'] = [projection_start, projection_end]
            print(f"Processing step {i+1}/{steps} with projections {projection_start} to {projection_end}") 


        azimutalAngle, polarAngle = getProjectionAngles(goldenAngle = config['goldenAngle'], proj_start = projection_start, proj_end=projection_end)

        loaded_kvec = loadmat(os.path.join(config['path'], config['kvecFile']))
        kvec = loaded_kvec['kVec_Meas_AxsEco'].T
        reso = 1 / (2 * kvec[2, -1]) * 1000  # [mm]

        params['reconstruction']['resolution'] = reso

        kvec = getCoordinates(azimutalAngle, polarAngle, kvec, params['acquisition']['coordinatesVersion'])
        kvec = getKvecRotation(kvec, params['reconstruction']['rotation'])

        density_compensation_vornoi = getDensityCompensation_Voronoi(kvec, params)
        density_compensation_vornoi_list.append(density_compensation_vornoi)

    with h5py.File(os.path.join(config['path'], f"densityCompensation_projections_{params['slidingwindowpros']}_N_windows_{steps}.h5"), "w") as f:
        f.create_dataset("densityCompensationVoronoi", data=np.array(density_compensation_vornoi_list))

if __name__ == "__main__":
    main()