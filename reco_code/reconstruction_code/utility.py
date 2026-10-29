import h5py
import numpy as np
from scipy.io import savemat
import os
import nibabel as nib
from matlab_functions_in_python import *

def _decode_if_possible(x):
    """Try to decode bytes / MATLAB-style strings."""
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8", errors="replace")

    if isinstance(x, np.ndarray) and x.dtype == np.uint8:
        try:
            b = bytes(x[x != 0])
            return b.decode("utf-8", errors="replace")
        except Exception:
            pass

    return x

def _convert_leaf(arr):
    """
    Convert squeezed numpy array into:
    - Python scalar if single value
    - Python list if multiple values
    """
    arr = np.squeeze(arr)

    # Try decoding strings
    arr = _decode_if_possible(arr)

    # Scalar → Python number/string
    if not isinstance(arr, np.ndarray):
        return arr

    if arr.shape == ():          # 0-d array
        return arr.item()

    # Array → list
    return arr.tolist()

def h5_to_dict(node):
    """
    Recursively convert HDF5 groups/datasets to dict.
    """
    # Dataset = leaf
    if isinstance(node, h5py.Dataset):
        data = np.array(node[...])
        return _convert_leaf(data)

    # Group = dict
    if isinstance(node, h5py.Group):
        out = {}
        for key, item in node.items():
            out[key] = h5_to_dict(item)
        return out

    raise TypeError(f"Unsupported HDF5 node: {type(node)}")

def ls_comb(m, c, coil_axis=-1):
        numerator = np.sum(c.conj() * m, axis=coil_axis)
        denuminator = np.linalg.norm(c, axis=coil_axis)
        return np.divide(numerator, denuminator, where=(denuminator != 0))  # not standard division because of 0 elements

def post_process_for_hernando(data, path_to_out, outname, TEs, B0, B0_strength, voxel_size):         
    
    gamma = 42.577478518e6  # gyromagnetic ratio in Hz/T
    CF = gamma * B0_strength  # Larmor frequency in Hz

    assert len(data.shape) == 3, "Input data must be 3D (n_TE, x, y)"
        
    data = data[:,np.newaxis, np.newaxis,...]#*1000
    data = data.transpose(-2,-1,1,2,0)
    data = data[::-1, ::-1,...]

    savemat(path_to_out + outname + '.mat', {'finalImages_array': data,
                                             'TE_mat': TEs,
                                             'no_TE': TEs.shape[0],
                                             'B0_mat': B0,
                                             'B0_strength': B0_strength,
                                             'voxel_size': voxel_size,
                                             'matrix_size': data.shape[:-1],
                                             'CF': CF})


def np_to_nii(data, affine, out_name, file_path, config_str):

    # ensure correct dimension ordering
    data = data.T

    assert len(data.shape) == 4, "Input data must be 4D (z, y, x, n_TE)"

    data = np.transpose(data, (2, 1, 0, 3))  # reorder to (x, y, z, n_TE)
    data = np.rot90(data, axes=(0, 1), k=1)  # rotate to match NIfTI orientation
    data = data[::-1,...]  # flip x and y axes to match NIfTI orientation

    data_abs = np.abs(data)
    data_phs = np.angle(data)

    if affine is None:
        affine=np.eye(data.ndim)
    else:
        affine=affine
    
    ext = nib.nifti1.Nifti1Extension(6, config_str.encode('utf-8'))

    img_abs = nib.Nifti1Image(data_abs, affine=affine)
    img_abs.header.set_sform(affine, code=1)
    img_abs.header.set_qform(affine, code=1)
    img_abs.header.extensions.append(ext)

    nib.save(img_abs, file_path + os.sep + out_name + '_abs.nii')


    img_phs = nib.Nifti1Image(data_phs, affine=affine)
    img_phs.header.set_sform(affine, code=1)
    img_phs.header.set_qform(affine, code=1)
    img_phs.header.extensions.append(ext)

    nib.save(img_phs, file_path + os.sep + out_name + '_phs.nii')

    # %%

def convert_raw_into_TE_windows(raw_data, TE_rastered, params, config):
    from scipy.io import loadmat


    N_coils, N_total_proj, N_samples = raw_data.shape

    n_to_keep, new_arr, dropped_indices, window_start_indices = get_indices_to_drop(TE_rastered=TE_rastered)

    N_windows = (N_total_proj-len(dropped_indices))//n_to_keep

    keep_mask = np.ones(TE_rastered.shape[0], dtype=bool)
    keep_mask[dropped_indices] = False

    k_all_new = raw_data[:, keep_mask, :]
    k_all_new = np.reshape(k_all_new, (N_coils, N_windows, n_to_keep, N_samples))

    #now estimate the coords
    kvec_theos = np.zeros((N_windows, 3, n_to_keep*N_samples))

    TE_rastered_new = TE_rastered[keep_mask,:]
    TE_rastered_new = TE_rastered_new.reshape((N_windows, n_to_keep, N_samples))

    for i in range(N_windows):
        azimutalAngle, polarAngle = getProjectionAngles(True, proj_start=window_start_indices[i], proj_end=window_start_indices[i]+n_to_keep)

        kvec_theo, reso = getKvecTheo(params)

        if params['reconstruction']['kvecMethod'] == 2:
            loaded_kvec = loadmat(os.path.join(config['path'],config['recon_parameter']['kvecFile']))
            kvec_meas = loaded_kvec['kVec_Meas_AxsEco'].T
            reso = 1 / (2 * kvec_meas[2, -1]) * 1000  # [mm]
            kvec_theo = getCoordinates(azimutalAngle, polarAngle, kvec_meas, params['acquisition']['coordinatesVersion'])
        elif params['reconstruction']['kvecMethod'] == 3:
            loaded_kvec = loadmat(os.path.join(config['path'],config['recon_parameter']['kvecFile']))
            kvec_meas = loaded_kvec['kVec_Meas_AxsEco'].T
            reso = 1 / (2 * kvec_meas[2, -1]) * 1000  # [mm]
            kvec_theo = getCoordinates(azimutalAngle, polarAngle, kvec_meas, params['acquisition']['coordinatesVersion'])
        else:
            # loaded_kvec = loadmat(os.path.join(config['path'],config['recon_parameter']['kvecFile']))
            # kvec_meas = loaded_kvec['kVec_Meas_AxsEco'].T
            kvec_theo = getCoordinates(azimutalAngle, polarAngle, kvec_theo, params['acquisition']['coordinatesVersion'])

        params['reconstruction']['resolution'] = reso
        
        kvec_theo = getKvecRotation(kvec_theo, params['reconstruction']['rotation'])

        kvec_theos[i,...] = kvec_theo

        # kvec_scaled = kvec_theo / np.max(np.abs(kvec_theo))

        # Nd = np.array(params["reconstruction"]["sizeFilling"], dtype=np.float32)  # (Nx,Ny,Nz)
        # coord = kvec_scaled * (Nd[:, None]) / 2

        # coord = np.transpose(coord)  # shape (3,M)
        # coord = coord.reshape((3, N_samples, n_to_keep), order="F")
        # coord = coord.T

        # coords[i,...] = coord

    return k_all_new, kvec_theos, TE_rastered_new, window_start_indices


def get_indices_to_drop(TE_rastered):
    TEs, count = np.unique(TE_rastered[:,0], return_counts=True)
    
    n_to_keep = np.min(count[1:-1])

    arr = TE_rastered[:,0]

    # 1. Find run boundaries (run-length encode)
    change_idx = np.where(np.diff(arr) != 0)[0] + 1
    group_starts = np.concatenate(([0], change_idx))
    group_ends = np.concatenate((change_idx, [len(arr)]))
    unique_vals = arr[group_starts]
    run_lengths = group_ends - group_starts

    print("Run lengths found:", run_lengths)

    # 2. Keep only runs with length >= n_to_keep, truncate each to exactly n_to_keep
    pieces = []
    dropped = []
    dropped_indices = []
    window_start_indices = []
    for start, end, val, length in zip(group_starts, group_ends, unique_vals, run_lengths):
        if length < n_to_keep:
            dropped.append((val, length))
            dropped_indices.extend(range(start, end))
            continue
        pieces.append(arr[start:start+n_to_keep])
        window_start_indices.append(start)  # <-- record it here
        if length > n_to_keep:
            dropped_indices.extend(range(start+n_to_keep, end))  # truncated-off tail

    new_arr = np.concatenate(pieces)
    dropped_indices = np.array(sorted(dropped_indices))
    window_start_indices = np.array(window_start_indices)  # one entry per window, in order

    print("New total length:", new_arr.shape)
    print("Dropped runs (val, length):", dropped)
    print("Number of runs kept:", len(pieces))
    print("Number of dropped indices:", len(dropped_indices))
    print("Check: new_arr + dropped should equal original length:",
        new_arr.shape[0] + len(dropped_indices) == len(arr))

    new_arr = new_arr.reshape(-1, 1)

    return n_to_keep, new_arr, dropped_indices, window_start_indices
