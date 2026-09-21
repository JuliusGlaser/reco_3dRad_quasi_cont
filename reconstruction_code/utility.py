import h5py
import numpy as np
from scipy.io import savemat
import os
import nibabel as nib

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

def post_process_for_hernando(data, path_to_out, outname, TEs=None, B0=None, B0_strength=None, voxel_size=None):         
    
    if B0_strength is not None:
        gamma = 42.577478518e6  # gyromagnetic ratio in Hz/T
        CF = gamma * B0_strength  # Larmor frequency in Hz

    assert len(data.shape) == 3, "Input data must be 3D (n_TE, x, y)"
        
    data = data[:,np.newaxis, np.newaxis,...]#*1000
    data = data.transpose(-2,-1,1,2,0)
    data = data[::-1, ::-1,...]

    if B0_strength is None:
        savemat(path_to_out + outname + '.mat', {'finalImages_array': data})
    else:
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
