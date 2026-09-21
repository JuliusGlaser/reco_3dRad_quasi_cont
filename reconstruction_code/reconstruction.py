from enum import KEEP
import os
import numpy as np
import cupy as cp
import h5py

import twixtools
import pickle
import sigpy as sp
from sigpy.mri import app, cc, nlrecon, nlop
from pathlib import Path
from scipy.ndimage import convolve
from RadialSlidingRecon.utility import *
import argparse


from time import time

import yaml
from yaml import Loader
from scipy.io import loadmat
from RadialSlidingRecon.matlab_functions_in_python import *
import sys
from pprint import pprint



def getCoilSensitivityMaps(k, params, config, device, coil_filename):
    print("Estimating coil sensitivity maps from first projection 3D...")
    
    if config['coil_estimation_3d']:
        print(">> Using 3D coil estimation")
        images_for_csm = k[...].astype(np.complex64, copy=False)
        images_for_csm = sp.fft(images_for_csm, axes=(-1,))
        

        c = app.EspiritCalib(sp.to_device(images_for_csm[...], device=device),
                            calib_width=25,
                            thresh=0.05,
                            kernel_width=6,
                            crop=0,
                            max_iter=50,
                            sets=1,
                            device=device,
                            output_eigenvalue=False,
                            show_pbar=True).run()
        mps = sp.to_device(c)
        mps = np.array(mps).squeeze()

    else:
        images_for_csm = k[...]
        N_slice = images_for_csm.shape[-1]
        mps = []
        for s in range(N_slice):
            print('  ' + str(s).zfill(3))


            c = app.EspiritCalib(sp.to_device(images_for_csm[...,s], device=device),
                                calib_width=20,
                                thresh=0.1,
                                kernel_width=6,
                                crop=0.1,
                                max_iter=50,
                                sets=1,
                                device=device,
                                output_eigenvalue=False,
                                show_pbar=True).run()
            mps.append(sp.to_device(c))

        mps = np.array(mps).squeeze()
        mps = np.transpose(mps, (1,2,3,0)) 

    
    # mps = np.transpose(mps, (1,2,3,0)) 
    with h5py.File(os.path.join(params['path'],  coil_filename), 'w') as f:
        f.create_dataset('mps', data=mps)

    return mps

def preProcessData(raw_data_selected, density_compensation_vornoi, kvec, params, shift_ppm, A_preprocess, device):

    N_channels = params['acquisition']['channels']
    N_selectedProjections = params['reconstruction']['selectiveProjections'][1] - params['reconstruction']['selectiveProjections'][0]
    N_samples = params['acquisition']['radialSamples']

    data_shifted, params = getSpatialDataTransform(raw_data_selected, kvec, params, shift_ppm)

    data_filtered, _ = getFilter(data_shifted, kvec, params)

    if params['reconstruction']['kvecMethod'] == 2:
        data_shifted_dc = data_shifted * density_compensation_vornoi
        data_filtered = data_filtered * density_compensation_vornoi
    else:
        data_shifted_dc, _ = getDensityCompensation(data_shifted, kvec)
        data_filtered, _ = getDensityCompensation(data_filtered, kvec)
        

    k = sp.to_device(data_shifted_dc.squeeze(),device=device)
    k_filt = sp.to_device(data_filtered.squeeze(),device=device)

    reco = A_preprocess.H * k

    reco_filt = A_preprocess.H * k_filt

    return data_shifted.squeeze(), reco, reco_filt 

def combine_coils(coil_combine_method, images):
        
    # which channel combination method? Sum-of-squares is common:
    if coil_combine_method == 1:
        raise NotImplementedError("Didn't know what to do here, implement this if you want to use it")
    
    elif coil_combine_method == 2:
        print(images.shape)
        finalImages = sp.util.rss(images, axes=0, keepdims=False)
    elif coil_combine_method == 3:

        finalImages = getCoilCombine(sp.to_device(images))

    return finalImages

def getMask(k, params, config, device, mask_filename):
    if Path(os.path.join(params['path'],  mask_filename)).exists():
        with h5py.File(os.path.join(params['path'],  mask_filename), 'r') as f:
            mask = f['mask'][:]
            print(">> Loaded mask from file for iterative reconstruction")
        return mask
    else:
        print(">> Creating mask")
        images_for_csm = k[...].astype(np.complex64, copy=False)
        images_for_csm = sp.fft(images_for_csm, axes=(-1,))
        
        _, e = app.EspiritCalib(sp.to_device(images_for_csm[...], device=device),
                            calib_width=25,
                            thresh=0.05,
                            kernel_width=6,
                            crop=0,
                            max_iter=50,
                            sets=2,
                            device=device,
                            output_eigenvalue=True,
                            show_pbar=False).run()
        e = sp.to_device(e)
        e = np.array(e).squeeze()
        mask = e[0,...] > config['iter_reco_parameter']['mask_threshold']
        print(mask.shape)
        with h5py.File(os.path.join(params['path'],  mask_filename), 'w') as f:
            f.create_dataset('mask', data=mask)
            f.create_dataset('eigenvalues', data=e)

        return mask

def getBoundingBox(mask, config) -> tuple[int, int, int, int]:
    """
    Compute the bounding box of a binary mask.

    Args:
        mask: 3D NumPy array (x,y,z), non-zero pixels = object

    Returns:
        (x_min, y_min, x_max, y_max) in pixel coordinates
    """
    print("DEPRECATED: REMOVE")
    return
    if config['slices_around_center'] > 0:
        z_effective = (config['slices_around_center']+2)*2
    else:
        z_effective = mask.shape[-1]

    z_begin = mask.shape[-1]//2

    bounding_box_params = np.zeros((4, 1), dtype=np.int32)

    mask_i = mask[..., z_begin]
    # Find rows and columns where mask is non-zero
    rows = np.any(mask_i, axis=-2)
    cols = np.any(mask_i, axis=-1)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    bounding_box_params[:, 0] = [x_min, y_min, x_max, y_max]

    usual_patches_to_calc = mask.shape[-1]**2
    redu_patches_to_calc = 0

    redu_patches_to_calc += (bounding_box_params[2, 0] - bounding_box_params[0, 0]) * (bounding_box_params[3, 0] - bounding_box_params[1, 0])

    print(f">>> using bounding boxes reduces N_patches of LLR from {usual_patches_to_calc} to {redu_patches_to_calc} = {redu_patches_to_calc/usual_patches_to_calc*100:.2f}% of the original patches")
    return bounding_box_params

def fmt_list(lst):
    return "".join(str(x) for x in lst)

def get_subspace(sig, num_coeffs=5, error_bound=1e-5, prior_err=True):
    """Compute linear subspace coefficients of MR signal.

    Args:
        sig (array): MR signal
        num_coeffs (int): expected number of coefficients
        error_bound (float): relative error bound between the subspace signal
                            and the ground-truth signal
        prior_err (boolean): iteratively increase the number of coefficients
                            in order to reach the error_bound

    Returns:
        U_sub (array): truncated subspace matrix

    References:
        Huang C., Graff C. G., Clarkson E. W.,
        Bilgin A., Altbach M. I. (2012).
        T2 mapping from highly undersampled data by
        reconstruction of principal component coefficient maps
        using compressed sensing.
        Magn. Reson. Med., 67, 1355-1366.

        Tamir J. I., Uecker M., Chen W., Lai P., Alley M. T.,
        Vasanawala S. S., Lustig M. (2017).
        T2 shuffling: Sharp, multicontrast, volumetric fast spin-echo imaging.
        Magn. Reson. Med., 77, 180-195.

    Author:
        Zhengguo Tan <zhengguo.tan@gmail.com>
    """
    def get_rel_error(recov_sig, full_sig):
        return np.linalg.norm(full_sig - recov_sig).item() \
            / np.linalg.norm(full_sig).item()

    # reshape sig to (number of echoes, number of dictionary atoms)
    sig2 = np.reshape(sig, (sig.shape[0], -1))

    # singular value decomposition
    U, S, VH = np.linalg.svd(sig2, full_matrices=False)

    while True:
        # truncate the U matrix
        U_sub = U[:, :num_coeffs]

        # recover from U_sub
        recov_sig = U_sub @ U_sub.T @ sig2

        err = get_rel_error(recov_sig, sig2)

        if (err > error_bound) and (prior_err):
            num_coeffs += 1
        else:
            print('\nEventual number of subspace coefficients: '
                  + str(num_coeffs))
            print('Eventual error: ' + str(err))
            return U_sub

def calc_fat_modu_k_space(gradient_times,  # second
                            GYRO: float = 42.57747892E6,  # Hz/T
                            B0: float = 0.55  # T
                            ):
    """
    calculate fat modulation

    Input:
        gradient_times - gradient times in second
        GYRO - Gyromagnetic ratio
        B0 - Magnetic field strength
    """
    device = sp.backend.get_device(gradient_times)
    xp = device.xp

    with device:

        # TE = TE.ravel()

        ppm = xp.array([-3.6, -3.2, -2.91, -2.47, -2.25, -1.73, +0.81]) * 1E-6
        # amp = xp.array([0.087, 0.693, 0.128, 0.004, 0.039, 0.048])
        amp = xp.array([0.0813, 0.6006, 0.0678, 0.0914, 0.0635, 0.0152, 0.0802])

        res = xp.zeros(gradient_times.shape, dtype=complex)

        for n in range(gradient_times.shape[0]):

            for p in range(len(ppm)):

                res[n, ...] += amp[p] * xp.exp(2j * np.pi * GYRO * B0 * ppm[p] * gradient_times[n])


    return res



def smooth_fieldmap_3d(fieldmap_init, smoothfmiter):
    mask = np.array([[0, 1, 0],
                      [1, 4, 1],
                      [0, 1, 0]], dtype=float) / 8.0

    out = fieldmap_init.copy()
    for _ in range(smoothfmiter):
        for z in range(out.shape[2]):
            out[:, :, z] = convolve(out[:, :, z], mask, mode='constant', cval=0.0)

    return out

def RadialImageReconstruction_SlidingWindow(config, raw_data, params):

    try:
        device = sp.Device(0)               # 0  for gpu, -1 for cpu
        print(">> Using GPU for reconstruction")
    except:
        print(">> No GPU found, using CPU instead.")
        device = sp.Device(-1)

    # mask = np.zeros
    xp = device.xp

    config_str = yaml.safe_dump(config)

    iter_reco_config = config['iter_reco_parameter']

    ppm_shift_str = ""
    for i in config['shift']:
        ppm_shift_str += '_' + str(i)

    nShifts = len(params['offresonant']['shift'])

    steps, meanTimes, T_rounded = get_quasi_continuous_params(params)
    
    if config['steps'] > 0:
        steps = config['steps']

    N_channels = params['acquisition']['channels']

    if config['coil_compression']:
        N_channels_cc = config['N_virtual_coils']
        raw_data, evecs = cc.scc(raw_data[:,:], P=N_channels_cc, coil_dim=0)
        print(f"> Compressed Channels from {N_channels} to {N_channels_cc}")
        params['acquisition']['channels'] = N_channels_cc
        N_channels = N_channels_cc

    if config['coil_estimation_3d']:
        coil_estimation_str = '3d'
    else:
        coil_estimation_str = '2d'
        
    N_selectedProjections = config['slidingwindowpros']
    N_window_step = config['slidingwindowstep']
    N_samples = params['acquisition']['radialSamples']
    us_factor = config['undersampling_factor']
    slidingwindowpros = config['slidingwindowpros']

    TE_rastered = np.zeros((meanTimes.shape[0], N_selectedProjections, N_samples), dtype=np.float32)
    curr_projection = 0
    # print(params['acquisition']['echospacing'], params['acquisition']['projections'], params['acquisition']['echos'], params['acquisition']['TE'][0])
    for te in range(meanTimes.shape[0]):
        TE_rastered[te, :, :] = getTEArray_Rastered_pyth(params['acquisition']['echospacing'], params['acquisition']['projections'], params['acquisition']['echos'], params['acquisition']['TE'][0], [curr_projection, curr_projection+N_selectedProjections], N_samples, params['acquisition']['dwelltime'][0])
        curr_projection += N_window_step

    avg_TE_rastered = np.mean(TE_rastered, axis=(1))
    avg_TE_rastered = avg_TE_rastered[:,np.newaxis, :]
    print("T_rounded:", T_rounded.shape)
    print("TE_rastered:", TE_rastered.shape)
    print("avg_TE_rastered:", avg_TE_rastered.shape)
    # np.savetxt('TE_rastered.txt', TE_rastered[0])
    # np.savetxt('meanTimes.txt', meanTimes)
    # np.savetxt('T_rounded.txt', T_rounded)
    # np.savetxt('avg_TE_rastered.npy', avg_TE_rastered[:,0,:])
    FOV = config['recon_parameter']['FOV']
    N_slices = FOV
    fixed_output_slice = config['Fixed_output_slice']
    if fixed_output_slice >= N_slices:
        fixed_output_slice = -1
        print(f"Warning: Fixed_output_slice ({fixed_output_slice}) is larger than the number of slices ({N_slices}). Setting fixed_output_slice to -1 (no fixed slice).")
    
    N_slices_around_center_to_save = config['slices_around_center_to_save']

    us_factor = config['undersampling_factor']
    mask_us = np.zeros(tuple([1] + [N_selectedProjections] + [N_samples]), dtype=np.complex64)
    mask_us[:,::us_factor,:] = 1
    mask_us = sp.to_device(mask_us, device=device) 

    file_name = config['file']

    base_filename = f'{file_name}_win_{slidingwindowpros}_steps_{steps}_us_{us_factor}_C_{N_channels}_FOV_{FOV}_ppm{ppm_shift_str}'

    # if mps where already computed for this dataset -> load them, if not compute them later (unless data is single channel)
    mps = None
    mask = None
    bounding_box = None

    if config['coil_compression']:
        coil_filename = 'mps_virtual_' + str(N_channels) + f'_projections_{slidingwindowpros}_usfactor_{us_factor}_FOV_' +str(params['reconstruction']['size']['fov'][0])+ '_' + coil_estimation_str + f'.h5'
    else:
        coil_filename = 'mps_'+f'_projections_{slidingwindowpros}_usfactor_{us_factor}_FOV_' +str(params['reconstruction']['size']['fov'][0])+ '_' + coil_estimation_str + f'.h5'
    if Path(os.path.join(params['path'],  coil_filename)).exists():
        with h5py.File(os.path.join(params['path'],  coil_filename), 'r') as f:
            mps = f['mps'][:]
            mps = mps[:N_channels,...]
            mps_conj = mps[...].conj()
            coilcombine_den = np.linalg.norm(mps, axis=0)
    elif N_channels == 1:
        mps = np.ones(tuple([1] + params["reconstruction"]["sizeFilling"]), dtype=np.complex64)
        mps_conj = mps[...].conj()
        coilcombine_den = np.linalg.norm(mps, axis=0)

    if config['iterativeReco']:
        reg_type = iter_reco_config['reg']
        #Regu params
        if reg_type == 'LLR':
            blk_size = iter_reco_config['blk_size']
            blk_stride = iter_reco_config['blk_stride']
            lamda = iter_reco_config['lambda']
            regu = 'LLR_3dRad'
            randshift = iter_reco_config['randshift']
            regu_kspace = False
            regu_axes = [-1,-2]
            blk_shape = tuple(blk_size)
            blk_strides = tuple(blk_stride)
            normalization = False
            thresh = 'soft'
            deep_model = None
            scale = 1
            rho = iter_reco_config['rho']
            con_step = iter_reco_config['con_step']
            con_stride = iter_reco_config['con_stride']
            use_con_mask = iter_reco_config['use_con_mask']
            if con_step == 0:
                con_step = None
                con_stride = None
            
        else:
            lamda = iter_reco_config['lambda']
            regu = reg_type
            regu_kspace = False
            regu_axes = [-2,-3]
            blk_shape = None
            blk_strides = None
            normalization = False
            thresh = "soft"
            deep_model = None
            scale = 1
            rho = iter_reco_config['rho']

        recon_type = config['recon_type']
        max_iter = iter_reco_config['max_iter_ADMM']
        max_iter_cg = iter_reco_config['max_iter_CG']
        solver = iter_reco_config['solver']
        gpu_exclusively = iter_reco_config['gpu_exclusive_reco']

        N_slices_around_center = iter_reco_config['slices_around_center']
        

        show_pbar = True
        if solver == 'ADMM':
            show_pbar = False
        if solver == 'ConjugateGradient':
            proxg = None

        iters_to_save_step = iter_reco_config['iters_to_save_step']

        iterations_to_save = list(range(iters_to_save_step, max_iter + 1, iters_to_save_step))

        if config['custom_out_name'] == False:
            if reg_type == 'LLR':
                if config['iter_reco_parameter']['use_con_mask']:
                    full_filename = os.path.join(params['path'], f'reco_iter_{base_filename}_{reg_type}_blk_{fmt_list(blk_size)}_{fmt_list(blk_stride)}_it_{max_iter}_CG_{max_iter_cg}_l_{lamda}_r_{rho}_winsel_mask_s_{N_slices_around_center}.h5')
                else:
                    full_filename = os.path.join(params['path'], f'reco_iter_{base_filename}_recon_type_{recon_type}_{reg_type}_blk_{fmt_list(blk_size)}_{fmt_list(blk_stride)}_it_{max_iter}_CG_{max_iter_cg}_l_{lamda}_r_{rho}_winsel_{con_step}_{con_stride}_s_{N_slices_around_center}.h5')
            else:
                full_filename = os.path.join(params['path'], f'reco_iter_{base_filename}_recon_type_{recon_type}_{reg_type}_it_{max_iter}_CG_{max_iter_cg}_l_{lamda}_r_{rho}.h5')
        else:
            custom_filename = config['custom_out_name']
            full_filename = os.path.join(params['path'], f'reco_{custom_filename}.h5')

        filename_init_sol = os.path.join(params['path'], f'reconstructed_iterative_TIK_{base_filename}.h5')
        filename_init_sol_complete = os.path.join(params['path'], f'reconstructed_iterative_TIK_{base_filename}_complete.h5')


    full_filename_inis_complete = os.path.join(params['path'], f'initially_reconstructed_{base_filename}_complete.h5')
    full_filename_inis = os.path.join(params['path'], f'initially_reconstructed_{base_filename}.h5')

    initial_x_all = np.zeros(tuple([steps, nShifts] + params["reconstruction"]["sizeFilling"]), dtype=np.complex64)
    initial_x_all_filtered = np.zeros(tuple([steps, nShifts] + params["reconstruction"]["sizeFilling"]), dtype=np.complex64)


    if config['iterativeReco'] or config['save_complete']:
        k_all = np.zeros(tuple([steps, nShifts, N_channels] + [N_selectedProjections] + [N_samples]), dtype=np.complex64)
        coords_all = np.zeros(tuple([steps] + [N_selectedProjections] + [N_samples] + [3]), dtype=np.float32)

    if Path(os.path.join(params['path'],  full_filename_inis_complete)).exists() and config['iterativeReco']:
        with h5py.File(os.path.join(params['path'],  full_filename_inis_complete), 'r') as f:
            initial_x_all = f['initial_img'][:]
            initial_x_all_filtered = f['initial_img_filtered'][:]
            coords_all = f['coords_all'][:]
            k_all = f['k_all'][:]
    else:
        for i in range(steps):

            projection_start = i* params['slidingwindowstep']
            projection_end = projection_start + params['slidingwindowpros']
            params['reconstruction']['selectiveProjections'] = [projection_start, projection_end]
            print(f">>> Processing step {i+1}/{steps} with projections {projection_start} to {projection_end}") 

            azimutalAngle, polarAngle = getProjectionAngles(params, 1)

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

            kvec_scaled = kvec_theo / np.max(np.abs(kvec_theo))

            Nd = np.array(params["reconstruction"]["sizeFilling"], dtype=np.float32)  # (Nx,Ny,Nz)
            coord = kvec_scaled * (Nd[:, None]) / 2

            # coord = np.transpose(coord)  # shape (3,M)
            coord = coord.reshape((3, N_samples, N_selectedProjections), order="F")
            coord = coord.T

            N_selectedProjections = params['reconstruction']['selectiveProjections'][1] - params['reconstruction']['selectiveProjections'][0]

            A_preprocess = sp.linop.Diag(
                    [
                        sp.linop.Multiply(tuple([1] + [N_selectedProjections] + [N_samples]), mask_us) *
                        sp.linop.NUFFT(ishape=tuple([1] + params["reconstruction"]["sizeFilling"]), coord=coord, oversamp=params['reconstruction']['gridOS'], width=2, toeplitz=False)
                        

                        for c in range(N_channels)
                    ], 
                    iaxis=0,
                    oaxis=0
                )

            raw_data_selected = raw_data[:, params['reconstruction']['selectiveProjections'][0]:params['reconstruction']['selectiveProjections'][1],:]
            raw_data_selected = raw_data_selected * sp.to_device(mask_us, device=sp.get_device(raw_data_selected))

            if params['reconstruction']['kvecMethod'] == 2:
                # Load precomputed Voronoi density compensation from file
                with h5py.File(os.path.join(config['path'], f'densityCompensation_projections_{N_selectedProjections}_N_windows_{steps}.h5'), 'r') as f:
                    n_TE = params['reconstruction']['selectiveProjections'][0]//params['slidingwindowstep']
                    density_compensation_vornoi = f['densityCompensationVoronoi'][n_TE,...]
                    density_compensation_vornoi = density_compensation_vornoi[np.newaxis, np.newaxis, :]
                    # density_compensation_vornoi = np.ones_like(density_compensation_vornoi)
            else:
                density_compensation_vornoi = None


            k_shifts = np.zeros(([nShifts] + [N_channels] + [N_selectedProjections] + [N_samples]), dtype=np.complex64)
            initial_x_shifts_comb = np.zeros(([nShifts] + params['reconstruction']['sizeFilling']), dtype=np.complex64)
            initial_x_filtered_comb = np.zeros(tuple([nShifts] + params["reconstruction"]["sizeFilling"]), dtype=np.complex64)

            for i_shift in range(nShifts):
                print(">> Calculating initial solution")
                k, initial_x, initial_x_filt = preProcessData(raw_data_selected, density_compensation_vornoi, kvec_theo, params, config['shift'][i_shift], A_preprocess, device)
                initial_x = sp.to_device(initial_x)
                initial_x_filt = sp.to_device(initial_x_filt)
                print(">> Initial solution calculated")
                k_shifts[i_shift,...] = sp.to_device(k)

                if config['iterativeReco'] or config['recon_parameter']['multichannel_combination'] == 4:
                    if mps is None:
                        # Estimate CSM from first few projections
                        print(">> Calculating CSM")
                        k_for_csm = sp.fft(initial_x_filt, axes=(-3,-2))
                        mps = getCoilSensitivityMaps(k_for_csm, params, config, device, coil_filename)
                        mps_conj = mps[...].conj()
                        coilcombine_den = np.linalg.norm(mps, axis=0)

                    initial_x_comb = np.divide(np.sum(mps_conj * initial_x, axis=0), 
                                                coilcombine_den, where=(coilcombine_den != 0))
                    initial_x_filt_comb = np.divide(np.sum(mps_conj * initial_x_filt, axis=0), 
                                                coilcombine_den, where=(coilcombine_den != 0))
                else:
                    coil_combine_method = config['recon_parameter']['multichannel_combination'] 
                    initial_x_comb = combine_coils(coil_combine_method, initial_x)
                    initial_x_filt_comb = combine_coils(coil_combine_method, initial_x_filt)

                initial_x_shifts_comb[i_shift,...] = initial_x_comb
                initial_x_filtered_comb[i_shift,...] = initial_x_filt_comb
                    
            initial_x_all[i,...] = initial_x_shifts_comb
            initial_x_all_filtered[i,...] = initial_x_filtered_comb

            if config['iterativeReco'] or config['save_complete']:
                coords_all[i,...] = coord
                k_all[i,...] = k_shifts

        sub_basis = None #TODO: define subspace basis if needed for iterative reconstruction
            # to reduce size of the saved file, we only save certain slices
        if config['save_h5']:
            if config['save_complete']:
                with h5py.File(full_filename_inis_complete, 'w') as f:
                    f.create_dataset('initial_img', data=sp.to_device(initial_x_all))
                    f.create_dataset('initial_img_filtered', data=sp.to_device(initial_x_all_filtered))
                    f.create_dataset('k_all', data=sp.to_device(k_all))
                    f.create_dataset('coords_all', data=sp.to_device(coords_all))
                    f.create_dataset("config", data=config_str)
            else:
                with h5py.File(full_filename_inis, 'w') as f:
                    f.create_dataset('initial_img', data=sp.to_device(initial_x_all[...,N_slices//2-N_slices_around_center_to_save:N_slices//2+N_slices_around_center_to_save]))
                    f.create_dataset('initial_img_filtered', data=sp.to_device(initial_x_all_filtered[...,N_slices//2-N_slices_around_center_to_save:N_slices//2+N_slices_around_center_to_save]))
                    f.create_dataset("config", data=config_str)
        
        if config['save_nii']:
            print("> Saving initial solution as NIfTI files...")
            for i in range(nShifts):
                out_name = f'initially_reconstructed_{base_filename}'
                np_to_nii(initial_x_all[:,i,...], affine=None, out_name=out_name, file_path=config['path'], config_str=config_str)

                out_name = f'initially_reconstructed_filtered_{base_filename}'
                np_to_nii(initial_x_all_filtered[:,i,...], affine=None, out_name=out_name, file_path=config['path'], config_str=config_str)

        if config['save_mat']:
            if nShifts > 1:
                raise NotImplementedError("Didn't implement saving multiple shifts in .mat file, implement this if you want to use it")
            else:
                print('> WARNING: currently this is only outputting the center axial slice')
                print("> WARNING: B0 matrix not defined yet")
                post_process_for_hernando(initial_x_all_filtered[...,N_slices//2].squeeze(),
                                           params['path'],
                                           f'initially_reconstructed_filtered_{base_filename}',
                                           meanTimes,
                                           0,
                                           params['scanner']['fieldstrength'],
                                           reso)

    print(initial_x_all.shape)
    if config['iterativeReco']:
        # print(f"> Starting reconstruction for shift {i} / {nShifts}")

        mps = sp.to_device(mps, device=device)
        coords_all = sp.to_device(coords_all, device=device)
        
        # Define forward operator A for iterative reconstruction
        # As everey TE has its own set of spokes, we need to define a diagonal operator
        # For better performance, we perform coil splitted reconstruction
        sub_basis = None
        if sub_basis is not None:
            img_shape = params["reconstruction"]["sizeFilling"]
            Ncontrast, Ncoef = sub_basis.shape

            ishape = [Ncoef] + [1] + img_shape

            sub_ishape = [Ncoef] + [np.prod(ishape[1:])]
            sub_oshape = [Ncontrast] +  [1] + img_shape

            B1 = sp.linop.Reshape(sub_ishape, ishape)
            B2 = sp.linop.MatMul(B1.oshape, sub_basis)
            B3 = sp.linop.Reshape(sub_oshape, B2.oshape)

            B = B3 * B2 * B1
        else:

            ishape = [steps] + [1] + params["reconstruction"]["sizeFilling"]

            B = sp.linop.Identity(ishape)

    
        if gpu_exclusively:
            PI_op = sp.linop.Diag(
                        [
                            sp.linop.Vstack(
                                [   
                                    sp.linop.Multiply(tuple([1] + [1] + [slidingwindowpros] + [N_samples]), mask_us[xp.newaxis, ...]) *
                                    sp.linop.NUFFT(ishape=tuple([1] + [1] + params["reconstruction"]["sizeFilling"]), coord=coords_all[TE,...], oversamp=params['reconstruction']['gridOS'], width=4, toeplitz=False) *
                                    # sp.linop.Transpose(tuple([steps] + [1] + params["reconstruction"]["sizeFilling"]), axes=(0,1,4,3,2)) * 
                                    sp.linop.Multiply(tuple([1] + [1] + params["reconstruction"]["sizeFilling"]), mps[c,...])
                                    for c in range(N_channels)
                                ], 
                                axis=1,
                                # bypass_device = True 
                            ) for TE in range(steps)
                        ], 
                        iaxis=0,
                        oaxis=0    
                    ) 
            A_one_Device = PI_op * B
        else:
            PI_op = sp.linop.Diag(
                        [
                            sp.linop.Vstack(
                            [   
                                sp.linop.ToDevice(shape=tuple([1] + [1] + [slidingwindowpros] + [N_samples]), odevice=sp.Device(-1), idevice=sp.Device(0)) * 
                                sp.linop.Multiply(tuple([1] + [1] + [slidingwindowpros] + [N_samples]), mask_us[xp.newaxis, ...]) *
                                sp.linop.NUFFT(ishape=tuple([1] + [1] + params["reconstruction"]["sizeFilling"]), coord=coords_all[TE,...], oversamp=params['reconstruction']['gridOS'], width=4, toeplitz=False) *
                                sp.linop.Multiply(tuple([1] + [1] + params["reconstruction"]["sizeFilling"]), mps[c,...]) *
                                sp.linop.ToDevice(shape=tuple([1] + [1] + params["reconstruction"]["sizeFilling"]), odevice=device, idevice=sp.Device(-1))
                                for c in range(N_channels)
                            ], 
                            axis=1,
                            # bypass_device = True 
                            ) for TE in range(steps)
                        ], 
                        iaxis=0,
                        oaxis=0    
                    )
            
            A_one_Device = sp.linop.Diag(
                        [
                            sp.linop.Vstack(
                            [   
                                sp.linop.Multiply(tuple([1] + [1] + [slidingwindowpros] + [N_samples]), mask_us[xp.newaxis, ...]) *
                                sp.linop.NUFFT(ishape=tuple([1] + [1] + params["reconstruction"]["sizeFilling"]), coord=coords_all[TE,...], oversamp=params['reconstruction']['gridOS'], width=4, toeplitz=False) *
                                sp.linop.Multiply(tuple([1] + [1] + params["reconstruction"]["sizeFilling"]), mps[c,...])
                                for c in range(N_channels)
                            ], 
                            axis=1,
                            # bypass_device = True 
                            ) for TE in range(steps)
                        ], 
                        iaxis=0,
                        oaxis=0    
                    ) 

        with h5py.File(full_filename, 'w') as f:
            f.create_dataset("config", data=config_str)
            # f.create_dataset("sub_basis", data=sub_basis)
            
        
        print(f'>> Starting iterative reconstruction with solver {solver}...')

        k = sp.to_device(k_all.squeeze(), device=device)
        # k = k * sp.to_device(mask_us, device=device)
        k_device = device   

        A_for_scale = A_one_Device

        if scale == 0:
            print(">> Calculating scale for data consistency term...")
            t = time()
            xp = k_device.xp
            with k_device:
                x0 = A_for_scale.H(k)
                x1 = np.linalg.norm(x0, axis=0)
                x2 = np.linalg.norm(x1, axis=0)
                x1 = xp.sort(x2.flatten())
                x_med = abs(x1[len(x1)//2])
                x_p90 = abs(x1[int(len(x1) * 0.9)])
                x_max = abs(x1[-1])
                if (x_max - x_p90) < 2 * (x_p90 - x_med):
                    scale = x_p90
                else:
                    scale = x_max
            print(f">> Scale calculated in {time() - t:.2f} seconds")

        if gpu_exclusively:
            k = sp.to_device(k_all.squeeze(), device=device)
            k = k * sp.to_device(mask_us, device=device)
            k_device = device
        else:
            k = sp.to_device(k_all.squeeze())
            k = k * sp.to_device(mask_us, device=sp.Device(-1))
            k_device = sp.Device(-1)

        # print('scale: ' + str(scale))
        scale = sp.to_device(scale, device=sp.get_device(k))
        
        if iter_reco_config['initial_sol_iterative']:
            
            if Path(os.path.join(params['path'],  filename_init_sol_complete)).exists():
                with h5py.File(os.path.join(params['path'],  filename_init_sol_complete), 'r') as f:
                    initial_x_all = f['final_images'][:]
                    print(">> Loaded initial solution from file for iterative reconstruction")
            else:
                print(">> Performing iterative reconstruction for each TE separately for initial solution")
                res = np.zeros_like(sp.to_device(initial_x_all[...]))
                # res = np.zeros(tuple([max_iter] + params["reconstruction"]["sizeFilling"]), dtype=np.complex44)
                
                for echo_t in range(steps):

                    print("Processing TE {}/{}...".format(echo_t+1, steps))

                    k_TE = sp.to_device(k[echo_t,...].squeeze(), device)
                    img_TE = sp.to_device(initial_x_all[echo_t,...], device)
                    coords_TE = coords_all[echo_t,...]

                    A_for_ini =  sp.linop.Vstack(
                                    [   
                                        sp.linop.Multiply(tuple([1] + [slidingwindowpros] + [N_samples]), mask_us[ ...]) *
                                        sp.linop.NUFFT(ishape=tuple([1] + params["reconstruction"]["sizeFilling"]), coord=coords_TE, oversamp=params['reconstruction']['gridOS'], width=2, toeplitz=False) *
                                        # sp.linop.Transpose(tuple([steps] + [1] + params["reconstruction"]["sizeFilling"]), axes=(0,1,4,3,2)) * 
                                        sp.linop.Multiply(tuple([1] + params["reconstruction"]["sizeFilling"]), mps[c,...])
                                        for c in range(N_channels)
                                    ], 
                                    axis=0,
                                    # bypass_device = True 
                                ) 
                    
                    trafos, proxg, g, strength = app._get_regularization(
                                                                        A_for_ini.ishape, regu="TIK", lamda=0.001,
                                                                        regu_kspace=regu_kspace, regu_axes=regu_axes,
                                                                        blk_shape=blk_shape, blk_strides=blk_strides,
                                                                        slices_around_center=N_slices_around_center,
                                                                        normalization=normalization,
                                                                        thresh=thresh, deep_model=deep_model)
                
                    it = 0

                    iterative_solver = sp.app.LinearLeastSquares(A=A_for_ini, y=k_TE, x=img_TE, lamda=strength,
                                            G=trafos, proxg=proxg, g=g, scale=scale, rho = 0.05,
                                            max_iter=50, max_cg_iter=max_iter_cg, solver="ConjugateGradient",
                                            show_pbar=False, verbose=True)
                    
                    reco_t = time()
                    while not iterative_solver.alg.done():
                        iterative_solver.alg.update()

                    print(f"Reconstruction time: {(time() - reco_t)//60:.0f}:{(time() - reco_t)%60:02.0f} min")

                    res[echo_t,...] = sp.to_device(iterative_solver._output())

                
                if config['save_all_slices'] or config['save_complete']:
                    with h5py.File(filename_init_sol_complete, 'w') as f:
                        f.create_dataset('final_images', data=sp.to_device(res, device=sp.Device(-1)))
                        f.create_dataset("config", data=config_str)
                else:
                    with h5py.File(filename_init_sol, 'w') as f:
                        f.create_dataset('final_images', data=res[...,N_slices//2-2-N_slices_around_center:N_slices//2+N_slices_around_center+2])
                        f.create_dataset("config", data=config_str)
                    

                initial_x_all = res
        
        k /= scale

        if recon_type == 'basic' and reg_type == 'LLR':

            A = PI_op * B
            print(A.ishape)
            trafos, proxg, g, strength = app._get_regularization(
                                                                A.ishape, regu=regu, lamda=lamda,
                                                                regu_kspace=regu_kspace, regu_axes=regu_axes,
                                                                blk_shape=blk_shape, blk_strides=blk_strides,
                                                                randshift = randshift,
                                                                slices_around_center=N_slices_around_center,
                                                                con_step=con_step, con_stride=con_stride,
                                                                use_con_mask=use_con_mask,
                                                                bounding_box = bounding_box,
                                                                normalization=normalization,
                                                                thresh=thresh, deep_model=deep_model)
            if gpu_exclusively:
                if iter_reco_config['initial_sol_filt']:
                    img = sp.to_device(initial_x_all_filtered, device=device)
                else:
                    img = sp.to_device(initial_x_all, device=device)
            else:
                if iter_reco_config['initial_sol_filt']:
                    img = sp.to_device(initial_x_all_filtered, device=sp.Device(-1))
                else:
                    img = sp.to_device(initial_x_all, device=sp.Device(-1))

            img = sp.to_device(B.H(initial_x_all), device=sp.Device(-1))
            print('>> Starting LLR reconstruction with ADMM solver...')
            t = time()
            iterative_solver = sp.app.LinearLeastSquares(A=A, y=k, x=img, lamda=strength,
                                        G=trafos, proxg=proxg, g=g, scale=scale,
                                        max_iter=max_iter, max_cg_iter=max_iter_cg, solver=solver, A_one_Device=A_one_Device,
                                        show_pbar=False, verbose=True)
            print(f">> LLR reconstruction initialized in {time() - t:.2f} seconds")
            while not iterative_solver.alg.done():
            
                iter_t = time()
                iterative_solver.alg.update()
                print(f"Iteration time ADMM: {(time() - iter_t)//60:.0f}:{(time() - iter_t)%60:02.0f} min")

                current_out = sp.to_device(iterative_solver._output(), device=sp.Device(-1))

                if iterative_solver.alg.iter in iterations_to_save and iterative_solver.alg.iter != max_iter:
                    with h5py.File(full_filename, 'r+') as f:
                        if config['save_all_slices']:
                            f.create_dataset(f'images_iter_{iterative_solver.alg.iter}', data=current_out)
                        elif fixed_output_slice >= 0:
                            f.create_dataset(f'images_iter_{iterative_solver.alg.iter}', data=current_out[...,fixed_output_slice])
                        else:
                            f.create_dataset(f'images_iter_{iterative_solver.alg.iter}', data=current_out[...,N_slices//2-N_slices_around_center_to_save:N_slices//2+N_slices_around_center_to_save])

            out_rot = iterative_solver._output()
            with h5py.File(full_filename, 'r+') as f:
                if config['save_all_slices']:
                    f.create_dataset('final_images', data=sp.to_device(out_rot, device=sp.Device(-1)))
                elif fixed_output_slice >= 0:
                    f.create_dataset(f'final_images', data=sp.to_device(out_rot[...,fixed_output_slice]))
                else:
                    f.create_dataset('final_images', data=sp.to_device(out_rot[...,N_slices//2-N_slices_around_center_to_save:N_slices//2+N_slices_around_center_to_save]))

        elif recon_type == 'basic':
            if fixed_output_slice >= 0:
                res = np.zeros_like(sp.to_device(initial_x_all[...,fixed_output_slice]))
            else:
                res = np.zeros_like(sp.to_device(initial_x_all[...,N_slices//2-N_slices_around_center_to_save:N_slices//2+N_slices_around_center_to_save]))
            # res = np.zeros(tuple([max_iter] + params["reconstruction"]["sizeFilling"]), dtype=np.complex64)
            print(res.shape)
            for echo_t in range(steps):

                print("Processing TE {}/{}...".format(echo_t+1, steps))

                k_TE = sp.to_device(k[[echo_t],...], device)
                if iter_reco_config['initial_sol_filt']:
                    img_TE = sp.to_device(initial_x_all_filtered[[echo_t],...], device)
                else:
                    img_TE = sp.to_device(initial_x_all[[echo_t],...], device)
                coords_TE = coords_all[[echo_t],...]

                A =  PI_op.linops[echo_t]

                trafos, proxg, g, strength = app._get_regularization(
                                                                    A.ishape, regu=regu, lamda=lamda,
                                                                    regu_kspace=regu_kspace, regu_axes=regu_axes,
                                                                    blk_shape=blk_shape, blk_strides=blk_strides,
                                                                    slices_around_center=N_slices_around_center,
                                                                    normalization=normalization,
                                                                    thresh=thresh, deep_model=deep_model)
                it = 0

                iterative_solver = sp.app.LinearLeastSquares(A=A, y=k_TE, x=img_TE, lamda=strength,
                                        G=trafos, proxg=proxg, g=g, scale=scale, rho = rho,
                                        max_iter=max_iter, max_cg_iter=max_iter_cg, solver=solver,
                                        show_pbar=False, verbose=True)
                while not iterative_solver.alg.done():
            
                    iter_t = time()
                    iterative_solver.alg.update()
                    print(f"Iteration time: {(time() - iter_t)//60:.0f}:{(time() - iter_t)%60:02.0f} min")

                    # res[it,...] = sp.to_device(iterative_solver._output())
                    it += 1

                if fixed_output_slice >= 0:
                    res[echo_t,...] = sp.to_device(iterative_solver._output()[...,fixed_output_slice])
                else:
                    res[echo_t,...] = sp.to_device(iterative_solver._output()[...,N_slices//2-N_slices_around_center_to_save:N_slices//2+N_slices_around_center_to_save])
            
            with h5py.File(full_filename, 'r+') as f:
                f.create_dataset('final_images', data=res)
        
        elif recon_type == 'k_s_sepa':

            k = k[:,np.newaxis,...]  # Add a new axis for the Fat dimension

            t_tau = calc_fat_modu_k_space(TE_rastered*1e-6, B0=7)  # convert to seconds
            D_t = np.ones((steps,2,1,1, N_samples), dtype=np.complex64)  # Placeholder for D(t), should be defined based on fat peaks and echo times
            D_t[:,1,...] = t_tau[:,np.newaxis,...]  # Assuming t_tau is the modulation for the Fat dimension
            
            # print("Sigma shape:", Sigma.ishape, "Sigma oshape:", Sigma.oshape)

            E = sp.linop.Diag(
                        [
                            sp.linop.Vstack(
                            [   
                                sp.linop.ToDevice(shape=tuple([1] + [2] + [1] + [slidingwindowpros] + [N_samples]), odevice=sp.Device(-1), idevice=sp.Device(0)) * 
                                # sp.linop.Sum(tuple([1] + [2] + [1] + [slidingwindowpros] + [N_samples]), axes=(1,), keepdims=True) *
                                # sp.linop.Multiply(tuple([1] + [2] + [1] + [slidingwindowpros] + [N_samples]), mult=D_t[TE,...]) *
                                sp.linop.Multiply(tuple([1] + [2] + [1] + [slidingwindowpros] + [N_samples]), mask_us[xp.newaxis, ...]) *
                                sp.linop.NUFFT(ishape=tuple([1] + [2] + [1] + params["reconstruction"]["sizeFilling"]), coord=coords_all[TE,...], oversamp=params['reconstruction']['gridOS'], width=4, toeplitz=False) *
                                sp.linop.Multiply(tuple([1] + [2] + [1] + params["reconstruction"]["sizeFilling"]), mps[c,...]) *
                                sp.linop.ToDevice(shape=tuple([1] + [2] + [1] + params["reconstruction"]["sizeFilling"]), odevice=device, idevice=sp.Device(-1))
                                for c in range(N_channels)
                            ], 
                            axis=2,
                            # bypass_device = True 
                            ) for TE in range(steps)
                        ], 
                        iaxis=0,
                        oaxis=0    
                    )
            
            M = sp.linop.Multiply(ishape=E.oshape, mult=D_t)
            # print("M shape:", M.ishape, "M oshape:", M.oshape)
            Sigma = sp.linop.Sum(ishape=M.oshape, axes=(1,), keepdims=True)
            
            A_one_Device = sp.linop.Diag(
                        [
                            sp.linop.Vstack(
                            [   
                                sp.linop.Sum(tuple([1] + [2] + [1] + [slidingwindowpros] + [N_samples]), axes=(1,), keepdims=True) *
                                sp.linop.Multiply(tuple([1] + [2] + [1] + [slidingwindowpros] + [N_samples]), mult=D_t[TE,...]) *
                                sp.linop.Multiply(tuple([1] + [2] + [1] + [slidingwindowpros] + [N_samples]), mask_us[xp.newaxis, ...]) *
                                sp.linop.NUFFT(ishape=tuple([1] + [2] + [1] + params["reconstruction"]["sizeFilling"]), coord=coords_all[TE,...], oversamp=params['reconstruction']['gridOS'], width=4, toeplitz=False) *
                                sp.linop.Multiply(tuple([1] + [2] + [1] + params["reconstruction"]["sizeFilling"]), mps[c,...])
                                for c in range(N_channels)
                            ], 
                            axis=2,
                            # bypass_device = True 
                            ) for TE in range(steps)
                        ], 
                        iaxis=0,
                        oaxis=0    
                    ) 
            
            
            A = Sigma * M * E
            trafos, proxg, g, strength = app._get_regularization(
                                                                A.ishape, regu='TIK', lamda=lamda,
                                                                regu_kspace=regu_kspace, regu_axes=regu_axes,
                                                                blk_shape=blk_shape, blk_strides=blk_strides,
                                                                randshift = False,
                                                                slices_around_center=N_slices_around_center,
                                                                normalization=normalization,
                                                                thresh=thresh, deep_model=deep_model)
            if gpu_exclusively:
                if iter_reco_config['initial_sol_filt']:
                    img = sp.to_device(initial_x_all_filtered, device=device)
                else:
                    img = sp.to_device(initial_x_all, device=device)
            else:
                if iter_reco_config['initial_sol_filt']:
                    img = sp.to_device(initial_x_all_filtered, device=sp.Device(-1))
                else:
                    img = sp.to_device(initial_x_all, device=sp.Device(-1))

            img = np.concatenate([img, img], axis=1)  # for water and fat images
            img = img[:,:,np.newaxis,...]  # Add a new axis for the Fat dimension
            img = sp.to_device(img, device=sp.Device(-1))
            k = sp.to_device(k, device=sp.Device(-1))
            print('>> Starting k-space separated reconstruction with ADMM solver...')
            t = time()
            iterative_solver = sp.app.LinearLeastSquares(A=A, y=k, x=img, lamda=strength,
                                        G=trafos, proxg=proxg, g=g, scale=scale,
                                        max_iter=max_iter, max_cg_iter=max_iter_cg, solver='ADMM',
                                        show_pbar=False, verbose=True)
            
            while not iterative_solver.alg.done():
            
                iter_t = time()
                iterative_solver.alg.update()
                print(f"Iteration time ADMM: {(time() - iter_t)//60:.0f}:{(time() - iter_t)%60:02.0f} min")

                current_out = sp.to_device(iterative_solver._output(), device=sp.Device(-1))

                if iterative_solver.alg.iter in iterations_to_save and iterative_solver.alg.iter != max_iter:
                    with h5py.File(full_filename, 'r+') as f:
                        if config['save_all_slices']:
                            f.create_dataset(f'images_iter_{iterative_solver.alg.iter}', data=current_out)
                        elif fixed_output_slice >= 0:
                            f.create_dataset(f'images_iter_{iterative_solver.alg.iter}', data=current_out[...,fixed_output_slice])
                        else:
                            f.create_dataset(f'images_iter_{iterative_solver.alg.iter}', data=current_out[...,N_slices//2-N_slices_around_center_to_save:N_slices//2+N_slices_around_center_to_save])

            out_rot = iterative_solver._output()
            with h5py.File(full_filename, 'r+') as f:
                if config['save_all_slices']:
                    f.create_dataset('final_images', data=sp.to_device(out_rot, device=sp.Device(-1)))
                elif fixed_output_slice >= 0:
                    f.create_dataset(f'final_images', data=sp.to_device(out_rot[...,fixed_output_slice]))
                else:
                    f.create_dataset('final_images', data=sp.to_device(out_rot[...,N_slices//2-N_slices_around_center_to_save:N_slices//2+N_slices_around_center_to_save]))

        elif recon_type == 'model_based':

            moba_par = config['moba_parameter']
            img = np.ones([1] + [1] + [1] + params["reconstruction"]["sizeFilling"], dtype=np.complex64) 
            img = np.concatenate([img, img, img], axis=0)  # for water and fat images
            # k = k.swapaxes(0,1)  # shape (N_channels, steps, N_selectedProjections, N_samples)
            scaling = moba_par['scaling']
            print(scaling)
            img = img *scaling
            initial_sol = img
            if moba_par['init_fieldmap']:
                with h5py.File(moba_par['init_file_path'], 'r') as f: #TODO: fix hardcode
                    field_map = np.real(f['field_map'][:])
                    # field_map = field_map[12:187, 12:187, 12:187]
                    fat_map = f['fat_map'][:]
                    wat_map = f['wat_map'][:]
                    
                if moba_par['init_fat_water']:
                    initial_sol[0,0, ...] = wat_map*scaling
                    initial_sol[1,0, ...] = fat_map *scaling
                initial_sol[2,0, 0, ...] = field_map*scaling #sp.fft(field_map, axes=(-3,-2,-1))

            # field_map = smooth_fieldmap_3d(field_map, 10)*1E-6

            

            initial_sol = sp.to_device(initial_sol)
            k = sp.to_device(k)
            k = k[:, np.newaxis, ...]  # add coil dimension
            mps = sp.to_device(mps)
            print(">> Starting nonlinear reconstruction with FWZ model...")
            
            iterative_solver = nlrecon.kinv(y=k, image_shape=list(initial_sol.shape), model='FWZ', coil=mps, coord=coords_all, 
                                            coil_shape=list(mps.shape), x=initial_sol, rvc=[False, False, True], echo_times=meanTimes[:, np.newaxis]*1E-3, gradient_times=TE_rastered*1e-6, 
                                            inner_iter=moba_par['max_iter_CG'], outer_iter=moba_par['max_iter_IRGNM'], alpha=moba_par['alpha'], redu=moba_par['redu'], inner_tol=moba_par['inner_tol'], fix_fieldmap=moba_par['fix_fieldmap'], trafo_MGRE=moba_par['trafo_MGRE'])
            
            iter = 0
            while not iterative_solver.done():
                iterative_solver.update()
                with h5py.File(full_filename, 'r+') as f:
                    if fixed_output_slice >= 0:
                        out_data = iterative_solver.x
                        f.create_dataset(f'images_iter_{iter}', data=out_data[...,fixed_output_slice])
                    else:
                        out_data = iterative_solver.x
                        f.create_dataset(f'images_iter_{iter}', data=out_data[...,N_slices//2-N_slices_around_center_to_save:N_slices//2+N_slices_around_center_to_save])
                iter += 1
            

            with h5py.File(full_filename, 'r+') as f:
                if fixed_output_slice >= 0:
                    out_data = iterative_solver.x

                    f.create_dataset('final_images', data=out_data[...,fixed_output_slice])
                else:
                    out_data = iterative_solver.x
                    f.create_dataset('final_images', data=out_data[...,N_slices//2-N_slices_around_center_to_save:N_slices//2+N_slices_around_center_to_save])
    
                
def paramsPreperation(config, mapped):
    params = createParams(config, mapped)
    params['slidingwindowpros'] = config['slidingwindowpros']
    params['slidingwindowstep'] = config['slidingwindowstep']
    params['offresonant']['shift'] = config['shift'] # [ppm]  
    params['reconstruction']['grad_delay'] = config['grad_delay'] # [µs]

    return params

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to reconstruction config")
    pre_args, _ = parser.parse_known_args()
    pre_args.config
    stream = open(pre_args.config, 'r') 
    print(pre_args.config)  
    # stream = open('config.yaml', 'r')                   #TODO: adapt config file for your purpose
    config = yaml.load(stream, Loader)
    stream.close()

    print(">> Configuration loaded:")
    pprint(config)
    sys.stdout.flush()
    path = config['path']
    file = config['file']

    full_path = os.path.join(path, file)

    print(f">> Loading data from {full_path}...")
    t = time()
    mapped = twixtools.map_twix(full_path)

    print(f"Preparing parameters...")
    params = paramsPreperation(config, mapped)

    if not os.path.exists(os.path.join(params['path'], params['filename'] + '_raw_data.h5')):
        t = time()
        print(f"Preprocessing data...")
        raw_data = PrepRaw(params, mapped)
        f = h5py.File(os.path.join(params['path'], params['filename'] + '_raw_data.h5'), 'w')
        f.create_dataset('raw_data', data=raw_data)
        f.close()
        print(f"Data loading and preprocessing completed in {(time() - t)//60:.0f}:{(time() - t)%60:.0f} min.")
    else:
        with h5py.File(os.path.join(params['path'], params['filename'] + '_raw_data.h5'), 'r') as f:
            raw_data = f['raw_data'][:]

    full_t = time()
    
    RadialImageReconstruction_SlidingWindow(config, raw_data, params)    

    print(f"Total reconstruction time: {(time() - full_t)//60:.0f}:{(time() - full_t)%60:02.0f} min")

if __name__ == "__main__":
    main()
