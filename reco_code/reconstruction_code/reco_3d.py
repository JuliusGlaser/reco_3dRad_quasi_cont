
import os
import numpy as np
import cupy as cp
import h5py

import twixtools
import pickle
import sigpy as sp
from sigpy.mri import app, cc, nlrecon, nlop
from sigpy import backend, alg
from pathlib import Path
from scipy.ndimage import convolve
from RadialSlidingRecon.utility import *
import argparse


from time import time

import yaml
from yaml import Loader
from scipy.io import loadmat
from RadialSlidingRecon.matlab_functions_in_python import *
from RadialSlidingRecon.reconstruction import *
from scipy.ndimage import convolve

import sys
from pprint import pprint

def summarize(a, name=""):
    a = sp.to_device(a, device=backend.cpu_device)
    a = np.asarray(a)
    print(f"--- {name} ---")
    print("shape:", a.shape)
    print("mean:", a.mean().real)
    print("std:", a.std(ddof=1))          # magnitude-based automatically for complex dtype
    idx_min = np.argmin(np.abs(a))
    idx_max = np.argmax(np.abs(a))
    print("min:", a.flat[idx_min].real)   # MATLAB picks by magnitude, prints real part
    print("max:", a.flat[idx_max].real)
    print("sum:", a.sum().real)
    print("norm:", np.linalg.norm(a))
    print()

def smooth_fieldmap(fieldmap_init, smoothfmiter):
    mask = np.array([[0, 1, 0],
                      [1, 4, 1],
                      [0, 1, 0]], dtype=float) / 8.0

    out = fieldmap_init.copy()
    for _ in range(smoothfmiter):
        out[:, :,] = convolve(out[:, :,], mask, mode='constant', cval=0.0)

    return out

def smooth_fieldmap_3d(fieldmap_init, smoothfmiter):
    mask = np.array([[0, 1, 0],
                      [1, 4, 1],
                      [0, 1, 0]], dtype=float) / 8.0

    out = fieldmap_init.copy()
    for _ in range(smoothfmiter):
        for z in range(out.shape[2]):
            out[:, :, z] = convolve(out[:, :, z], mask, mode='constant', cval=0.0)

    return out

def main():

    # Load config
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

    fixed_output_slice = config['Fixed_output_slice']

    us_fac_spokes = config['undersampling_factor']
    us_fac_TE = config['TE_us']

    if config['calc_equi_TE_windows']:
        
        
        path = config['path']
        file = config['file']
        full_path = os.path.join(path, file)
        
        print(f">> Loading data from {full_path}...")
        t = time()
        mapped = twixtools.map_twix(full_path)
        print("len mapped", len(mapped))
    
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

        N_channels = params['acquisition']['channels']

        if config['coil_compression']:
            N_channels_cc = config['N_virtual_coils']
            raw_data, evecs = cc.scc(raw_data[:,:], P=N_channels_cc, coil_dim=0)
            print(f"> Compressed Channels from {N_channels} to {N_channels_cc}")
            params['acquisition']['channels'] = N_channels_cc
            N_channels = N_channels_cc
            
        N_selectedProjections = config['slidingwindowpros']
        N_window_step = config['slidingwindowstep']
        N_samples = params['acquisition']['radialSamples']

        steps, meanTimes, T_rounded = get_quasi_continuous_params(params)
        

        TE_rastered = np.zeros((meanTimes.shape[0], N_selectedProjections, N_samples), dtype=np.float32)
        curr_projection = 0
        # print(params['acquisition']['echospacing'], params['acquisition']['projections'], params['acquisition']['echos'], params['acquisition']['TE'][0])
        for te in range(meanTimes.shape[0]):
            TE_rastered[te, :, :] = getTEArray_Rastered_pyth(params['acquisition']['echospacing'], params['acquisition']['projections'], params['acquisition']['echos'], params['acquisition']['TE'][0], [curr_projection, curr_projection+N_selectedProjections], N_samples, params['acquisition']['dwelltime'][0])
            curr_projection += N_window_step
    
        TE_rastered = TE_rastered.reshape((TE_rastered.shape[0]*TE_rastered.shape[1], -1))
    
        k_windowed, kvec_theos, TE_rastered_new, window_start_indices = convert_raw_into_TE_windows(raw_data, TE_rastered, params, config)
        print("k_windowed.shape: ", k_windowed.shape)
        print("kvec_theos.shape: ", kvec_theos.shape)
        print("kvec_theos.shape: ", kvec_theos.transpose(1,0,2).reshape(3,-1).shape)
        print("window_start_indices.shape: ", window_start_indices.shape)

        k_windowed_shifted = np.zeros_like(k_windowed)
        for win in range(k_windowed.shape[1]):
            data_shifted, params = getSpatialDataTransform(k_windowed[:,win], kvec_theos[win], params, 0)
            k_windowed_shifted[:,win] = data_shifted

        N_selectedProjections = k_windowed.shape[2]
        
        coords = np.zeros((k_windowed.shape[1], N_selectedProjections, N_samples, 3), dtype=np.float32)
        print("coords.shape: ", coords.shape)

        for i in range(k_windowed.shape[1]):
            projection_start = window_start_indices[i]
            projection_end = projection_start + N_selectedProjections
            params['reconstruction']['selectiveProjections'] = [projection_start, projection_end]
            # print(f">>> projections {projection_start} to {projection_end}")

            azimutalAngle, polarAngle = getProjectionAngles(params["acquisition"]["goldenAngle"], proj_start=projection_start, proj_end=projection_end)

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

            coords[i,...] = coord

        
        kdata = np.swapaxes(k_windowed_shifted, 0,1)
        kdata = kdata[:,np.newaxis,...]

        print("TE_rastered.shape: ", TE_rastered.shape)
        TE_rastered = TE_rastered_new*1E-6
        # print("TE_rastered.shape: ", TE_rastered.shape)
        # print(TE_rastered[0])

        mps_filename = config['moba_parameter']['mps_name']
        with h5py.File(path + os.sep + mps_filename, 'r') as f:
            mps = f['mps'][:]
    else:
        path = config['path']
        kdat_filename = 'initially_reconstructed_meas_MID00753_FID01064_MR_DA_3D_RAD_80k_1000mum.dat_equi_TE_win_comb_1_win_10000_steps_255_us_4_C_17_FOV_200_ppm_0_complete.h5'
        mps_filename = 'mps_virtual_17_projections_20000_usfactor_1_FOV_200_3d.h5'
        meanTimes_filename = 'TE_rastered_470_N_windows_255.h5'
    
        with h5py.File(path + os.sep + kdat_filename, 'r') as f:
            kdata = f['k_all'][:]
            # kdata = kdata[::us_fac, :,:,0:58]  # add coil dimension
            
            coords = f['coords_all'][:]
            # coords = coords[::us_fac, 0:58, ...]
            
    
        with h5py.File(path + os.sep + mps_filename, 'r') as f:
            mps = f['mps'][:]
    
        with h5py.File(path + os.sep + meanTimes_filename, 'r') as f:
            
            TE_rastered = f['TE_rastered'][:]*1e-6
            
            # meanTimes = TE_rastered[:,0,0]
            # print("TE_rastered: ", TE_rastered[0,1000,0])

    if us_fac_TE > 1:
        n = TE_rastered.shape[0]
        idx_us_TE = np.floor(np.arange(0, n, us_fac_TE)).astype(int)
        TE_rastered = TE_rastered[idx_us_TE, ...]
        coords = coords[idx_us_TE, :, ...]
        kdata = kdata[idx_us_TE, :,:, :]  # add coil dimension
    if us_fac_spokes > 1:
        n = TE_rastered.shape[1]
        idx_us_spokes = np.floor(np.arange(0, n, us_fac_spokes)).astype(int)
        TE_rastered = TE_rastered[:,idx_us_spokes , ...]
        coords = coords[:, idx_us_spokes, :, ...]
        kdata = kdata[:,:, :, idx_us_spokes, :]  # add coil dimension
        

    Nproj = 384
    Nread = config['slidingwindowpros']
    FOV = config['recon_parameter']['FOV']
    Neco = 1019 #TODO: don't hardcode! int((20000*6 - Nread) / config['slidingwindowstep'] + 1)
    field_strength = 6.980936 #2.89 #

    

    Ncoil = config['N_virtual_coils']

    path = config['path']

    summarize(kdata, 'k_after_loading')
    

    # scale k_data
    factor = 100 / np.sqrt(np.vdot(kdata, kdata))
    print("Factor: ", factor)
    kdata = kdata * factor

    summarize(kdata, 'k_after_scaling')


    img = np.zeros([1] + [1] + [1] + [FOV, FOV, FOV], dtype=np.complex128) 
    img = np.concatenate([img, img, img, img], axis=0)  # for water and fat images
    # k = k.swapaxes(0,1)  # shape (N_channels, steps, N_selectedProjections, N_samples)


    moba_par = config['moba_parameter']

    scaling = moba_par['scaling']
    print(scaling)
    img = img *scaling
    initial_sol = img
    if moba_par['init_fieldmap']:
        with h5py.File(moba_par['init_file_path'], 'r') as f: #TODO: fix hardcode
            
            field_map = np.array(f['field_map'][:])
            r2star = np.array(f['r2star_map'][:])
            if moba_par['init_fat_water']:
                # fat_map = np.array(f['fat_map'][:])
                wat_map = np.array(f['wat_map'][:])
                initial_sol[0,0, 0, ...] = wat_map*scaling #sp.fft(wat_map, axes=(-3,-2,-1))
                # initial_sol[1,0, 0, ...] = fat_map*scaling #sp.fft(fat_map, axes=(-3,-2,-1))
            if moba_par['smooth_fieldmap']:
                field_map = smooth_fieldmap_3d(field_map, 10)
            
            if moba_par['fm_normalization']== 'max':
                field_map = field_map / np.max(field_map)
            elif moba_par['fm_normalization']== 'minmax':
                field_map = (field_map - field_map.min()) / (field_map.max() - field_map.min())
            else:
                print("No normalization applied to field map")
            # field_map = field_map*2-1
            # field_map = field_map[12:187, 12:187, 12:187]

        print("field_map shape: ", field_map.shape)

        initial_sol[2,0, 0, ...] = field_map*scaling #sp.fft(field_map, axes=(-3,-2,-1))
        initial_sol[3,0, 0, ...] = r2star*scaling #sp.fft(field_map, axes=(-3,-2,-1))
        

    # field_map = smooth_fieldmap_3d(field_map, 10)*1E-6
    with h5py.File(moba_par['dcf_path'], 'r') as f:
        dcf = f['densityCompensationVoronoi'][...]
    # dcf = dcf[::4, ...]
    if us_fac_TE > 1:
        dcf = dcf[idx_us_TE, ...]
    if us_fac_spokes > 1:
        dcf = dcf[:,idx_us_spokes,...]

    factor=1
    if factor > 1:
        comb = np.zeros((dcf.shape[0] // factor, dcf.shape[1] * factor, dcf.shape[2]), dtype=dcf.dtype)
        for win in range(comb.shape[0]):
            comb[win, ...] = dcf[win * factor:(win + 1) * factor, ...].reshape(
                factor * dcf.shape[1], dcf.shape[2])
        dcf = comb
    dcf = np.sqrt(dcf)

        

    initial_sol = sp.to_device(initial_sol)
    k = sp.to_device(kdata)
    # k = k[:, np.newaxis, ...]  # add coil dimension
    mps = sp.to_device(mps)
    print(">> Starting nonlinear reconstruction with FWZ model...")

    iterative_solver = NonLinearLeastSquares(y=k, mps=mps, coords=coords, TE_rastered=TE_rastered, x=initial_sol, avg_TE=meanTimes,
                                inner_iter=moba_par['max_iter_CG'], outer_iter=moba_par['max_iter_IRGNM'], alpha=moba_par['alpha'], redu=moba_par['redu'], 
                                inner_tol=moba_par['inner_tol'], field_strength=field_strength, max_iter_inner=moba_par['max_iter_inner'], 
                                TV_weight_water=moba_par['TV_weight_water'], TV_weight_fat=moba_par['TV_weight_fat'], dcf_sqrt=dcf, fat_model=moba_par['fat_model'])

    full_filename = path + os.sep + config['custom_out_name'] + '.h5'
    with h5py.File(full_filename, 'w') as f:
        pass
    fixed_output_slice = config['Fixed_output_slice']
    if fixed_output_slice < 0:
        fixed_output_slice = 100
    N_slices = 1

    iter = 0
    while not iterative_solver.done():
        iterative_solver.update()
        with h5py.File(full_filename, 'r+') as f:
            out_data = iterative_solver.x
            f.create_dataset(f'images_iter_{iter}', data=out_data[...,fixed_output_slice])
        iter += 1


    with h5py.File(full_filename, 'r+') as f:
        out_data = iterative_solver.x

        f.create_dataset('final_images', data=out_data[...])

class NonLinearLeastSquares(alg.Alg):
    """Abstraction for non-linear least square (nlls) algorithms.

    Args:
        A (nlop): non-linear operator.
        y (array): measurements.
        x (None or array): initial guess of unknown. Default None.
        x0 (None or array): previous estimate of unknown. Default None.
        outer_iter (int): outer iteration steps. Default 6.
        alpha (float): regularization strength. Default 1.
        redu (float): reduction rate along outer iteration. Default 2.
        trafos (None or linop): transformation on x. Default None.
        proxf (None or prox): proximal operator. Default None.
        inner_iter (int): inner iteration steps. Default 100.
        inner_tol (float): inner iteration tolerance. Default 0.01.

    References:
        Bauer F., Kannengiesser S. (2007).
        An alternative approach to the image reconstruction for parallel data acquisition in MRI.
        Math. Meth. Appl. Sci. 30, 1437-1451.

        Uecker M., Hohage T., Block K. T., Frahm J. (2008)
        Image reconstruction by regularized nonlinear inversion -- joint estimation of coil sensitivities and image content.
        Magn. Reson. Med. 60, 674-682.

        Tan Z., Roeloffs V., Voit D., Joseph A. A., Untenberger M., Merboldt K. D., Frahm J. (2016).
        Model-based reconstruction for real-time phase-contrast flow MRI: Improved spatiotemporal accuracy.
        Magn. Reson. Med. 77, 1082-1093.
    """
    def __init__(self, y, x=None, x0=None, coords=None, mps=None, TE_rastered=None, avg_TE=None, field_strength=2.89,
                 outer_iter=6, alpha=1., redu=2.,
                 trafos=None, proxf=None, max_iter_inner=3,
                 inner_iter=100, inner_tol=0.01, TV_weight_water=0, TV_weight_fat=0, coil_batch_size=1, dcf_sqrt=None, fat_model='invivo'):
        
        self.y = y
        self.mps = backend.to_device(mps, backend.Device(0))
        self.device = backend.get_device(y)
        self.coords = coords
        self.TE_rastered = TE_rastered
        self.TE = TE_rastered[:,0,0]
        self.field_strength = field_strength

        self.fatModulation = backend.to_device(self.calc_fat_modu_k_space(self.TE_rastered, B0=self.field_strength, fat_model=fat_model), backend.Device(0))
        summarize(self.fatModulation, 'fatModulation')

        print(self.y.shape)
        self.Neco, _, self.Ncoil, self.Nproj, self.Nread = self.y.shape
        self.FOV = x.shape[-1]

        self.coil_batch_size = min(coil_batch_size, self.Ncoil)
        self.coil_slices = [
            slice(s, min(s + self.coil_batch_size, self.Ncoil))
            for s in range(0, self.Ncoil, self.coil_batch_size)
        ]
        print("coil batches:", [(sl.start, sl.stop) for sl in self.coil_slices])
        print("device: ", self.device)
        xp = self.device.xp

        self.trafos = trafos
        self.proxf = proxf

        # outer iteration
        self.outer_iter = outer_iter  # max_iter
        self.max_iter_inner = max_iter_inner
        self.alpha = alpha
        self.redu = redu
        self.TV_weight_water = TV_weight_water
        self.TV_weight_fat = TV_weight_fat

        self.fat_scale = 1

        # inner iteration
        self.inner_iter = inner_iter
        self.inner_tol = inner_tol


        self.dcf_sqrt = dcf_sqrt
        self.define_FFT()          # now AFTER dcf_sqrt exists
        self.y = self.y * self.dcf_sqrt[:, np.newaxis, np.newaxis, ...]
        self.dcf_sqrt = backend.to_device(self.dcf_sqrt, backend.Device(0))
        with self.device:
            # initialize x
            self.x = sp.to_device(x, device=self.device)

            # initialize x0
            # self.x0 = xp.zeros(self.x.shape, dtype=y.dtype)

        super().__init__(outer_iter)
    
    def _update(self):
        """Perfom outer iteration steps of the nonlinear problem.
        """
        xp = self.device.xp

        with self.device:
            self.dx = xp.zeros_like(self.x)

            self.r = self.y - self.forward_op(self.x)

            # residual
            resid = xp.linalg.norm(self.r).item()

            print("iter: " + "%2d"%(self.iter) + "; alpha: " + "%.2f"%(self.alpha) + "; resid: " + "%4.3f"%(resid))

            # self.p = self.A.adjoint(self.x, self.r)

            # self.p += self.alpha * (self.x0 - self.x)

            # update dx
            for i in range(self.max_iter_inner):
                print('Inner iteration: ', i)
                t = time()
                self.lls()
            print(f'\n\n\n Done with NL_CG \n Time passed: {time()-t}\n\n\n')

            self.x += 1. * self.dx  #here the matlab code uses some scaling factor t

            self.alpha /= self.redu


    def _done(self):
        """Stopping criteria for the outer iteration.
        """
        return self.iter >= self.outer_iter


    def lls(self):
        """Inner linear least square (lls) solver 
        for the linearized non-linear problem.
        """
        xp = self.device.xp

        with self.device:

            # tolerance
            # tol = self.inner_tol * xp.linalg.norm(self.p).item()
            
            if self.trafos is None and self.proxf is None:
                """NL Conjugate gradient method
                """

                alg_lls = self.NL_ConjugateGradient(
                                                        self.A_der, self.A_adjoint, par_data=self.r,
                                                        x_lin=self.x, dx=self.dx, mps=self.mps,
                                                        fatModulation=self.fatModulation, coords=self.coords, FFT_ops=self.FFT_ops,
                                                        TE_rastered=self.TE_rastered,
                                                        verbose=True, max_iter=self.inner_iter, tol=0, alpha_reg=self.alpha,
                                                        par_dataWeight=1, TV_weight_water=self.TV_weight_water, TV_weight_fat= self.TV_weight_fat
                                                    )
                #alg.ConjugateGradient(AHA_L2, self.p, self.dx, max_iter=self.inner_iter, tol=tol)
                while not alg_lls.done():
                    # print("  inner iter: " + "%3d"%(alg_lls.iter) + "; resid: " + "%4.6f"%(alg_lls.resid))
                    alg_lls.update()
                self.dx = alg_lls.inner_x

            return self.dx

    def calc_fat_modu_k_space(self, gradient_times,  # second
                                GYRO: float = 42.57747892E6,  # Hz/T
                                B0: float = 0.55,  # T
                                fat_model: str = 'invivo') -> np.ndarray:
        """
        calculate fat modulation

        Input:
            gradient_times - gradient times in second
            GYRO - Gyromagnetic ratio
            B0 - Magnetic field strength
        """
        device = sp.backend.get_device(gradient_times)
        print("Gradient times shape", gradient_times.shape)

        with device:

            # TE = TE.ravel()

            # ppm = np.array([-3.2]) * 1E-6
            # amp = np.array([0.087, 0.693, 0.128, 0.004, 0.039, 0.048])
            # amp = np.array([0.6006])
            # ppm = np.array([-3.80, -3.40, -2.60, -1.94, -0.39, 0.60]) * 1E-6
            # amp   = np.array([0.087, 0.693, 0.128, 0.004, 0.039, 0.048])

            if fat_model == 'phantom':
                ppm = np.array([-3.926, -3.5, -3.211, -2.782, -2.563, -2.037, -0.724, -0.510, 0.513]) * 1E-6
                amp = np.array([0.0879, 0.6478, 0.0130, 0.0936, 0.0468, 0.0031, 0.0230, 0.0158, 0.0690])
                print("Using phantom fat model")
            elif fat_model == 'invivo':
                ppm = np.array([-3.6, -3.2, -2.91, -2.47, -2.25, -1.73, 0.81]) * 1E-6
                amp = np.array([0.0813, 0.6006, 0.0678, 0.0914, 0.0635, 0.0152, 0.0802])
                print("Using in-vivo fat model")
            
            res = np.zeros(gradient_times.shape, dtype=np.complex64)

            for n in range(gradient_times.shape[0]):

                for p in range(len(ppm)):

                    res[n, ...] += amp[p] * np.exp(2j * np.pi * GYRO * B0 * ppm[p] * gradient_times[n])

        print("fat modulation shape: ", res.shape)
        return res

    def define_FFT(self):
        dcf = self.dcf_sqrt
        sizes = sorted({sl.stop - sl.start for sl in self.coil_slices})
 
        # self.FFT_ops[n_coils_in_batch][TE_i]
        self.FFT_ops = {}
        for n in sizes:
            ops = []
            for TE_i in range(self.coords.shape[0]):
                ops.append(
                    sp.linop.Multiply((2 * n, 1, self.Nproj, self.Nread), dcf[[TE_i], ...]) *
                    sp.linop.NUFFT(
                        ishape=(2 * n, 1, self.FOV, self.FOV, self.FOV),  # coils x {water,fat}
                        coord=self.coords[TE_i, ...], oversamp=2, width=4, toeplitz=False))
            self.FFT_ops[n] = ops

    def forward_op(self, x):
        device = backend.Device(0)
        xp = device.xp
 
        kdata = xp.zeros((self.Neco, 1, self.Ncoil, self.Nproj, self.Nread), dtype=xp.complex64)
 
        water    = backend.to_device(x[0, ...], device)
        fat      = backend.to_device(x[1, ...], device)
        fieldmap = backend.to_device(x[2, ...], device)
        r2star   = backend.to_device(x[3, ...], device)
 
        for ie in range(self.Neco):
            TE = self.TE[ie]
            exp_term = xp.exp(1j * 2 * xp.pi * fieldmap * TE) * xp.exp(-r2star * TE)  # coil-independent
 
            for sl in self.coil_slices:
                n = sl.stop - sl.start
                coilmaps_exp = self.mps[sl, xp.newaxis] * exp_term                    # (n, 1, FOV, FOV, FOV)
 
                stacked = xp.concatenate([coilmaps_exp * water, coilmaps_exp * fat], axis=-5)  # (2n, 1, ...)
 
                k_all = self.FFT_ops[n][ie] * stacked
                k_water, k_fat = k_all[:n], k_all[n:]
 
                kdata[ie, 0, sl, ...] = k_water[:, 0, ...] + self.fat_scale * self.fatModulation[ie, xp.newaxis, ...] * k_fat[:, 0, ...]
 
        return backend.to_device(kdata, backend.cpu_device)
 
    def A_der(self, x, dx):
        device = backend.Device(0)
        xp = device.xp
 
        x = backend.to_device(x, device)
        dx = backend.to_device(dx, device)
 
        d_kdata = xp.zeros((len(self.TE), 1, self.Ncoil, self.Nproj, self.Nread), dtype=xp.complex64)
 
        water, fat, fieldmap, r2star = x[0, ...], x[1, ...], x[2, ...], x[3, ...]
        d_water, d_fat, d_fieldmap, d_r2star = dx[0, ...], dx[1, ...], dx[2, ...], dx[3, ...]
 
        for ie in range(self.Neco):
            TE = self.TE[ie]
 
            # coil-independent -> computed once per echo, outside the batch loop
            exp_term = xp.exp(1j * 2 * xp.pi * fieldmap * TE) * xp.exp(-r2star * TE)
            scalar_term1 = 1j * 2 * xp.pi * TE * water * d_fieldmap + d_water - TE * water * d_r2star
            scalar_term2 = 1j * 2 * xp.pi * TE * fat   * d_fieldmap + d_fat   - TE * fat   * d_r2star
 
            for sl in self.coil_slices:
                n = sl.stop - sl.start
                coilmaps_exp = self.mps[sl, xp.newaxis] * exp_term
 
                stacked = xp.concatenate([coilmaps_exp * scalar_term1,
                                          coilmaps_exp * scalar_term2], axis=-5)      # (2n, 1, ...)
 
                k_all = self.FFT_ops[n][ie] * stacked
                k1, k2 = k_all[:n], k_all[n:]
 
                d_kdata[ie, 0, sl, ...] = k1[:, 0, ...] + self.fat_scale * self.fatModulation[ie, xp.newaxis, ...] * k2[:, 0, ...]
 
        return backend.to_device(d_kdata, backend.cpu_device)
 
    def A_adjoint(self, x0, d_kdata):
        device = backend.Device(0)
        xp = device.xp
        x0 = backend.to_device(x0, device)
 
        water, fat, fieldmap, r2star = x0[0, ...], x0[1, ...], x0[2, ...], x0[3, ...]
        d_kdata = backend.to_device(d_kdata, device)
 
        dx_water    = xp.zeros_like(water)
        dx_fat      = xp.zeros_like(fat)
        dx_fieldmap = xp.zeros_like(fieldmap)
        dx_r2star   = xp.zeros_like(r2star)
 
        for ie in range(self.Neco):
            TE = self.TE[ie]
            exp_term = xp.exp(1j * 2 * xp.pi * fieldmap * TE) * xp.exp(-r2star * TE)
            fatmod_conj = xp.conj(self.fatModulation[ie, xp.newaxis, ...])
 
            # accumulate the coil-combined images over all coil batches
            comb1 = xp.zeros_like(water)   # water-path
            comb2 = xp.zeros_like(water)   # fat-path
 
            for sl in self.coil_slices:
                n = sl.stop - sl.start
                coilmaps_exp = self.mps[sl, xp.newaxis] * exp_term
 
                raw = d_kdata[ie, 0, sl, :, :][:, xp.newaxis, ...]            # (n, 1, Nproj, Nread)
                weighted = fatmod_conj * raw
 
                stacked_k = xp.concatenate([raw, weighted], axis=-4)          # (2n, 1, Nproj, Nread)
                img_all = self.FFT_ops[n][ie].H * stacked_k
                temp1, temp2 = img_all[:n], img_all[n:]
 
                comb1 += xp.sum(xp.conj(coilmaps_exp) * temp1, axis=0)
                comb2 += self.fat_scale * xp.sum(xp.conj(coilmaps_exp) * temp2, axis=0)
 
            # the parameter-gradient terms are linear in comb1/comb2 -> apply once per echo
            dx_water    += comb1
            dx_fat      += comb2
            dx_fieldmap += xp.conj(1j * 2 * xp.pi * TE * water) * comb1 + xp.conj(1j * 2 * xp.pi * TE * fat) * comb2
            dx_r2star   += xp.conj(-TE * fat) * comb2 + xp.conj(-TE * water) * comb1
 
        dx = xp.zeros_like(x0)
        dx[0, ...] = dx_water
        dx[1, ...] = dx_fat
        dx[2, ...] = xp.real(dx_fieldmap)
        dx[3, ...] = xp.clip(xp.real(dx_r2star), 0, None)
 
        return backend.to_device(dx, backend.cpu_device)
        
    def get_coords(self):
        self.Nproj = 256
        self.Nread = 512
        self.Neco = 3
        self.Ncoil = 4
        self.TE = np.array([0.0014100000, 0.0030100000, 0.0046100000])
        blipangle = 0.0349
        samptime = 2.5E-6

        # Generating rastered TE
        TE_rastered = np.zeros((self.Neco, 1, self.Nread))

        for ie in range(self.Neco):
            idx = np.arange(-self.Nread // 2, self.Nread // 2)  # -Nread/2 : Nread/2 - 1
            timemap = self.TE[ie] + idx * samptime
            # print("Timemap ", timemap[0:10])
            if ie % 2 == 1:
                timemap = np.flip(timemap)
            TE_rastered[ie, 0, :] = timemap

        # Golden angle in radians
        GA = 111.246117975 / 180 * np.pi

        # phi: shape (Nread, Nproj, Neco)
        phi = np.zeros((self.Nread, self.Nproj, self.Neco))
        for ip in range(self.Nproj):
            for ie in range(self.Neco):
                phi[:, ip, ie] = np.pi / 2 + ip * GA + ie * blipangle

        # rho: radial coordinate, centered, normalized
        rho = np.linspace(0, self.Nread - 1, self.Nread) - (self.Nread - 1) / 2
        rho = rho[:, None, None]                       # shape (Nread, 1, 1)
        rho = np.tile(rho, (1, self.Nproj, self.Neco))            # shape (Nread, Nproj, Neco)
        rho = rho / self.Nread

        # Generate k-space trajectory as complex values kx + i*ky
        traj = (rho * np.exp(1j * phi)).astype(np.complex64)
        traj = np.transpose(traj, (2,1,0))  # shape (Nread, Nproj, Neco)

        coords = np.stack([np.real(traj), np.imag(traj)], axis=-1) * self.Nproj  # shape (Nread, Nproj, Neco, 2)
        return coords, TE_rastered

    class NL_ConjugateGradient(alg.Alg):
 
        def __init__(self, A_der, A_adj, x_lin, dx, mps, fatModulation, coords, TE_rastered, FFT_ops,
                        max_iter=100, tol=0, verbose=False,
                        par_dataWeight=1, par_data=None, alpha_reg=0.01,
                        ls_alpha=0.01, ls_beta=0.6, TV_weight_water=0, TV_weight_fat=0):
    
            self.A_der = A_der
            self.A_adj = A_adj
    
            self.x_lin = x_lin               # fixed Gauss-Newton linearization point
            self.inner_x = dx.copy()         # increment being solved for, warm-started from dx
            self.tol = tol
            self.verbose = verbose
            self.device = backend.get_device(x_lin)
    
            xp = self.device.xp
    
            self.alpha = alpha_reg           # Tikhonov weight  (par.alpha_n)
            self.two_alpha = 2 * alpha_reg
            self.ls_alpha = ls_alpha         # Armijo constant  (par.alpha)
            self.ls_beta = ls_beta           # backtracking factor (par.beta)
            self.t0 = 1.0
            self.TV_weight_water = TV_weight_water
            self.TV_weight_fat = TV_weight_fat

            self.pnorm_im = 1 #TODO: figure out what these two do
            self.pnorm_off = 2
            self.mu = 1e-6 * (xp.abs(x_lin).max().item()) ** 2
    
            self.par_dataWeight = par_dataWeight
            self.two_dataWeight = 2 * par_dataWeight
            self.par_data = par_data
    
            self.coords = coords
            self.TE_rastered = TE_rastered
            self.FFT_ops = FFT_ops
            self.fatModulation = fatModulation
            self.mps = mps
    
            # --- one-time forward pass to seed the cache ---
            with self.device:
                self.Ax = self.A_der(self.x_lin, self.inner_x)   # cached Jacobian @ x0 applied to current point
                self.g0 = self.gradient(self.inner_x, self.Ax)
    
            self.inner_dx = -self.g0                          # CG search direction (MATLAB local "dx")
    
            self.gtg = xp.real(xp.vdot(self.g0, self.g0)).item()
            self.k = 0
            self.normdx0 = xp.linalg.norm(self.inner_dx).item()
    
            self.break_cond = False
    
            super().__init__(max_iter)
    
        def _update(self):
            xp = self.device.xp
            with self.device:
                DAx = self.Ax                              # reused from previous iteration / init — NOT recomputed
                DAd = self.A_der(self.x_lin, self.inner_dx)    # only forward-operator call needed this iteration

                TV_w, TV_dw, TV_f, TV_df = self.preobjective(self.inner_x, self.inner_dx)
                f0 = self.objective(self.inner_x, self.inner_dx, DAx, DAd, TV_w, TV_dw, TV_f, TV_df, 0.)
                # #TODO: remove this part
                # gd = xp.real(xp.vdot(self.g0.ravel(), self.inner_dx.ravel())).item()
                # print(f"gd = {gd:.6e}, f0 = {float(f0):.10e}")
                # for e in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
                #     fe = self.objective(self.inner_x, self.inner_dx, DAx, DAd, TV_w, TV_dw, TV_f, TV_df, e)
                #     print(f"  t={e:.0e}  (f(t)-f0)/t = {(float(fe)-float(f0))/e:.6e}")

                # x = xp.random.randn(*self.x_lin[0].shape) + 1j*xp.random.randn(*self.x_lin[0].shape)
                # y = xp.random.randn(*self.TV_for(x).shape) + 1j*xp.random.randn(*self.TV_for(x).shape)
                # print(xp.vdot(self.TV_for(x), y), xp.vdot(x, self.TV_adj(y)))   # must match
                # #########
    
                t = self.t0
                iter_ls = 0
                f1 = self.objective(self.inner_x, self.inner_dx, DAx, DAd, TV_w, TV_dw, TV_f, TV_df, t)
    
                gd = xp.real(xp.vdot(self.g0.ravel(), self.inner_dx.ravel()))
                c2 = -xp.abs(gd)
                while f1 > f0 + self.ls_alpha * t * c2 and iter_ls < 100:
                    t *= self.ls_beta
                    f1 = self.objective(self.inner_x, self.inner_dx, DAx, DAd, TV_w, TV_dw, TV_f, TV_df, t)
                    iter_ls += 1
    
                if self.verbose:
                    print("  iter_ls: " + "%3d" % (iter_ls) + "; t: " + "%4.6f" % (t))
    
                if iter_ls > 2:
                    self.t0 *= self.ls_beta
                elif iter_ls < 1:
                    self.t0 /= self.ls_beta
    
                self.inner_x = self.inner_x + t * self.inner_dx
    
                # DAx at the accepted step is just DAx + t*DAd — exploits linearity of A_der,
                # so gradient() never needs to call A_der a third time.
                Ax_new = DAx + t * DAd
                self.Ax = Ax_new
    
                g1 = self.gradient(self.inner_x, Ax_new)

                gtg_new = xp.real(xp.vdot(g1, g1)).item()

                # Polak-Ribière+ instead of Fletcher-Reeves (uses OLD g0, so compute before overwriting it)
                bk = max(0.0, xp.real(xp.vdot(g1, g1 - self.g0)).item() / (self.gtg + 1e-30))

                self.inner_dx = -g1 + bk * self.inner_dx

                # Safeguard: if the new direction isn't a descent direction, restart with steepest descent
                if xp.real(xp.vdot(g1, self.inner_dx)).item() >= 0:
                    self.inner_dx = -g1

                self.gtg = gtg_new
                self.g0 = g1
                self.k += 1

                self.norm_d = xp.linalg.norm(self.inner_dx).item()
                self.break_cond = ((self.norm_d / self.normdx0) < self.tol)
    
                if self.verbose:
                    print(f"  iter: {self.iter:3d}; norm_d: {self.norm_d:.6f}")
                    print(f"  iter: {self.iter:3d}; f1: {f1:.6f}; f0: {f0:.6f}; t: {t:.6f}")
                    print(f"  gtg_new: {gtg_new:.6f}; bk: {bk:.6f}")
    
        def _done(self):
            return (self.iter >= self.max_iter) or self.break_cond
    
        def summarize(self, a, name=""):
            a = np.asarray(a)
            print(f"--- {name} ---")
            print("shape:", a.shape)
            print("mean:", a.mean().real)
            print("std:", a.std(ddof=1))
            idx_min = np.argmin(np.abs(a))
            idx_max = np.argmax(np.abs(a))
            print("min:", a.flat[idx_min].real)
            print("max:", a.flat[idx_max].real)
            print("sum:", a.sum().real)
            print("norm:", np.linalg.norm(a))
            print()
    
        # --- internal helpers that take an already-computed A_der(x0, ·) result ---
        # These avoid ever calling A_der more than once per iteration.
    
        def _gradient_from_Ax(self, DAdx):
            grad_data = self.par_dataWeight * 2 * self.A_adj(self.x_lin, DAdx - self.par_data)
            return grad_data
    
        # --- public API kept for backwards compatibility with any external callers ---
    
        def gradient(self, dx, DAdx):
            """Kept for API compatibility. Recomputes A_der; prefer the internal
            cached path inside _update() for the hot loop."""
            gradData = self._gradient_from_Ax(DAdx)
            grad_TV = np.zeros_like(dx)
            if self.TV_weight_water > 0 or self.TV_weight_fat > 0:
                grad_TV[0,...] = self.TV_weight_water*self.gradTV(dx[0,...] + self.x_lin[0,...])
                grad_TV[1,...] = self.TV_weight_fat*self.gradTV(dx[1,...] + self.x_lin[1,...])

           
            full_grad = self.two_alpha*grad_TV + gradData + self.two_alpha*dx
            return full_grad

        def gradTV(self, x):
            p    = self.pnorm_im
            fd   = self.TV_for(x)
            grad = p * fd *((fd * fd.conj()).real + self.mu) ** (p/2-1)
            grad = self.TV_adj(grad)
            return grad
    
        def objective(self, dx, d, DAx, DAd, TV_w, TV_dw, TV_f, TV_df, t):
            xp = self.device.xp
            res = DAx + t * DAd - self.par_data
            res = xp.linalg.norm(res.ravel()) ** 2

            if self.TV_weight_water > 0 or self.TV_weight_fat > 0:
                p = self.pnorm_im
                tmp1 = TV_w + t*TV_dw
                tmp2 = TV_f + t*TV_df
                TV_res = self.TV_weight_water*(((tmp1 * tmp1.conj()).real + self.mu) ** (p/2)) + self.TV_weight_fat*(((tmp2 * tmp2.conj()).real + self.mu) ** (p/2))
            else:
                TV_res = 0

            tmp = dx + t * d
            tmp = xp.linalg.norm(tmp.ravel()) ** 2
            # print("Tikhinov weight: ", self.alpha * tmp)
            # print("TV weight: ", self.TV_weight*xp.sum(TV_res))
            # print("Data weight: ",self.par_dataWeight * res )
            return self.par_dataWeight * res + self.alpha * tmp + self.two_alpha*xp.sum(TV_res)
    
        def preobjective(self, x, dx):

            if self.TV_weight_water > 0 or self.TV_weight_fat > 0:
                TV_w = self.TV_for(x[0,...] + self.x_lin[0,...])
                TV_dw = self.TV_for(dx[0,...])
                TV_f = self.TV_for(x[1,...] + self.x_lin[1,...])
                TV_df = self.TV_for(dx[1,...])
            else:
                TV_w = 0
                TV_dw = 0
                TV_f = 0
                TV_df = 0

            return TV_w, TV_dw, TV_f, TV_df

        def TV_for(self, x):
            device = backend.Device(0)
            xp = device.xp
            # print("x in TV_for shape: ", x.shape)
            x = backend.to_device(x, device)
    
            Dx = xp.zeros_like(x)
            Dy = xp.zeros_like(x)
            Dz = xp.zeros_like(x)
    
            Dy[..., :, :-1, :] = x[..., :, 1:, :] - x[..., :, :-1, :]   # along columns
            Dx[..., :-1, :, :] = x[..., 1:, :, :] - x[..., :-1, :, :]   # along rows
            Dz[..., :, :, :-1] = x[..., :, :, 1:] - x[..., :, :, :-1]   # along z
    
            return backend.to_device(xp.stack((Dx, Dy, Dz), axis=-1))
    
        def TV_adj(self, g):
            device = backend.Device(0)
            xp = device.xp
            g = backend.to_device(g, device)
            # print("g in TV_adj shape: ", g.shape)
            """Adjoint of D. g: (ny, nx, nz, 3) -> (ny, nx, nz)."""
            Dx, Dy, Dz = g[..., 0], g[..., 1], g[..., 2]
            res = xp.zeros_like(Dx)
    
            res[..., :, :-1, :] -= Dy[..., :, :-1, :]
            res[..., :, 1:, :]  += Dy[..., :, :-1, :]
    
            res[..., :-1, :, :] -= Dx[..., :-1, :, :]
            res[..., 1:, :, :]  += Dx[..., :-1, :, :]
    
            res[..., :, :, :-1] -= Dz[..., :, :, :-1]
            res[..., :, :, 1:]  += Dz[..., :, :, :-1]
    
            return backend.to_device(res)


    




if __name__ == "__main__":
    main()