#!/usr/bin/env python
# -*- coding: utf-8 -*-

#################################################################################################

# Remove the chromatic OPD variation and a residual OPD and offset from the differential phase
# of the data reduced by the MATISSE pipeline.

# Requires:
# - A directory containing the pipeline's output subdirectories (one OB = one directory)
#   These OB subdirectories have names like 'mat_raw_estimates.2022-11-09T04_58_16.HAWAII-2RG.rb'
# - In each subdirectory, at least the nrjReal, nrjImag, RAW_VIS2 and TARGET_CAL files.

# Define your parameters in the input parameters section

# Authors:
#   M. Houllé

#################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy

from astropy.io import fits

from common_tools import refractive_index, compute_UV_coords, wrap

####################################################################

### Input parameters ###

# Raw data directory
path_raw = '/store/projects/MATISSE/jscigliuto/2023-10-30/corrfluxtrue/Iter1'

# Corrected data directory
path_results = '/store/projects/MATISSE/jscigliuto/HR8799e_phase_corrected'

# (dRA, dDec) coordinate offsets of the targets. Used for properly naming files.
# Use one of these:
# 1) Coordinates matching the SEQ OFFSET ALPHA and SEQ OFFSET DELTA of the OIFITS headers.
#    If no matching coordinates, the target will be named 'unknown'.
target_coords = {'planet': (-223.8, 332.1), 'star': (0.0, 0.0), 'antiplanet': (223.8, -332.1)}
# 2) No coordinates. Will assume the star is (0, 0) and everything else a planet (be careful if there is an antiplanet!)
# target_coords = None
# 3) Special case for the Beta Pic Nov 2022 data, which had no SEQ OFFSET keywords.
# target_coords = 'beta_pic'

# Flag abnormal exposures with a list of (i_OB, i_exposure)
# Ex: ((1, 6), (2, 3), (2, 5))
# i_OB and i_exposure must be from the final chronological order (not EXPNO)
flagged_exposures = ()

# Selected wavelength range for OPD fitting
#wmin, wmax = 2.82, 4.19
wmin, wmax = 3.1, 4.0

# Plot the data selection process (slow, only if necessary)
plot_data_selection = False

# Initial OPD and offset guesses
# Can use previous fit results or custom guesses

## Use previous fit as guesses:
# use_previous_fit = True
# mjds_prev_fit = fits.getdata('results_20230724_v7_guessFromPrevFit/mjds.fits')
# opds_prev_fit = fits.getdata('results_20230724_v7_guessFromPrevFit/opd_fit_smooth.fits')
# offsets_prev_fit = fits.getdata('results_20230724_v7_guessFromPrevFit/offset_fit_smooth.fits')

## Use custom guesses:
use_previous_fit = False
#opd_guesses = np.array([10, 10, -1.5, 10, 10, 20])
#offset_guesses = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

# Search window of the OPD and offset fit 
opd_min = -40 # µm
opd_max = 40 # µm
offset_min = None # rad
offset_max = None # rad
bounds = ((opd_min, opd_max), (offset_min, offset_max))

####################################################################

### Functions ###

# Model and residuals functions for chi-2 fitting

def model_high_order(wl, n_air, deltaPath):
    # Model the phase introduced by the extra layer of air
    phase_layer = 2 * np.pi * (n_air - 1) * deltaPath / wl
    # Remove from it the linear OPD component (as performed by the fringe tracker)
    # We are left with the high-order variations located near the telluric absorption bands
    wl_mask_lin = (wl > 3.2) & (wl < 3.8)
    wlm = wl[wl_mask_lin]
    phasem = phase_layer[wl_mask_lin]
    slope = np.sum((phasem-phasem.mean())*(1/wlm-np.mean(1/wlm))) / np.sum((1/wlm-np.mean(1/wlm))**2)
    return wrap(phase_layer - slope/wl)

def model_low_order(params, wl):
    # Model a phase composed of an OPD component and an offset
    opd, offset = params
    return wrap(2 * np.pi * opd / wl + offset)

def residuals(params, wl, phase, phase_err):
    phase_model = model_low_order(params, wl)
    return np.sum(wrap(phase - phase_model)**2 / phase_err**2)

# Other functions

def find_target_by_coords(target_coords, coord_to_test):
    for target, target_coord in target_coords.items():
        if target_coord == coord_to_test:
            return target
    return 'unknown'

####################################################################

### Main program ###

# BCD baseline reorder
bcd_base_reorder = {'OUT-OUT': [0, 1, 2, 3, 4, 5],
                     'OUT-IN': [0, 1, 4, 5, 2, 3],
                     'IN-OUT': [0, 1, 3, 2, 5, 4],
                      'IN-IN': [0, 1, 5, 4, 3, 2]}
# BCD sign correction
bcd_sign = {'OUT-OUT': [1, 1, 1, 1, 1, 1],
             'OUT-IN': [1, -1, 1, 1, 1, 1],
             'IN-OUT': [-1, 1, 1, 1, 1, 1],
              'IN-IN': [-1, -1, 1, 1, 1, 1]}
# Final baseline and triangle order
base_order = [(3,4), (1,2), (2,3), (2,4), (1,3), (1,4)]
base_order_name = ('U3-U4', 'U1-U2', 'U2-U3', 'U2-U4', 'U1-U3', 'U1-U4')
t3_order = [(1,2,3), (1,2,4), (1,3,4), (2,3,4)]

# Create output directory if necessary
if not os.path.isdir(path_results):
    os.makedirs(path_results)
    if plot_data_selection:
        os.makedirs(path_results + '/fitting_masks')
    os.makedirs(path_results + '/best_fits')
    os.makedirs(path_results + '/corrected_data')

#obs_dirs = sorted([path_raw + '/' + d for d in os.listdir(path_raw) if not d.startswith('.')])

obs_dirs = sorted([path_raw + '/' + d for d in os.listdir(path_raw) if 'HAWAII-2RG.rb' in d and int(d[d.find('T')+1:d.find('T')+3])<3])
print(obs_dirs)

# Loop over OBs
for i_OB, obs_dir in enumerate(obs_dirs):
    print(i_OB, obs_dir)

    obs_time = obs_dir[obs_dir.find('.HAWAII')-8:obs_dir.find('.HAWAII')]
    target_cal_files = sorted([file for file in os.listdir(obs_dir) if 'TARGET_CAL' in file or 'CALIB_CAL' in file])
    n_exp = len(target_cal_files)

    mjd_target_cal = np.zeros(n_exp)
    expno_raw_vis2 = np.zeros(n_exp, dtype=int)

    for i_exp in range(n_exp):
        # List of MJDs of the TARGET_CAL files
        if os.path.exists(f'{obs_dir}/TARGET_CAL_{i_exp+1:04d}.fits'):
            mjd_target_cal[i_exp] = fits.getval(f'{obs_dir}/TARGET_CAL_{i_exp+1:04d}.fits', 'MJD-OBS')
        elif os.path.exists(f'{obs_dir}/CALIB_CAL_{i_exp+1:04d}.fits'):
            mjd_target_cal[i_exp] = fits.getval(f'{obs_dir}/CALIB_CAL_{i_exp+1:04d}.fits', 'MJD-OBS')
        else:
            raise ValueError('No TARGET_CAL or CALIB_CAL.')
        # List of exposure numbers of the RAW_VIS2 files
        expno_raw_vis2[i_exp] = fits.getval(f'{obs_dir}/RAW_VIS2_{i_exp+1:04d}.fits', 'ESO TPL EXPNO')

    # List of indices corresponding to the sorted TARGET_CAL MJDs
    i_sort_mjd_target_cal = np.argsort(mjd_target_cal)
    
    # Loop over exposures
    for i_exp in range(n_exp):
        
        # Open the calibration file
        if os.path.exists(f'{obs_dir}/TARGET_CAL_{i_exp+1:04d}.fits'):
            hdul_cal = fits.open(f'{obs_dir}/TARGET_CAL_{i_exp+1:04d}.fits')
        elif os.path.exists(f'{obs_dir}/CALIB_CAL_{i_exp+1:04d}.fits'):
            hdul_cal = fits.open(f'{obs_dir}/CALIB_CAL_{i_exp+1:04d}.fits')
        else:
            raise ValueError('No TARGET_CAL or CALIB_CAL.')
        date_obs = hdul_cal[0].header['DATE-OBS'].replace(':', '', 2).split('.')[0]
        expno = hdul_cal[0].header['ESO TPL EXPNO']
        tartyp = hdul_cal['IMAGING_DATA'].data['TARTYP']
        n_valid_frames = np.sum(tartyp == 'T')

        # Exposure numbers in chronological order
        chronological_expno = np.argwhere(i_sort_mjd_target_cal == i_exp).squeeze()

        # Attribute exposure to a given target
        offset_RA = hdul_cal[0].header['ESO SEQ OFFSET ALPHA']
        offset_Dec = hdul_cal[0].header['ESO SEQ OFFSET DELTA']
        if target_coords is None:
            if (offset_RA, offset_Dec) == (0, 0):
                target = 'star'
            else:
                target = 'planet'
        elif target_coords == 'beta_pic':
            # The star is observed in the last 4 exposures of each OB
            if chronological_expno in range(n_exp-4, n_exp):
                target = 'star'
            else:
                target = 'planet'
        elif type(target_coords) is dict:
            target = find_target_by_coords(target_coords, (offset_RA, offset_Dec))
        else:
            raise ValueError("target_coords has an invalid format (use a dict of coordinates, None or 'beta_pic')")

        # Flag some abnormal exposures
        if (i_OB+1, i_exp+1) in flagged_exposures:
            flag = '_flagged'
        else:
            flag = ''

        # Get the BCD configurations  
        bcd1 = hdul_cal[0].header['ESO INS BCD1 NAME']
        bcd2 = hdul_cal[0].header['ESO INS BCD2 NAME']
        bcd = f'{bcd1}-{bcd2}'

        # Get the MJDs
        mjds = hdul_cal['IMAGING_DATA'].data['TIME']
        mjds_valid = mjds[tartyp == 'T']

        # Get the coordinates of the array
        ra, dec = hdul_cal[0].header['RA'], hdul_cal[0].header['DEC']
        lat, long = hdul_cal[0].header['ESO ISS GEOLAT'], hdul_cal[0].header['ESO ISS GEOLON']
        sta_xyz = hdul_cal['ARRAY_GEOMETRY'].data['STAXYZ']

        ## Get the path lengths
        static_lengths = np.zeros(4)
        start_OPLs = np.zeros(4)
        end_OPLs = np.zeros(4)
        OPLs = np.zeros((4, n_valid_frames))
        deltaPaths = np.zeros((6, n_valid_frames))
        for i_tel in range(4):
            # Static paths (m)
            static_lengths[i_tel] = hdul_cal[0].header[f'ESO ISS CONF A{i_tel+1}L']
            # Optical path lengths (m)
            start_OPLs[i_tel] = hdul_cal[0].header[f'ESO DEL DLT{i_tel+1} OPL START']
            end_OPLs[i_tel] = hdul_cal[0].header[f'ESO DEL DLT{i_tel+1} OPL END'] 
        # Compute optical path lengths of individual frames with linear interpolation
        start_mjd, end_mjd = mjds[0], mjds[-1] + hdul_cal[0].header['EXPTIME']
        relative_mjds = (mjds_valid - start_mjd) / (end_mjd - start_mjd)
        OPLs = (np.outer(end_OPLs - start_OPLs, relative_mjds).T + start_OPLs).T
        # Total path difference (microns)
        deltaPaths = np.array([static_lengths[3] + OPLs[3] - static_lengths[2] - OPLs[2],
                               static_lengths[1] + OPLs[1] - static_lengths[0] - OPLs[0],
                               static_lengths[2] + OPLs[2] - static_lengths[1] - OPLs[1],
                               static_lengths[3] + OPLs[3] - static_lengths[1] - OPLs[1],
                               static_lengths[2] + OPLs[2] - static_lengths[0] - OPLs[0],
                               static_lengths[3] + OPLs[3] - static_lengths[0] - OPLs[0]])
        deltaPaths *= 1e6 # m -> microns

        ## Get the ambient conditions
        # Relative humidity
        humidity = hdul_cal[0].header['ESO ISS AMBI RHUM'] / 100
        # Temperature (°C)
        Tlab = hdul_cal[0].header['ESO ISS TEMP LAB1']
        T1 = hdul_cal[0].header['ESO ISS TEMP TUN1']
        T2 = hdul_cal[0].header['ESO ISS TEMP TUN2']
        T3 = hdul_cal[0].header['ESO ISS TEMP TUN3']
        T4 = hdul_cal[0].header['ESO ISS TEMP TUN4']
        temperature = (T1+T2+T3+T4)/4
        # Pressure (hPa)
        pressure = hdul_cal[0].header['ESO ISS AMBI PRES']

        # Open the associated MATISSE file and get the real and imaginary correlated fluxes
        data_real = fits.getdata(f'{obs_dir}/nrjReal_{expno}.fits')
        data_imag = fits.getdata(f'{obs_dir}/nrjImag_{expno}.fits')
                
        # Extract quantities
        wl_real_all  = data_real[:-1, 0]
        cf_real_all  = data_real[:-1, 1:-1]
        err_real_all = data_real[:-1, -1]

        wl_imag_all  = data_imag[:-1, 0]
        cf_imag_all  = data_imag[:-1, 1:-1]
        err_imag_all = data_imag[:-1, -1]

        # Dimensions
        n_frames, n_base, n_wave = cf_real_all.shape

        # Check if the number of frames corresponds to the fringe tracker TARTYP
        if n_frames != np.sum(tartyp == 'T'):
            print('Data dimensions and valid tartyp not matching!')
            print('Frames:', n_frames, 'Valid tartyp:', np.sum(tartyp == 'T'), 'Tartyp:', tartyp)
                
        # Reorder the baselines for this BCD configuration
        cf_real_all  = cf_real_all[:, bcd_base_reorder[bcd]]
        cf_imag_all  = cf_imag_all[:, bcd_base_reorder[bcd]]
                
        # Apply the signs associated to this BCD configuration
        cf_imag_all  = np.swapaxes(np.swapaxes(cf_imag_all, 1, 2) * bcd_sign[bcd], 1, 2)
                
        # Wavelength sanity check
        if np.all(wl_real_all == wl_imag_all):
            wl_all = wl_real_all
        else:
            raise ValueError('Wavelengths stored in the real and imaginary arrays are not equal.')
                
        # Complexify
        cf_all = cf_real_all + 1j * cf_imag_all

        # Compute the chromatic refractive index
        n_air_all = refractive_index(wl_all, T=temperature, P=pressure, h=humidity, N_CO2=417)

        # Loop on frames
        for i_frame in range(n_frames):

            print(obs_time, i_exp+1, i_frame+1)

            wl_full = wl_all[i_frame]
            err_real_full = err_real_all[i_frame]
            err_imag_full = err_imag_all[i_frame]
            n_air_full = n_air_all[i_frame]

            # Wavelength mask for OPD fitting
            wl_mask = (wl_full > wmin) & (wl_full < wmax)

            # Compute the UV coordinates
            # U, V = np.array(bcd_sign[bcd]) * compute_UV_coords(ra, dec, lat, long, mjds_valid[i_frame], sta_xyz, base_order)
            U, V = compute_UV_coords(ra, dec, lat, long, mjds_valid[i_frame], sta_xyz, base_order)

            # Get the initial guesses
            if use_previous_fit:
                i_mjd_prev = np.argwhere(mjds_prev_fit == mjds_valid[i_frame]).squeeze()
                opd_guesses = opds_prev_fit[i_mjd_prev, :]
                offset_guesses = offsets_prev_fit[i_mjd_prev, :]

            # Initialization
            cf_corr = np.zeros((n_base, n_wave), dtype=complex)
            amp_err_final = np.zeros((n_base, n_wave))
            phase_err_final = np.zeros((n_base, n_wave))
            opds_frame = np.zeros(n_base)
            offsets_frame = np.zeros(n_base)

            # Loop on baselines
            for i_base in range(n_base):

                # Data from full wavelength range
                cf_full = cf_all[i_frame, i_base]
                amp_full = np.abs(cf_full)
                amp_err_full = np.sqrt((np.real(cf_full)*err_real_full)**2 + (np.imag(cf_full)*err_imag_full)**2) / amp_full
                phase_full = np.angle(cf_full)
                phase_err_full = np.sqrt((np.imag(cf_full)*err_real_full)**2 + (np.real(cf_full)*err_imag_full)**2) / amp_full**2

                amp_err_final[i_base] = amp_err_full
                phase_err_final[i_base] = phase_err_full

                # First mask based on reduced wavelength range
                cf = cf_full[wl_mask]
                phase = phase_full[wl_mask]
                phase_err = phase_err_full[wl_mask]
                
                wl = wl_full[wl_mask]
                err_real = err_real_full[wl_mask]
                err_imag = err_imag_full[wl_mask]
                n_air = n_air_full[wl_mask]
    
                # Second mask based on data quality (exclude points with extremely low or high phase errors)
                phase_err_min, phase_err_max = np.quantile(phase_err, 0.02), np.quantile(phase_err, 0.98)
                mask_err = (phase_err > phase_err_min) & (phase_err < phase_err_max)

                wl = wl[mask_err]
                phase = phase[mask_err]
                phase_err = phase_err[mask_err]
                err_real = err_real[mask_err]
                err_imag = err_imag[mask_err]
                n_air = n_air[mask_err]
                
                if plot_data_selection:

                    ## Quality check plots
                    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                    ax1, ax2, ax3, ax4 = axes.flat

                    # Phase vs. wavelength
                    ax1.plot(wl_full, np.angle(np.exp(1j*phase_full)), '+', label='All data')
                    ax1.axvline(wmin, c='k', ls='--', lw=0.5)
                    ax1.axvline(wmax, c='k', ls='--', lw=0.5)
                    ax1.plot(wl, np.angle(np.exp(1j*phase)), '+', c='tab:orange', label='Selected data')
                    ax1.set_ylabel('Phase')
                    ax1.set_xlabel(r'Wavelength [$\mu$m]')
                    ax1.legend(loc='lower right')

                    # Phase error vs. wavelength
                    ax2.semilogy(wl_full, phase_err_full, '+', label='All data')
                    ax2.axvline(wmin, c='k', ls='--', lw=0.5)
                    ax2.axvline(wmax, c='k', ls='--', lw=0.5)
                    ax2.semilogy(wl, phase_err, '+', c='tab:orange', label='Selected data')
                    ax2.axhline(phase_err_min, ls='--', c='k', lw=0.5)
                    ax2.axhline(phase_err_max, ls='--', c='k', lw=0.5)
                    ax2.set_xlabel(r'Wavelength [$\mu$m]')
                    ax2.set_ylabel('Phase error')
                    ax2.legend(loc='lower right')

                    # Phase S/N vs. wavelength
                    ax3.semilogy(wl_full, np.abs(phase_full/phase_err_full), '+', label='All data')
                    ax3.axvline(wmin, c='k', ls='--', lw=0.5)
                    ax3.axvline(wmax, c='k', ls='--', lw=0.5)
                    ax3.semilogy(wl, np.abs(phase/phase_err), '+', c='tab:orange', label='Selected data')
                    ax3.set_xlabel(r'Wavelength [$\mu$m]')
                    ax3.set_ylabel('Phase S/N')
                    ax3.legend(loc='lower right')

                    # Phase vs. wavenumber
                    ax4.plot(1/wl_full, np.angle(np.exp(1j*phase_full)), '+', label='All data')
                    ax4.axvline(1/wmin, c='k', ls='--', lw=0.5)
                    ax4.axvline(1/wmax, c='k', ls='--', lw=0.5)
                    ax4.plot(1/wl, np.angle(np.exp(1j*phase)), '+', c='tab:orange', label='Selected data')
                    ax4.axvline(1/3.2, c='k', ls='--', lw=0.5)
                    ax4.axvline(1/3.8, c='k', ls='--', lw=0.5)
                    ax4.plot(1/wl[(wl>3.2)&(wl<3.8)], np.angle(np.exp(1j*phase[(wl>3.2)&(wl<3.8)])), '+', c='tab:green', label='Selected for OPD fitting')
                    ax4.set_xlabel(r'Wavenumber [$\mu$m$^{-1}$]')
                    ax4.set_ylabel('Phase')
                    ax4.legend(loc='lower left', fontsize='small')

                    fig.tight_layout()
                    fig.savefig(f'{path_results}/fitting_masks/fitting_mask_{obs_time}_exp{i_exp+1}_frame{i_frame+1}_base{i_base+1}.pdf')

                    plt.close(fig)

                ## Model fitting

                # Step 1: model the high-order variations with the chromatic refractive index and OPDs
                phase_ho_model = model_high_order(wl, n_air, deltaPaths[i_base, i_frame])

                # Remove this high-order model from the data
                # We are in principle only left with the linear residual OPD that was not corrected
                # by the fringe tracker + an offset.
                phase_corr = phase - phase_ho_model

                # Step 2: model the residual OPD and offset

                # Pre-fitting on a defined range. Helps to find good initial guesses for the chi2 minimization.
                opds_to_test = np.linspace(opd_min, opd_max, 50)
                offset = np.zeros_like(opds_to_test)
                chi2 = np.zeros_like(opds_to_test)
                for i_opd, opd_to_test in enumerate(opds_to_test):
                    model_without_offset = model_low_order((opd_to_test, 0), wl)
                    offset[i_opd] = np.angle(np.mean(np.exp(1j * (phase_corr - model_without_offset))))
                    model_with_offset = model_low_order((opd_to_test, offset[i_opd]), wl)
                    chi2[i_opd] = np.sum(wrap(phase_corr - model_with_offset)**2 / phase_err**2)
                i_chi2_min = np.argmin(chi2)
                opd_best_prefit = opds_to_test[i_chi2_min]
                offset_best_prefit = offset[i_chi2_min]

                # Chi-2 minimization
                initial_guess = (opd_best_prefit, offset_best_prefit)
                result = scipy.optimize.minimize(residuals, initial_guess, args=(wl, phase_corr, phase_err), bounds=bounds)
                opd_best, offset_best = result.x

                opds_frame[i_base] = opd_best
                offsets_frame[i_base] = wrap(offset_best)

                # Compute model on full wavelength range
                phase_ho_model_full = model_high_order(wl_full, n_air_full, deltaPaths[i_base, i_frame])
                phase_lo_model_full = model_low_order(result.x, wl_full)
                phase_model_full = phase_ho_model_full + phase_lo_model_full

                # Plot the phase data and fit
                fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
                fig.subplots_adjust(hspace=0)
                ax[0].plot(wl_full, phase_full, '+', label='Data')
                ax[0].plot(wl_full, wrap(phase_model_full), label='Best model')
                #ax[0].plot(wl_full, wrap(phase_lo_model_full), 'purple', label='Achromatic dispersion')
                ax[1].plot(wl_full, wrap(phase_full - phase_model_full), label='Residuals')
                
                # 10 degrees error level
                ''' 
                plt.axhline(-10*np.pi/180, c='k', ls='--', lw=0.5)
                plt.axhline(10*np.pi/180, c='k', ls='--', lw=0.5)
                plt.axhspan(-10*np.pi/180, 10*np.pi/180, alpha=0.2)
                # 3 degrees error level
                plt.axhline(-3*np.pi/180, c='k', ls='--', lw=0.5)
                plt.axhline(3*np.pi/180, c='k', ls='--', lw=0.5)
                plt.axhspan(-3*np.pi/180, 3*np.pi/180, alpha=0.3)
                # Wavelength limits
                plt.axvline(wmin, c='k', ls='--', lw=0.5)
                plt.axvline(wmax, c='k', ls='--', lw=0.5)
                '''
                plt.legend()
                fig.tight_layout()
                fig.savefig(f"{path_results}/best_fits/best_fit_{date_obs}_OB{i_OB+1}_exp{chronological_expno+1}_frame{i_frame+1}_base{i_base+1}_{target}{flag}.pdf")
                plt.close()
                

                ## Phase correction
                cf_corr[i_base] = cf_full * np.exp(-1j * phase_model_full)
                
            ## Compute final interferometric quantities 
            cf_corr_amp = np.abs(cf_corr)
            cf_corr_phase = np.angle(cf_corr)    
            # Closure phases:
            # base_order = [(3,4), (1,2), (2,3), (2,4), (1,3), (1,4)]
            # 123 -> 1*2*conj4, 124 -> 1*3*conj0, 134 -> 4*0*conj5, 234 -> 2*0*conj3
            t3 = np.array([cf_corr[1] * cf_corr[2] * np.conj(cf_corr[4]),
                           cf_corr[1] * cf_corr[3] * np.conj(cf_corr[0]),
                           cf_corr[4] * cf_corr[0] * np.conj(cf_corr[5]),
                           cf_corr[2] * cf_corr[0] * np.conj(cf_corr[3])])
            t3_amp = np.abs(t3)
            t3_phase = np.angle(t3)
            t3_amp_err = np.sqrt(np.array([(cf_corr_amp[1]*cf_corr_amp[2]*amp_err_final[4])**2 + (cf_corr_amp[1]*cf_corr_amp[4]*amp_err_final[2])**2 + (cf_corr_amp[2]*cf_corr_amp[4]*amp_err_final[1])**2,
                                           (cf_corr_amp[1]*cf_corr_amp[3]*amp_err_final[0])**2 + (cf_corr_amp[1]*cf_corr_amp[0]*amp_err_final[3])**2 + (cf_corr_amp[0]*cf_corr_amp[3]*amp_err_final[1])**2,
                                           (cf_corr_amp[4]*cf_corr_amp[0]*amp_err_final[5])**2 + (cf_corr_amp[4]*cf_corr_amp[5]*amp_err_final[0])**2 + (cf_corr_amp[0]*cf_corr_amp[5]*amp_err_final[4])**2,
                                           (cf_corr_amp[2]*cf_corr_amp[0]*amp_err_final[3])**2 + (cf_corr_amp[2]*cf_corr_amp[3]*amp_err_final[0])**2 + (cf_corr_amp[3]*cf_corr_amp[0]*amp_err_final[2])**2]))
            t3_phase_err = np.sqrt(np.array([phase_err_final[1]**2 + phase_err_final[2]**2 + phase_err_final[4]**2,
                                             phase_err_final[1]**2 + phase_err_final[3]**2 + phase_err_final[0]**2,
                                             phase_err_final[4]**2 + phase_err_final[0]**2 + phase_err_final[5]**2,
                                             phase_err_final[2]**2 + phase_err_final[0]**2 + phase_err_final[3]**2]))
            U1 = np.array([U[1], U[1], U[4], U[2]])
            V1 = np.array([V[1], V[1], V[4], V[2]])
            U2 = np.array([U[2], U[3], U[0], U[0]])
            V2 = np.array([V[2], V[3], V[0], V[0]])

            ## Create the corrected OIFITS

            # Copy some extensions from the RAW_VIS2 files
            hdul_rawvis2 = fits.open(f'{obs_dir}/RAW_VIS2_{np.argwhere(expno_raw_vis2 == expno)[0, 0]+1:04d}.fits')
            
            # Primary extension (main header)
            hdu0 = fits.PrimaryHDU(header=hdul_rawvis2[0].header)
            for i_base_opd, opd in enumerate(opds_frame):
                hdu0.header[f'HIERARCH FIT OPD{i_base_opd+1}'] = (opds_frame[i_base_opd], f'OPD fit in base {i_base_opd+1} [microns]')
                hdu0.header[f'HIERARCH FIT OFFSET{i_base_opd+1}'] = (offsets_frame[i_base_opd], f'Phase offset fit in base {i_base_opd+1} [rad]')

            # OI_TARGET and OI_ARRAY
            hdu_target = fits.BinTableHDU(name='OI_TARGET', data=hdul_rawvis2['OI_TARGET'].data, header=hdul_rawvis2['OI_TARGET'].header)
            hdu_array = fits.BinTableHDU(name='OI_ARRAY', data=hdul_rawvis2['OI_ARRAY'].data, header=hdul_rawvis2['OI_ARRAY'].header)

            # OI_WAVELENGTH
            wl_col = fits.Column(name='EFF_WAVE', array=wl_full*1e-6, unit='m', format='D')
            wband_col = fits.Column(name='EFF_BAND', array=np.full_like(wl_full, 1.4261867e-9), unit='m', format='D')
            hdu_wave = fits.BinTableHDU.from_columns([wl_col, wband_col], name='OI_WAVELENGTH')

            # OI_VIS
            visamp_col = fits.Column(name='VISAMP', array=cf_corr_amp, format=f'{n_wave}D')
            visphi_col = fits.Column(name='VISPHI', array=np.rad2deg(cf_corr_phase), unit='deg', format=f'{n_wave}D')
            visamperr_col = fits.Column(name='VISAMPERR', array=amp_err_final, format=f'{n_wave}D')
            visphierr_col = fits.Column(name='VISPHIERR', array=np.rad2deg(phase_err_final), unit='deg', format=f'{n_wave}D')
            ucoord_col = fits.Column(name='UCOORD', array=U, unit='m', format='D')
            vcoord_col = fits.Column(name='VCOORD', array=V, unit='m', format='D')
            mjd_col = fits.Column(name='MJD', array=np.full(6, mjds_valid[i_frame]), format='D')
            inttime_col = fits.Column(name='INT_TIME', array=np.full(6, hdul_cal[0].header['EXPTIME']), unit='s', format='D')
            staindex_col = fits.Column(name='STA_INDEX', array=np.array(base_order)+31, format='2I')
            targetid_col = fits.Column(name='TARGET_ID', array=np.full(6, 1), format='I')
            hdu_vis = fits.BinTableHDU.from_columns([targetid_col, mjd_col, inttime_col, visamp_col, visphi_col, 
                                                     visamperr_col, visphierr_col, ucoord_col, vcoord_col, 
                                                     staindex_col], name='OI_VIS')

            # OI_T3
            t3amp_col = fits.Column(name='T3AMP', array=t3_amp, format=f'{n_wave}D')
            t3phi_col = fits.Column(name='T3PHI', array=np.rad2deg(t3_phase), unit='deg', format=f'{n_wave}D')
            t3amperr_col = fits.Column(name='T3AMPERR', array=t3_amp_err, format=f'{n_wave}D')
            t3phierr_col = fits.Column(name='T3PHIERR', array=np.rad2deg(t3_phase_err), unit='deg', format=f'{n_wave}D')
            u1coord_col = fits.Column(name='U1COORD', array=U1, unit='m', format='D')
            u2coord_col = fits.Column(name='U2COORD', array=U2, unit='m', format='D')
            v1coord_col = fits.Column(name='V1COORD', array=V1, unit='m', format='D')
            v2coord_col = fits.Column(name='V2COORD', array=V2, unit='m', format='D')
            mjd_col = fits.Column(name='MJD', array=np.full(4, mjds_valid[i_frame]), format='D')
            inttime_col = fits.Column(name='INT_TIME', array=np.full(4, hdul_cal[0].header['EXPTIME']), unit='s', format='D')
            staindex_t3_col = fits.Column(name='STA_INDEX', array=np.array(t3_order)+31, format='3I')
            targetid_col = fits.Column(name='TARGET_ID', array=np.full(4, 1), format='I')
            hdu_t3 = fits.BinTableHDU.from_columns([targetid_col, mjd_col, inttime_col, t3amp_col, t3phi_col,
                                                    t3amperr_col, t3phierr_col, u1coord_col, u2coord_col,
                                                    v1coord_col, v2coord_col, staindex_t3_col], name='OI_T3')
            
            # Add mandatory keywords
            for hdu_i in (hdu_target, hdu_array, hdu_wave, hdu_vis, hdu_t3):
                hdu_i.header['EXTVER'] = 1
                hdu_i.header['OI_REVN'] = 2
            for hdu_i in (hdu_vis, hdu_t3):
                hdu_i.header['DATE-OBS'] = hdu0.header['DATE-OBS']
            for hdu_i in (hdu_array, hdu_vis, hdu_t3):
                hdu_i.header['ARRNAME'] = 'VLTI'
            for hdu_i in (hdu_wave, hdu_vis, hdu_t3):
                hdu_i.header['INSNAME'] = 'MATISSE'
            hdu_vis.header['AMPTYP'] = 'correlated flux'
            hdu_vis.header['PHITYP'] = 'differential'

            # Stack in a HDU list and save
            hdul = fits.HDUList([hdu0, hdu_target, hdu_array, hdu_wave, hdu_vis, hdu_t3])
            #hdul.writeto(f'{path_results}/corrected_data/{obs_time}_exp{i_exp+1}_frame{i_frame+1}.fits', overwrite=True)
            hdul.writeto(f"{path_results}/corrected_data/{date_obs}_OB{i_OB+1}_exp{chronological_expno+1}_frame{i_frame+1}_{target}{flag}.fits", overwrite=True)

            hdul.close()
            hdul_rawvis2.close()
        
        hdul_cal.close()

# Make some plots and save OPD and offsets
        
all_corr_files = [file for file in os.listdir(f'{path_results}/corrected_data/')]
fig1, axes1 = plt.subplots(6, 1, figsize=(10, 6))
axes1=axes1.flat
fig2, axes2 = plt.subplots(6, 1, figsize=(10, 6))
axes2=axes2.flat

mjds = np.zeros(len(all_corr_files))
opds = np.zeros((len(all_corr_files), 6))
offsets = np.zeros((len(all_corr_files), 6))
color_BCD = {'IN-IN': 'tab:blue', 'IN-OUT':'tab:orange', 'OUT-IN':'tab:green', 'OUT-OUT':'tab:red'}

for i_file, file in enumerate(all_corr_files):
    hdu = fits.open(f'{path_results}/corrected_data/{file}')
    mjd = hdu['OI_VIS'].data['MJD']
    mjds[i_file] = mjd[0]
    # Get the BCD configurations  
    bcd1 = hdu[0].header['ESO INS BCD1 NAME']
    bcd2 = hdu[0].header['ESO INS BCD2 NAME']
    bcd = f'{bcd1}-{bcd2}'
    # Plot the OPDs and offsets
    for i in range(6):
        opd = hdu[0].header[f'FIT OPD{i+1}']
        offset = hdu[0].header[f'FIT OFFSET{i+1}']
        axes1[i].plot(mjd[i], opd, '+', c=color_BCD[bcd])
        axes2[i].plot(mjd[i], np.rad2deg(offset), '+')
        opds[i_file, i] = opd
        offsets[i_file, i] = offset

for bcd in ('IN-IN', 'IN-OUT', 'OUT-IN', 'OUT-OUT'):
    axes1[i].plot([], [], '+', c=color_BCD[bcd], label=bcd)
fig1.legend()

for i in range(6):
    axes1[i].set_ylabel(r'OPD [$\mu$m]')
    axes2[i].set_ylabel('Offset [rad]')
    if i != 5:
        axes1[i].set_xticklabels([])
        axes2[i].set_xticklabels([])
axes1[-1].set_xlabel('MJD')
axes2[-1].set_xlabel('MJD')
fig1.tight_layout()
fig1.savefig(f'{path_results}/OPD_fit.pdf')
fig2.tight_layout()
fig2.savefig(f'{path_results}/offset_fit.pdf')

i_mjd_sort = np.argsort(mjds)
mjds_sort = mjds[i_mjd_sort]
opds_sort = opds[i_mjd_sort, :]
offsets_sort = offsets[i_mjd_sort, :]

opds_smooth = np.zeros_like(opds_sort)
offsets_smooth = np.zeros_like(opds_sort)

for i_mjd in range(mjds.size):
    i_avg = [*np.arange(np.max((0, i_mjd-20)), i_mjd), *np.arange(i_mjd+1, np.min((mjds_sort.size, i_mjd+21)))]
    opds_smooth[i_mjd, :] = np.mean(opds_sort[i_avg, :], axis=0)
    offsets_smooth[i_mjd, :] = np.arctan2(np.mean(np.sin(offsets_sort[i_avg, :]), axis=0), np.mean(np.cos(offsets_sort[i_avg, :]), axis=0))

fits.writeto(f'{path_results}/mjds.fits', mjds_sort, overwrite=True)
fits.writeto(f'{path_results}/opd_fit.fits', opds_sort, overwrite=True)
fits.writeto(f'{path_results}/opd_fit_smooth.fits', opds_smooth, overwrite=True)
fits.writeto(f'{path_results}/offset_fit.fits', offsets_sort, overwrite=True)
fits.writeto(f'{path_results}/offset_fit_smooth.fits', offsets_smooth, overwrite=True)

plt.figure()
plt.plot(mjds_sort, opds_smooth, 'x')
plt.xlabel('MJD')
plt.ylabel(r'OPD [$\mu$m]')
plt.savefig(f'{path_results}/OPD_fit_smooth.pdf')
plt.figure()
plt.plot(mjds_sort, np.rad2deg(offsets_smooth), 'x')
plt.xlabel('MJD')
plt.ylabel('Offset [deg]')
plt.savefig(f'{path_results}/offset_fit_smooth.pdf')