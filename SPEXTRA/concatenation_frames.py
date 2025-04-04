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
path_results = '/store/projects/MATISSE/jscigliuto/HR8799e_phase_corrected_concatenated'

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

####################################################################

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
        
        # Integrate over an exposure (all frames)
        cf_real_all = np.mean(cf_real_all, axis=0)
        cf_imag_all = np.mean(cf_imag_all, axis=0)
        err_real_all = np.sqrt(np.sum(err_real_all**2, axis=0))/n_frames
        err_imag_all = np.sqrt(np.sum(err_imag_all**2, axis=0))/n_frames
                
        # Complexify
        cf_all = cf_real_all + 1j * cf_imag_all
        cf_all_amp = np.abs(cf_all)
        cf_all_phi = np.angle(cf_all)
        amp_err_full = np.sqrt((cf_real_all*err_real_all)**2 + (cf_imag_all*err_imag_all)**2) / cf_all_amp
        phase_err_full = np.sqrt((cf_imag_all*err_real_all)**2 + (cf_real_all*err_imag_all)**2) / cf_all_amp**2

        
        # Loop on frames
        U_exp, V_exp = np.zeros((n_frames, n_base)), np.zeros((n_frames, n_base))
        for i_frame in range(n_frames):

            print(obs_time, i_exp+1, i_frame+1)

            wl_full = wl_all[i_frame]

            # Compute the UV coordinates
            U_exp[i_frame], V_exp[i_frame] = compute_UV_coords(ra, dec, lat, long, mjds_valid[i_frame], sta_xyz, base_order)

        U_exp_mean, V_exp_mean = np.mean(U_exp, axis=0), np.mean(V_exp, axis=0)

        ## Create the corrected OIFITS

        # Copy some extensions from the RAW_VIS2 files
        hdul_rawvis2 = fits.open(f'{obs_dir}/RAW_VIS2_{np.argwhere(expno_raw_vis2 == expno)[0, 0]+1:04d}.fits')

        # Primary extension (main header)
        hdu0 = fits.PrimaryHDU(header=hdul_rawvis2[0].header)

        # OI_TARGET and OI_ARRAY
        hdu_target = fits.BinTableHDU(name='OI_TARGET', data=hdul_rawvis2['OI_TARGET'].data, header=hdul_rawvis2['OI_TARGET'].header)
        hdu_array = fits.BinTableHDU(name='OI_ARRAY', data=hdul_rawvis2['OI_ARRAY'].data, header=hdul_rawvis2['OI_ARRAY'].header)

        # OI_WAVELENGTH
        wl_col = fits.Column(name='EFF_WAVE', array=wl_full*1e-6, unit='m', format='D')
        wband_col = fits.Column(name='EFF_BAND', array=np.full_like(wl_full, 1.4261867e-9), unit='m', format='D')
        hdu_wave = fits.BinTableHDU.from_columns([wl_col, wband_col], name='OI_WAVELENGTH')

        # OI_VIS
        visamp_col = fits.Column(name='VISAMP', array=cf_all_amp, format=f'{n_wave}D')
        visphi_col = fits.Column(name='VISPHI', array=np.rad2deg(cf_all_phi), unit='deg', format=f'{n_wave}D')
        visamperr_col = fits.Column(name='VISAMPERR', array=amp_err_full, format=f'{n_wave}D')
        visphierr_col = fits.Column(name='VISPHIERR', array=np.rad2deg(phase_err_full), unit='deg', format=f'{n_wave}D')
        ucoord_col = fits.Column(name='UCOORD', array=U_exp_mean, unit='m', format='D')
        vcoord_col = fits.Column(name='VCOORD', array=V_exp_mean, unit='m', format='D')
        mjd_col = fits.Column(name='MJD', array=np.full(6, np.mean(mjds_valid)), format='D')
        inttime_col = fits.Column(name='INT_TIME', array=np.full(6, hdul_cal[0].header['EXPTIME']), unit='s', format='D')
        staindex_col = fits.Column(name='STA_INDEX', array=np.array(base_order)+31, format='2I')
        targetid_col = fits.Column(name='TARGET_ID', array=np.full(6, 1), format='I')
        hdu_vis = fits.BinTableHDU.from_columns([targetid_col, mjd_col, inttime_col, visamp_col, visphi_col, 
                                                visamperr_col, visphierr_col, ucoord_col, vcoord_col, 
                                                staindex_col], name='OI_VIS')

        # Add mandatory keywords
        for hdu_i in (hdu_target, hdu_array, hdu_wave, hdu_vis):
            hdu_i.header['EXTVER'] = 1
            hdu_i.header['OI_REVN'] = 2
        hdu_vis.header['DATE-OBS'] = hdu0.header['DATE-OBS']
        for hdu_i in (hdu_array, hdu_vis):
            hdu_i.header['ARRNAME'] = 'VLTI'
        for hdu_i in (hdu_wave, hdu_vis):
            hdu_i.header['INSNAME'] = 'MATISSE'
        hdu_vis.header['AMPTYP'] = 'correlated flux'
        hdu_vis.header['PHITYP'] = 'differential'

        # Stack in a HDU list and save
        hdul = fits.HDUList([hdu0, hdu_target, hdu_array, hdu_wave, hdu_vis])
        hdul.writeto(f"{path_results}/{date_obs}_OB{i_OB+1}_exp{chronological_expno+1}_{target}{flag}.fits", overwrite=True)

        hdul.close()
        hdul_rawvis2.close()

    hdul_cal.close()

