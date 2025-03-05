#!/usr/bin/env python
# -*- coding: utf-8 -*-

#################################################################################################

# Bin the OIFITS data by chunks of 5 spectral channels.

# Requires:
# - A directory with all reduced frame files from phase_correction.py

# Authors:
#   M. Houllé

#################################################################################################

import numpy as np
import os
from astropy.io import fits

#################################################################################################

# Inputs

path_oifits = '/data/home/jscigliuto/Pipeline/corrPhase1/'

#################################################################################################

path_oifits_corrected = path_oifits 

if not os.path.isdir(path_oifits + '/corrected_data_5bin/'):
    os.makedirs(path_oifits + '/corrected_data_5bin/')

files = sorted(os.listdir(path_oifits_corrected))

for file in files:
    print(file)

    hdul = fits.open(path_oifits_corrected + file)

    # Extract quantities
    cf_amp     = hdul['OI_VIS'].data['VISAMP']
    cf_amp_err = hdul['OI_VIS'].data['VISAMPERR']
    cf_phi     = np.deg2rad(hdul['OI_VIS'].data['VISPHI'])
    cf_phi_err = np.deg2rad(hdul['OI_VIS'].data['VISPHIERR'])
    t3_amp     = hdul['OI_T3'].data['T3AMP']
    t3_amp_err = hdul['OI_T3'].data['T3AMPERR']
    t3_phi     = np.deg2rad(hdul['OI_T3'].data['T3PHI'])
    t3_phi_err = np.deg2rad(hdul['OI_T3'].data['T3PHIERR'])
    wl         = hdul['OI_WAVELENGTH'].data['EFF_WAVE']

    U          = hdul['OI_VIS'].data['UCOORD']
    V          = hdul['OI_VIS'].data['VCOORD']
    U1         = hdul['OI_T3'].data['U1COORD']
    U2         = hdul['OI_T3'].data['U2COORD']
    V1         = hdul['OI_T3'].data['V1COORD']
    V2         = hdul['OI_T3'].data['V2COORD']

    mjd        = hdul['OI_VIS'].data['MJD']
    mjd_t3     = hdul['OI_T3'].data['MJD']
    int_time   = hdul['OI_VIS'].data['INT_TIME']
    int_time_t3 = hdul['OI_T3'].data['INT_TIME']
    sta_index  = hdul['OI_VIS'].data['STA_INDEX']
    sta_index_t3 = hdul['OI_T3'].data['STA_INDEX']
    target_id  = hdul['OI_VIS'].data['TARGET_ID']
    target_id_t3 = hdul['OI_T3'].data['TARGET_ID']

    n_base = cf_amp.shape[0]
    n_triangle = t3_amp.shape[0]

    # Complexify
    cf = cf_amp * np.exp(1j * cf_phi)
    cf_real_err = np.sqrt((np.cos(cf_phi) * cf_amp_err)**2 + (cf_amp * np.sin(cf_phi) * cf_phi_err)**2)
    cf_imag_err = np.sqrt((np.sin(cf_phi) * cf_amp_err)**2 + (cf_amp * np.cos(cf_phi) * cf_phi_err)**2)

    t3 = t3_amp * np.exp(1j * t3_phi)
    t3_real_err = np.sqrt((np.cos(t3_phi) * t3_amp_err)**2 + (t3_amp * np.sin(t3_phi) * t3_phi_err)**2)
    t3_imag_err = np.sqrt((np.sin(t3_phi) * t3_amp_err)**2 + (t3_amp * np.cos(t3_phi) * t3_phi_err)**2)

    # Bin
    wl = wl.reshape(-1, 5).mean(axis=1)
    
    cf_real = np.real(cf).reshape(n_base, -1, 5).mean(axis=2)
    cf_imag = np.imag(cf).reshape(n_base, -1, 5).mean(axis=2)
    cf_real_err = np.sqrt((cf_real_err**2).reshape(n_base, -1, 5).mean(axis=2))
    cf_imag_err = np.sqrt((cf_imag_err**2).reshape(n_base, -1, 5).mean(axis=2))
    
    cf = cf_real + 1j * cf_imag
    cf_amp = np.abs(cf)
    cf_phi = np.angle(cf)
    cf_amp_err = np.sqrt((cf_real*cf_real_err)**2 + (cf_imag*cf_imag_err)**2) / cf_amp
    cf_phi_err = np.sqrt((cf_imag*cf_real_err)**2 + (cf_real*cf_imag_err)**2) / cf_amp**2

    t3_real = np.real(t3).reshape(n_triangle, -1, 5).mean(axis=2)
    t3_imag = np.imag(t3).reshape(n_triangle, -1, 5).mean(axis=2)
    t3_real_err = np.sqrt((t3_real_err**2).reshape(n_triangle, -1, 5).mean(axis=2))
    t3_imag_err = np.sqrt((t3_imag_err**2).reshape(n_triangle, -1, 5).mean(axis=2))
    
    t3 = t3_real + 1j * t3_imag
    t3_amp = np.abs(t3)
    t3_phi = np.angle(t3)
    t3_amp_err = np.sqrt((t3_real*t3_real_err)**2 + (t3_imag*t3_imag_err)**2) / t3_amp
    t3_phi_err = np.sqrt((t3_imag*t3_real_err)**2 + (t3_real*t3_imag_err)**2) / t3_amp**2

    n_wave = wl.size

    ## Create the corrected OIFITS

    hdu0 = fits.PrimaryHDU(header=hdul[0].header)

    hdu_target = fits.BinTableHDU(name='OI_TARGET', data=hdul['OI_TARGET'].data, header=hdul['OI_TARGET'].header)
    hdu_array = fits.BinTableHDU(name='OI_ARRAY', data=hdul['OI_ARRAY'].data, header=hdul['OI_ARRAY'].header)

    wl_col = fits.Column(name='EFF_WAVE', array=wl, unit='m', format='D')
    wband_col = fits.Column(name='EFF_BAND', array=np.full_like(wl, 1.4261867e-9*5), unit='m', format='D')
    hdu_wave = fits.BinTableHDU.from_columns([wl_col, wband_col], name='OI_WAVELENGTH')

    visamp_col = fits.Column(name='VISAMP', array=cf_amp, format=f'{n_wave}D')
    visphi_col = fits.Column(name='VISPHI', array=np.rad2deg(cf_phi), unit='deg', format=f'{n_wave}D')
    visamperr_col = fits.Column(name='VISAMPERR', array=cf_amp_err, format=f'{n_wave}D')
    visphierr_col = fits.Column(name='VISPHIERR', array=np.rad2deg(cf_phi_err), unit='deg', format=f'{n_wave}D')
    ucoord_col = fits.Column(name='UCOORD', array=U, unit='m', format='D')
    vcoord_col = fits.Column(name='VCOORD', array=V, unit='m', format='D')
    mjd_col = fits.Column(name='MJD', array=mjd, format='D')
    inttime_col = fits.Column(name='INT_TIME', array=int_time, unit='s', format='D')
    staindex_col = fits.Column(name='STA_INDEX', array=sta_index, format='2I')
    targetid_col = fits.Column(name='TARGET_ID', array=target_id, format='I')
    hdu_vis = fits.BinTableHDU.from_columns([targetid_col, mjd_col, inttime_col, visamp_col, visphi_col, 
                                                visamperr_col, visphierr_col, ucoord_col, vcoord_col, 
                                                staindex_col], name='OI_VIS')

    t3amp_col = fits.Column(name='T3AMP', array=t3_amp, format=f'{n_wave}D')
    t3phi_col = fits.Column(name='T3PHI', array=np.rad2deg(t3_phi), unit='deg', format=f'{n_wave}D')
    t3amperr_col = fits.Column(name='T3AMPERR', array=t3_amp_err, format=f'{n_wave}D')
    t3phierr_col = fits.Column(name='T3PHIERR', array=np.rad2deg(t3_phi_err), unit='deg', format=f'{n_wave}D')
    u1coord_col = fits.Column(name='U1COORD', array=U1, unit='m', format='D')
    u2coord_col = fits.Column(name='U2COORD', array=U2, unit='m', format='D')
    v1coord_col = fits.Column(name='V1COORD', array=V1, unit='m', format='D')
    v2coord_col = fits.Column(name='V2COORD', array=V2, unit='m', format='D')
    mjd_t3_col = fits.Column(name='MJD', array=mjd_t3, format='D')
    inttime_t3_col = fits.Column(name='INT_TIME', array=int_time_t3, unit='s', format='D')
    staindex_t3_col = fits.Column(name='STA_INDEX', array=sta_index_t3, format='3I')
    targetid_t3_col = fits.Column(name='TARGET_ID', array=target_id_t3, format='I')
    hdu_t3 = fits.BinTableHDU.from_columns([targetid_t3_col, mjd_t3_col, inttime_t3_col, t3amp_col, t3phi_col,
                                            t3amperr_col, t3phierr_col, u1coord_col, u2coord_col,
                                            v1coord_col, v2coord_col, staindex_t3_col], name='OI_T3')

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

    hdul_bin = fits.HDUList([hdu0, hdu_target, hdu_array, hdu_wave, hdu_vis, hdu_t3])
    #hdul.writeto(f'{path_results}/corrected_data/{obs_time}_exp{i_exp+1}_frame{i_frame+1}.fits', overwrite=True)
    hdul_bin.writeto(f"{path_oifits}/corrected_data_5bin/{file[:file.find('.fits')]}_bin.fits", overwrite=True)

    hdul.close()