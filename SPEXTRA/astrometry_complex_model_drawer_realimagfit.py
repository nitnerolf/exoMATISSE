#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Plot the model over the data and the chi2 maps.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, sys

from astropy.io import fits
from numpy.polynomial import Polynomial

from common_tools import wrap, reorder_baselines, get_all_baseline_names, mas2rad

matplotlib.use('Agg')
plt.rcParams["figure.dpi"] = 100
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


### Parameters ###

# Input OIFITS directory (be careful to use the same OIFITS you used on the cluster)
path_oifits = '/data/home/jscigliuto/Pipeline/corrPhase0_MACAO/'

# Output directory
path_output = '/data/home/jscigliuto/Pipeline/Result_betPic0_MACAO/'

use_bin_data = False
# Selected wavelength range for displaying the results
# Can be different from the wavelength range selected for fitting
wmin, wmax = 3.0, 4.1 # microns
#wmin, wmax = 2.7, 5.0 # microns

# SkyCalc radiance and transmission spectra (not used yet)
#skycalc_radiance_file = 'skycalc_radiance_all.txt'
#skycalc_transmission_file = 'skycalc_radiance_all.txt'

# Telescope pointing relative to the star
#x_pointing, y_pointing = 280, 455 # mas
#x_pointing, y_pointing = 290, 440 # mas

# Selected grid of delta(alpha, delta) astrometries to test
#x = np.arange(250, 300, 1) # mas
#y = np.arange(425, 475, 1) # mas
# x = np.arange(280-10, 280+11, 0.1) # mas
# y = np.arange(455-10, 455+11, 0.1) # mas  
x = np.arange(279-15, 279+15, 0.4) # mas
y = np.arange(455-15, 455+15, 0.4) # mas
# x = np.arange(280-65, 280+66, 1) # mas
# y = np.arange(455-65, 455+66, 1) # mas

# Hypothesis on the planet-to-star contrast (not fitted with the astrometry)
Cps_file = '/data/home/jscigliuto/Pipeline/Templates/contrast_template_bt-settl_startpl.fits'
#Cps_file = 'flat 7e-4'
#Cps = 7e-4

# Get the errors on the data from the pipeline or from the data itself
error_type = 'errors_from_pipeline' # 'errors_from_data'


### Main program ###

# def model(wl, transmission_ratio, stellar_residuals_poly_coeffs, Cps, freqProj_iB):
#     # Compute the fitted polynomial stellar residual contrast
#     Css = Polynomial(stellar_residuals_poly_coeffs)(wl)
#     model = transmission_ratio * Cps + Css * np.exp(2j * np.pi * freqProj_iB)
#     amp_model = np.abs(model)
#     phi_model = np.angle(model)
#     return amp_model, phi_model

# def model(wl, transmission_ratio, stellar_residuals_poly_coeffs, Cps, freqProj_iB):
#     # Compute the fitted polynomial stellar residual contrast
#     Css = Polynomial(stellar_residuals_poly_coeffs)(wl)
#     model =  Cps + Css * np.exp(2j * np.pi * freqProj_iB)
#     amp_model = np.abs(model)
#     phi_model = np.angle(model)
#     return amp_model, phi_model

# transmission_ratio = []

def model(wl, transmission_ratio, stellar_residuals_poly_coeffs, Cps, freqProj_iB):
    # Compute the fitted polynomial stellar residual contrast
    n_coeffs = stellar_residuals_poly_coeffs.size // 2
    Css_real = Polynomial(stellar_residuals_poly_coeffs[:n_coeffs])(wl)
    Css_imag = Polynomial(stellar_residuals_poly_coeffs[n_coeffs:])(wl)
    model = transmission_ratio * Cps + Css_real * np.cos(2 * np.pi * freqProj_iB) + 1j * Css_imag * np.sin(2 * np.pi * freqProj_iB)
    amp_model = np.abs(model)
    phi_model = np.angle(model)
    return amp_model, phi_model

# def model(transmission_ratio, stellar_residuals_poly_coeffs, Cps, freqProj_iB):
#     Css = Polynomial(stellar_residuals_poly_coeffs)(wl)
#     model = transmission_ratio * Cps + Css * np.exp(2j * np.pi * freqProj_iB)
#     amp_model = np.abs(model)
#     phi_model = np.angle(model)
#     return amp_model, phi_model

# Search files
oifits_files = sorted([file for file in os.listdir(path_oifits) if '.fits' in file and 'flagged' not in file])
results_files = sorted([file[:-16] for file in os.listdir(path_output + 'astrometry_fits_files/') if '_fit_params.fits' in file])

# Create output directory if necessary
if not os.path.os.path.isdir(path_output):
    raise ValueError("Results path not found.")
if not os.path.isdir(path_output + 'astrometry_figures'):
    os.makedirs(path_output + 'astrometry_figures')

if not os.path.isdir(path_output + 'astrometry_figures/amp_model'):
    os.makedirs(path_output + 'astrometry_figures/amp_model')
if not os.path.isdir(path_output + 'astrometry_figures/amp_model_allbase'):
    os.makedirs(path_output + 'astrometry_figures/amp_model_allbase')
if not os.path.isdir(path_output + 'astrometry_figures/amp_model_allframes'):
    os.makedirs(path_output + 'astrometry_figures/amp_model_allframes')

if not os.path.isdir(path_output + 'astrometry_figures/phase_model'):
    os.makedirs(path_output + 'astrometry_figures/phase_model')
if not os.path.isdir(path_output + 'astrometry_figures/phase_model_allbase'):
    os.makedirs(path_output + 'astrometry_figures/phase_model_allbase')
if not os.path.isdir(path_output + 'astrometry_figures/phase_model_allframes'):
    os.makedirs(path_output + 'astrometry_figures/phase_model_allframes')

if not os.path.isdir(path_output + 'astrometry_figures/real_model_allframes'):
    os.makedirs(path_output + 'astrometry_figures/real_model_allframes')
if not os.path.isdir(path_output + 'astrometry_figures/imag_model_allframes'):
    os.makedirs(path_output + 'astrometry_figures/imag_model_allframes')

if not os.path.isdir(path_output + 'astrometry_figures/t3phi_model_allbase'):
    os.makedirs(path_output + 'astrometry_figures/t3phi_model_allbase')
if not os.path.isdir(path_output + 'astrometry_figures/t3phi_model_allframes'):
    os.makedirs(path_output + 'astrometry_figures/t3phi_model_allframes')

if not os.path.isdir(path_output + 'astrometry_figures/chi2_maps_allbase'):
    os.makedirs(path_output + 'astrometry_figures/chi2_maps_allbase')
if not os.path.isdir(path_output + 'astrometry_figures/chi2_maps_amp_allbase'):
    os.makedirs(path_output + 'astrometry_figures/chi2_maps_amp_allbase')
if not os.path.isdir(path_output + 'astrometry_figures/chi2_maps_phi_allbase'):
    os.makedirs(path_output + 'astrometry_figures/chi2_maps_phi_allbase')

if not os.path.isdir(path_output + 'astrometry_figures/fitted_polynomials_allframe'):
    os.makedirs(path_output + 'astrometry_figures/fitted_polynomials_allframe')
if not os.path.isdir(path_output + 'astrometry_figures/relativediff_fitted_polynomials_allframe'):
    os.makedirs(path_output + 'astrometry_figures/relativediff_fitted_polynomials_allframe')

# Baseline names
base_name = get_all_baseline_names()

# BCD names
#bcd_names = ('IN_IN', 'OUT_IN', 'IN_OUT', 'OUT_OUT')

# Grid of tested delta(alpha, delta) offsets from the star (mas)
xx, yy = np.meshgrid(x, y)

# Associated PAs and separations
pa = np.arctan2(xx, yy)
sep = np.sqrt(xx**2 + yy**2)

# Initialize output chi2 maps (allframe: total chi2 of all baselines + frames)
chi2_map_allframe = np.zeros((x.size, y.size))
chi2_map_amp_allframe = np.zeros((x.size, y.size))
chi2_map_phi_allframe = np.zeros((x.size, y.size))

# First run to find the best astrometry from the all-frames minimal chi2
for i, file in enumerate(results_files):
    # Co-add the chi2 maps of all files
    chi2_map = fits.getdata(path_output + f'astrometry_fits_files/{file}_chi2_map_allbase.fits')
    chi2_map_amp = fits.getdata(path_output + f'astrometry_fits_files/{file}_chi2_map_real_allbase.fits')
    chi2_map_phi = fits.getdata(path_output + f'astrometry_fits_files/{file}_chi2_map_imag_allbase.fits')
    chi2_map_allframe += chi2_map / len(results_files)
    chi2_map_amp_allframe += chi2_map_amp / len(results_files)
    chi2_map_phi_allframe += chi2_map_phi / len(results_files)

# Determine the best astrometry from the maximum of the chi2 map
idx_allframe, idy_allframe = np.unravel_index(np.argmin(chi2_map_allframe), chi2_map_allframe.shape)
xp_allframe, yp_allframe = x[idx_allframe], y[idy_allframe]
sep_p_allframe = np.sqrt(xp_allframe**2 + yp_allframe**2)
pa_p_allframe = np.arctan2(xp_allframe, yp_allframe)

# Search the planet files
# files_planet = sorted([file for file in results_files if 'unknown' in file and 'flagged' not in file])
files_planet = sorted([file for file in results_files if 'planet' in file and 'flagged' not in file])

# Find all the available OB numbers
OB_list = set([file[file.find('OB')+2:file.find('_exp')] for file in files_planet])
OB_list = list(map(int, OB_list))
print('OB list:', OB_list)
# Load the stellar average quantities
stellar_averages = {}
for i_OB in OB_list:
    print('i_OB:', i_OB)
    i_OB = i_OB - 1
    stellar_averages[i_OB] = {}
    cf_amp_star_data = fits.getdata(path_output + f'/stellar_OB_averages/star_avg_visamp_OB{i_OB}.fits')
    stellar_averages[i_OB]['cf_amp_star'], stellar_averages[i_OB]['cf_amp_err_star'] = cf_amp_star_data
    cf_phi_star_data = fits.getdata(path_output + f'/stellar_OB_averages/star_avg_visphi_OB{i_OB}.fits')
    stellar_averages[i_OB]['cf_phi_star'], stellar_averages[i_OB]['cf_phi_err_star'] = cf_phi_star_data
    # t3_phi_star_data = fits.getdata(path_output + f'/stellar_OB_averagei_err_star'] = t3_phi_star_data

## Loop over the planet files
for i, file in enumerate(files_planet):
    print(i,file)
    
    # Read the observations
    hdul = fits.open(path_oifits + file + '.fits')
    
    # Reorder the baselines
    hdul = reorder_baselines(hdul)
    
    # Extract quantities
    cf_amp     = hdul['OI_VIS'].data['VISAMP']
    cf_amp_err = hdul['OI_VIS'].data['VISAMPERR']
    cf_phi     = np.deg2rad(hdul['OI_VIS'].data['VISPHI'])
    cf_phi_err = np.deg2rad(hdul['OI_VIS'].data['VISPHIERR'])
    t3_phi     = np.deg2rad(hdul['OI_T3'].data['T3PHI'])
    t3_phi_err = np.deg2rad(hdul['OI_T3'].data['T3PHIERR'])
    U          = hdul['OI_VIS'].data['UCOORD']
    V          = hdul['OI_VIS'].data['VCOORD']
    wl         = hdul['OI_WAVELENGTH'].data['EFF_WAVE'] * 1e6
    mjd        = hdul['OI_VIS'].data['MJD']

    hdul.close()

    Cps = fits.getdata(Cps_file)
    print('Cps shape before reshape:', Cps.shape)
    if use_bin_data == True:
        Cps = Cps.reshape(-1, 5).mean(axis=1)
    # Cps = np.ones_like(wl) * 7e-4
    
    # Filter wavelengths
    wl_mask = (wl > wmin) & (wl < wmax)

    print('Cps & wl_mask shape:', Cps.shape, wl_mask.shape)
    
    cf_amp     = cf_amp[:, wl_mask]
    cf_amp_err = cf_amp_err[:, wl_mask]
    cf_phi     = cf_phi[:, wl_mask]
    cf_phi_err = cf_phi_err[:, wl_mask]
    t3_phi     = t3_phi[:, wl_mask]
    t3_phi_err = t3_phi_err[:, wl_mask]
    wl  = wl[wl_mask]
    Cps = Cps[wl_mask]

    # Dimensions
    n_base = 6
    n_wave = wl.size

    # Load the stellar average quantities for this OB
    i_OB = file[file.find('OB')+2:file.find('_exp')]
    i_OB = int(i_OB)
    i_OB = i_OB - 1
    cf_amp_star, cf_amp_err_star = stellar_averages[i_OB]['cf_amp_star'], stellar_averages[i_OB]['cf_amp_err_star']
    cf_phi_star, cf_phi_err_star = stellar_averages[i_OB]['cf_phi_star'], stellar_averages[i_OB]['cf_phi_err_star']
    # t3_phi_star, t3_phi_err_star = stellar_averages[i_OB]['t3_phi_star'], stellar_averages[i_OB]['t3_phi_err_star']
    
    # Calibrate with the stellar coherent flux
    cf_amp_cal = cf_amp / cf_amp_star[:, wl_mask]
    cf_phi_cal = wrap(cf_phi - cf_phi_star[:, wl_mask])
    # t3_phi_cal = wrap(t3_phi - t3_phi_star[:, wl_mask])

    cf_real_cal = cf_amp_cal * np.cos(cf_phi_cal)
    cf_imag_cal = cf_amp_cal * np.sin(cf_phi_cal)

    # Errors
    if error_type == 'errors_from_pipeline':
        cf_amp_cal_err = cf_amp_cal * np.sqrt((cf_amp_err/cf_amp)**2 + (cf_amp_err_star[:, wl_mask]/cf_amp_star[:, wl_mask])**2)
        cf_phi_cal_err = np.sqrt(cf_phi_err**2 + cf_phi_err_star[:, wl_mask]**2)
        # t3_phi_cal_err = np.sqrt(t3_phi_err**2 + t3_phi_err_star[:, wl_mask]**2)
        cf_real_cal_err = np.sqrt((np.cos(cf_phi_cal) * cf_amp_cal_err)**2 + (cf_amp_cal * np.sin(cf_phi_cal) * cf_phi_cal_err)**2)
        cf_imag_cal_err = np.sqrt((np.sin(cf_phi_cal) * cf_amp_cal_err)**2 + (cf_amp_cal * np.cos(cf_phi_cal) * cf_phi_cal_err)**2)
    elif error_type == 'errors_from_data':
        exposure_name = file[:file.find('_frame')]
        cf_amp_cal_err = fits.getdata(path_output + f'/error_estimates/{exposure_name}_cf_amp_cal_err.fits')[:, wl_mask]
        cf_phi_cal_err = fits.getdata(path_output + f'/error_estimates/{exposure_name}_cf_phi_cal_err.fits')[:, wl_mask]
        # t3_phi_cal_err = fits.getdata(path_output + f'/error_estimates/{exposure_name}_t3_phi_cal_err.fits')[:, wl_mask]
    elif error_type == 'none':
        raise ValueError('Not yet implemented!')
        #cf_ratio_err = np.ones_like(cf_ratio)
    else:
        raise ValueError('Error type not recognized.')

    # Baseline-PA coverage in the UV space
    PAcov = np.arctan2(U, V)
    Bcov  = np.sqrt(U**2 + V**2)
    
    # Compute the baselines projected on the all-frame best astrometry
    Bproj_allframe = Bcov * mas2rad(sep_p_allframe) * np.cos(PAcov - pa_p_allframe)
    freqProj_allframe = np.outer(Bproj_allframe, 1/(wl*1e-6))
    
    # Read the fitted parameters and the chi2 map
    params = fits.getdata(path_output + f'/astrometry_fits_files/{file}_fit_params.fits')
    chi2_map = fits.getdata(path_output + f'/astrometry_fits_files/{file}_chi2_map.fits')
    chi2_map_amp = fits.getdata(path_output + f'/astrometry_fits_files/{file}_chi2_map_real.fits')
    chi2_map_phi = fits.getdata(path_output + f'/astrometry_fits_files/{file}_chi2_map_imag.fits')
    
    transmission_ratio = params[..., 0]
    stellar_residuals_poly_coeffs = params[..., 1:]
    #stellar_residuals_poly_coeffs = params.copy()
    
    # Plot the fitted model parameters
    if i == 0:
        fig_params, axes_params = plt.subplots(params.shape[-1], 1)
        fig_poly, axes_poly = plt.subplots(2, 3)
        axes_poly = axes_poly.flat

        fig_params_allbase, axes_params_allbase = plt.subplots(params.shape[-1], 1)
        fig_poly_allbase, axes_poly_allbase = plt.subplots(2, 3)
        axes_poly_allbase = axes_poly_allbase.flat

        fig_params_allframe, axes_params_allframe = plt.subplots(params.shape[-1], 1)
    
    fig_poly_allframe, axes_poly_allframe = plt.subplots(2, 3)
    axes_poly_allframe = axes_poly_allframe.flat
    fig_diff_poly_allframe, axes_diff_poly_allframe = plt.subplots(2, 3)
    axes_diff_poly_allframe = axes_diff_poly_allframe.flat

    # Co-add the chi2 maps of the 6 baselines within this frame
    chi2_map_allbase = fits.getdata(path_output + f'/astrometry_fits_files/{file}_chi2_map_allbase.fits')
    
    # Determine the best astrometry for this frame from the minimum of the all-baseline chi2 map
    idx_allbase, idy_allbase = np.unravel_index(np.argmin(chi2_map_allbase), chi2_map_allbase.shape)
    xp_allbase, yp_allbase = x[idx_allbase], y[idy_allbase]
    sep_p_allbase = np.sqrt(xp_allbase**2 + yp_allbase**2)
    pa_p_allbase = np.arctan2(xp_allbase, yp_allbase)

    # Compute the spatial frequencies projected on the all-baseline best astrometry of this frame
    Bproj_allbase = Bcov * np.cos(PAcov - pa_p_allbase) * mas2rad(sep_p_allbase)
    freqProj_allbase = np.outer(Bproj_allbase, 1/(wl*1e-6))

    # Plot the model over the data
    fig, axes = plt.subplots(6, 1, figsize=(10, 6))
    axes = axes.flat
    fig_allbase, axes_allbase = plt.subplots(6, 1, figsize=(10, 6))
    axes_allbase = axes_allbase.flat
    fig_allframe, axes_allframe = plt.subplots(6, 1, figsize=(6, 8))
    axes_allframe = axes_allframe.flat
    
    fig_phi, axes_phi = plt.subplots(6, 1, figsize=(10, 6))
    axes_phi = axes_phi.flat
    fig_phi_allbase, axes_phi_allbase = plt.subplots(6, 1, figsize=(10, 6))
    axes_phi_allbase = axes_phi_allbase.flat
    fig_phi_allframe, axes_phi_allframe = plt.subplots(6, 1, figsize=(6, 8), layout='tight')  #"compressed")
    axes_phi_allframe = axes_phi_allframe.flat

    fig_real_allframe, axes_real_allframe = plt.subplots(6, 1, figsize=(6, 8))
    axes_real_allframe = axes_real_allframe.flat
    fig_imag_allframe, axes_imag_allframe = plt.subplots(6, 1, figsize=(6, 8))
    axes_imag_allframe = axes_imag_allframe.flat

    # Loop over the baselines of this frame
    for iB in range(n_base):
    
        # Determine the best astrometry from the minimum of the chi2 map
        idx, idy = np.unravel_index(np.argmin(chi2_map[iB]), chi2_map[iB].shape)
        xp, yp = x[idx], y[idy]
        sep_p = np.sqrt(xp**2 + yp**2)
        pa_p = np.arctan2(xp, yp)
        
        # Extract the model parameters fitted at these coordinates
        params_best = params[iB, idx, idy, :]
        transmission_ratio = params_best[0]
        stellar_residuals_poly_coeffs = params_best[1:]
        #stellar_residuals_poly_coeffs = params_best
        
        # Compute the spatial frequencies projected at these coordinates
        Bproj = Bcov * mas2rad(sep_p) * np.cos(PAcov - pa_p)
        freqProj = np.outer(Bproj, 1/(wl*1e-6))
        
        # Compute the whole model
        amp_model, phi_model = model(wl, transmission_ratio, stellar_residuals_poly_coeffs, Cps, freqProj[iB])

        # Remove the slope from the phase model, like in the data
        wl_mask_lin = (wl > 3.1) & (wl < 4.0)
        mean_inv_wl = np.mean(1 / wl[wl_mask_lin])
        mean_phi_model = np.mean(phi_model[wl_mask_lin])
        slope = np.sum((1/wl[wl_mask_lin] - mean_inv_wl) * (phi_model[wl_mask_lin] - mean_phi_model)) / np.sum((1/wl[wl_mask_lin] - mean_inv_wl) ** 2)
        intercept = mean_phi_model - slope * mean_inv_wl
        phi_model = wrap(phi_model - slope / wl - intercept)
        
        # Plot the model and the data
        
        # Amplitudes 
        axes[iB].plot(wl, cf_amp_cal[iB], 'r+')#, label=f'Correlated flux ratio {base_name[iB]}')
        #axes[iB].plot(wl, cf_ratio[iexp*6+iB]+cf_ratio_err[iexp*6+iB], 'k+')#, label=f'Correlated flux ratio {base_name[iB]}')
        axes[iB].plot(wl, amp_model)
        axes[iB].set_ylim(0, 0.002)
        axes[iB].set_ylabel(base_name[iB])

        # Differential phases
        axes_phi[iB].plot(wl, np.rad2deg(cf_phi_cal[iB]), 'r+')
        axes_phi[iB].plot(wl, np.rad2deg(phi_model))
        axes_phi[iB].set_ylim(-120, 120)
        axes_phi[iB].set_ylabel(base_name[iB])

        # Fitted parameters
        for ip in range(params.shape[-1]):
            axes_params[ip].plot(mjd[iB]-np.floor(mjd[iB]), params_best[ip], '+', c=colors[iB])
        #axes_poly[iB].plot(wl, np.abs(Css))
        axes_poly[iB].plot(wl, Polynomial(stellar_residuals_poly_coeffs)(wl))

        ###
        ### Redo everything with the parameters of the all-baseline best astrometry ###
        ###

        params_best_allbase = params[iB, idx_allbase, idy_allbase, :]
        transmission_ratio = params_best_allbase[0]
        stellar_residuals_poly_coeffs = params_best_allbase[1:]
        
        # Compute the whole model
        amp_model, phi_model = model(wl, transmission_ratio, stellar_residuals_poly_coeffs, Cps, freqProj_allbase[iB])

        # Remove the slope from the phase model, like in the data
        wl_mask_lin = (wl > 3.1) & (wl < 4.0)
        mean_inv_wl = np.mean(1 / wl[wl_mask_lin])
        mean_phi_model = np.mean(phi_model[wl_mask_lin])
        slope = np.sum((1/wl[wl_mask_lin] - mean_inv_wl) * (phi_model[wl_mask_lin] - mean_phi_model)) / np.sum((1/wl[wl_mask_lin] - mean_inv_wl) ** 2)
        intercept = mean_phi_model - slope * mean_inv_wl
        phi_model = wrap(phi_model - slope / wl - intercept)

        # Plot the model and the data
        
        # Amplitudes 
        axes_allbase[iB].plot(wl, cf_amp_cal[iB], 'r+')#, label=f'Correlated flux ratio {base_name[iB]}')
        axes_allbase[iB].plot(wl, amp_model)
        axes_allbase[iB].set_ylim(0, 0.002)
        axes_allbase[iB].set_ylabel(base_name[iB])
        
        # Phases
        axes_phi_allbase[iB].plot(wl, np.rad2deg(cf_phi_cal[iB]), 'r+')
        axes_phi_allbase[iB].plot(wl, np.rad2deg(phi_model))
        axes_phi_allbase[iB].set_ylim(-120, 120)
        axes_phi_allbase[iB].set_ylabel(base_name[iB])
        
        # Fitted parameters
        for ip in range(params.shape[-1]):
            axes_params_allbase[ip].plot(mjd[iB]-np.floor(mjd[iB]), params_best_allbase[ip], '+', c=colors[iB])
        #axes_poly_allbase[iB].plot(wl, np.abs(Css))
        axes_poly_allbase[iB].plot(wl, Polynomial(stellar_residuals_poly_coeffs)(wl))
        

        ###     
        ### Redo everything with the parameters of the all-frame best astrometry ###
        ###

        params_best_allframe = params[iB, idx_allframe, idy_allframe, :]
        transmission_ratio = params_best_allframe[0]
        stellar_residuals_poly_coeffs = params_best_allframe[1:]
        #stellar_residuals_poly_coeffs = params_best_allframe
        
        # Compute the whole model
        amp_model, phi_model = model(wl, transmission_ratio, stellar_residuals_poly_coeffs, Cps, freqProj_allframe[iB])

        # Remove the slope from the phase model, like in the data
        wl_mask_lin = (wl > 3.1) & (wl < 4.0)
        mean_inv_wl = np.mean(1 / wl[wl_mask_lin])
        mean_phi_model = np.mean(phi_model[wl_mask_lin])
        slope = np.sum((1/wl[wl_mask_lin] - mean_inv_wl) * (phi_model[wl_mask_lin] - mean_phi_model)) / np.sum((1/wl[wl_mask_lin] - mean_inv_wl) ** 2)
        intercept = mean_phi_model - slope * mean_inv_wl
        phi_model = wrap(phi_model - slope / wl - intercept)

        real_model = amp_model * np.cos(phi_model)
        imag_model = amp_model * np.sin(phi_model)

        # Plot the model and the data
        
        # Amplitudes
        #axes_allframe[iB].plot(wl, cf_amp_cal[iB], '+', c='orangered')#, label=f'Correlated flux ratio {base_name[iB]}')
        #axes_allframe[iB].plot(wl, cf_amp_cal[iB], '.', c='tomato')
        #axes_allframe[iB].plot(wl, cf_amp_cal[iB], '.', markersize=5, c='tomato')
        axes_allframe[iB].plot(wl, cf_amp_cal[iB], '+', markersize=5, c='tomato')
        axes_allframe[iB].fill_between(wl, cf_amp_cal[iB]-cf_amp_cal_err[iB], cf_amp_cal[iB]+cf_amp_cal_err[iB],  color='tomato', alpha=0.2, zorder=-100)
        axes_allframe[iB].plot(wl, amp_model, c='purple')
        axes_allframe[iB].set_ylim(-0.002, 0.008)
        #axes_allframe[iB].set_ylim(0, 0.008)
        axes_allframe[iB].set_ylabel(base_name[iB])
        axes_allframe[iB].spines['top'].set_visible(False)
        axes_allframe[iB].spines['right'].set_visible(False)
        if iB != 5:
            axes_allframe[iB].set_xticklabels([])
        
        # Phases
        axes_phi_allframe[iB].plot(wl, np.rad2deg(cf_phi_cal[iB]), '+', markersize=5, c='mediumseagreen')
        #axes_phi_allframe[iB].errorbar(wl, np.rad2deg(cf_phi_cal[iB]), yerr=np.rad2deg(cf_phi_cal_err[iB]), fmt='none', elinewidth=0.5, color='mediumseagreen')
        axes_phi_allframe[iB].fill_between(wl, np.rad2deg(cf_phi_cal[iB]-cf_phi_cal_err[iB]), np.rad2deg(cf_phi_cal[iB]+cf_phi_cal_err[iB]), color='mediumseagreen', alpha=0.2, zorder=-100)
        #axes_phi_allframe[iB].plot(wl, np.rad2deg(cf_phi_cal[iB]), 'r+')
        axes_phi_allframe[iB].plot(wl, np.rad2deg(phi_model), c='darkblue', zorder=100)
        axes_phi_allframe[iB].set_ylim(-120, 120)
        #axes_phi_allframe[iB].set_ylim(-180, 180)
        axes_phi_allframe[iB].set_ylabel(base_name[iB])
        axes_phi_allframe[iB].set_yticks([-100, 0, 100], ['-100°', '0', '100°'])
        axes_phi_allframe[iB].spines['top'].set_visible(False)
        axes_phi_allframe[iB].spines['right'].set_visible(False)
        #axes_phi_allframe[iB].set_xlim(4.5, 5.0)
        if iB != 5:
            axes_phi_allframe[iB].set_xticklabels([])

        # Real parts
        #axes_allframe[iB].plot(wl, cf_amp_cal[iB], '+', c='orangered')#, label=f'Correlated flux ratio {base_name[iB]}')
        #axes_allframe[iB].plot(wl, cf_amp_cal[iB], '.', c='tomato')
        #axes_allframe[iB].plot(wl, cf_amp_cal[iB], '.', markersize=5, c='tomato')
        axes_real_allframe[iB].plot(wl, cf_real_cal[iB], '+', markersize=5, c='tomato')
        axes_real_allframe[iB].fill_between(wl, cf_real_cal[iB]-cf_real_cal_err[iB], cf_real_cal[iB]+cf_real_cal_err[iB],  color='tomato', alpha=0.2, zorder=-100)
        axes_real_allframe[iB].plot(wl, real_model, c='purple')
        axes_real_allframe[iB].set_ylim(-0.004, 0.004)
        #axes_real_allframe[iB].set_ylim(0, 0.008)
        axes_real_allframe[iB].set_ylabel(base_name[iB])
        axes_real_allframe[iB].spines['top'].set_visible(False)
        axes_real_allframe[iB].spines['right'].set_visible(False)
        if iB != 5:
            axes_real_allframe[iB].set_xticklabels([])

        # Imaginary parts
        #axes_allframe[iB].plot(wl, cf_amp_cal[iB], '+', c='orangered')#, label=f'Correlated flux ratio {base_name[iB]}')
        #axes_allframe[iB].plot(wl, cf_amp_cal[iB], '.', c='tomato')
        #axes_allframe[iB].plot(wl, cf_amp_cal[iB], '.', markersize=5, c='tomato')
        axes_imag_allframe[iB].plot(wl, cf_imag_cal[iB], '+', markersize=5, c='tomato')
        axes_imag_allframe[iB].fill_between(wl, cf_imag_cal[iB]-cf_imag_cal_err[iB], cf_imag_cal[iB]+cf_imag_cal_err[iB],  color='tomato', alpha=0.2, zorder=-100)
        axes_imag_allframe[iB].plot(wl, imag_model, c='purple')
        axes_imag_allframe[iB].set_ylim(-0.004, 0.004)
        #axes_imag_allframe[iB].set_ylim(0, 0.008)
        axes_imag_allframe[iB].set_ylabel(base_name[iB])
        axes_imag_allframe[iB].spines['top'].set_visible(False)
        axes_imag_allframe[iB].spines['right'].set_visible(False)
        if iB != 5:
            axes_imag_allframe[iB].set_xticklabels([])
        
        # Fitted parameters
        for ip in range(params.shape[-1]):
            axes_params_allframe[ip].plot(mjd[iB]-np.floor(mjd[iB]), params_best_allframe[ip], '+', c=colors[iB])
        #axes_poly_allframe[iB].plot(wl, np.abs(Css))
        n_coeffs = stellar_residuals_poly_coeffs.size // 2
        axes_poly_allframe[iB].plot(wl, Polynomial(stellar_residuals_poly_coeffs[:n_coeffs])(wl))
        axes_poly_allframe[iB].plot(wl, Polynomial(stellar_residuals_poly_coeffs[n_coeffs:])(wl))
        fig_poly_allframe.savefig(path_output + f'astrometry_figures/fitted_polynomials_allframe/{file}_fitted_polynomials_allframe.png')
        axes_diff_poly_allframe[iB].plot(wl, Polynomial(stellar_residuals_poly_coeffs[n_coeffs:])(wl)/Polynomial(stellar_residuals_poly_coeffs[:n_coeffs])(wl))
        fig_diff_poly_allframe.savefig(path_output + f'astrometry_figures/relativediff_fitted_polynomials_allframe/{file}_relativediff_fitted_polynomials_allframe.png')

        plt.close('all')
    
    # Save the figures

    axes[-1].set_xlabel(r'Wavelength [$\mu$m]')
    fig.tight_layout()
    fig.savefig(path_output + f'/astrometry_figures/amp_model/{file}_amp_model.png')

    axes_allbase[-1].set_xlabel(r'Wavelength [$\mu$m]')
    fig_allbase.tight_layout()
    fig_allbase.savefig(path_output + f'/astrometry_figures/amp_model_allbase/{file}_amp_model_allbase.png')
    
    axes_allframe[-1].set_xlabel(r'Wavelength [$\mu$m]')
    fig_allframe.suptitle('Amplitude of the coherent flux ratio')
    fig_allframe.tight_layout()
    fig_allframe.savefig(path_output + f'/astrometry_figures/amp_model_allframes/{file}_amp_model_allframes.pdf')
    
    axes_phi[-1].set_xlabel(r'Wavelength [$\mu$m]')
    fig_phi.tight_layout()
    fig_phi.savefig(path_output + f'/astrometry_figures/phase_model/{file}_phi_model.png')

    axes_phi_allbase[-1].set_xlabel(r'Wavelength [$\mu$m]')
    fig_phi_allbase.tight_layout()
    fig_phi_allbase.savefig(path_output + f'/astrometry_figures/phase_model_allbase/{file}_phi_model_allbase.png')

    axes_phi_allframe[-1].set_xlabel(r'Wavelength [$\mu$m]')
    fig_phi_allframe.suptitle('Phase of the coherent flux ratio')
    fig_phi_allframe.tight_layout()
    fig_phi_allframe.savefig(path_output + f'/astrometry_figures/phase_model_allframes/{file}_phi_model_allframes.pdf')
    
    axes_real_allframe[-1].set_xlabel(r'Wavelength [$\mu$m]')
    fig_real_allframe.suptitle('Real part of the coherent flux ratio')
    fig_real_allframe.tight_layout()
    fig_real_allframe.savefig(path_output + f'/astrometry_figures/real_model_allframes/{file}_real_model_allframes.pdf')

    axes_imag_allframe[-1].set_xlabel(r'Wavelength [$\mu$m]')
    fig_imag_allframe.suptitle('Imaginary part of the coherent flux ratio')
    fig_imag_allframe.tight_layout()
    fig_imag_allframe.savefig(path_output + f'/astrometry_figures/imag_model_allframes/{file}_imag_model_allframes.pdf')      

    # Plot the all-baseline chi2 map
    fig, ax = plt.subplots()
    #im = ax.imshow(np.clip(chi2_map_allbase.T, np.quantile(chi2_map_allbase.T, 0.05), np.quantile(chi2_map_allbase.T, 0.95)), origin='lower', extent=(x.min(), x.max(), y.min(), y.max()))
    im = ax.imshow(chi2_map_allbase.T, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()))

    ax.axvline(xp_allbase, c='w', lw=0.5, ls='--')
    ax.axhline(yp_allbase, c='w', lw=0.5, ls='--')
    #ax.axvline(279.392, c='orange', lw=0.5, ls='--')
    #ax.axhline(455.423, c='orange', lw=0.5, ls='--')

    ax.set_ylabel(r'$\Delta\delta$ [mas]')
    ax.set_xlabel(r'$\Delta\alpha$ [mas]')
    
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path_output + f'/astrometry_figures/chi2_maps_allbase/{file}_chi2map_allbase.png')

    # Plot the all-baseline chi2 map of the amplitude
    chi2_map_allbase = fits.getdata(path_output + f'/astrometry_fits_files/{file}_chi2_map_real_allbase.fits')
    fig, ax = plt.subplots()
    im = ax.imshow(chi2_map_allbase.T, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()))
    
    ax.axvline(xp_allbase, c='w', lw=0.5, ls='--')
    ax.axhline(yp_allbase, c='w', lw=0.5, ls='--')
    #ax.axvline(279.392, c='orange', lw=0.5, ls='--')
    #ax.axhline(455.423, c='orange', lw=0.5, ls='--')

    ax.set_ylabel(r'$\Delta\delta$ [mas]')
    ax.set_xlabel(r'$\Delta\alpha$ [mas]')
    
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path_output + f'astrometry_figures/chi2_maps_amp_allbase/{file}_chi2map_real_allbase.png')

    # Plot the all-baseline chi2 map of the phase
    chi2_map_allbase = fits.getdata(path_output + f'astrometry_fits_files/{file}_chi2_map_imag_allbase.fits')
    fig, ax = plt.subplots()
    im = ax.imshow(chi2_map_allbase.T, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()))
    
    ax.axvline(xp_allbase, c='w', lw=0.5, ls='--')
    ax.axhline(yp_allbase, c='w', lw=0.5, ls='--')
    #ax.axvline(279.392, c='orange', lw=0.5, ls='--')
    #ax.axhline(455.423, c='orange', lw=0.5, ls='--')

    ax.set_ylabel(r'$\Delta\delta$ [mas]')
    ax.set_xlabel(r'$\Delta\alpha$ [mas]')
    
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path_output + f'astrometry_figures/chi2_maps_phi_allbase/{file}_chi2map_imag_allbase.png')

    ###
    ### Compute and plot the closure phases
    ###

    triangle_names = ('U1-U2-U3', 'U1-U2-U4', 'U1-U3-U4', 'U2-U3-U4')
    #triangle_orders = np.array([[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]])
    base_in_triangles = np.array([[0, 3, 1], [0, 4, 2], [1, 5, 2], [3, 5, 4]])

    fig_t3phi_allbase, axes_t3phi_allbase = plt.subplots(4, 1, figsize=(10, 6))
    axes_t3phi_allbase = axes_t3phi_allbase.flat
    fig_t3phi_allframe, axes_t3phi_allframe = plt.subplots(4, 1, figsize=(10, 6))
    axes_t3phi_allframe = axes_t3phi_allframe.flat

    for iT in range(4):
        base1, base2, base3 = base_in_triangles[iT]

        # All-baseline best astrometry
        amp_model = np.zeros((6, wl.size), dtype=float)
        phi_model = np.zeros((6, wl.size), dtype=float)
        cf_model = np.zeros((6, wl.size), dtype=complex)
        for iB in range(6):
            params_best_allbase = params[iB, idx_allbase, idy_allbase, :]
            transmission_ratio = params_best_allbase[0]
            stellar_residuals_poly_coeffs = params_best_allbase[1:]
            #stellar_residuals_poly_coeffs = params_best_all
        
            # Compute the whole model
            amp_model[iB], phi_model[iB] = model(wl, transmission_ratio, stellar_residuals_poly_coeffs, Cps, freqProj_allbase[iB])

            # Remove the slope from the phase model, like in the data
            wl_mask_lin = (wl > 3.1) & (wl < 4.0)
            mean_inv_wl = np.mean(1 / wl[wl_mask_lin])
            mean_phi_model = np.mean(phi_model[iB, wl_mask_lin])
            slope = np.sum((1/wl[wl_mask_lin] - mean_inv_wl) * (phi_model[iB, wl_mask_lin] - mean_phi_model)) / np.sum((1/wl[wl_mask_lin] - mean_inv_wl) ** 2)
            intercept = mean_phi_model - slope * mean_inv_wl
            phi_model[iB] = wrap(phi_model[iB] - slope / wl - intercept)

            # Complexify
            cf_model[iB] = amp_model[iB] * np.exp(1j * phi_model[iB])

        # Compute the model of the closure phases
        # t3_model = cf_model[base1] * cf_model[base2] * np.conj(cf_model[base3])
        # t3phi_model = np.angle(t3_model)
        
        # axes_t3phi_allbase[iT].plot(wl, np.rad2deg(t3_phi_cal[iT]), 'r+')#, label=f'Correlated flux ratio {base_name[iT]}')
        # axes_t3phi_allbase[iT].plot(wl, np.rad2deg(t3phi_model))
        # axes_t3phi_allbase[iT].set_ylim(-120, 120)
        # axes_t3phi_allbase[iT].set_ylabel(triangle_names[iT])
        
        ###
        ### Redo everything with the parameters from the all-frame best astrometry
        ###

        amp_model = np.zeros((6, wl.size), dtype=float)
        phi_model = np.zeros((6, wl.size), dtype=float)
        cf_model = np.zeros((6, wl.size), dtype=complex)
        for iB in range(6):
            # All-frame best astrometry
            params_best_allframe = params[iB, idx_allframe, idy_allframe, :]
            transmission_ratio = params_best_allframe[0]
            stellar_residuals_poly_coeffs = params_best_allframe[1:]
            #stellar_residuals_poly_coeffs = params_best_all
            
            amp_model[iB], phi_model[iB] = model(wl, transmission_ratio, stellar_residuals_poly_coeffs, Cps, freqProj_allframe[iB])

            # Remove the slope from the phase model, like in the data
            wl_mask_lin = (wl > 3.1) & (wl < 4.0)
            mean_inv_wl = np.mean(1 / wl[wl_mask_lin])
            mean_phi_model = np.mean(phi_model[iB, wl_mask_lin])
            slope = np.sum((1/wl[wl_mask_lin] - mean_inv_wl) * (phi_model[iB, wl_mask_lin] - mean_phi_model)) / np.sum((1/wl[wl_mask_lin] - mean_inv_wl) ** 2)
            intercept = mean_phi_model - slope * mean_inv_wl
            phi_model[iB] = wrap(phi_model[iB] - slope / wl - intercept)

            # Complexify
            cf_model[iB] = amp_model[iB] * np.exp(1j * phi_model[iB])

        # t3_model = cf_model[base1] * cf_model[base2] * np.conj(cf_model[base3])
        # t3phi_model = np.angle(t3_model)
        
        # axes_t3phi_allframe[iT].plot(wl, np.rad2deg(t3_phi_cal[iT]), 'r+')#, label=f'Correlated flux ratio {base_name[iT]}')
        # axes_t3phi_allframe[iT].plot(wl, np.rad2deg(t3phi_model))
        # axes_t3phi_allframe[iT].set_ylim(-120, 120)
        # axes_t3phi_allframe[iT].set_ylabel(triangle_names[iT])

        plt.close('all')

    # Save the closure phase figures

    # axes_t3phi_allbase[-1].set_xlabel(r'Wavelength [$\mu$m]')
    # fig_t3phi_allbase.tight_layout()
    # fig_t3phi_allbase.savefig(path_output + f'astrometry_figures/t3phi_model_allbase/{file}_t3phi_model_allbase.png')

    # axes_t3phi_allframe[-1].set_xlabel(r'Wavelength [$\mu$m]')
    # fig_t3phi_allframe.tight_layout()
    # fig_t3phi_allframe.savefig(path_output + f'astrometry_figures/t3phi_model_allframes/{file}_t3phi_model_allframes.png')
    

# Plot the total chi2 map (all frames co-added)

fig, ax = plt.subplots()
im = ax.imshow(chi2_map_allframe.T, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()))

ax.axvline(xp_allframe, c='w', lw=0.5, ls='--')
ax.axhline(yp_allframe, c='w', lw=0.5, ls='--')
#ax.axvline(279.392, c='orange', lw=0.5, ls='--')
#ax.axhline(455.423, c='orange', lw=0.5, ls='--')

ax.set_ylabel(r'$\Delta\delta$ [mas]')
ax.set_xlabel(r'$\Delta\alpha$ [mas]')

fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(path_output + 'astrometry_figures/chi2map_allframes.png')

# Plot the total chi2 map of the amplitude (all frames co-added)

fig, ax = plt.subplots()
im = ax.imshow(chi2_map_amp_allframe.T, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()))

idx_allframe, idy_allframe = np.unravel_index(np.argmin(chi2_map_amp_allframe), chi2_map_amp_allframe.shape)
xp_allframe, yp_allframe = x[idx_allframe], y[idy_allframe]

ax.axvline(xp_allframe, c='w', lw=0.5, ls='--')
ax.axhline(yp_allframe, c='w', lw=0.5, ls='--')
#ax.axvline(279.392, c='orange', lw=0.5, ls='--')
#ax.axhline(455.423, c='orange', lw=0.5, ls='--')

ax.set_ylabel(r'$\Delta\delta$ [mas]')
ax.set_xlabel(r'$\Delta\alpha$ [mas]')

fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(path_output + 'astrometry_figures/chi2map_real_allframes.png')

# Plot the total chi2 map of the phase (all frames co-added)

fig, ax = plt.subplots()
im = ax.imshow(chi2_map_phi_allframe.T, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()))

idx_allframe, idy_allframe = np.unravel_index(np.argmin(chi2_map_phi_allframe), chi2_map_phi_allframe.shape)
xp_allframe, yp_allframe = x[idx_allframe], y[idy_allframe]

ax.axvline(xp_allframe, c='w', lw=0.5, ls='--')
ax.axhline(yp_allframe, c='w', lw=0.5, ls='--')
#ax.axvline(279.392, c='orange', lw=0.5, ls='--')
#ax.axhline(455.423, c='orange', lw=0.5, ls='--')

ax.set_ylabel(r'$\Delta\delta$ [mas]')
ax.set_xlabel(r'$\Delta\alpha$ [mas]')

fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(path_output + 'astrometry_figures/chi2map_imag_allframes.png')

# Save the all-frame chi2 map
fits.writeto(path_output + 'astrometry_figures/chi2map_allframes.fits', chi2_map_allframe, overwrite=True)

# Save the fitted parameters plot
fig_params.savefig(path_output + 'astrometry_figures/fitted_parameters.png')
fig_poly.savefig(path_output + 'astrometry_figures/fitted_polynomials.png')

fig_params_allbase.savefig(path_output + 'astrometry_figures/fitted_parameters_allbase.png')
fig_poly_allbase.savefig(path_output + 'astrometry_figures/fitted_polynomials_allbase.png')

fig_params_allframe.savefig(path_output + 'astrometry_figures/fitted_parameters_allframe.png')