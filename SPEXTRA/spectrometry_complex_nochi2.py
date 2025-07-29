#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### Compute the cross-correlation map for a single file

import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
import astropy.constants as cst

from astropy.io import fits
from astropy.table import Table
from numpy.polynomial import Polynomial
from PyAstronomy.pyasl import instrBroadGaussFast
import spectres 
from common_tools import wrap, reorder_baselines, mas2rad

#matplotlib.use('Qt5Agg')
#plt.rcParams["figure.dpi"] = 100


### Parameters ###

# Input OIFITS directory
#path_oifits = './postprocessing/results_20230724_v7_guessFromPrevFit/corrected_data/'
#path_oifits = './postprocessing/results_20231030_phiIns_allbase_starFirst/corrected_data/'
path_oifits = '/data/home/jscigliuto/Pipeline/corrPhase0_MACAO/'

# Output results directory
#path_output = './results_20230911_mp_errdata_snrmask_modelcontrast/'
path_output = '/data/home/jscigliuto/Pipeline/Result_betPic0_MACAO/'

use_bin_data = False
# Selected wavelength range for computing the spectrum
wmin, wmax = 2.5, 5.5 # microns

# Choose to reject an additional wavelength interval
reject_an_interval = False
wmin_itv, wmax_itv = 3.58, 3.61

# Grid of (alpha, delta) coordinates
#x = np.arange(250, 300, 1) # mas
#y = np.arange(425, 475, 1) # mas
#x = np.arange(280-65, 280+66, 1) # mas
#y = np.arange(455-65, 455+66, 1) # mas
#x = np.arange(280-10, 280+11, 0.1) # mas
#y = np.arange(455-10, 455+11, 0.1) # mas
x = np.arange(279-15, 279+15, 0.4) # mas
y = np.arange(455-15, 455+15, 0.4) # mas
# x = np.arange(-300, -250, 1) # mas
# y = np.arange(-475, -425, 1) # mas

# Polynomial order
n_poly = 1

# Weighted least squares
weighted_least_squares = 'errors_from_pipeline' #'errors_from_data', 'errors_from_pipeline'


### Function definitions ###

# def model(params, transmission_ratio, stellar_part):
#     model = np.outer(transmission_ratio, params) + stellar_part
#     amp_model = np.abs(model)
#     phi_model = np.angle(model)
#     return amp_model, phi_model

# def residuals(params, transmission_ratio, stellar_part, amp, phi, amp_err, phi_err):
#     amp_model, phi_model = model(params, transmission_ratio, stellar_part)
#     # Remove the slope from the phase model, like in the data
#     wl_mask_lin = (wl > 3.1) & (wl < 4.0)
#     mean_inv_wl = np.mean(1 / wl[wl_mask_lin])
#     mean_phi_model = np.mean(phi_model[wl_mask_lin])
#     slope = np.sum((1/wl[wl_mask_lin] - mean_inv_wl) * (phi_model[wl_mask_lin] - mean_phi_model)) / np.sum((1/wl[wl_mask_lin] - mean_inv_wl) ** 2)
#     intercept = mean_phi_model - slope * mean_inv_wl
#     phi_model = wrap(phi_model - slope / wl - intercept)
#     # Residuals
#     chi2_amp = ((amp - amp_model) / amp_err) ** 2
#     chi2_phi = (wrap(phi - phi_model) / phi_err) ** 2
#     chi2 = np.sum(chi2_amp) + np.sum(chi2_phi)
#     chi2_red = chi2 / (amp.size + phi.size - params.size)
#     return chi2_red


### Main program ###

# Search files
# files = sorted([file for file in os.listdir(path_oifits) if 'unknown' in file and 'flagged' not in file])
files = sorted([file for file in os.listdir(path_oifits) if 'planet' in file and 'flagged' not in file])
files_star = sorted([file for file in os.listdir(path_oifits) if 'star' in file and 'flagged' not in file])

# Create output directory if necessary
if not os.path.isdir(path_output + '/spectrometry'):
    os.makedirs(path_output + '/spectrometry')

# Pre-reading to get the wavelengths
hdul = fits.open(path_oifits + files[0])
wl = hdul['OI_WAVELENGTH'].data['EFF_WAVE'] * 1e6
wband = hdul['OI_WAVELENGTH'].data['EFF_BAND'] * 1e6
n_wave = wl.size
hdul.close()

# Filter wavelengths
if reject_an_interval:
    wl_mask = (wl > wmin) & (wl < wmax) & ((wl < wmin_itv) | (wl > wmax_itv))
else:
    wl_mask = (wl > wmin) & (wl < wmax)
wl    = wl[wl_mask]
wband = wband[wl_mask]

# Initialization of concatenated arrays
cf_amp_all = np.zeros((1, wl.size))
cf_amp_err_all = np.zeros((1, wl.size))
cf_phi_all = np.zeros((1, wl.size))
cf_phi_err_all = np.zeros((1, wl.size))
# t3_phi_all = np.zeros((1, wl.size))
# t3_phi_err_all = np.zeros((1, wl.size))
U_all = np.zeros(1)
V_all = np.zeros(1)
# stellar_coeffs_all = np.zeros((1, n_poly+1))
stellar_coeffs_all = np.zeros((1, 2*(n_poly+1)))
transmission_all = np.zeros(1)

# Get the astrometry from the minimum of the chi2 map
chi2_map_tot = fits.getdata(path_output + '/astrometry_figures/chi2map_allframes.fits')
idx, idy = np.unravel_index(np.argmin(chi2_map_tot), chi2_map_tot.shape)

# Find all the available OB numbers
OB_list = set([file[file.find('OB')+2:file.find('_exp')] for file in files])

# Load the stellar average quantities
stellar_averages = {}
for i_OB in OB_list:
    i_OB = int(i_OB)
    i_OB = i_OB - 1
    print('i_OB:', i_OB)
    stellar_averages[i_OB] = {}
    cf_amp_star_data = fits.getdata(path_output + f'/stellar_OB_averages/star_avg_visamp_OB{i_OB}.fits')
    stellar_averages[i_OB]['cf_amp_star'], stellar_averages[i_OB]['cf_amp_err_star'] = cf_amp_star_data
    cf_phi_star_data = fits.getdata(path_output + f'/stellar_OB_averages/star_avg_visphi_OB{i_OB}.fits')
    stellar_averages[i_OB]['cf_phi_star'], stellar_averages[i_OB]['cf_phi_err_star'] = cf_phi_star_data
    # t3_phi_star_data = fits.getdata(path_output + f'/stellar_OB_averages/star_avg_t3phi_OB{i_OB}.fits')
    # stellar_averages[i_OB]['t3_phi_star'], stellar_averages[i_OB]['t3_phi_err_star'] = t3_phi_star_data

# Concatenate all the frames and baselines
for i, file in enumerate(files):
    print(f'Concatenating {file}...')
    
    params = fits.getdata(path_output + f'/astrometry_fits_files/{file[:-5]}_fit_params.fits')
    chi2_map = fits.getdata(path_output + f'/astrometry_fits_files/{file[:-5]}_chi2_map.fits')
    
    # Read the data
    hdul = fits.open(path_oifits + file)
    
    # Reorder the baselines
    hdul = reorder_baselines(hdul)
    
    # Extract quantities
    cf_amp     = hdul['OI_VIS'].data['VISAMP']
    cf_amp_err = hdul['OI_VIS'].data['VISAMPERR']
    cf_phi     = np.deg2rad(hdul['OI_VIS'].data['VISPHI'])
    cf_phi_err = np.deg2rad(hdul['OI_VIS'].data['VISPHIERR'])
    # t3_phi     = np.deg2rad(hdul['OI_T3'].data['T3PHI'])
    # t3_phi_err = np.deg2rad(hdul['OI_T3'].data['T3PHIERR'])
    U  = hdul['OI_VIS'].data['UCOORD']
    V  = hdul['OI_VIS'].data['VCOORD']
    wl = hdul['OI_WAVELENGTH'].data['EFF_WAVE'] * 1e6

    hdul.close()
    
    # Filter wavelengths
    cf_amp     = cf_amp[:, wl_mask]
    cf_amp_err = cf_amp_err[:, wl_mask]
    cf_phi     = cf_phi[:, wl_mask]
    cf_phi_err = cf_phi_err[:, wl_mask]
    # t3_phi     = t3_phi[:, wl_mask]
    # t3_phi_err = t3_phi_err[:, wl_mask]
    wl  = wl[wl_mask]

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
    
    # Errors
    if weighted_least_squares == 'errors_from_pipeline':
        cf_amp_cal_err = cf_amp_cal * np.sqrt((cf_amp_err/cf_amp)**2 + (cf_amp_err_star[:, wl_mask]/cf_amp_star[:, wl_mask])**2)
        cf_phi_cal_err = np.sqrt(cf_phi_err**2 + cf_phi_err_star[:, wl_mask]**2)
        # t3_phi_cal_err = np.sqrt(t3_phi_err**2 + t3_phi_err_star[:, wl_mask]**2)
    elif weighted_least_squares == 'errors_from_data':
        exposure_name = file[:file.find('_frame')]
        cf_amp_cal_err = fits.getdata(path_output + f'/error_estimates/{exposure_name}_cf_amp_cal_err.fits')[:, wl_mask]
        cf_phi_cal_err = fits.getdata(path_output + f'/error_estimates/{exposure_name}_cf_phi_cal_err.fits')[:, wl_mask]
        # t3_phi_cal_err = fits.getdata(path_output + f'/error_estimates/{exposure_name}_t3_phi_cal_err.fits')[:, wl_mask]
    elif weighted_least_squares == 'none':
        raise ValueError('Not yet implemented!')
        #cf_ratio_err = np.ones_like(cf_ratio)
    else:
        raise ValueError('Weighted least squares mode not recognized.')
    
    # Stack data all together
    cf_amp_all = np.concatenate((cf_amp_all, cf_amp_cal))
    cf_amp_err_all = np.concatenate((cf_amp_err_all, cf_amp_cal_err))
    cf_phi_all = np.concatenate((cf_phi_all, cf_phi_cal))
    cf_phi_err_all = np.concatenate((cf_phi_err_all, cf_phi_cal_err))
    # t3_phi_all = np.concatenate((t3_phi_all, t3_phi_cal))
    # t3_phi_err_all = np.concatenate((t3_phi_err_all, t3_phi_cal_err))
    U_all = np.concatenate((U_all, U))
    V_all = np.concatenate((V_all, V))
    transmission_all = np.concatenate((transmission_all, params[:, idx, idy, 0]))
    stellar_coeffs_all = np.concatenate((stellar_coeffs_all, params[:, idx, idy, 1:]))

# Delete the null first element of the concatenation
cf_amp_all = np.delete(cf_amp_all, 0, 0)
cf_amp_err_all = np.delete(cf_amp_err_all, 0, 0)
cf_phi_all = np.delete(cf_phi_all, 0, 0)
cf_phi_err_all = np.delete(cf_phi_err_all, 0, 0)
# t3_phi_all = np.delete(t3_phi_all, 0, 0)
# t3_phi_err_all = np.delete(t3_phi_err_all, 0, 0)
U_all = np.delete(U_all, 0, 0)
V_all = np.delete(V_all, 0, 0)
stellar_coeffs_all = np.delete(stellar_coeffs_all, 0, 0)
transmission_all = np.delete(transmission_all, 0, 0)

# Dimensions of the full stack of observations
nb, nw = cf_amp_all.shape
print(f'{nb} baselines over {nb//6} frames, {nw} wavelengths')

# Complex coherent flux
cf_all = cf_amp_all * np.exp(1j * cf_phi_all)
    
# Load Skycalc radiance and transmission
# skycalc_file = np.loadtxt('skycalc_radiance_all.txt')
# wl_skycalc = skycalc_file[:, 0] / 1e3
# sky_radiance = skycalc_file[:, 1]
# f_interp = scipy.interpolate.interp1d(wl_skycalc, sky_radiance)
# sky_radiance = f_interp(wl)

# skycalc_file = np.loadtxt('skycalc_transmission_all.txt')
# wl_skycalc = skycalc_file[:, 0] / 1e3
# sky_transmission = skycalc_file[:, 1]
# f_interp = scipy.interpolate.interp1d(wl_skycalc, sky_transmission)
# sky_transmission = f_interp(wl)

# Grid of (alpha, delta) coordinates
xx, yy = np.meshgrid(x, y)

# Associated PAs and separations
pa = np.arctan2(xx, yy)
sep = np.sqrt(xx**2+yy**2)

# Baseline-PA coverage in the UV space
PAcov = np.arctan2(U_all, V_all)
Bcov  = np.sqrt(U_all**2 + V_all**2)

# Get the astrometry from the minimum of the total chi2 map
xp, yp = x[idx], y[idy]
sep_p = np.sqrt(xp**2 + yp**2)
pa_p = np.arctan2(xp, yp)
print('Astrometry from the total chi2 map: ', xp, yp, sep_p, np.rad2deg(pa_p))

# Compute the spatial frequencies projected on these coordinates
Bproj = Bcov * np.cos(PAcov - pa_p) * mas2rad(sep_p)
freqProj = np.outer(Bproj, 1/(wl*1e-6))

print(f'Median transmission ratio: {np.median(transmission_all)}')
print(f'Median stellar polynomial coefficients: {np.median(stellar_coeffs_all, axis=0)}')

# Compute the stellar residual component (previously fitted with astrometry)
stellar_part = np.zeros_like(cf_amp_all, dtype=complex)
#Css = np.zeros_like(cf_amp_all, dtype=complex)
Css_real = np.zeros_like(cf_amp_all, dtype=complex)
Css_imag = np.zeros_like(cf_amp_all, dtype=complex)
Cps = np.zeros_like(cf_amp_all, dtype=float)
# for iB in range(nb):
#     Css[iB] = Polynomial(stellar_coeffs_all[iB])(wl)
#     stellar_part[iB] = Css[iB] * np.exp(2j * np.pi * freqProj[iB])

#     Cps[iB] = np.real(cf_all[iB] - stellar_part[iB])

#     amp_Cps = np.abs(Cps[iB])
#     phi_Cps = np.angle(Cps[iB])
#     # Remove the slope from the phase model, like in the data
#     wl_mask_lin = (wl > 3.1) & (wl < 4.0)
#     mean_inv_wl = np.mean(1/wl[wl_mask_lin])
#     mean_Cps = np.mean(phi_Cps[wl_mask_lin])
#     slope = np.sum((1/wl[wl_mask_lin] - mean_inv_wl) * (phi_Cps[wl_mask_lin] - mean_Cps)) / np.sum((1/wl[wl_mask_lin] - mean_inv_wl) ** 2)
#     intercept = mean_Cps - slope * mean_inv_wl
#     phi_Cps = wrap(phi_Cps - slope / wl - intercept)
    
#     Cps[iB] = amp_Cps * np.exp(1j * phi_Cps)

for iB in range(nb):
    #Css[iB] = Polynomial(stellar_coeffs_all[iB])(wl)
    #stellar_part[iB] = Css[iB] * np.exp(2j * np.pi * freqProj[iB])
    Css_real[iB] = Polynomial(stellar_coeffs_all[iB, :n_poly+1])(wl)
    Css_imag[iB] = Polynomial(stellar_coeffs_all[iB, n_poly+1:])(wl)
    #stellar_part[iB] = Css[iB] * np.exp(2j * np.pi * freqProj[iB])
    stellar_part[iB] = Css_real[iB] * np.cos(2 * np.pi * freqProj[iB]) + 1j * Css_imag[iB] * np.sin(2 * np.pi * freqProj[iB])

    amp_stellar_part = np.abs(stellar_part[iB])
    phi_stellar_part = np.angle(stellar_part[iB])
    # Remove the slope from the phase model, like in the data
    wl_mask_lin = (wl > 3.1) & (wl < 4.0)
    mean_inv_wl = np.mean(1/wl[wl_mask_lin])
    mean_stellar_part = np.mean(phi_stellar_part[wl_mask_lin])
    slope = np.sum((1/wl[wl_mask_lin] - mean_inv_wl) * (phi_stellar_part[wl_mask_lin] - mean_stellar_part)) / np.sum((1/wl[wl_mask_lin] - mean_inv_wl) ** 2)
    intercept = mean_stellar_part - slope * mean_inv_wl
    phi_stellar_part = wrap(phi_stellar_part - slope / wl - intercept)
    
    stellar_part[iB] = amp_stellar_part * np.exp(1j * phi_stellar_part)

# Css = np.zeros_like(cf_amp_all, dtype=complex)
# Cps = np.zeros_like(cf_amp_all, dtype=float)
# for iB in range(nb):
#     Css[iB] = Polynomial(stellar_coeffs_all[iB, :n_poly+1])(wl) + 1j * Polynomial(stellar_coeffs_all[iB, n_poly+1:])(wl)


#Cps = (np.real(cf_all - stellar_part).T / transmission_all).T
#Cps = np.real((cf_all.T / transmission_all).T - stellar_part)
Cps = np.real(cf_all - stellar_part)
#Cps = np.real(Cps)
#Cps = np.real((cf_all - Css) * np.exp(-2j * np.pi * freqProj))

cf_real_err_all = np.sqrt((np.cos(cf_phi_all) * cf_amp_err_all)**2 + (cf_amp_all * np.sin(cf_phi_all) * cf_phi_err_all)**2)
#cf_real_err_all = (cf_real_err_all.T / transmission_all).T

# Method 1: inverse-variance weighted mean. Associated errors computed through propagation or sample estimation
#weights = 1 / cf_real_err_all**2
#contrast = np.sum(Cps * weights, axis=0) / np.sum(weights, axis=0)
##contrast_err = 1 / np.sqrt(np.sum(1/cf_real_err_all**2, axis=0))
##contrast_err = np.sqrt(np.sum(weights * (Cps - contrast)**2, axis=0) / np.sum(weights, axis=0))
#contrast_err = np.sqrt(np.sum((weights*(Cps-contrast))**2, axis=0) / np.sum(weights, axis=0)**2)
##contrast_err = np.sqrt(np.sum(weights * (Cps - contrast)**2, axis=0) / (np.sum(weights, axis=0) - np.sum(weights**2, axis=0) / np.sum(weights, axis=0)))

# Method 2: same but the variances are corrected by a chi2 factor.
#chi2r = np.mean(((Cps - np.mean(Cps, axis=0)) / cf_real_err_all) ** 2, axis=0)
# plt.figure()
# plt.semilogy(wl, chi2r, label='Chi2')
# plt.semilogy(wl, np.sqrt(chi2r), label='sqrt(Chi2)')
# plt.savefig(path_output + 'spectrometry/chi2_contrast.png')
# weights = 1 / (chi2r * cf_real_err_all**2)
# contrast = np.sum(Cps * weights, axis=0) / np.sum(weights, axis=0)
# contrast_err = np.sqrt(np.sum(weights * (Cps - contrast)**2, axis=0) / np.sum(weights, axis=0))

# Method 3: simple mean and standard deviation estimation
#contrast = np.mean(Cps, axis=0)
#contrast_err = np.std(Cps, axis=0) / np.sqrt(Cps.shape[0])
#contrast_err = np.sqrt(np.sum(cf_real_err_all**2, axis=0)) / Cps.shape[0]

# Method 4: median and median absolute deviations
#contrast = np.median(Cps, axis=0)
#contrast_err = 1.4826 * np.median(np.abs(Cps - contrast), axis=0) / np.sqrt(Cps.shape[0]) # 1.4826: scale factor between MAD and StD in normal distributions
#contrast_err = np.sqrt(np.mean((Cps - contrast)**2, axis=0)) / np.sqrt(Cps.shape[0])

# Method 5: mean with outlier rejection
contrast = np.zeros_like(wl)
contrast_err = np.zeros_like(wl)
for iw in range(wl.size):
    low_limit = np.quantile(Cps[:, iw], 0.0)
    high_limit = np.quantile(Cps[:, iw], 1.0)
    Cps_valid = (Cps[:, iw] > low_limit) & (Cps[:, iw] < high_limit)
    contrast[iw] = np.mean(Cps[Cps_valid, iw])
    contrast_err[iw] = np.std(Cps[Cps_valid, iw]) / np.sqrt(Cps_valid.sum())

# Method 5: weighted mean with outlier rejection
# contrast = np.zeros_like(wl)
# contrast_err = np.zeros_like(wl)
# weights = 1 / cf_real_err_all**2
# weighted_Cps = Cps * weights
# for iw in range(wl.size):
#     low_limit = np.quantile(weighted_Cps[:, iw], 0.05)
#     high_limit = np.quantile(weighted_Cps[:, iw], 0.95)
#     Cps_valid = (weighted_Cps[:, iw] > low_limit) & (weighted_Cps[:, iw] < high_limit)
#     contrast[iw] = np.sum(Cps[Cps_valid, iw] * weights[Cps_valid, iw], axis=0) / np.sum(weights[Cps_valid, iw], axis=0)
#     contrast_err[iw] = np.sqrt(np.sum((weights[Cps_valid, iw]*(Cps[Cps_valid, iw]-contrast[iw]))**2, axis=0) / np.sum(weights[Cps_valid, iw], axis=0)**2)

chi2r = np.mean(((Cps - contrast) / cf_real_err_all) ** 2, axis=0)
plt.figure()
plt.semilogy(wl, chi2r, label='Chi2')
plt.semilogy(wl, np.sqrt(chi2r), label='sqrt(Chi2)')
plt.legend()
plt.savefig(path_output + '/spectrometry/chi2_contrast.png')

cov_Cps = np.corrcoef(Cps.T)
plt.figure()
plt.imshow(cov_Cps, cmap='jet')
plt.colorbar()
plt.savefig(path_output + '/spectrometry/corr_Cps.png')

Cps_base = np.zeros((Cps.shape[0]//6, 6, wl.size))
for i in range(6):
    Cps_base[:, i] = Cps[i::6]
Cps_base = Cps_base.reshape(Cps.shape[0]//6, -1)
cov_Cps_wlbase = np.corrcoef(Cps_base.T)
plt.figure()
plt.imshow(cov_Cps_wlbase, cmap='jet')
plt.colorbar()
plt.savefig(path_output + '/spectrometry/corr_Cps_wlbase.pdf')

# Error distribution plots

#print(np.min((Cps-contrast)/contrast_err), np.max((Cps-contrast)/contrast_err))
wlmask1 = (wl > 3.0) & (wl < 4.0)
#all_devs = np.ravel((Cps[:, wlmask1]-contrast[wlmask1])/(contrast_err[wlmask1]*chi2r[wlmask1]))
all_devs = np.ravel((Cps[:, wlmask1]-contrast[wlmask1])/contrast_err[wlmask1])

plt.figure()
plt.hist(all_devs, bins=500, log=True)
plt.savefig(path_output + '/spectrometry/error_histogram.png')

plt.figure()
bins = np.arange(-10, 10.01, 0.1)
half_bins = (bins[:-1] + bins[1:]) / 2
plt.hist(all_devs, bins=bins, log=True)

sigma = 10
normal_dist = np.diff(bins) * all_devs.size * 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(half_bins/sigma)**2)
#lorentz_dist = np.diff(bins) * all_devs.size * 1 / (np.pi * 30 * (1 + (half_bins/30)**2))
plt.plot(half_bins[normal_dist >= 1], normal_dist[normal_dist >= 1])
#plt.plot(half_bins[lorentz_dist >= 1], lorentz_dist[lorentz_dist >= 1])
#plt.plot(half_bins, np.diff(bins) * all_devs.size * 1/(25*np.sqrt(2*np.pi)) * np.exp(-(half_bins/25)**2))

#plt.ylim(1e-5)
plt.savefig(path_output + '/spectrometry/error_histogram_zoom.png')

# Plot the contrast S/N
plt.figure()
snr = contrast/contrast_err
plt.plot(wl, contrast/contrast_err)
plt.xlabel(r'Wavelength [$\mu$m]')
plt.ylabel('S/N')
plt.tight_layout()
plt.savefig(path_output + '/spectrometry/snr_contrast.png')

print(f'S/N: median (3-4 um): {np.median(snr[(wl>3.0)&(wl<4.0)])}, mean: {np.mean(snr[(wl>3.0)&(wl<4.0)])},\
      max: {np.max(snr[(wl>3.0)&(wl<4.0)])}')

### Plots ###

# Contrast plot
fig, ax = plt.subplots(figsize=(15 , 5))
#fig, ax = plt.subplots()
#ax.plot(wl, contrast, '+', label='MATISSE contrast')
ax.errorbar(wl, contrast, yerr=contrast_err, fmt='+', label='MATISSE contrast')
#ax.plot(wl, contrast, '+', c='tab:red')
#ax.set_ylim(0, 0.001)
ax.set_xlabel(r'Wavelength [$\mu$m]')
ax.set_ylabel('Planet-to-star contrast')
#ax.set_ylim(1e-4, 12e-4)
#ax.set_ylim(3e-4, 12e-4)
ax.set_ylim(0, 14e-4)
#ax.set_xlim(2.9, 4.15)
ax.set_xlim(2.76, 5.0)

#resid = residuals(res.x, wl, freqProj[iB], sep_p, transmission_ratio, stellar_residuals_poly_coeffs, cf_ratio)

contrast_tpl = fits.getdata('/data/home/jscigliuto/Pipeline/Templates/contrast_template_bt-settl_startpl.fits')
if use_bin_data:
    contrast_tpl = contrast_tpl.reshape(-1, 5).mean(axis=1)
contrast_tpl = contrast_tpl[wl_mask]
#ax.plot(wl, 1.1*contrast_tpl, zorder=100, label=r'Model: (BT-Settl, $T_{eff}$ = 1700 K, log(g) = 4.0) / (ISO Beta Pic spectrum)')

#ax.legend(loc='upper left')
ax.legend(fontsize='small', loc='lower right')
fig.tight_layout()
fig.savefig(path_output + '/spectrometry/contrast_nochi2.png')


## ISO stellar spectrum
# file_star = np.loadtxt('/data/home/jscigliuto/Pipeline/Templates/ISO_betaPic.txt')
# w_star, spec_star = file_star[:, 0], file_star[:, 1]
# spec_star_conv = instrBroadGaussFast(w_star, spec_star, resolution=np.mean((wl/wband)/5), equid=True)
# spec_star_conv = instrBroadGaussFast(w_star, spec_star, resolution=3000, equid=True)
# f_interp = scipy.interpolate.interp1d(w_star, spec_star_conv, fill_value='extrapolate')
# spec_star_conv = f_interp(wl)

## Model of the stellar spectrum
file_star = np.loadtxt('/data/home/jscigliuto/Pipeline/Templates/BT-NextGen_T7890K_lg3.8_M0.0_R15.5_res300.800.txt')
w_star, spec_star_conv_SI = file_star[:, 0], file_star[:, 1]
spec_star_conv_SI = spectres.spectres(wl[::-1], w_star, spec_star_conv_SI)
spec_star_conv = spec_star_conv_SI * 1e6
spec_star_conv = spec_star_conv * (wl[::-1]*1e-6)**2 / cst.c.value
spec_star_conv *= 1e26 # Jy
spec_star_conv = spec_star_conv[::-1]

# print('wlstar et specconvstarSI shape:', w_star.shape, spec_star_conv_SI.shape)


# Compute the MATISSE planetary spectrum
spec_planet = contrast * spec_star_conv
spec_planet_err = contrast_err * spec_star_conv


# Plot the stellar spectrum and its convolution
plt.figure()
# plt.scatter(w_star, spec_star_conv_SI, s=0.1, label='Stellar spectrum')
plt.scatter(wl, spec_star_conv_SI, s=0.1, label='Stellar spectrum')
plt.scatter(wl, spec_star_conv, s=0.1, c='tab:orange', label='Convolved stellar spectrum')
plt.xlim(3, 4)
plt.legend()
plt.title('Stellar spectrum (ISO)')
plt.xlabel(r'Wavelength [$\mu$m]')
plt.ylabel('Flux [Jy]')
#plt.savefig(path_output + 'spectrometry/spectrum_star_iso.png')
#plt.ylim(8, 17)

# Planetary spectrum model (BT-Settl / 1700 K / 4.0)
planet_tpl = fits.getdata('/data/home/jscigliuto/Pipeline/Templates/planet_spectrum_template_bt-settl.fits')
if use_bin_data:
    planet_tpl = planet_tpl.reshape(-1, 5).mean(axis=1)
planet_tpl = planet_tpl[wl_mask]
#plt.plot(wl, planet_tpl)

# Plot the planetary spectrum
plt.figure(figsize=(15 , 5))
#plt.plot(wl, spec_planet*1e3, '+', label='MATISSE spectrum')
plt.errorbar(wl, spec_planet*1e3, yerr=spec_planet_err*1e3, fmt='+', label='MATISSE spectrum')
# plt.plot(wl, sky_transmission*3+20, '--', label='SkyCalc transmission model [arbitrary units]')
# plt.plot(wl, 2*sky_radiance/1e7+15, '--', label='SkyCalc radiance model [arbitrary units]')
#plt.plot(wl, 1.1*planet_tpl*1e3, label=r'Model: BT-Settl, $T_{eff}$ = 1700 K, log(g) = 4.0', zorder=100)

# for i in range(1, 4):
#    spec_naco = fits.getdata(f'./templates/NACO/out_specfinal_weigthSD{i}.fits')
#    plt.plot(spec_naco[:, 0], spec_naco[:, 1]/10)

plt.xlabel(r'Wavelength [$\mu$m]')
plt.ylabel('Flux [mJy]')
plt.legend(loc='upper left')
#plt.ylim(4, 25)
#plt.ylim(1, 14)
plt.ylim(0, 30)
#plt.axvline(4.15, ls='--', c='k')
#plt.axvline(2.85, ls='--', c='k')
#plt.axvline(4.55, ls='--', c='k')
#plt.ylim(6, 13)
#plt.xlim(3, 4.15)
plt.xlim(2.76, 5.0)
#plt.plot(wl, C/np.sum(C), label='MATISSE contrast')
#plt.plot(wl, 1.1*planet_tpl*1e3, label=r'Model: BT-Settl, $T_{eff}$ = 1700 K, log(g) = 4.0', zorder=100)
plt.tight_layout()
plt.savefig(path_output + '/spectrometry/spectrum_planet_simple_nochi2.png')

spec_planet_SI = spec_planet * 1e-26 * (cst.c.value / (wl*1e-6)**2) * 1e-6
spec_planet_err_SI = spec_planet_err * 1e-26 * (cst.c.value / (wl*1e-6)**2) * 1e-6 

plt.figure(figsize=(15,5))
plt.errorbar(wl, spec_planet_SI, yerr=spec_planet_err_SI, label='MATISSE spectrum')
plt.xlabel(r'Wavelength [$\mu$m]')
plt.ylabel(r'Flux [W/m$^2$/µm]')
plt.legend(loc='upper left')
plt.ylim(0, 3e-15)
plt.xlim(2.76, 5.0)
plt.tight_layout()
plt.savefig(path_output + '/spectrometry/spectrum_planet_simple_nochi2_SI.png')

# Plot the planetary spectrum
from cycler import cycler
cycler_custom = cycler(color=('b', 'r', 'g', 'k', 'orange', 'purple'))
fig, ax = plt.subplots(figsize=(15 , 5))
#plt.plot(wl, (Cps*spec_star_conv).T*1e3, ms=0.1, marker='+', ls='', c='skyblue')
ax.plot(wl, (Cps*spec_star_conv).T*1e3, ms=0.1, marker='+', ls='', c='skyblue')
ax.set_prop_cycle(cycler_custom)
plt.errorbar(wl, spec_planet*1e3, yerr=spec_planet_err*1e3, fmt='+', label='MATISSE spectrum', c='steelblue', zorder=100)
plt.xlabel(r'Wavelength [$\mu$m]')
plt.ylabel('Flux [mJy]')
plt.legend(loc='upper left')
plt.ylim(-10, 30)
plt.xlim(2.76, 5.0)
plt.tight_layout()
plt.axvline(2.80, c='k')
plt.axvline(2.85, c='k')
plt.axvline(2.87, c='k')
plt.axvline(2.90, c='k')
plt.axvline(4.15, c='k')
plt.axvline(4.50, c='k')
plt.axvline(4.55, c='k')
plt.axvline(4.57, c='k')
plt.axvline(4.60, c='k')
plt.axvline(5.00, c='k')
plt.savefig(path_output + '/spectrometry/spectrum_planet_simple_nochi2_allspectra.png')

# Plot the planetary spectrum
plt.figure(figsize=(15 , 5))
#plt.plot(wl, (Cps*spec_star_conv).T*1e3, ms=0.1, marker='+', ls='', c='skyblue')
plt.plot(wl, spec_planet*1e3, label='MATISSE spectrum', c='steelblue', zorder=100)
plt.fill_between(wl, (spec_planet-spec_planet_err)*1e3, (spec_planet+spec_planet_err)*1e3)
plt.xlabel(r'Wavelength [$\mu$m]')
plt.ylabel('Flux [mJy]')
plt.legend(loc='upper left')
plt.ylim(-10, 25)
plt.xlim(2.76, 5.0)
plt.tight_layout()
plt.savefig(path_output + '/spectrometry/spectrum_planet_nochi2_filled.png')

# Save the spectra
fits.writeto(path_output + '/spectrometry/contrast_planet_nochi2.fits', np.array([wl, contrast, contrast_err]), overwrite=True)
fits.writeto(path_output + '/spectrometry/spectrum_planet_nochi2.fits', np.array([wl, spec_planet, spec_planet_err]), overwrite=True)

# Save the spectra in the ForMoSA format

wres = (wl / wband) / 5
instrument_col = np.full(wl.size, 'MATISSE-LM-MED')

spec_planet_formosa = spec_planet * 1e-26 * (cst.c.value / (wl*1e-6)**2) * 1e-6
spec_planet_err_formosa = spec_planet_err * 1e-26 * (cst.c.value / (wl*1e-6)**2) * 1e-6

plt.figure()
plt.errorbar(wl, spec_planet_formosa, yerr=spec_planet_err_formosa)
plt.ylim(0, 3e-15)
plt.xlabel(r'Wavelength [$\mu$m]')
plt.ylabel('Flux [W/m2/um]')
plt.savefig(path_output + '/spectrometry/spectrum_ForMoSA.png')

t = Table([wl[::-1], spec_planet_formosa[::-1], spec_planet_err_formosa[::-1], wres[::-1], instrument_col], names=('WAV', 'FLX', 'ERR', 'res', 'INS'))
#t.write(path_output + 'spectrometry/spectrum_BetaPicb_formosa.fits', format='fits', overwrite=True)
t.write(path_output + '/spectrometry/spectrum_BetaPicb_MATISSE-LM.fits', format='fits', overwrite=True)

t = Table([wl[::-1], contrast[::-1], contrast_err[::-1], wres[::-1], instrument_col], names=('WAV', 'CONTRAST', 'ERR', 'res', 'INS'))
#t.write(path_output + 'spectrometry/contrast_BetaPicb_formosa.fits', format='fits', overwrite=True)
t.write(path_output + '/spectrometry/contrast_BetaPicb_MATISSE-LM.fits', format='fits', overwrite=True)



# Add the GRAVITY spectrum

#hdul_grav = fits.open('/Users/mhoulle/Desktop/betPic-MATISSE-2022-11-09/GRAVITY_data/fits/spectrum.fit')
hdul_grav = fits.open('/Users/mhoulle/Desktop/betPic-MATISSE-2022-11-09/GRAVITY_data/BetaPictorisb_MR.fits')
#wl_grav = hdul_grav['WAVELENGTH'].data
wl_grav = hdul_grav['SPECTRUM'].data['WAVELENGTH'] * 1e6
#spec_grav = hdul_grav[0].data
#spec_err_grav = np.sqrt(np.diag(hdul_grav['COV'].data))
contrast_grav = hdul_grav['SPECTRUM'].data['CONTRAST']
contrast_err_grav = np.sqrt(np.diag(hdul_grav['SPECTRUM'].data['COVARIANCE_CONTRAST'].data))

# Matthieu Ravet's stellar template
file_star_grav = np.loadtxt('/Users/mhoulle/Desktop/betPic-MATISSE-2022-11-09/templates/stellar_mod_GRAVITY_R500.txt')
w_star_grav, spec_star_grav = file_star_grav[:, 0], file_star_grav[:, 1]
file_star_mat = np.loadtxt('/Users/mhoulle/Desktop/betPic-MATISSE-2022-11-09/templates/stellar_mod_MATISSE.txt')
w_star_mat, spec_star_mat = file_star_mat[:, 0], file_star_mat[:, 1]

spec_grav = contrast_grav * spec_star_grav
spec_err_grav = contrast_err_grav * spec_star_grav

res_grav = 500 * np.ones_like(wl_grav)
ins_grav = np.full(wl_grav.size, 'GRAVITY-K-MED')

spec_mat = contrast[::-1] * spec_star_mat
spec_err_mat = contrast_err[::-1] * spec_star_mat

wl_gravmat = np.hstack((wl_grav, wl[::-1]))
spec_gravmat = np.hstack((spec_grav, spec_mat))
spec_err_gravmat = np.hstack((spec_err_grav, spec_err_mat))
res_gravmat = np.hstack((res_grav, wres[::-1]))
ins_gravmat = np.hstack((ins_grav, instrument_col))

t = Table([wl_gravmat, spec_gravmat, spec_err_gravmat, res_gravmat, ins_gravmat], names=('WAV', 'FLX', 'ERR', 'res', 'INS'))
t.write(path_output + '/spectrometry/spectrum_BetaPicb_formosa_GRA+MAT.fits', format='fits', overwrite=True)