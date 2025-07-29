#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Fits the planet astrometry and stellar contamination in a single planet frame.

# Authors:
#   M. Houllé

import numpy as np
import scipy
import multiprocessing as mp
import sys
import os

from datetime import datetime
from astropy.io import fits
from numpy.polynomial import Polynomial
from common_tools import wrap, reorder_baselines, mas2rad


### Function definitions ###

def model(params, wl, freqProj, Cps):
    # Fitted parameters
    trans_ratio = params[0]
    stellar_residuals_poly_coeffs = params[1:]
    # Complex polynomial modelling the stellar residuals
    n_coeffs = stellar_residuals_poly_coeffs.size // 2
    Css_real = Polynomial(stellar_residuals_poly_coeffs[:n_coeffs])(wl)
    Css_imag = Polynomial(stellar_residuals_poly_coeffs[n_coeffs:])(wl)
    # Model
    cf_model = trans_ratio * Cps + Css_real * np.cos(2 * np.pi * freqProj) + 1j * Css_imag * np.sin(2 * np.pi * freqProj)
    amp_model = np.abs(cf_model)
    phi_model = np.angle(cf_model)
    return amp_model, phi_model

def residuals(params, wl, freqProj, Cps, real_err_mask, imag_err_mask, cf_real, cf_real_err, cf_imag, cf_imag_err):
    # Model
    amp_model, phi_model = model(params, wl, freqProj, Cps)

    # Remove the slope from the phase model, like in the data
    wl_mask_lin = (wl > 3.1) & (wl < 4.0)
    mean_inv_wl = np.mean(1 / wl[wl_mask_lin])
    mean_phi_model = np.mean(phi_model[wl_mask_lin])
    slope = np.sum((1/wl[wl_mask_lin] - mean_inv_wl) * (phi_model[wl_mask_lin] - mean_phi_model)) / np.sum((1/wl[wl_mask_lin] - mean_inv_wl) ** 2)
    intercept = mean_phi_model - slope * mean_inv_wl
    phi_model = wrap(phi_model - slope / wl - intercept)

    # Recombine
    cf_model = amp_model * np.exp(1j * phi_model)
    
    # Residuals
    chi2_real = ((cf_real - np.real(cf_model)) / cf_real_err) ** 2
    chi2_imag = ((cf_imag - np.imag(cf_model)) / cf_imag_err) ** 2
    chi2 = np.sum(chi2_real[real_err_mask]) + np.sum(chi2_imag[imag_err_mask])
    chi2_red = chi2 / (real_err_mask.sum() + imag_err_mask.sum() - params.size)
    return chi2_red

def residuals_full_output(params, wl, freqProj, Cps, real_err_mask, imag_err_mask, cf_real, cf_real_err, cf_imag, cf_imag_err):
    # Model
    amp_model, phi_model = model(params, wl, freqProj, Cps)
    
    # Remove the slope from the phase model, like in the data
    wl_mask_lin = (wl > 3.1) & (wl < 4.0)
    mean_inv_wl = np.mean(1 / wl[wl_mask_lin])
    mean_phi_model = phi_model[wl_mask_lin].mean()
    slope = np.sum((1/wl[wl_mask_lin] - mean_inv_wl) * (phi_model[wl_mask_lin] - mean_phi_model)) / np.sum((1/wl[wl_mask_lin] - mean_inv_wl) ** 2)
    intercept = mean_phi_model - slope * mean_inv_wl
    phi_model = wrap(phi_model - slope / wl - intercept)
    
    # Real and imaginary parts
    cf_model = amp_model * np.exp(1j * phi_model)   
    
    # Residuals
    chi2_real = ((cf_real - np.real(cf_model)) / cf_real_err) ** 2
    chi2_real = np.sum(chi2_real[real_err_mask])
    chi2_real_red = chi2_real / (real_err_mask.sum() - (params.size-1)//2 - 1)

    chi2_imag = ((cf_imag - np.imag(cf_model)) / cf_imag_err) ** 2
    chi2_imag = np.sum(chi2_imag[imag_err_mask])
    chi2_imag_red = chi2_imag / (imag_err_mask.sum() - (params.size-1)//2)

    chi2 = chi2_real + chi2_imag
    chi2_red = chi2 / (real_err_mask.sum() + imag_err_mask.sum() - params.size)

    return chi2_red, chi2_real_red, chi2_imag_red

def fit(ix, iy, Bcov, PAcov, pa, sep, wl, n_base, params_init, Cps, real_err_mask, imag_err_mask, 
        cf_real, cf_real_err, cf_imag, cf_imag_err, bounds):
    # Compute projected spatial frequencies
    Bproj = Bcov * mas2rad(sep[iy, ix]) * np.cos(PAcov - pa[iy, ix]) # scalar product planet vector - baseline vector
    freqProj = np.outer(Bproj, 1/(wl*1e-6))

    fit_params_xy = np.zeros((n_base, len(params_init)))
    chi2_map_xy = np.zeros(n_base)
    chi2_map_real_xy = np.zeros(n_base)
    chi2_map_imag_xy = np.zeros(n_base)

    for iB in range(n_base):
        # Least-squares fitting
        res = scipy.optimize.minimize(residuals, params_init,
                        args=(wl, freqProj[iB], Cps, real_err_mask[iB], imag_err_mask[iB], 
                                cf_real[iB], cf_real_err[iB], cf_imag[iB], cf_imag_err[iB]),
                        bounds=bounds)# tol=1e-1, options={'maxiter': 10000000})
        # Save the fitted parameters
        fit_params_xy[iB] = res.x
        # Save the likelihood map
        chi2_map_xy[iB], chi2_map_real_xy[iB], chi2_map_imag_xy[iB] = residuals_full_output(fit_params_xy[iB], wl, freqProj[iB], Cps, 
                                                                                          real_err_mask[iB], imag_err_mask[iB], cf_real[iB],
                                                                                          cf_real_err[iB], cf_imag[iB], cf_imag_err[iB])
    
    return (ix, iy), fit_params_xy, chi2_map_xy, chi2_map_real_xy, chi2_map_imag_xy


if __name__ == '__main__':

    start_time = datetime.now()

    ###############################################################
    ### Input parameters (can be modified for your desired fit) ###
    ###############################################################

    # Are you using binned data?
    use_bin_data = False

    # Selected wavelength range for fitting
    #wmin, wmax = 3.0, 4.1 # µm (L band)
    wmin, wmax = 2.87, 5.0 # µm (L-M bands)

    # Reject an additional wavelength interval from fitting
    reject_interval = True
    witv_min, witv_max = 4.15, 4.57 # µm (CO2 feature between L and M)

    # SkyCalc radiance and transmission spectra (not used yet)
    #skycalc_radiance_file = 'skycalc_radiance_all.txt'
    #skycalc_transmission_file = 'skycalc_radiance_all.txt'

    # Selected grid of delta(alpha, delta) astrometries to test
    #x = np.arange(250, 300, 1) # mas
    #y = np.arange(425, 475, 1) # mas
    #x = np.arange(280-65, 280+66, 1) # mas
    #y = np.arange(455-65, 455+66, 1) # mas
    x = np.arange(279-15, 279+15, 0.4) # mas
    y = np.arange(455-15, 455+15, 0.4) # mas
    # x = np.arange(-280-10, -280+10.2, 0.4) # mas
    # y = np.arange(-455-10, -455+10.2, 0.4) # mas  
    # x = np.arange(-300, -250, 1) # mas
    # y = np.arange(-475, -425, 1) # mas
    # x = np.arange(250, 260, 1) # mas
    # y = np.arange(425, 429, 1) # mas

    # Hypothesis on the planet-to-star contrast (not fitted here)
    Cps_file = '/data/home/jscigliuto/Pipeline/Templates/contrast_template_bt-settl_startpl.fits'
    # Cps_file = 'flat 1.5e-4'

    # Degree of the polynomial used for fitting the stellar residual contrast
    n_poly = 1

    # Initial guesses on the fitted parameters
    #Css = 1e-4 # contrast of the stellar residuals at the planet location
    #stellar_residuals_poly_coeffs_init = (1e-6, 1e-5, 1e-4, 6e-4, 1e-6, 1e-5, 1e-4, 6e-4)
    #stellar_residuals_poly_coeffs_init = (1e-6, 1e-5, 1e-4, 6e-4)
    #stellar_residuals_poly_coeffs_init = (4.82483246e-06, 4.16249894e-06, -5.37115119e-05, 1.63427183e-05)
    #stellar_residuals_poly_coeffs_init = (1.32557020e-05,  2.61251921e-05, -6.78750772e-05,  1.76543933e-05) * 2
    # stellar_residuals_poly_coeffs_init = (1.32557020e-05,  2.61251921e-05, -6.78750772e-05,  1.76543933e-05) * 2
    stellar_residuals_poly_coeffs_init = [1e-5] * (n_poly+1) * 2
    #stellar_residuals_poly_coeffs_init = Css * np.ones((n_poly+1)*2) # Initial coefficients of the stellar residual polynomial
    transmission_ratio_init = 1.0 # Initial transmission ratio between stellar and planet exposures

    # Uncertainties to use for the weighted least squares
    # 'errors_from_data': errors estimated from the standard deviation of the frames within one exposure
    # 'errors_from_pipeline': errors provided by the MATISSE reduction pipeline
    weighted_least_squares = 'errors_from_pipeline'


    ###############################################################

    ### Start of script ###

    ### Save the input parameters ###

    # OIFITS input directory
    path_oifits = sys.argv[2]

    # Output directory
    path_output = sys.argv[3]

    with open(path_output + '/summary.txt', 'w') as params_txt:
        params_txt.write(f'Processing started at {str(start_time)}\n')
        params_txt.write(f'Coherent processing, parallelized\n')
        params_txt.write(f'Script: {os.path.basename(__file__)}\n')
        params_txt.write(f'Data directory: {path_oifits}\n')
        params_txt.write(f'Results directory: {path_output}\n')
        if reject_interval:
            params_txt.write(f'Wavelength range for fitting: [{wmin} - {witv_min}] and [{witv_max} - {wmax}] microns\n')
        else:
            params_txt.write(f'Wavelength range for fitting: [{wmin} - {wmax}] microns\n')
        params_txt.write(f'Grid of tested planet coordinates, x-axis: {x.min():.2f} to {x.max():.2f} mas, step of {x[1]-x[0]:.2f} mas\n')
        params_txt.write(f'Grid of tested planet coordinates, y-axis: {y.min():.2f} to {y.max():.2f} mas, step of {y[1]-y[0]:.2f} mas\n')
        params_txt.write(f'Contrast template: {Cps_file}\n')
        params_txt.write(f'Degree of the stellar residuals polynomial: {n_poly}\n')
        params_txt.write(f'Initial hypothesis on the polynomial coefficients: {stellar_residuals_poly_coeffs_init}\n')
        params_txt.write(f'Errors used for least squares fitting: {weighted_least_squares}\n')

    ### Read the data ###
    
    # File to be processed
    file = sys.argv[1]
    
    # Open the file
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
    U          = hdul['OI_VIS'].data['UCOORD']
    V          = hdul['OI_VIS'].data['VCOORD']
    wl         = hdul['OI_WAVELENGTH'].data['EFF_WAVE'] * 1e6

    hdul.close()

    # Get the contrast template
    Cps = fits.getdata(Cps_file)
    if use_bin_data:
        Cps = Cps.reshape(-1, 5).mean(axis=1)
    # Cps = np.ones_like(wl) * 1.5e-4
    
    # Filter wavelengths
    if reject_interval:
        wl_mask = (wl > wmin) & (wl < wmax) & ((wl < witv_min) | (wl > witv_max))
    else:
        wl_mask = (wl > wmin) & (wl < wmax)
    
    cf_amp     = cf_amp[:, wl_mask]
    cf_amp_err = cf_amp_err[:, wl_mask]
    cf_phi     = cf_phi[:, wl_mask]
    cf_phi_err = cf_phi_err[:, wl_mask]
    # t3_phi     = t3_phi[:, wl_mask]
    # t3_phi_err = t3_phi_err[:, wl_mask]
    wl  = wl[wl_mask]
    Cps = Cps[wl_mask]

    # Dimensions
    n_base = 6
    n_wave = wl.size

    # OB number
    i_OB = int(file[file.find('OB')+2:file.find('_exp')])

    files = sorted([file for file in os.listdir(path_oifits) if '.fits' in file and 'flagged' not in file])
    # OB_list_planet = set([int(file[file.find('OB')+2:file.find('_exp')]) for file in files if '_unknown' in file])
    OB_list_planet = set([int(file[file.find('OB')+2:file.find('_exp')]) for file in files if '_planet' in file])
    OB_list_star = set([int(file[file.find('OB')+2:file.find('_exp')]) for file in files if '_star' in file])

    OB_association = {key: value for key, value in zip(OB_list_planet, OB_list_star)}

    # Load the stellar average quantities for this OB
    cf_amp_star_data = fits.getdata(path_output + f'/stellar_OB_averages/star_avg_visamp_OB{OB_association[i_OB]}.fits')
    cf_amp_star, cf_amp_err_star = cf_amp_star_data[0], cf_amp_star_data[1]
    cf_phi_star_data = fits.getdata(path_output + f'/stellar_OB_averages/star_avg_visphi_OB{OB_association[i_OB]}.fits')
    cf_phi_star, cf_phi_err_star = cf_phi_star_data[0], cf_phi_star_data[1]
    # t3_phi_star_data = fits.getdata(path_output + f'/stellar_OB_averages/star_avg_t3phi_OB{i_OB}.fits')
    # t3_phi_star, t3_phi_err_star = t3_phi_star_data[0], t3_phi_star_data[1]
    
    # Calibrate the planet coherent flux with the stellar coherent flux (coherent flux ratio)
    cf_amp_cal = cf_amp / cf_amp_star[:, wl_mask]
    cf_phi_cal = wrap(cf_phi - cf_phi_star[:, wl_mask])
    # t3_phi_cal = wrap(t3_phi - t3_phi_star[:, wl_mask])

    # Complexify
    cf_cal = cf_amp_cal * np.exp(1j * cf_phi_cal)
    cf_real_cal = np.real(cf_cal)
    cf_imag_cal = np.imag(cf_cal)
    
    # Compute errors of the coherent flux ratio
    if weighted_least_squares == 'errors_from_pipeline':
        cf_amp_cal_err = cf_amp_cal * np.sqrt((cf_amp_err/cf_amp)**2 + (cf_amp_err_star[:, wl_mask]/cf_amp_star[:, wl_mask])**2)
        cf_phi_cal_err = np.sqrt(cf_phi_err**2 + cf_phi_err_star[:, wl_mask]**2)
        # t3_phi_cal_err = np.sqrt(t3_phi_err**2 + t3_phi_err_star[:, wl_mask]**2)
        cf_real_cal_err = np.sqrt((np.cos(cf_phi_cal) * cf_amp_cal_err)**2 + (cf_amp_cal * np.sin(cf_phi_cal) * cf_phi_cal_err)**2)
        cf_imag_cal_err = np.sqrt((np.sin(cf_phi_cal) * cf_amp_cal_err)**2 + (cf_amp_cal * np.cos(cf_phi_cal) * cf_phi_cal_err)**2)
    elif weighted_least_squares == 'errors_from_data':
        exposure_name = file[:file.find('_frame')]
        cf_amp_cal_err = fits.getdata(path_output + f'/error_estimates/{exposure_name}_cf_amp_cal_err.fits')[:, wl_mask]
        cf_phi_cal_err = fits.getdata(path_output + f'/error_estimates/{exposure_name}_cf_phi_cal_err.fits')[:, wl_mask]
        # t3_phi_cal_err = fits.getdata(path_output + f'/error_estimates/{exposure_name}_t3_phi_cal_err.fits')[:, wl_mask]
        cf_real_cal_err = np.sqrt((np.cos(cf_phi_cal) * cf_amp_cal_err)**2 + (cf_amp_cal * np.sin(cf_phi_cal) * cf_phi_cal_err)**2)
        cf_imag_cal_err = np.sqrt((np.sin(cf_phi_cal) * cf_amp_cal_err)**2 + (cf_amp_cal * np.cos(cf_phi_cal) * cf_phi_cal_err)**2)
    elif weighted_least_squares == 'none':
        raise ValueError('Not yet implemented!')
    else:
        raise ValueError('Weighted least squares mode not recognized.')
    
    # # Filter based on data quality (exclude extreme amplitude and phase errors)
    # amp_err_mask = (cf_amp_cal_err.T > np.quantile(cf_amp_cal_err, 0.02, axis=1)).T \
    #             & (cf_amp_cal_err.T < np.quantile(cf_amp_cal_err, 0.98, axis=1)).T
    # phi_err_mask = (cf_phi_cal_err.T > np.quantile(cf_phi_cal_err, 0.02, axis=1)).T \
    #             & (cf_phi_cal_err.T < np.quantile(cf_phi_cal_err, 0.98, axis=1)).T

    snr_real = np.abs(cf_real_cal / cf_real_cal_err)
    snr_imag = np.abs(cf_imag_cal / cf_imag_cal_err)

    # Filter based on data quality (exclude low SNRs)
    real_err_mask = (snr_real.T > np.quantile(snr_real, 0.05, axis=1)).T # \
                #& (snr_real.T < np.quantile(snr_real, 0.98, axis=1)).T
    imag_err_mask = (snr_imag.T > np.quantile(snr_imag, 0.05, axis=1)).T # \
                #& (snr_imag.T < np.quantile(snr_imag, 0.98, axis=1)).T
    
    # Grid of tested (alpha, delta) offsets from the star (mas) 
    xx, yy = np.meshgrid(x, y)

    # Associated PAs and separations
    pa  = np.arctan2(xx, yy)
    sep = np.sqrt(xx**2 + yy**2)

    # Baseline-PA coverage in the UV space
    PAcov = np.arctan2(U, V)
    Bcov  = np.sqrt(U**2 + V**2)
    
    params_init = np.array([transmission_ratio_init, *stellar_residuals_poly_coeffs_init])
    n_params = params_init.size
    #params_init = np.array(stellar_residuals_poly_coeffs_init)
    #n_params = params_init.size
    bounds = [(0, None), *[(None, None)]*(n_poly+1)*2]
    # bounds = [(0, None), *[(None, None)]*(n_poly+1)]
    # bounds = [*[(None, None)]*(n_poly+1)]
  

    # Initialize output arrays
    chi2_map = np.zeros((n_base, x.size, y.size))
    chi2_map_real = np.zeros((n_base, x.size, y.size))
    chi2_map_imag = np.zeros((n_base, x.size, y.size))
    fit_params = np.zeros((n_base, x.size, y.size, len(params_init)))
    
    ### Least-squares fitting position by position and baseline by baseline ###

    with open(path_output + '/log.txt', 'a') as log_txt:
        log_txt.write(f'Start multiprocessing for {file}\n')
        log_txt.write(f'params_init shape:{np.shape(params_init)} Bounds shape: {np.shape(bounds)} Bounds:{bounds}\n')
        
    
    time_start_fit = datetime.now()

    # Queue the fits of each position (one position fit per CPU)
    pool = mp.Pool(processes=mp.cpu_count())
    args = [(ix, iy, Bcov, PAcov, pa, sep, wl, n_base, params_init, Cps, real_err_mask, imag_err_mask,
             cf_real_cal, cf_real_cal_err, cf_imag_cal, cf_imag_cal_err, bounds) 
            for ix in range(x.size) for iy in range(y.size)]
    results = pool.starmap(fit, args)

    with open(path_output + '/log.txt', 'a') as log_txt:
        log_txt.write(f'Stellar polynomial coefficents: {fit_params}\n')

    # Extract the results and reorder them into the output arrays
    for result in results:
        coords, params, resid, resid_real, resid_imag = result
        ix, iy = coords
        chi2_map[:, ix, iy] = resid
        chi2_map_real[:, ix, iy] = resid_real
        chi2_map_imag[:, ix, iy] = resid_imag
        fit_params[:, ix, iy, :] = params
    
    with open(path_output + '/log.txt', 'a') as log_txt:
        log_txt.write(f'End of multiprocessing for {file} after ' + str(datetime.now() - time_start_fit) + '\n')

    ## Sum of chi2 maps of all baselines
    # Cancel the free parameter factor of the reduced chi2 maps
    chi2_map_nored = (chi2_map.T * (real_err_mask.sum(axis=1) + imag_err_mask.sum(axis=1) - n_params)).T
    chi2_map_real_nored = (chi2_map_real.T * (real_err_mask.sum(axis=1) - n_poly - 2)).T
    chi2_map_imag_nored = (chi2_map_imag.T * (imag_err_mask.sum(axis=1) - n_poly - 1)).T
    # Sum and reintroduce the total free parameter factor
    chi2_map_allbase = np.sum(chi2_map_nored, axis=0) / (real_err_mask.sum() + imag_err_mask.sum() - 6 * n_params)
    chi2_map_real_allbase = np.sum(chi2_map_real_nored, axis=0) / (real_err_mask.sum() - 6 * (n_poly + 2))
    chi2_map_imag_allbase = np.sum(chi2_map_imag_nored, axis=0) / (imag_err_mask.sum() - 6 * (n_poly + 1))

    # Save the results
    fits.writeto(path_output + f'/astrometry_fits_files/{file[:-5]}_chi2_map.fits', chi2_map, overwrite=True)
    fits.writeto(path_output + f'/astrometry_fits_files/{file[:-5]}_chi2_map_real.fits', chi2_map_real, overwrite=True)
    fits.writeto(path_output + f'/astrometry_fits_files/{file[:-5]}_chi2_map_imag.fits', chi2_map_imag, overwrite=True)
    fits.writeto(path_output + f'/astrometry_fits_files/{file[:-5]}_chi2_map_allbase.fits', chi2_map_allbase, overwrite=True)
    fits.writeto(path_output + f'/astrometry_fits_files/{file[:-5]}_chi2_map_real_allbase.fits', chi2_map_real_allbase, overwrite=True)
    fits.writeto(path_output + f'/astrometry_fits_files/{file[:-5]}_chi2_map_imag_allbase.fits', chi2_map_imag_allbase, overwrite=True)
    fits.writeto(path_output + f'/astrometry_fits_files/{file[:-5]}_fit_params.fits', fit_params, overwrite=True)

    # End of script
    end_time = datetime.now()
    with open(path_output + '/summary.txt', 'a') as params_txt:
        params_txt.write(f"Processing ended at {str(end_time)}\n")
        params_txt.write(f"Duration: {str(end_time - start_time)}")
