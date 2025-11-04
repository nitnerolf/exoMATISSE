# Importation
import numpy as np
from astropy.io import fits
import os 
import matplotlib.pyplot as plt
import scipy 
import shutil
from datetime import datetime
import multiprocessing as mp
from common_tools import reorder_baselines, wrap, mas2rad


# Path of the OiFits files (outputs of the pipeline, phase corrected)
# path_oifits = '/Users/jscigliuto/Nextcloud/DATA/HD72946B/corrected_data_wo_rmnrec/' #HD 72946 B
path_oifits = '/Users/jscigliuto/Desktop/Licallo_backup/Pipeline/betaPicb/corrPhaseMathis_MACAO/corrected_data_bin/' #beta Pic b

# Output path 
# path_output = '/Users/jscigliuto/exoMATISSE/exoMATISSE/data_hd72946b/test/' #HD 72946 B
path_output = '/Users/jscigliuto/exoMATISSE/exoMATISSE/data_betaPicb/' #beta Pic b

# Science case
sci_case = 'bright' #'faint'   

# Baseline order and names
base_order_name = ('U3-U4', 'U1-U2', 'U2-U3', 'U2-U4', 'U1-U3', 'U1-U4')


### Initializations 
# Planet offsets coords mentioned in the OB [mas]
# Offset_RA = 106
# Offset_Dec = -145

# # Grid of coordinates to determine the astrometry of the planet
# x      = np.arange(Offset_RA-50, Offset_RA+50, 0.4)
# y      = np.arange(Offset_Dec-50, Offset_Dec+50, 0.4)
# xp, yp = np.meshgrid(x, y) #grid of coords to look for the planet position

# 
n_poly = 1 # Degree of the polynomial to model the stellar speckle

# Wavelength range to fit
wmin = 3.0e-6
wmax = 4.15e-6


# List the Cps files
Cps_path = 'Contrast_fits/'
files_Cps = sorted([file for file in os.listdir(path_output + Cps_path) if '.fits' in file])
print('Cps path', path_output + Cps_path)

# Path to Cps model 
# Cps_model_path = '/Users/jscigliuto/Nextcloud/Py/MATISSE-DataProcessing/Spectrum/computed_spectra/contrast_template_bt-settl_hd72946_ph1ld.fits' #HD 72946 B
Cps_model_path = '/Users/jscigliuto/Nextcloud/Py/MATISSE-DataProcessing/Spectrum/computed_spectra/contrast_template_bt-settl_betPic.fits' #beta Pic b

# Get the contrast template
use_bin_data = True

# Create output directories if they do not exist
if not os.path.isdir(path_output + '/chi2_maps'):
    os.makedirs(path_output + '/chi2_maps')
if os.path.isdir(path_output + '/chi2_maps'):
    shutil.rmtree(path_output + '/chi2_maps')
    os.makedirs(path_output + '/chi2_maps')
if not os.path.isdir(path_output + '/chi2_maps_fits'):
    os.makedirs(path_output + '/chi2_maps_fits')
if os.path.isdir(path_output + '/chi2_maps_fits'):
    shutil.rmtree(path_output + '/chi2_maps_fits')
    os.makedirs(path_output + '/chi2_maps_fits')
if not os.path.isdir(path_output + '/fitted_params'):
    os.makedirs(path_output + '/fitted_params')
if os.path.isdir(path_output + '/fitted_params'):
    shutil.rmtree(path_output + '/fitted_params')
    os.makedirs(path_output + '/fitted_params')



def model_bright_split(params, wl, spat_freq, Cps_model):
    """
    Model with separate polynomials for real and imaginary parts
    """
    # Extract parameters
    alpha = params[0]
    
    stellar_coeffs = params[1:]
    n_coeffs = stellar_coeffs.size // 2
    stellar_coeffs_real = stellar_coeffs[:n_coeffs]
    stellar_coeffs_imag = stellar_coeffs[n_coeffs:]
    
    # Polynômes séparés
    stellar_poly_real = np.polyval(stellar_coeffs_real, wl)
    stellar_poly_imag = np.polyval(stellar_coeffs_imag, wl)
    # stellar_poly_real = np.polynomial.Polynomial(stellar_coeffs_real)(wl)
    # stellar_poly_imag = np.polynomial.Polynomial(stellar_coeffs_imag)(wl)
    
    # Compute model 
    cf_model = alpha * Cps_model + stellar_poly_real * np.cos(2 * np.pi * spat_freq) + 1j * stellar_poly_imag * np.sin(2 * np.pi * spat_freq)

    # Go back to amplitude/phase space
    amp_model = np.abs(cf_model)
    phi_model = np.angle(cf_model)

    return amp_model, phi_model

def residuals_split(params, spat_freq, wl, Cps_model, Cps_real_cal, 
                    Cps_real_cal_err, Cps_imag_cal, Cps_imag_cal_err):
    """
    Residuals with separate polynomials for real and imaginary parts
    """
    # Models
    amp_model, phi_model = model_bright_split(params, wl, spat_freq, Cps_model)

    # Reconstruct complex quantities    
    cf_model = amp_model * np.exp(1j * phi_model)
    cf_data  = Cps_real_cal + 1j * Cps_imag_cal

    # Compute residuals
    chi2_real = ((np.real(cf_data) - np.real(cf_model)) / Cps_real_cal_err)**2
    chi2_imag = ((np.imag(cf_data) - np.imag(cf_model)) / Cps_imag_cal_err)**2
    chi2      = np.sum(chi2_real + chi2_imag)
    chi2_red  = chi2 / (len(wl) - len(params))

    return chi2_red


def residuals_fixed_alpha_split(stellar_coeffs, alpha, spat_freq, wl, Cps_model, 
                                Cps_real_cal, Cps_real_cal_err, 
                                Cps_imag_cal, Cps_imag_cal_err):
    """
    Residuals with fixed alpha, optimizes only stellar_coeffs (real and imag)
    stellar_coeffs contains [coeffs_real, coeffs_imag]
    """
    params = np.concatenate([[alpha], stellar_coeffs])
    return residuals_split(params, spat_freq, wl, Cps_model, 
                          Cps_real_cal, Cps_real_cal_err, 
                          Cps_imag_cal, Cps_imag_cal_err)

def global_fit_objective_split(alpha, files_Cps, all_data_dict, n_poly):
    """
    Globaly optimizes alpha, while fitting stellar_coeffs (real + imag) independently for each file
    """
    total_chi2 = 0
    
    for file_name in files_Cps:
        data = all_data_dict[file_name]
        
        # Initialisation: n_poly+1 coeffs pour réel + n_poly+1 coeffs pour imag
        stellar_coeffs_init = [1e-4] * (2 * (n_poly + 1))
        bounds_stellar = [(None, None)] * (2 * (n_poly + 1))
        
        for i_base in range(6):
            res = scipy.optimize.minimize(
                residuals_fixed_alpha_split, 
                stellar_coeffs_init,
                args=(alpha, data['spat_freq'][i_base], data['wl'], 
                      Cps_model, data['Cps_real_cal'][i_base], 
                      data['Cps_real_cal_err'][i_base],
                      data['Cps_imag_cal'][i_base], 
                      data['Cps_imag_cal_err'][i_base]),
                bounds=bounds_stellar,
                method='L-BFGS-B'
            )
            total_chi2 += res.fun
    
    return total_chi2



if sci_case == 'bright':
    n_base = 6

    start_time = datetime.now()
    with open(path_output + '/log.txt', 'w') as log_txt:
        log_txt.write(f'Starting the fitting' + str(start_time) + '\n')
    
    # Load all the data in one dictionary
    all_data_dict = {}
    
    for i_file, file_Cps in enumerate(files_Cps):
        print(f"Loading file {i_file+1}/{len(files_Cps)}: {file_Cps}")
        
        # Load Cps data per file
        hdu = fits.open(path_output + Cps_path + file_Cps)
        wl = hdu['WAVELENGTH'].data  
        Cps_real_cal = hdu['CPS_REAL'].data
        Cps_real_cal_err = hdu['CPS_REAL_ERR'].data
        Cps_imag_cal = hdu['CPS_IMAG'].data
        Cps_imag_cal_err = hdu['CPS_IMAG_ERR'].data
        U = hdu['U'].data
        V = hdu['V'].data
        Bproj = np.sqrt(U**2+V**2) * mas2rad(534) * np.cos(np.arctan2(U,V) - np.deg2rad(31.52)) # at the found position Houllé+2025
        spat_freq = np.outer(Bproj, 1/wl) 

        # Apply wavelength mask (wmin, wmax)
        wl_mask = (wl > wmin) & (wl < wmax)
        wl = wl[wl_mask]
        Cps_real_cal = Cps_real_cal[:, wl_mask]
        Cps_real_cal_err = Cps_real_cal_err[:, wl_mask]
        Cps_imag_cal = Cps_imag_cal[:, wl_mask]
        Cps_imag_cal_err = Cps_imag_cal_err[:, wl_mask]
        spat_freq = spat_freq[:, wl_mask]

        Cps_model = fits.getdata(Cps_model_path)
        if use_bin_data:
            Cps_model = Cps_model.reshape(-1, 5).mean(axis=1)
        Cps_model = Cps_model[wl_mask]

        # Store the data in all_data_dict[file_Cps]
        all_data_dict[file_Cps] = {
            'wl': wl,
            'Cps_real_cal': Cps_real_cal,
            'Cps_real_cal_err': Cps_real_cal_err,
            'Cps_imag_cal': Cps_imag_cal,
            'Cps_imag_cal_err': Cps_imag_cal_err,
            'spat_freq':  spat_freq
        }
    
    # Global optimization of alpha 
    print("Optimizing the alpha...")
    alpha_init = 1.0
    
    result_alpha = scipy.optimize.minimize_scalar(
        lambda a: global_fit_objective_split(a, files_Cps, all_data_dict, n_poly),
        # bounds=(0, 10),
        method='brent'
    )
    
    alpha_optimal = result_alpha.x
    print(f"Best alpha: {alpha_optimal}")
    
    # Final fit of stellar_coeffs per file with optimal alpha fixed
    for i_file, file_Cps in enumerate(files_Cps):
        print(f"Fit final file {i_file+1}/{len(files_Cps)}")

        data = all_data_dict[file_Cps]
        fitted_stellar_coeffs = np.zeros((n_base, 2 * (n_poly + 1)))
        chi2_map_file = np.zeros((n_base))

        stellar_coeffs_init = [1e-4] * (2 * (n_poly + 1))
        
        for i_base in range(n_base):
            res = scipy.optimize.minimize(
                residuals_fixed_alpha_split,
                stellar_coeffs_init,
                args=(alpha_optimal, data['spat_freq'][i_base], 
                      data['wl'], Cps_model,
                      data['Cps_real_cal'][i_base], 
                      data['Cps_real_cal_err'][i_base],
                      data['Cps_imag_cal'][i_base], 
                      data['Cps_imag_cal_err'][i_base]),
                bounds=[(None, None)] * (2 * (n_poly + 1)),
                method='L-BFGS-B'
            )
            fitted_stellar_coeffs[i_base] = res.x
            chi2_map_file[i_base] = res.fun
        
        # Save fitted parameters
        np.save(path_output + f'/fitted_params/stellar_coeffs_{file_Cps}.npy', fitted_stellar_coeffs)
        np.save(path_output + f'/chi2_maps/chi2_{file_Cps}.npy', chi2_map_file)

    np.save(path_output + '/fitted_params/alpha_global.npy', alpha_optimal)
    print(f"\nAlpha global sauvegardé: {alpha_optimal}")
        
    with open(path_output + '/log.txt', 'a') as log_txt:
        log_txt.write(f'End of fitting for {file_Cps} at ' + str(datetime.now()) + '\n')
        log_txt.write(f'Total duration: ' + str(datetime.now() - start_time) + '\n')


# Plot 
for i_file, file_Cps in enumerate(files_Cps):
    # Load fitted parameters
    alpha_optimal = np.load(path_output + f'/fitted_params/alpha_global.npy')
    fitted_stellar_coeffs = np.load(path_output + f'/fitted_params/stellar_coeffs_{file_Cps}.npy')
    
    # Load data
    data = all_data_dict[file_Cps]
    wl = data['wl']
    spat_freq = data['spat_freq']
    Cps_real_cal = data['Cps_real_cal']
    Cps_real_cal_err = data['Cps_real_cal_err']
    Cps_imag_cal = data['Cps_imag_cal']
    Cps_imag_cal_err = data['Cps_imag_cal_err']
    
    n_base = spat_freq.shape[0]
    
    # Prepare figure
    fig_Cps_real, ax_Cps_real = plt.subplots(n_base, 1, figsize=(8, 3*n_base), sharex=True)
    fig_Cps_imag, ax_Cps_imag = plt.subplots(n_base, 1, figsize=(8, 3*n_base), sharex=True)
    
    for i_base in range(n_base):
        # Compute model with fitted parameters
        params = np.concatenate([[alpha_optimal], fitted_stellar_coeffs[i_base]])
        amp_model, phi_model = model_bright_split(params, wl, spat_freq[i_base], Cps_model)
        
        cf_model = amp_model * np.exp(1j * phi_model)
        
        Cps_real_model = np.real(cf_model)
        Cps_imag_model = np.imag(cf_model)

        
        # Plot real part
        ax_Cps_real[i_base].errorbar(wl*1e6, Cps_real_cal[i_base], yerr=np.sqrt(Cps_real_cal_err[i_base]**2), fmt='o', label='Data', color='blue', alpha=0.5)
        ax_Cps_real[i_base].plot(wl*1e6, Cps_real_model, label='Model', color='red')
        ax_Cps_real[i_base].plot(wl*1e6, alpha_optimal * Cps_real_model + fitted_stellar_coeffs[i_base][0]*wl*1e6 + fitted_stellar_coeffs[i_base][1], label='Stellar Poly', color='purple', linestyle='--')
        ax_Cps_real[i_base].plot(wl*1e6, alpha_optimal * Cps_real_model - fitted_stellar_coeffs[i_base][0]*wl*1e6 + fitted_stellar_coeffs[i_base][1], color='purple', linestyle='--')
        ax_Cps_real[i_base].set_ylabel(f'{base_order_name[i_base]} Cps Real')
        ax_Cps_real[i_base].set_ylim(-3e-3, 3e-3)
        ax_Cps_real[i_base].set_xlim(3.0, 4.15)
        ax_Cps_real[i_base].legend()

        # Plot imaginary part
        ax_Cps_imag[i_base].errorbar(wl*1e6, Cps_imag_cal[i_base], yerr=np.sqrt(Cps_imag_cal_err[i_base]**2), fmt='o', label='Data', color='green', alpha=0.5)
        ax_Cps_imag[i_base].plot(wl*1e6, Cps_imag_model, label='Model', color='orange')
        ax_Cps_imag[i_base].plot(wl*1e6, fitted_stellar_coeffs[i_base][2]*wl*1e6 + fitted_stellar_coeffs[i_base][3], label='Stellar Poly', color='purple', linestyle='--')
        ax_Cps_imag[i_base].plot(wl*1e6, -fitted_stellar_coeffs[i_base][2]*wl*1e6 + fitted_stellar_coeffs[i_base][3], color='purple', linestyle='--')
        ax_Cps_imag[i_base].set_ylabel(f'{base_order_name[i_base]} Cps Imag')
        ax_Cps_imag[i_base].set_ylim(-4e-3, 4e-3)
        ax_Cps_imag[i_base].set_xlim(3.0, 4.15)
        ax_Cps_imag[i_base].legend()
    
    ax_Cps_real[-1].set_xlabel('Wavelength [µm]')
    ax_Cps_imag[-1].set_xlabel('Wavelength [µm]')
    fig_Cps_real.suptitle(f'Fitted Cps - {file_Cps}')
    fig_Cps_imag.suptitle(f'Fitted Cps - {file_Cps}')
    fig_Cps_real.savefig(path_output + f'/fitted_params/fitted_Cps_real_{file_Cps[:-5]}.png', dpi=300, bbox_inches='tight')
    fig_Cps_imag.savefig(path_output + f'/fitted_params/fitted_Cps_imag_{file_Cps[:-5]}.png', dpi=300, bbox_inches='tight')
    plt.close(fig_Cps_real)
    plt.close(fig_Cps_imag)