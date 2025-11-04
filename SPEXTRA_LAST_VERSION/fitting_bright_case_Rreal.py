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
Offset_RA = 106
Offset_Dec = -145

# Grid of coordinates to determine the astrometry of the planet
x      = np.arange(Offset_RA-50, Offset_RA+50, 0.4)
y      = np.arange(Offset_Dec-50, Offset_Dec+50, 0.4)
xp, yp = np.meshgrid(x, y) #grid of coords to look for the planet position

# 
n_poly = 1 # Degree of the polynomial to model the stellar speckle
stellar_coeffs_init = [1e-4] * (n_poly + 1)  # Initial coeffs for the stellar speckle polynomial
alpha = 1. # Multiplicative factor to scale the Cps


# List the Cps files
Cps_path = 'Contrast_fits/'
files_Cps = sorted([file for file in os.listdir(path_output + Cps_path) if '.fits' in file])
print(path_output+Cps_path)

# Path to Cps model 
Cps_model_path = '/Users/jscigliuto/Nextcloud/Py/MATISSE-DataProcessing/Spectrum/computed_spectra/contrast_template_bt-settl_hd72946_ph1ld.fits'

# Get the contrast template
use_bin_data = True
Cps_model = fits.getdata(Cps_model_path)
if use_bin_data:
    Cps_model = Cps_model.reshape(-1, 5).mean(axis=1)


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


def global_fit_objective(alpha, files_Cps, all_data_dict):
    """
    Fonction objectif globale qui fit alpha globalement 
    et stellar_coeffs indépendamment pour chaque fichier
    """
    total_chi2 = 0
    
    for file_name in files_Cps:
        data = all_data_dict[file_name]
        
        # Fit stellar_coeffs (with fixed alpha) for each baseline
        stellar_coeffs_init = [1e-4] * (n_poly + 1)
        bounds_stellar = [(None, None)] * (n_poly + 1)
        
        for i_base in range(n_base):
            res = scipy.optimize.minimize(
                residuals_fixed_alpha, 
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

#Exoplanets/BDs brighter than their host star at its position (beta Pic b)
def model_bright(params, wl, spat_freq, Cps_model):
    # Extract parameters
    alpha = params[0]
    stellar_coeffs = params[1:]
    stellar_poly = np.polyval(stellar_coeffs, wl)

    # Compute model 
    cf_model = alpha * Cps_model + stellar_poly * np.exp(-2j * np.pi * spat_freq)

    # Go back to amplitude/phase space
    amp_model = np.abs(cf_model)
    phi_model = wrap(np.angle(cf_model))

    return amp_model, phi_model

def residuals(params, spat_freq, wl, Cps_model, Cps_real_cal, Cps_real_cal_err, Cps_imag_cal, Cps_imag_cal_err, sci_case=sci_case):

    # Models
    if sci_case == 'faint':
        amp_model, phi_model = model_faint(params, wl, spat_freq, Cps_model)
    elif sci_case == 'bright':
        amp_model, phi_model = model_bright(params, wl, spat_freq, Cps_model)

    # Reconstruct complex quantities    
    cf_model = amp_model * np.exp(1j * phi_model)
    cf_data  = Cps_real_cal + 1j * Cps_imag_cal

    # Compute residuals
    chi2_real = ((np.real(cf_data) - np.real(cf_model)) / Cps_real_cal_err)**2
    chi2_imag = ((np.imag(cf_data) - np.imag(cf_model)) / Cps_imag_cal_err)**2
    chi2      = np.sum(chi2_real + chi2_imag)
    chi2_red  = chi2 / (len(wl) - len(params))

    return chi2_red


def residuals_fixed_alpha(stellar_coeffs, alpha, spat_freq, wl, Cps_model, 
                          Cps_real_cal, Cps_real_cal_err, 
                          Cps_imag_cal, Cps_imag_cal_err):
    """
    Résidus avec alpha fixé, optimise seulement stellar_coeffs
    """
    params = np.concatenate([[alpha], stellar_coeffs])
    return residuals(params, spat_freq, wl, Cps_model, 
                    Cps_real_cal, Cps_real_cal_err, 
                    Cps_imag_cal, Cps_imag_cal_err)



if sci_case == 'bright':
    stellar_coeffs_init = [1e-4] * (n_poly + 1)
    n_base = 6

    start_time = datetime.now()
    with open(path_output + '/log.txt', 'w') as log_txt:
        log_txt.write(f'Starting the fitting' + str(start_time) + '\n')
    
    # Load all the data in one dictionary
    all_data_dict = {}
    
    for i_file, file_Cps in enumerate(files_Cps):
        print(f"Chargement fichier {i_file+1}/{len(files_Cps)}: {file_Cps}")
        
        # Load Cps data per file
        hdu = fits.open(path_output + Cps_path + file_Cps)
        wl = hdu['WAVELENGTH'].data  
        Cps_real_cal = hdu['CPS_REAL'].data
        Cps_real_cal_err = hdu['CPS_REAL_ERR'].data
        Cps_imag_cal = hdu['CPS_IMAG'].data
        Cps_imag_cal_err = hdu['CPS_IMAG_ERR'].data
        U = hdu['U'].data
        V = hdu['V'].data
        
        # Store in all_data_dict[file_Cps]
        
        all_data_dict[file_Cps] = {
            'wl': wl,
            'Cps_real_cal': Cps_real_cal,
            'Cps_real_cal_err': Cps_real_cal_err,
            'Cps_imag_cal': Cps_imag_cal,
            'Cps_imag_cal_err': Cps_imag_cal_err,
            'spat_freq': np.sqrt(U**2+V**2) * mas2rad(534) * np.cos(31.52) # Pré-calculé pour une position
        }
    
    # Global optimization of alpha 
    print("Optimisation de alpha global...")
    alpha_init = 1.0
    
    result_alpha = scipy.optimize.minimize_scalar(
        lambda a: global_fit_objective(a, files_Cps, all_data_dict),
        bounds=(0, 10),
        method='bounded'
    )
    
    alpha_optimal = result_alpha.x
    print(f"Alpha optimal: {alpha_optimal}")
    
    # Final fit of stellar_coeffs per file with optimal alpha fixed
    for i_file, file_Cps in enumerate(files_Cps):
        print(f"Fit final fichier {i_file+1}/{len(files_Cps)}")
        
        data = all_data_dict[file_Cps]
        fitted_stellar_coeffs = np.zeros((n_base, n_poly + 1))
        
        for i_base in range(n_base):
            res = scipy.optimize.minimize(
                residuals_fixed_alpha,
                stellar_coeffs_init,
                args=(alpha_optimal, data['spat_freq'][i_base], 
                      data['wl'], Cps_model,
                      data['Cps_real_cal'][i_base], 
                      data['Cps_real_cal_err'][i_base],
                      data['Cps_imag_cal'][i_base], 
                      data['Cps_imag_cal_err'][i_base]),
                bounds=[(None, None)] * (n_poly + 1),
                method='L-BFGS-B'
            )
            fitted_stellar_coeffs[i_base] = res.x
        
        # Save fitted parameters
        np.save(path_output + f'/fitted_params/alpha_global.npy', alpha_optimal)
        np.save(path_output + f'/fitted_params/stellar_coeffs_{file_Cps}.npy', fitted_stellar_coeffs)
        
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
    fig_Cps, ax_Cps = plt.subplots(n_base, 1, figsize=(8, 3*n_base), sharex=True)
    
    for i_base in range(n_base):
        # Compute model with fitted parameters
        params = np.concatenate([[alpha_optimal], fitted_stellar_coeffs[i_base]])
        amp_model, phi_model = model_bright(params, wl, spat_freq[i_base], Cps_model)
        
        cf_model = amp_model * np.exp(1j * phi_model)
        
        Cps_real_model = np.real(cf_model)
        Cps_imag_model = np.imag(cf_model)
        
        # Plot real part
        ax_Cps[i_base].errorbar(wl*1e6, Cps_real_cal[i_base], yerr=np.sqrt(Cps_real_cal_err[i_base]**2), fmt='o', label='Data', color='blue', alpha=0.5)
        ax_Cps[i_base].plot(wl*1e6, Cps_real_model, label='Model', color='red')
        ax_Cps[i_base].set_ylabel(f'{base_order_name[i_base]} Cps Real')
        ax_Cps[i_base].set_ylim(-1e-3, 8e-3)
        ax_Cps[i_base].legend()
    
    ax_Cps[-1].set_xlabel('Wavelength [µm]')
    fig_Cps.suptitle(f'Fitted Cps - {file_Cps}')
    fig_Cps.savefig(path_output + f'/fitted_params/fitted_Cps_{file_Cps[:-5]}.png', dpi=300, bbox_inches='tight')
    plt.close(fig_Cps)