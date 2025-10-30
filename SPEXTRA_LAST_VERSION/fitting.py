# Importation
import numpy as np
from astropy.io import fits
import os 
import matplotlib.pyplot as plt
import scipy 
import shutil
import multiprocessing as mp
from common_tools import reorder_baselines, wrap, mas2rad


# Path of the OiFits files (outputs of the pipeline, phase corrected)
path_oifits = '/Users/jscigliuto/Nextcloud/DATA/HD72946B/corrected_data_wo_rmnrec/'

# Output path 
path_output = '/Users/jscigliuto/exoMATISSE/exoMATISSE/data_hd72946b/test/'

# Science case
sci_case = 'faint'  #'bright' 

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
params_init = np.array([alpha, *stellar_coeffs_init]) 
n_params = params_init.size
bounds_params = [(0, None), *[(None, None)] * (n_poly + 1)]
n_base = 6  # Number of baselines

# List the Cps files
Cps_path = 'Contrast_fits/'
files_Cps = sorted([file for file in os.listdir(path_output + Cps_path) if '.fits' in file])
print(path_output+Cps_path)

# Path to Cps model 
Cps_model_path = '/Users/jscigliuto/Nextcloud/Py/MATISSE-DataProcessing/Spectrum/computed_spectra/contrast_template_bt-settl_hd72946_ph1ld.fits'

# Get the contrast template
use_bin_data = False
Cps_model = fits.getdata(Cps_model_path)
if use_bin_data:
    Cps_model = Cps_model.reshape(-1, 5).mean(axis=1)


## Functions 
def fit(ix, iy, Bcov, PAcov, PA, sep, wl, n_base, params_init, Cps_model,
        Cps_real_cal, Cps_real_cal_err, Cps_imag_cal, Cps_imag_cal_err, bounds_params):

    Bproj     = Bcov * mas2rad(sep[ix,iy]) * np.cos(PAcov - PA[ix,iy])
    spat_freq = np.outer(Bproj, 1/wl)  # Spatial frequency [rad^-1]

    fitted_params = np.zeros((n_base, len(params_init)))
    chi2_map = np.zeros((n_base))
    chi2_real_map = np.zeros((n_base))
    chi2_imag_map = np.zeros((n_base))

    for i_base in range(n_base):
        res = scipy.optimize.minimize(residuals, params_init, args=(spat_freq[i_base], wl, Cps_model,
                                         Cps_real_cal[i_base], Cps_real_cal_err[i_base],
                                         Cps_imag_cal[i_base], Cps_imag_cal_err[i_base]),
                                         bounds=bounds_params, method='L-BFGS-B') #BFGS
        fitted_params[i_base] = res.x

        chi2_real_map[i_base], chi2_imag_map[i_base], chi2_map[i_base] = residuals_all_chi2(wl, res.x, spat_freq[i_base], Cps_model,
                                         Cps_real_cal[i_base], Cps_real_cal_err[i_base],
                                         Cps_imag_cal[i_base], Cps_imag_cal_err[i_base])

    return (ix, iy), fitted_params, chi2_map, chi2_real_map, chi2_imag_map

def residuals(params, spat_freq, wl, Cps_model, Cps_real_cal, Cps_real_cal_err, Cps_imag_cal, Cps_imag_cal_err):
    # Model 
    amp_model, phi_model = model(params, wl, spat_freq, Cps_model)

    # Reconstruct complex quantities    
    cf_model = amp_model * np.exp(1j * phi_model)
    cf_data  = Cps_real_cal + 1j * Cps_imag_cal

    # Compute residuals
    chi2_real = ((np.real(cf_data) - np.real(cf_model)) / Cps_real_cal_err)**2
    chi2_imag = ((np.imag(cf_data) - np.imag(cf_model)) / Cps_imag_cal_err)**2
    chi2      = np.sum(chi2_real + chi2_imag)
    chi2_red  = chi2 / (len(wl) - len(params))

    return chi2_red

def residuals_all_chi2(wl, params, spat_freq, Cps_model, Cps_real_cal, 
                       Cps_real_cal_err, Cps_imag_cal, Cps_imag_cal_err):
    # Model 
    amp_model, phi_model = model(params, wl, spat_freq, Cps_model)

    # Reconstruct complex quantities    
    cf_model = amp_model * np.exp(1j * phi_model)
    cf_data  = Cps_real_cal + 1j * Cps_imag_cal

    # Compute residuals
    chi2_real = ((np.real(cf_data) - np.real(cf_model)) / Cps_real_cal_err)**2
    chi2_real_red = np.sum(chi2_real) / (len(wl) - len(params) - 1)
    chi2_imag = ((np.imag(cf_data) - np.imag(cf_model)) / Cps_imag_cal_err)**2
    chi2_imag_red = np.sum(chi2_imag) / (len(wl) - len(params))
    chi2      = np.sum(chi2_real + chi2_imag)
    chi2_red  = chi2 / (len(wl) - len(params))

    return chi2_red, chi2_real_red, chi2_imag_red



def model(params, wl, spat_freq, Cps_model):
    # Extract parameters
    alpha = params[0]
    stellar_coeffs = params[1:]
    # stellar_poly = np.polynomial.Polynomial(stellar_coeffs)(wl)
    stellar_poly = np.polyval(stellar_coeffs, wl)

    # Compute model 
    cf_model = alpha * Cps_model * np.exp(-2j * np.pi * spat_freq) + stellar_poly

    # Go back to amplitude/phase space
    amp_model = np.abs(cf_model)
    phi_model = wrap(np.angle(cf_model))

    return amp_model, phi_model

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


### Determine the astrometry 
if __name__ == "__main__":
    for i_file, file_Cps in enumerate(files_Cps):
        print('Fitting:', file_Cps)
        
        num_OB_planet    = int(file_Cps[file_Cps.find('OB')+2:file_Cps.find('_exp')])
        num_exp_planet   = int(file_Cps[file_Cps.find('_exp')+4:file_Cps.find('_frame')])
        num_frame_planet = int(file_Cps[file_Cps.find('_frame')+6:file_Cps.find('.fits')])

        # Initialize
        chi2_map = np.zeros((n_base, xp.size, yp.size))
        chi2_real_map = np.zeros((n_base, xp.size, yp.size))
        chi2_imag_map = np.zeros((n_base, xp.size, yp.size))
        fitted_params = np.zeros((n_base, xp.size, yp.size, n_params))


        # Extract Cps quantities
        hdul_Cps = fits.open(path_output + Cps_path + file_Cps)
        U  = hdul_Cps['U'].data
        V  = hdul_Cps['V'].data
        wl = hdul_Cps['WAVELENGTH'].data
        Cps_real_cal     = hdul_Cps['CPS_REAL'].data
        Cps_imag_cal     = hdul_Cps['CPS_IMAG'].data
        Cps_real_cal_err = hdul_Cps['CPS_REAL_ERR'].data
        Cps_imag_cal_err = hdul_Cps['CPS_IMAG_ERR'].data

        if sci_case == 'faint':
            
            # Position angle (PA) and separation grid
            PA = np.arctan2(yp, xp) 
            sep = np.sqrt(xp**2 + yp**2)

            # Baseline-PA coverage in the UV-space
            PAcov = np.arctan2(U,V)
            Bcov = np.sqrt(U**2 + V**2)
            
            
            # Loop over the grid of coordinates to find the planet position
            # for ix in range(xp.shape[0]):
            #     for iy in range(xp.shape[1]):
            #         print(ix)
            #         print(iy)
            pool = mp.Pool(processes=mp.cpu_count())
            args = [(ix, iy, Bcov, PAcov, PA, sep, wl, n_base, params_init, Cps_model, Cps_real_cal, 
                    Cps_real_cal_err, Cps_imag_cal, Cps_imag_cal_err, bounds_params) for ix in range(xp.shape[0]) for iy in range(xp.shape[1])]
                
            results = pool.starmap(fit, args)
            
            for result in results:
                coords, fitted_params_fit, chi2_map_fit, chi2_real_map_fit, chi2_imag_map_fit = result
                ix, iy = coords
                chi2_map[:, ix, iy]         = chi2_map_fit
                chi2_real_map[:, ix, iy]    = chi2_real_map_fit
                chi2_imag_map[:, ix, iy]    = chi2_imag_map_fit
                fitted_params[:, ix, iy, :] = fitted_params_fit
            
            fits.writeto(path_output + f'/fitted_params/{file_Cps[:-5]}_fit_params.fits', fitted_params_fit, overwrite=True)
            fits.writeto(path_output + f'/chi2_maps_fits/{file_Cps[:-5]}_chi2_map.fits', chi2_map_fit, overwrite=True)
            fits.writeto(path_output + f'/chi2_maps_fits/{file_Cps[:-5]}_chi2_real_map.fits', chi2_real_map_fit, overwrite=True)
            fits.writeto(path_output + f'/chi2_maps_fits/{file_Cps[:-5]}_chi2_imag_map.fits', chi2_imag_map_fit, overwrite=True)

            xmin, xmax = x.min(), x.max()
            ymin, ymax = y.min(), y.max()
            for i_base in range(n_base):
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(chi2_map[i_base], origin='lower', extent=(xmin, xmax, ymin, ymax), aspect='auto')
                ax.set_title(f'Chi2 map - {file_Cps} - Baseline {base_order_name[i_base]}')
                ax.set_xlabel('Offset RA (mas)')
                ax.set_ylabel('Offset Dec (mas)')
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label('reduced chi2')
                fig.savefig(path_output + f'/chi2_maps/{file_Cps[:-5]}_chi2_base{i_base}.png', dpi=300, bbox_inches='tight')
                plt.close(fig)
            
    