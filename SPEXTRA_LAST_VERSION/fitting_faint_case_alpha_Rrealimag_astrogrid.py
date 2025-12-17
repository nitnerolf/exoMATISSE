# Importation
import numpy as np
from astropy.io import fits
import os 
import matplotlib.pyplot as plt
import scipy 
import shutil
from datetime import datetime
import multiprocessing as mp
from functools import partial
from common_tools import mas2rad
from time import time as get_time

####################################################################################################################################################################################
### INPUTS SET UP 
# Output path 
path_output = '/Users/jscigliuto/exoMATISSE/exoMATISSE/data_hd72946b/' #HD 72946 B

# Science case
sci_case = 'faint'   

# Planet offsets coords mentioned in the OB [mas]
Offset_RA = 106
Offset_Dec = -145

# Grid of coordinates to determine the astrometry of the planet
x      = np.arange(Offset_RA-17, Offset_RA+17, .5) 
y      = np.arange(Offset_Dec-17, Offset_Dec+17, .5)
xp, yp = np.meshgrid(x, y)  # grid of coords to look for the planet position

# Degree of the polynomial to model the stellar speckle
n_poly = 1 

# Wavelength range to fit
wmin = 3.0e-6
wmax = 4.15e-6

# Number of cores to use for parallelization (chi2 maps)
n_cores = 10

# Path to Cps model 
Cps_model_path = '/Users/jscigliuto/Nextcloud/Py/MATISSE-DataProcessing/Spectrum/computed_spectra/contrast_template_bt-settl_hd72946_ph1ld.fits' 

# Flag for binned data
use_bin_data = True

### INITIALIZATION 
# Baseline order and names
base_order_name = ('U3-U4', 'U1-U2', 'U2-U3', 'U2-U4', 'U1-U3', 'U1-U4')
# Cps files path 
Cps_path = 'Contrast_fits/'

####################################################################################################################################################################################
### FUNCTIONS

def mas2deg(angle):
    return angle/1000/3600
def deg2mas(angle):
    return angle*1000*3600
def mas2rad(angle):
    return np.deg2rad(mas2deg(angle))
def rad2mas(angle):
    return deg2mas(np.rad2deg(angle))
def wrap(angle):
    return np.angle(np.exp(1j*angle))


def model_bright(params, wl, spat_freq, Cps_model):
    """
    Model with stellar polynomial (REAL only)
    
    cf_model = alpha * Cps_model * (cos(2π·f) + i·sin(2π·f)) + stellar_poly_real
    The modulations are hold by the contrast 
    """
    # Extract parameters
    alpha = params[0]
    stellar_coeffs_real = params[1:]  

    wl_norm = (wl - wl.mean()) / wl.std()
    
    # Polynomial for stellar speckle (REAL only)
    stellar_poly_real = np.polyval(stellar_coeffs_real, wl_norm)
    
    # Compute model: les modulations sont sur le contraste
    cf_model =  Cps_model * (np.cos(2 * np.pi * spat_freq) + 1j * np.sin(2 * np.pi * spat_freq)) + alpha * stellar_poly_real
    
    # Go back to amplitude/phase space
    amp_model = np.abs(cf_model)
    phi_model = np.angle(cf_model)

    return amp_model, phi_model


def prepare_global_data(files_Cps, all_data_dict, wl_mask, wl):
    """
    Build a big array with all Cps data from all files/baselines
    """
    all_Cps_real = []
    all_Cps_real_err = []
    all_Cps_imag = []
    all_Cps_imag_err = []
    all_real_err_mask = []
    all_imag_err_mask = []
    all_U = []
    all_V = []
    
    for i_file, file_Cps in enumerate(files_Cps):
        data = all_data_dict[file_Cps]
        
        all_Cps_real.append(data['Cps_real_cal'])
        all_Cps_real_err.append(data['Cps_real_cal_err'])
        all_Cps_imag.append(data['Cps_imag_cal'])
        all_Cps_imag_err.append(data['Cps_imag_cal_err'])
        all_real_err_mask.append(data['real_err_mask'])
        all_imag_err_mask.append(data['imag_err_mask'])
        all_U.append(data['U'])
        all_V.append(data['V'])
    
    all_Cps_real = np.array(all_Cps_real)
    all_Cps_real_err = np.array(all_Cps_real_err)
    all_Cps_imag = np.array(all_Cps_imag)
    all_Cps_imag_err = np.array(all_Cps_imag_err)
    all_real_err_mask = np.array(all_real_err_mask)
    all_imag_err_mask = np.array(all_imag_err_mask)
    all_U = np.array(all_U)
    all_V = np.array(all_V)
    
    return {
        'Cps_real': all_Cps_real,
        'Cps_real_err': all_Cps_real_err,
        'Cps_imag': all_Cps_imag,
        'Cps_imag_err': all_Cps_imag_err,
        'real_err_mask': all_real_err_mask,
        'imag_err_mask': all_imag_err_mask,
        'U': all_U,
        'V': all_V
    }


def global_fit_simultaneous(params, global_data, n_poly, Cps_model, wl, xp, yp, ix, iy, n_files):
    """
    Function that computes the global chi2 for all data simultaneously (vectorized)
    
    Model (1 alpha per baselines):
    params[0:n_files*6] = (n_files*6 valeurs)
    params[n_files*6:] = n_files * 6_baselines * (n_poly+1) coefficients

    Everything is fitted all together 
    """
    n_baselines = 6
    n_alphas = n_files * n_baselines
    alphas = params[0:n_alphas].reshape(n_files, n_baselines)  # shape: (n_files, 6)
    
    n_coeffs_per_baseline = n_poly + 1  # Only real coeffs
    
    # Position in the grid
    sep = np.sqrt(xp[ix, iy]**2 + yp[ix, iy]**2)
    PA = np.arctan2(xp[ix, iy], yp[ix, iy])
    
    # Load the data
    Cps_real = global_data['Cps_real']  # shape: (n_files, 6, n_wl)
    Cps_real_err = global_data['Cps_real_err']
    Cps_imag = global_data['Cps_imag']
    Cps_imag_err = global_data['Cps_imag_err']
    real_err_mask = global_data['real_err_mask']
    imag_err_mask = global_data['imag_err_mask']
    U = global_data['U']  # shape: (n_files, 6)
    V = global_data['V']
    
    # Compute all spatial frequencies for this position
    Bproj = np.sqrt(U**2 + V**2) * mas2rad(sep) * np.cos(np.arctan2(U, V) - PA)
    spat_freq = Bproj[:, :, np.newaxis] / wl[np.newaxis, np.newaxis, :]
    
    # Normalize wavelengths
    wl_norm = (wl - wl.mean()) / wl.std()
    
    # Reshape the stellar coeffs for all files/baselines
    stellar_coeffs_all = params[n_alphas:].reshape(n_files, n_baselines, n_coeffs_per_baseline)
    
    # Compute all stellar polynomials (REAL only)
    stellar_poly_real = np.zeros((n_files, n_baselines, wl.size))
    
    for i_file in range(n_files):
        for i_base in range(n_baselines):
            stellar_poly_real[i_file, i_base] = np.polyval(stellar_coeffs_all[i_file, i_base], wl_norm)

    # Compute the model (vectorized) 
    # alphas shape: (n_files, 6) -> reshape to (n_files, 6, 1) for broadcasting
    alphas_broadcast = alphas[:, :, np.newaxis]
    
    contrast_with_modulation =  Cps_model[np.newaxis, np.newaxis, :] * \
                              (np.cos(2 * np.pi * spat_freq) + 1j * np.sin(2 * np.pi * spat_freq))

    cf_model = contrast_with_modulation + alphas_broadcast * stellar_poly_real

    # Complexify data
    cf_data = Cps_real + 1j * Cps_imag
    
    # Compute the chi2 (vectorized)
    chi2_real = ((np.real(cf_data) - np.real(cf_model)) / Cps_real_err)**2
    chi2_imag = ((np.imag(cf_data) - np.imag(cf_model)) / Cps_imag_err)**2
    
    chi2_real_masked = chi2_real * real_err_mask
    chi2_imag_masked = chi2_imag * imag_err_mask
    
    total_chi2 = np.sum(chi2_real_masked) + np.sum(chi2_imag_masked)
    n_dof_total = np.sum(real_err_mask) + np.sum(imag_err_mask) - len(params)
    
    # Global chi2 red 
    chi2_red_global = total_chi2 / n_dof_total

    return chi2_red_global


def process_single_position_global(idx, xp, yp, global_data, n_poly, Cps_model, wl, n_files, path_output):
    """
    Fit all data simultaneously for a given position in the grid.
    All parameters (n_files*6 alphas + all stellar coeffs) are fitted in one optimization
    """
    start_time = get_time()
    ix, iy = np.unravel_index(idx, xp.shape)
    
    # Total number of parameters
    n_coeffs_per_baseline = n_poly + 1
    n_baselines = 6
    n_alphas = n_files * n_baselines
    n_params = n_alphas + (n_files * n_baselines * n_coeffs_per_baseline)
    
    # Initialization 
    params_init = np.zeros(n_params)
    params_init[0:n_alphas] = 1.0  # all alpha init

    coeff_idx = n_alphas  
    Cps_real = global_data['Cps_real']
    Cps_imag = global_data['Cps_imag']
    
    wl_norm = (wl - wl.mean()) / wl.std()
    
    for i_file in range(n_files):
        for i_base in range(n_baselines):
            # Initialiser avec la partie réelle uniquement
            offset_real_init = np.mean(Cps_real[i_file, i_base]) 
            slope_real_init = np.polyfit(wl_norm, Cps_real[i_file, i_base], 1)[0] 
            
            params_init[coeff_idx:coeff_idx+n_coeffs_per_baseline] = [
                slope_real_init, offset_real_init]
            
            coeff_idx += n_coeffs_per_baseline
    
    init_time = get_time() - start_time
    
    ## Define bounds 
    # Alpha bounds (n_files * 6 alphas)
    bounds = [(0.1, 10.0)] * n_alphas
    
    # Stellar coeffs bounds (REAL only)
    for i in range(n_files * n_baselines):
        bounds.append((None, None))  # slope real 
        bounds.append((None, None))  # offset real 

    optim_start_time = get_time()

    result = scipy.optimize.minimize(
        global_fit_simultaneous,
        params_init,
        args=(global_data, n_poly, Cps_model, wl, xp, yp, ix, iy, n_files),
        method='L-BFGS-B',  
        bounds=bounds,
    )
    
    optim_time = get_time() - optim_start_time
    total_time = get_time() - start_time
    
    alphas_optimal = result.x[0:n_alphas].reshape(n_files, n_baselines)
    fitted_coeffs = result.x[n_alphas:]

    chi2_total = chi2_reduced(result.x, global_data, n_poly, Cps_model, wl, xp, yp, ix, iy, n_files)

    # Log
    with open(path_output + '/timing_log.txt', 'a') as f:
        f.write(f"pos({xp[ix,iy]:.1f},{yp[ix,iy]:.1f}) | init={init_time:.2f}s | optim={optim_time:.2f}s | "
                f"total={total_time:.2f}s | chi2={chi2_total:.6f} | "
                f"iter={result.nit} | feval={result.nfev}\n")
        for i_file in range(n_files):
            alphas_str = ', '.join([f'{a:.4f}' for a in alphas_optimal[i_file]])
            f.write(f"  File {i_file}: alphas=[{alphas_str}]\n")
    
    return idx, ix, iy, alphas_optimal, chi2_total, fitted_coeffs

def chi2_reduced(params, global_data, n_poly, Cps_model, wl, xp, yp, ix, iy, n_files):
    """
    Calculate the reduced chi2 from fitted parameters
    """
    n_baselines = 6
    n_alphas = n_files * n_baselines
    n_coeffs_per_baseline = n_poly + 1
    
    # Extract alphas: shape (n_files, 6)
    alphas = params[0:n_alphas].reshape(n_files, n_baselines)
    
    # Extract stellar coefficients
    stellar_coeffs_all = params[n_alphas:].reshape(n_files, n_baselines, n_coeffs_per_baseline)
    
    # Position in the grid
    sep = np.sqrt(xp[ix, iy]**2 + yp[ix, iy]**2)
    PA = np.arctan2(xp[ix, iy], yp[ix, iy])
    
    # Load the data
    Cps_real = global_data['Cps_real']          # shape: (n_files, 6, n_wl)
    Cps_real_err = global_data['Cps_real_err']
    Cps_imag = global_data['Cps_imag']
    Cps_imag_err = global_data['Cps_imag_err']
    real_err_mask = global_data['real_err_mask']
    imag_err_mask = global_data['imag_err_mask']
    U = global_data['U']                        # shape: (n_files, 6)
    V = global_data['V']
    
    # Compute all spatial frequencies for this position
    Bproj = np.sqrt(U**2 + V**2) * mas2rad(sep) * np.cos(np.arctan2(U, V) - PA)
    spat_freq = Bproj[:, :, np.newaxis] / wl[np.newaxis, np.newaxis, :]
    
    # Normalize wavelengths for numerical stability
    wl_norm = (wl - wl.mean()) / wl.std()
    
    # Compute all stellar polynomials (REAL only)
    stellar_poly_real = np.zeros((n_files, n_baselines, wl.size))
    
    for i_file in range(n_files):
        for i_base in range(n_baselines):
            stellar_poly_real[i_file, i_base] = np.polyval(stellar_coeffs_all[i_file, i_base], wl_norm)
    
    # Compute the model (vectorized) avec alpha différent par baseline ET par frame
    # alphas shape: (n_files, 6) -> reshape to (n_files, 6, 1) for broadcasting
    alphas_broadcast = alphas[:, :, np.newaxis]
    
    contrast_with_modulation =  Cps_model[np.newaxis, np.newaxis, :] * \
                              (np.cos(2 * np.pi * spat_freq) + 1j * np.sin(2 * np.pi * spat_freq))

    cf_model = contrast_with_modulation + alphas_broadcast * stellar_poly_real

    # Complex data
    cf_data = Cps_real + 1j * Cps_imag
    
    # Chi-square for real and imaginary parts separately
    chi2_real = ((np.real(cf_data) - np.real(cf_model)) / Cps_real_err)**2
    chi2_imag = ((np.imag(cf_data) - np.imag(cf_model)) / Cps_imag_err)**2
    
    # Apply masks (only count valid data points)
    chi2_real_masked = chi2_real * real_err_mask
    chi2_imag_masked = chi2_imag * imag_err_mask
    
    # Total chi-square
    total_chi2 = np.sum(chi2_real_masked) + np.sum(chi2_imag_masked)

    # Number of data points
    n_data_points = np.sum(real_err_mask) + np.sum(imag_err_mask)
    
    # Number of parameters
    n_params = len(params)
    
    # Degrees of freedom
    n_dof = n_data_points - n_params

    # Reduced chi2
    chi2_red = total_chi2 / n_dof

    return chi2_red

####################################################################################################################################################################################
### SCRIPT

if __name__ == '__main__':                          
     
    # Create output directories if they do not exist
    for directory in ['/chi2_maps', '/chi2_maps_fits', '/fitted_params']:
        dir_path = path_output + directory
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    
    # Initialize timing log file
    with open(path_output + '/timing_log.txt', 'w') as f:
        f.write(f"Timing log started at {datetime.now()}\n")
        f.write(f"Grid size: {xp.shape[0]} x {xp.shape[1]} = {xp.size} positions\n")
        f.write(f"Using {n_cores} cores for parallelization\n")
        f.write(f"NEW MODEL: cf = alpha * Cps_model * (cos + i*sin) + stellar_poly_real\n")
        f.write(f"One alpha per file and per baseline - all fitted simultaneously\n")
        f.write(f"{'='*80}\n\n")
    
    files_Cps = sorted([file for file in os.listdir(path_output + Cps_path) if '.fits' in file])
    print('Cps path', path_output + Cps_path)
    
    if sci_case == 'faint':
        n_base = 6

        start_time = datetime.now()
        with open(path_output + '/log.txt', 'w') as log_txt:
            log_txt.write(f'Starting the fitting ' + str(start_time) + '\n')
            log_txt.write(f'Grid size: {xp.shape[0]} x {xp.shape[1]} = {xp.size} positions\n')
            log_txt.write(f'Using {n_cores} cores for parallelization\n')
            log_txt.write(f'NEW MODEL: cf = alpha * Cps_model * (cos + i*sin) + stellar_poly_real\n')
            log_txt.write(f'One alpha per file and per baseline - all fitted simultaneously\n')
        
        # ====================================================================
        # STEP 1: Load and prepare global data
        # ====================================================================
        print("=" * 70)
        print("STEP 1: Prepare global data")
        print("=" * 70)
        
        all_data_dict = {}
        
        for i_file, file_Cps in enumerate(files_Cps):
            print(f"Loading file {i_file+1}/{len(files_Cps)}: {file_Cps}")
            
            hdu = fits.open(path_output + Cps_path + file_Cps)
            wl = hdu['WAVELENGTH'].data
            Cps_real_cal = hdu['CPS_REAL'].data
            Cps_real_cal_err = hdu['CPS_REAL_ERR'].data
            Cps_imag_cal = hdu['CPS_IMAG'].data
            Cps_imag_cal_err = hdu['CPS_IMAG_ERR'].data
            U = hdu['U'].data
            V = hdu['V'].data
        
            wl_mask = (wl > wmin) & (wl < wmax)
            wl = wl[wl_mask]
            Cps_real_cal = Cps_real_cal[:, wl_mask]
            Cps_real_cal_err = Cps_real_cal_err[:, wl_mask]
            Cps_imag_cal = Cps_imag_cal[:, wl_mask]
            Cps_imag_cal_err = Cps_imag_cal_err[:, wl_mask]
            
            # Mask the outliers based on SNR
            snr_real = np.abs(Cps_real_cal / Cps_real_cal_err) 
            snr_imag = np.abs(Cps_imag_cal / Cps_imag_cal_err)
            real_err_mask = (snr_real.T > np.quantile(snr_real, 0.05, axis=1)).T & (snr_real.T < np.quantile(snr_real, 0.95, axis=1)).T
            imag_err_mask = (snr_imag.T > np.quantile(snr_imag, 0.05, axis=1)).T & (snr_imag.T < np.quantile(snr_imag, 0.95, axis=1)).T
            
            all_data_dict[file_Cps] = {
                'Cps_real_cal': Cps_real_cal,
                'Cps_real_cal_err': Cps_real_cal_err,
                'Cps_imag_cal': Cps_imag_cal,
                'Cps_imag_cal_err': Cps_imag_cal_err,
                'real_err_mask': real_err_mask,
                'imag_err_mask': imag_err_mask,
                'U': U,
                'V': V}
    
            hdu.close()

        global_data = prepare_global_data(files_Cps, all_data_dict, wl_mask, wl)
        n_files = len(files_Cps)

        # Load the model
        Cps_model = fits.getdata(Cps_model_path)
        if use_bin_data:
            Cps_model = Cps_model.reshape(-1, 5).mean(axis=1)
        Cps_model = Cps_model[wl_mask]
        
        # ====================================================================
        # STEP 2: Loop over grid positions (PARALLELIZED)
        # ====================================================================
        print("\n" + "=" * 70)
        print(f"STEP 2: Loop over grid positions (using {n_cores} cores)")
        print("=" * 70)

        step2_start_time = get_time()

        chi2_map_global = np.zeros(xp.shape)
        alpha_maps = np.zeros((n_files, 6, *xp.shape))  # shape: (n_files, 6, ix, iy)
        n_positions = xp.size

        process_func = partial(process_single_position_global,
            xp=xp, yp=yp, global_data=global_data, n_poly=n_poly, 
            Cps_model=Cps_model, wl=wl, n_files=n_files, path_output=path_output)

        print(f"Processing {n_positions} positions with {n_cores} cores...")

        with mp.Pool(processes=n_cores) as pool:
            results = []
            for i, result in enumerate(pool.imap(process_func, range(n_positions))):
                idx, ix, iy, alphas_opt, chi2, fitted_coeffs = result
                results.append(result)

                elapsed_time = get_time() - step2_start_time
                avg_time_per_pos = elapsed_time / (i + 1)
                estimated_remaining = avg_time_per_pos * (n_positions - (i + 1))
                
                if (i+1) % 10 == 0 or (i+1) == n_positions:
                    print(f"Completed {i+1}/{n_positions} positions | Time: {datetime.now()} | "
                          f"Current: ({xp[ix,iy]:.1f}, {yp[ix,iy]:.1f}) mas, chi2={chi2:.2f} | "
                          f"Elapsed: {elapsed_time/60:.1f}min | Avg: {avg_time_per_pos:.1f}s/pos | "
                          f"ETA: {estimated_remaining/60:.1f}min")

                    with open(path_output + '/timing_log.txt', 'a') as f:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"CHECKPOINT: {i+1}/{n_positions} positions completed\n")
                        f.write(f"  Total elapsed time: {elapsed_time:.2f}s ({elapsed_time/60:.2f}min)\n")
                        f.write(f"  Average time per position: {avg_time_per_pos:.2f}s\n")
                        f.write(f"  Estimated remaining time: {estimated_remaining:.2f}s ({estimated_remaining/60:.2f}min)\n")
                        f.write(f"  Current position: ({xp[ix,iy]:.1f}, {yp[ix,iy]:.1f}) mas\n")
                        f.write(f"  Current chi2: {chi2:.6f}\n")
                        f.write(f"{'='*80}\n\n")

        # Fill the alpha and chi2 maps 
        for idx, ix, iy, alphas_optimal, chi2_total, fitted_coeffs in results:
            chi2_map_global[ix, iy] = chi2_total
            for i_file in range(n_files):
                for i_base in range(6):
                    alpha_maps[i_file, i_base, ix, iy] = alphas_optimal[i_file, i_base]
        
        # Find best position
        idx_min = np.argmin(chi2_map_global)
        ix_best, iy_best = np.unravel_index(idx_min, xp.shape)
        x_best = xp[ix_best, iy_best]
        y_best = yp[ix_best, iy_best]
        chi2_best = chi2_map_global[ix_best, iy_best]
        alphas_best = alpha_maps[:, :, ix_best, iy_best]  # shape: (n_files, 6)
        sep_best = np.sqrt(x_best**2 + y_best**2)
        PA_best = np.rad2deg(np.arctan2(x_best, y_best))
        
        print("\n" + "=" * 70)
        print("BEST POSITION FOUND:")
        print("=" * 70)
        print(f"RA offset:  {x_best:.2f} mas")
        print(f"Dec offset: {y_best:.2f} mas")
        print(f"Separation: {sep_best:.2f} mas")
        print(f"PA:         {PA_best:.2f} deg")
        print(f"Chi2:       {chi2_best:.6f}")
        print("\nAlphas per file and baseline:")
        print("=" * 70)
        
        with open(path_output + '/log.txt', 'a') as log_txt:
            log_txt.write(f'\nBest position found:\n')
            log_txt.write(f'ix         = {ix_best:.2f}\n')
            log_txt.write(f'iy         = {iy_best:.2f}\n')
            log_txt.write(f'RA offset  = {x_best:.2f} mas\n')
            log_txt.write(f'Dec offset = {y_best:.2f} mas\n')
            log_txt.write(f'Separation = {sep_best:.2f} mas\n')
            log_txt.write(f'PA         = {PA_best:.2f} deg\n')
            log_txt.write(f'Chi2       = {chi2_best:.6f}\n')
            log_txt.write(f'\nAlphas per file and baseline:\n')
            for i_file, file_Cps in enumerate(files_Cps):
                log_txt.write(f'\n{file_Cps}:\n')
                for i_base in range(6):
                    log_txt.write(f'  {base_order_name[i_base]} = {alphas_best[i_file, i_base]:.6f}\n')
        
        # ====================================================================
        # STEP 3: Extract and save coefficients at best position
        # ====================================================================
        print("\n" + "=" * 70)
        print("STEP 3: Extract fitted coefficients at best position")
        print("=" * 70)
        
        # Find the result corresponding to the best position
        best_result = None
        for result in results:
            idx, ix, iy, alphas_opt, chi2, fitted_coeffs = result
            if ix == ix_best and iy == iy_best:
                best_result = result
                break

        if best_result is not None:
            _, _, _, alphas_best, chi2_best, fitted_coeffs_best = best_result
            
            # Extract and save coefficients for each file
            n_coeffs_per_baseline = n_poly + 1
            coeff_idx = 0
            
            for i_file, file_Cps in enumerate(files_Cps):
                stellar_coeffs_file = np.zeros((6, n_coeffs_per_baseline))
                alphas_file = alphas_best[i_file]  # Les 6 alphas pour ce fichier
                
                for i_base in range(6):
                    stellar_coeffs_file[i_base] = fitted_coeffs_best[coeff_idx : coeff_idx + n_coeffs_per_baseline]
                    coeff_idx += n_coeffs_per_baseline

                # Store in the main data dictionary for plots
                all_data_dict[file_Cps]['fitted_stellar_coeffs'] = stellar_coeffs_file
                all_data_dict[file_Cps]['fitted_alphas'] = alphas_file
                
                # Save to file
                np.save(path_output + f'/fitted_params/stellar_coeffs_{file_Cps}.npy', stellar_coeffs_file)
                np.save(path_output + f'/fitted_params/alphas_{file_Cps}.npy', alphas_file)
                
                print(f"Saved coefficients and alphas for {file_Cps}")
        
        # ====================================================================
        # STEP 4: Save results
        # ====================================================================
        print("\n" + "=" * 70)
        print("STEP 4: Save results")
        print("=" * 70)
        
        # Save maps
        np.save(path_output + '/chi2_maps_fits/chi2_map_global.npy', chi2_map_global)
        np.save(path_output + '/chi2_maps_fits/alpha_maps.npy', alpha_maps)
        np.save(path_output + '/chi2_maps_fits/xp.npy', xp)
        np.save(path_output + '/chi2_maps_fits/yp.npy', yp)
        
        # Save best position parameters
        best_pos_dict = {
            'ix_best': ix_best,
            'iy_best': iy_best,
            'x_best': x_best,
            'y_best': y_best,
            'sep_best': sep_best,
            'PA_best': PA_best,
            'alphas_best': alphas_best,  # shape: (n_files, 6)
            'chi2_best': chi2_best
        }
        np.save(path_output + '/fitted_params/best_position.npy', best_pos_dict)
        
        with open(path_output + '/log.txt', 'a') as log_txt:
            log_txt.write(f'End of fitting at ' + str(datetime.now()) + '\n')
            log_txt.write(f'Total duration: ' + str(datetime.now() - start_time) + '\n')

        # ====================================================================
        # STEP 5: Stack frames and models by groups 
        # ====================================================================
        print("\n" + "=" * 70)
        print("STEP 5: Stack frames and models by groups (simple averaging)")
        print("=" * 70)
        
        # Number of frames to stack
        n_frames_per_stack = 20
        
        # Compute the number of stacked frames
        n_stacks = n_files // n_frames_per_stack
        n_remaining = n_files % n_frames_per_stack
        
        if n_remaining > 0:
            print(f"Total files: {n_files}")
            print(f"Creating {n_stacks} stacks of {n_frames_per_stack} frames each")
            print(f"Plus 1 additional stack with the remaining {n_remaining} frames")
            n_stacks += 1
        else:
            print(f"Total files: {n_files}")
            print(f"Creating {n_stacks} stacks of {n_frames_per_stack} frames each")
        
        # Sauvegarder les paramètres de normalisation (CRUCIAL pour la spectrométrie)
        wl_mean = wl.mean()
        wl_std = wl.std()
        normalization_params = {
            'wl_mean': wl_mean,
            'wl_std': wl_std,
            'wl': wl
        }
        np.save(path_output + '/fitted_params/normalization_params.npy', normalization_params)
        print(f"✓ Saved normalization parameters: mean={wl_mean:.6e}, std={wl_std:.6e}")
        
        # Normalize wavelength
        wl_norm = (wl - wl_mean) / wl_std
        
        # Loop over each group 
        all_stacked_data = []
        
        for i_stack in range(n_stacks):
            print(f"\n{'='*70}")
            print(f"Processing stack group {i_stack+1}/{n_stacks}")
            print(f"{'='*70}")
            
            # Determine start aand end index for each group
            start_idx = i_stack * n_frames_per_stack
            end_idx = min((i_stack + 1) * n_frames_per_stack, n_files)
            n_frames_in_stack = end_idx - start_idx
            
            print(f"Stacking frames {start_idx} to {end_idx-1} ({n_frames_in_stack} frames)")
            
            # Prepare array for stacking
            stacked_Cps_real = np.zeros((6, wl.size))
            stacked_Cps_imag = np.zeros((6, wl.size))
            stacked_Cps_real_err = np.zeros((6, wl.size))
            stacked_Cps_imag_err = np.zeros((6, wl.size))
            stacked_model_real = np.zeros((6, wl.size))
            stacked_model_imag = np.zeros((6, wl.size))
            stacked_stellar_real = np.zeros((6, wl.size))
            stacked_modulations_real = np.zeros((6, wl.size))
            stacked_modulations_imag = np.zeros((6, wl.size))
            stacked_modulations_real_err = np.zeros((6, wl.size))
            stacked_modulations_imag_err = np.zeros((6, wl.size))
            stacked_masks_real = np.zeros((6, wl.size), dtype=bool)
            stacked_masks_imag = np.zeros((6, wl.size), dtype=bool)
            
            # Compteurs pour la moyenne (nombre de valeurs valides par pixel)
            # Valid values
            n_valid_real = np.zeros((6, wl.size))
            n_valid_imag = np.zeros((6, wl.size))
            
            # Loop over the files of the group
            for i_file in range(start_idx, end_idx):
                file_Cps = files_Cps[i_file]
                print(f"  Adding file {i_file - start_idx + 1}/{n_frames_in_stack}: {file_Cps}")
                
                # Load data of the group
                data = all_data_dict[file_Cps]
                Cps_real_cal = data['Cps_real_cal']
                Cps_real_cal_err = data['Cps_real_cal_err']
                Cps_imag_cal = data['Cps_imag_cal']
                Cps_imag_cal_err = data['Cps_imag_cal_err']
                real_err_mask = data['real_err_mask']
                imag_err_mask = data['imag_err_mask']
                U = data['U']
                V = data['V']
                
                # Load fitted parameters of the group
                alphas_file = alphas_best[i_file]
                stellar_coeffs_file = np.load(path_output + f'/fitted_params/stellar_coeffs_{file_Cps}.npy')
                
                # Compute spatial frequency at hte best position
                PA_best_rad = np.arctan2(x_best, y_best)
                Bproj = np.sqrt(U**2 + V**2) * mas2rad(sep_best) * np.cos(np.arctan2(U, V) - PA_best_rad)
                spat_freq = np.outer(Bproj, 1/wl)
                
                # Loop over baselines
                for i_base in range(6):
                    # Compute model
                    alpha = alphas_file[i_base]
                    stellar_poly_real = np.polyval(stellar_coeffs_file[i_base], wl_norm)

                    contrast_with_modulation = alpha * Cps_model * \
                        (np.cos(2 * np.pi * spat_freq[i_base]) + 1j * np.sin(2 * np.pi * spat_freq[i_base]))
                    cf_model = contrast_with_modulation + stellar_poly_real
                    
                    model_real = np.real(cf_model)
                    model_imag = np.imag(cf_model)
                    
                    # Modulations (real and imag)
                    modulations_real = Cps_real_cal[i_base] - stellar_poly_real
                    modulations_imag = Cps_imag_cal[i_base]
                    
                    # Mask unvalid value
                    Cps_real_masked = Cps_real_cal[i_base].copy()
                    Cps_imag_masked = Cps_imag_cal[i_base].copy()
                    Cps_real_err_masked = Cps_real_cal_err[i_base].copy()
                    Cps_imag_err_masked = Cps_imag_cal_err[i_base].copy()
                    model_real_masked = model_real.copy()
                    model_imag_masked = model_imag.copy()
                    stellar_poly_real_masked = stellar_poly_real.copy()
                    modulations_real_masked = modulations_real.copy()
                    modulations_imag_masked = modulations_imag.copy()
                
                    ## Fix unvalid values as NaN
                    # real part
                    Cps_real_masked[~real_err_mask[i_base]] = np.nan
                    Cps_real_err_masked[~real_err_mask[i_base]] = np.nan
                    model_real_masked[~real_err_mask[i_base]] = np.nan
                    stellar_poly_real_masked[~real_err_mask[i_base]] = np.nan
                    modulations_real_masked[~real_err_mask[i_base]] = np.nan
                    # imag part
                    Cps_imag_masked[~imag_err_mask[i_base]] = np.nan
                    Cps_imag_err_masked[~imag_err_mask[i_base]] = np.nan
                    model_imag_masked[~imag_err_mask[i_base]] = np.nan
                    modulations_imag_masked[~imag_err_mask[i_base]] = np.nan
                    
                    # Stacking 
                    stacked_Cps_real[i_base] += np.nan_to_num(Cps_real_masked, nan=0.0)
                    stacked_Cps_imag[i_base] += np.nan_to_num(Cps_imag_masked, nan=0.0)
                    stacked_model_real[i_base] += np.nan_to_num(model_real_masked, nan=0.0)
                    stacked_model_imag[i_base] += np.nan_to_num(model_imag_masked, nan=0.0)
                    stacked_stellar_real[i_base] += np.nan_to_num(stellar_poly_real_masked, nan=0.0)
                    stacked_modulations_real[i_base] += np.nan_to_num(modulations_real_masked, nan=0.0)
                    stacked_modulations_imag[i_base] += np.nan_to_num(modulations_imag_masked, nan=0.0)
                    
                    stacked_Cps_real_err[i_base] += np.nan_to_num(Cps_real_err_masked**2, nan=0.0)
                    stacked_Cps_imag_err[i_base] += np.nan_to_num(Cps_imag_err_masked**2, nan=0.0)

                    # Count the number of valid value (for average)
                    n_valid_real[i_base] += real_err_mask[i_base].astype(float)
                    n_valid_imag[i_base] += imag_err_mask[i_base].astype(float)
                    
                    # Accumulating the 2 masks
                    stacked_masks_real[i_base] = stacked_masks_real[i_base] | real_err_mask[i_base]
                    stacked_masks_imag[i_base] = stacked_masks_imag[i_base] | imag_err_mask[i_base]
            
            # Average with the number of valid value
            for i_base in range(6):
                # Avoid division by 0
                mask_real_nonzero = n_valid_real[i_base] > 0
                mask_imag_nonzero = n_valid_imag[i_base] > 0
                
                # Averaging
                stacked_Cps_real[i_base, mask_real_nonzero] /= n_valid_real[i_base, mask_real_nonzero]
                stacked_Cps_imag[i_base, mask_imag_nonzero] /= n_valid_imag[i_base, mask_imag_nonzero]
                stacked_model_real[i_base, mask_real_nonzero] /= n_valid_real[i_base, mask_real_nonzero]
                stacked_model_imag[i_base, mask_imag_nonzero] /= n_valid_imag[i_base, mask_imag_nonzero]
                stacked_stellar_real[i_base, mask_real_nonzero] /= n_valid_real[i_base, mask_real_nonzero]
                stacked_modulations_real[i_base, mask_real_nonzero] /= n_valid_real[i_base, mask_real_nonzero]
                stacked_modulations_imag[i_base, mask_imag_nonzero] /= n_valid_imag[i_base, mask_imag_nonzero]
                
                # Error on stacked data
                stacked_Cps_real_err[i_base, mask_real_nonzero] = np.sqrt(stacked_Cps_real_err[i_base, mask_real_nonzero]) / n_valid_real[i_base, mask_real_nonzero]
                stacked_Cps_imag_err[i_base, mask_imag_nonzero] = np.sqrt(stacked_Cps_imag_err[i_base, mask_imag_nonzero]) / n_valid_imag[i_base, mask_imag_nonzero]
                
                stacked_modulations_real_err[i_base, mask_real_nonzero] = stacked_Cps_real_err[i_base, mask_real_nonzero]
                stacked_modulations_imag_err[i_base, mask_imag_nonzero] = stacked_Cps_imag_err[i_base, mask_imag_nonzero]
            
            # Compute mean alpha for this group
            alphas_mean = np.mean(alphas_best[start_idx:end_idx], axis=0)
            
            # Save data for that group
            stacked_data = {
                'Cps_real': stacked_Cps_real,
                'Cps_imag': stacked_Cps_imag,
                'Cps_real_err': stacked_Cps_real_err,
                'Cps_imag_err': stacked_Cps_imag_err,
                'model_real': stacked_model_real,
                'model_imag': stacked_model_imag,
                'stellar_real': stacked_stellar_real,
                'modulations_real': stacked_modulations_real,
                'modulations_imag': stacked_modulations_imag,
                'modulations_real_err': stacked_modulations_real_err,
                'modulations_imag_err': stacked_modulations_imag_err,
                'masks_real': stacked_masks_real,
                'masks_imag': stacked_masks_imag,
                'alphas_mean': alphas_mean,
                'wl': wl,
                'n_frames': n_frames_in_stack,
                'start_idx': start_idx,
                'end_idx': end_idx
            }
            
            all_stacked_data.append(stacked_data)
            
            np.save(path_output + f'/fitted_params/stacked_data_group{i_stack:03d}.npy', stacked_data)
            print(f"✓ Saved stacked data for group {i_stack}")
            
            # ====================================================================
            # PLOT: Stacked modulations for this group 
            # ====================================================================
            
            # Plot: Stacked modulations real
            fig_mod_stacked_real, ax_mod_stacked_real = plt.subplots(6, 1, figsize=(10, 18), sharex=True)
            
            for i_base in range(6):
                mask = stacked_masks_real[i_base]
                wl_masked = wl[mask]
                Cps_model_masked = Cps_model[mask]
                
                modulations = stacked_modulations_real[i_base, mask]
                modulations_err = stacked_modulations_real_err[i_base, mask]
                model_modulations = stacked_model_real[i_base, mask] - stacked_stellar_real[i_base, mask]
                
                # Plot données avec fill_between pour les erreurs
                # ax_mod_stacked_real[i_base].fill_between(
                #     wl_masked*1e6, 
                #     modulations - modulations_err, 
                #     modulations + modulations_err,
                #     alpha=0.3, color='lightblue', label='Data ± σ')

                # data
                ax_mod_stacked_real[i_base].errorbar(wl_masked*1e6, modulations, yerr=modulations_err,fmt='+', color='blue', markersize=3, alpha=0.7, label='Data ± σ')
                ax_mod_stacked_real[i_base].plot(wl_masked*1e6, modulations, label='Data - Stellar', color='blue', linewidth=2)

                # model
                ax_mod_stacked_real[i_base].plot(wl_masked*1e6, model_modulations, label='Model - Stellar', color='red', linewidth=2, alpha=0.8)
                
                # envelope
                ax_mod_stacked_real[i_base].plot(wl_masked*1e6, alphas_mean[i_base] * Cps_model_masked, label=f'α={alphas_mean[i_base]:.3f} × Cps_model', color='black', linestyle='--', linewidth=1.5, alpha=0.6)
                ax_mod_stacked_real[i_base].plot(wl_masked*1e6, -alphas_mean[i_base] * Cps_model_masked, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
                
                ax_mod_stacked_real[i_base].set_ylabel(f'{base_order_name[i_base]}\nModulations Real')
                ax_mod_stacked_real[i_base].axhline(0, color='gray', linestyle=':', linewidth=0.5)
                ax_mod_stacked_real[i_base].legend(loc='upper right', fontsize=7)
                ax_mod_stacked_real[i_base].grid(alpha=0.3)
                ax_mod_stacked_real[i_base].set_ylim(-5e-3, 5e-3)

            ax_mod_stacked_real[-1].set_xlabel('Wavelength [µm]')
            ax_mod_stacked_real[-1].set_xlim(3.0, 4.15)
            fig_mod_stacked_real.suptitle(f'Stacked Modulations Real - Group {i_stack} (frames {start_idx}-{end_idx-1}, n={n_frames_in_stack})\n'f'Best pos: ({x_best:.1f}, {y_best:.1f}) mas', fontsize=14)
            fig_mod_stacked_real.tight_layout()
            fig_mod_stacked_real.savefig(path_output + f'/fitted_params/stacked_modulations_real_group{i_stack:03d}.png', dpi=300, bbox_inches='tight')
            plt.close(fig_mod_stacked_real)
            
            # Plot: Stacked modulations imag 
            fig_mod_stacked_imag, ax_mod_stacked_imag = plt.subplots(6, 1, figsize=(10, 18), sharex=True)
            
            for i_base in range(6):
                mask = stacked_masks_imag[i_base]
                wl_masked = wl[mask]
                Cps_model_masked = Cps_model[mask]
                
                modulations = stacked_modulations_imag[i_base, mask]
                modulations_err = stacked_modulations_imag_err[i_base, mask]
                model_modulations = stacked_model_imag[i_base, mask]
                
                # Plot données avec fill_between pour les erreurs
                # ax_mod_stacked_imag[i_base].fill_between(
                #     wl_masked*1e6, 
                #     modulations - modulations_err, 
                #     modulations + modulations_err,
                #     alpha=0.3, color='lightgreen', label='Data ± σ')

                # data
                ax_mod_stacked_imag[i_base].errorbar(wl_masked*1e6, modulations, yerr=modulations_err, fmt='+', color='green', markersize=3, alpha=0.7, label='Data ± σ')
                ax_mod_stacked_imag[i_base].plot(wl_masked*1e6, modulations, label='Data', color='green', linewidth=2)

                # model
                ax_mod_stacked_imag[i_base].plot(wl_masked*1e6, model_modulations,label='Model', color='orange', linewidth=2, alpha=0.8)
                
                # envelope
                ax_mod_stacked_imag[i_base].plot(wl_masked*1e6, alphas_mean[i_base] * Cps_model_masked, label=f'α={alphas_mean[i_base]:.3f} × Cps_model', color='black', linestyle='--', linewidth=1.5, alpha=0.6)
                ax_mod_stacked_imag[i_base].plot(wl_masked*1e6, -alphas_mean[i_base] * Cps_model_masked, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
                
                ax_mod_stacked_imag[i_base].set_ylabel(f'{base_order_name[i_base]}\nModulations Imag')
                ax_mod_stacked_imag[i_base].axhline(0, color='gray', linestyle=':', linewidth=0.5)
                ax_mod_stacked_imag[i_base].legend(loc='upper right', fontsize=7)
                ax_mod_stacked_imag[i_base].grid(alpha=0.3)
                ax_mod_stacked_imag[i_base].set_ylim(-5e-3, 5e-3)
            
            ax_mod_stacked_imag[-1].set_xlabel('Wavelength [µm]')
            ax_mod_stacked_imag[-1].set_xlim(3.0, 4.15)
            fig_mod_stacked_imag.suptitle(f'Stacked Modulations Imag - Group {i_stack} (frames {start_idx}-{end_idx-1}, n={n_frames_in_stack})\n'f'Best pos: ({x_best:.1f}, {y_best:.1f}) mas', fontsize=14)
            fig_mod_stacked_imag.tight_layout()
            fig_mod_stacked_imag.savefig(path_output + f'/fitted_params/stacked_modulations_imag_group{i_stack:03d}.png', dpi=300, bbox_inches='tight')
            plt.close(fig_mod_stacked_imag)
            
            print(f"✓ Saved plots for group {i_stack}")
        
        # Save the data of all group
        np.save(path_output + '/fitted_params/all_stacked_data.npy', all_stacked_data)
        
        print(f"\n{'='*70}")
        print(f"✓ Stacking complete: {n_stacks} groups created (simple averaging, no weighting)")
        print(f"{'='*70}")
        
        # Log 
        with open(path_output + '/log.txt', 'a') as log_txt:
            log_txt.write(f'\n{"="*70}\n')
            log_txt.write(f'STACKING RESULTS (Simple averaging, no weighting):\n')
            log_txt.write(f'{"="*70}\n')
            log_txt.write(f'Number of stacking groups: {n_stacks}\n')
            log_txt.write(f'Frames per group: {n_frames_per_stack}\n')
            for i_stack in range(n_stacks):
                stacked_data = all_stacked_data[i_stack]
                log_txt.write(f'\nGroup {i_stack}: frames {stacked_data["start_idx"]}-{stacked_data["end_idx"]-1} ({stacked_data["n_frames"]} frames)\n')
                log_txt.write(f'  Mean alphas:\n')
                for i_base in range(6):
                    log_txt.write(f'    {base_order_name[i_base]}: {stacked_data["alphas_mean"][i_base]:.6f}\n')

####################################################################################################################################################################################
### FIGURES
        
        # ====================================================================
        # PLOT: chi2 map & alpha maps
        # ====================================================================
        
        fig_chi2, ax_chi2 = plt.subplots(1, 1, figsize=(10, 8))
        
        im_chi2 = ax_chi2.contourf(xp, yp, chi2_map_global, levels=50, cmap='viridis')
        ax_chi2.plot(x_best, y_best, 'r*', markersize=20, label=f'Best position\n({x_best:.1f}, {y_best:.1f}) mas')
        ax_chi2.set_xlabel('RA offset [mas]')
        ax_chi2.set_ylabel('Dec offset [mas]')
        ax_chi2.set_title('Chi2 map')
        ax_chi2.legend()
        ax_chi2.grid(alpha=0.3)
        plt.colorbar(im_chi2, ax=ax_chi2, label=r'$\chi^2$')
        
        fig_chi2.savefig(path_output + '/chi2_maps_fits/chi2_map_global.png', dpi=300, bbox_inches='tight')
        plt.close(fig_chi2)

        # Plot : alpha map for each file
        for i_file, file_Cps in enumerate(files_Cps):
            fig_alphas, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i_base in range(6):
                im = axes[i_base].contourf(xp, yp, alpha_maps[i_file, i_base], levels=50, cmap='viridis')
                axes[i_base].plot(x_best, y_best, 'r*', markersize=15, 
                                label=f'α={alphas_best[i_file, i_base]:.4f}')
                axes[i_base].set_title(f'{base_order_name[i_base]}')
                axes[i_base].set_xlabel('RA offset [mas]')
                axes[i_base].set_ylabel('Dec offset [mas]')
                axes[i_base].legend()
                axes[i_base].grid(alpha=0.3)
                plt.colorbar(im, ax=axes[i_base], label=r'$\alpha$')
            
            fig_alphas.suptitle(f'Alpha maps - {file_Cps}', fontsize=16)
            fig_alphas.tight_layout()
            fig_alphas.savefig(path_output + f'/chi2_maps_fits/alpha_maps_{file_Cps[:-5]}.png', 
                              dpi=300, bbox_inches='tight')
            plt.close(fig_alphas)

    # ========================================================================
    # PLOT: stellar fits at best position
    # ========================================================================

    # Reload best position
    best_pos_dict = np.load(path_output + '/fitted_params/best_position.npy', allow_pickle=True).item()
    x_best = best_pos_dict['x_best']
    y_best = best_pos_dict['y_best']
    sep_best = best_pos_dict['sep_best']
    PA_best_rad = np.arctan2(x_best, y_best)

    for i_file, file_Cps in enumerate(files_Cps):
        print(f"Plot {i_file+1}/{len(files_Cps)}: {file_Cps}")
        
        # Load fitted parameters
        fitted_stellar_coeffs = np.load(path_output + f'/fitted_params/stellar_coeffs_{file_Cps}.npy')
        alphas_optimal = np.load(path_output + f'/fitted_params/alphas_{file_Cps}.npy')
        
        # Load data
        data = all_data_dict[file_Cps]
        
        # Recalculate the spatial frequency at best position
        U = data['U']
        V = data['V']
        Bproj = np.sqrt(U**2 + V**2) * mas2rad(sep_best) * np.cos(np.arctan2(U, V) - PA_best_rad)
        spat_freq = np.outer(Bproj, 1/wl)
        
        Cps_real_cal = data['Cps_real_cal']
        Cps_real_cal_err = data['Cps_real_cal_err']
        Cps_imag_cal = data['Cps_imag_cal']
        Cps_imag_cal_err = data['Cps_imag_cal_err']
        
        n_base = spat_freq.shape[0]

        fig_Cps_real, ax_Cps_real = plt.subplots(n_base, 1, figsize=(8, 3*n_base), sharex=True)
        fig_Cps_imag, ax_Cps_imag = plt.subplots(n_base, 1, figsize=(8, 3*n_base), sharex=True)
        fig_mod_real, ax_mod_real = plt.subplots(n_base, 1, figsize=(8, 3*n_base), sharex=True)
        fig_mod_imag, ax_mod_imag = plt.subplots(n_base, 1, figsize=(8, 3*n_base), sharex=True)
        
        for i_base in range(n_base):
            # Compute model with fitted parameters
            params = np.concatenate([[alphas_optimal[i_base]], fitted_stellar_coeffs[i_base]])
            amp_model, phi_model = model_bright(params, wl, spat_freq[i_base], Cps_model)
            
            # Coherent flux model
            cf_model = amp_model * np.exp(1j * phi_model)
            cf_real_model = np.real(cf_model) 
            cf_imag_model = np.imag(cf_model)
            
            # Plot real part
            real_err_mask = data['real_err_mask'][i_base]
            wl_real = wl[real_err_mask]
            Cps_real_cal_mask = Cps_real_cal[i_base][real_err_mask]
            Cps_real_cal_err_mask = Cps_real_cal_err[i_base][real_err_mask]
            cf_real_model_mask = cf_real_model[real_err_mask]
            Cps_model_mask = Cps_model[real_err_mask]

            wl_real_norm = (wl_real - wl_mean) / wl_std
            stellar_real_poly = np.polyval(fitted_stellar_coeffs[i_base], wl_real_norm)
            
            ax_Cps_real[i_base].errorbar(wl_real*1e6, Cps_real_cal_mask, 
                                          yerr=Cps_real_cal_err_mask, 
                                          fmt='o', label='Data', color='blue', alpha=0.5)
            ax_Cps_real[i_base].plot(wl_real*1e6, cf_real_model_mask, label='Model', color='red')
            ax_Cps_real[i_base].plot(wl_real*1e6, stellar_real_poly, label='Stellar contamination', 
                                     color='purple', linestyle='--')

            ax_Cps_real[i_base].set_ylabel(f'{base_order_name[i_base]} Cps Real')
            ax_Cps_real[i_base].set_ylim(-5e-2, 5e-2)
            ax_Cps_real[i_base].set_xlim(3.0, 4.15)
            ax_Cps_real[i_base].set_title(f'α = {alphas_optimal[i_base]:.4f}')
            ax_Cps_real[i_base].legend()

            # Plot imaginary part
            imag_err_mask = data['imag_err_mask'][i_base]
            wl_imag = wl[imag_err_mask]
            Cps_imag_cal_mask = Cps_imag_cal[i_base][imag_err_mask]
            Cps_imag_cal_err_mask = Cps_imag_cal_err[i_base][imag_err_mask]
            cf_imag_model_mask = cf_imag_model[imag_err_mask]

            ax_Cps_imag[i_base].errorbar(wl_imag*1e6, Cps_imag_cal_mask, 
                                          yerr=Cps_imag_cal_err_mask, 
                                          fmt='o', label='Data', color='green', alpha=0.5)
            ax_Cps_imag[i_base].plot(wl_imag*1e6, cf_imag_model_mask, label='Model', color='orange')
            
            ax_Cps_imag[i_base].set_ylabel(f'{base_order_name[i_base]} Cps Imag')
            ax_Cps_imag[i_base].set_ylim(-3.5e-2, 3.5e-2)
            ax_Cps_imag[i_base].set_xlim(3.0, 4.15)
            ax_Cps_imag[i_base].set_title(f'α = {alphas_optimal[i_base]:.4f}')
            ax_Cps_imag[i_base].legend()

            # Modulations real part
            if i_base == 5:
                ax_mod_real[i_base].set_ylabel('Modulations Real')
                ax_mod_real[i_base].set_xlabel('Wavelength [µm]')
                ax_mod_real[i_base].plot(wl_real*1e6, Cps_real_cal_mask - stellar_real_poly, 
                                        label='Contrast', color='blue', marker='+', alpha=0.8)
                ax_mod_real[i_base].plot(wl_real*1e6, cf_real_model_mask - stellar_real_poly, 
                                        label='Model', color='red', alpha=0.8)
            ax_mod_real[i_base].plot(wl_real*1e6, Cps_real_cal_mask - stellar_real_poly, 
                                    color='blue', marker='+', alpha=0.8)
            ax_mod_real[i_base].plot(wl_real*1e6, cf_real_model_mask - stellar_real_poly, 
                                    color='red', alpha=0.8)
            ax_mod_real[i_base].plot(wl_real*1e6, alphas_optimal[i_base] * Cps_model_mask, 
                                    label='Cps Model', color='black', linestyle='--', alpha=0.5)
            ax_mod_real[i_base].plot(wl_real*1e6, -alphas_optimal[i_base] * Cps_model_mask, 
                                    color='black', linestyle='--', alpha=0.5)
            ax_mod_real[i_base].set_xlim(3., 4.1)
            ax_mod_real[i_base].set_ylim(-1e-2, 1e-2)
            ax_mod_real[i_base].set_title(f'α = {alphas_optimal[i_base]:.4f}')

            # Modulations imaginary part
            if i_base == 5:
                ax_mod_imag[i_base].set_ylabel('Modulations Imag')
                ax_mod_imag[i_base].set_xlabel('Wavelength [µm]')
                ax_mod_imag[i_base].plot(wl_imag*1e6, Cps_imag_cal_mask, 
                                        label='Contrast', color='green', marker='+', alpha=0.8)
                ax_mod_imag[i_base].plot(wl_imag*1e6, cf_imag_model_mask, 
                                        label='Model', color='orange', alpha=0.8)
            ax_mod_imag[i_base].plot(wl_imag*1e6, Cps_imag_cal_mask, color='green', marker='+', alpha=0.8)
            ax_mod_imag[i_base].plot(wl_imag*1e6, cf_imag_model_mask, color='orange', alpha=0.8)
            ax_mod_imag[i_base].plot(wl_imag*1e6, alphas_optimal[i_base] * Cps_model_mask, 
                                    label='Cps Model', color='black', linestyle='--', alpha=0.5)
            ax_mod_imag[i_base].plot(wl_imag*1e6, -alphas_optimal[i_base] * Cps_model_mask, 
                                    color='black', linestyle='--', alpha=0.5)
            ax_mod_imag[i_base].set_xlim(3., 4.1)
            ax_mod_imag[i_base].set_ylim(-1e-2, 1e-2)
            ax_mod_imag[i_base].set_title(f'α = {alphas_optimal[i_base]:.4f}')

        ax_Cps_real[-1].set_xlabel('Wavelength [µm]')
        ax_Cps_imag[-1].set_xlabel('Wavelength [µm]')
        fig_Cps_real.suptitle(f'Fitted Cps - {file_Cps}\nBest pos: ({x_best:.1f}, {y_best:.1f}) mas')
        fig_Cps_imag.suptitle(f'Fitted Cps - {file_Cps}\nBest pos: ({x_best:.1f}, {y_best:.1f}) mas')
        fig_Cps_real.tight_layout()
        fig_Cps_imag.tight_layout()
        fig_mod_real.tight_layout()
        fig_mod_imag.tight_layout()
        
        fig_Cps_real.savefig(path_output + f'/fitted_params/fitted_Cps_real_{file_Cps[:-5]}.png', 
                             dpi=300, bbox_inches='tight')
        fig_Cps_imag.savefig(path_output + f'/fitted_params/fitted_Cps_imag_{file_Cps[:-5]}.png', 
                             dpi=300, bbox_inches='tight')
        fig_mod_real.savefig(path_output + f'/fitted_params/modulations_real_{file_Cps[:-5]}.png', 
                             dpi=300, bbox_inches='tight')
        fig_mod_imag.savefig(path_output + f'/fitted_params/modulations_imag_{file_Cps[:-5]}.png', 
                             dpi=300, bbox_inches='tight')
        
        plt.close('all')