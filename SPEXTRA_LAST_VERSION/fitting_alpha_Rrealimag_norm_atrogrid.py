# Fit a multiplicative factor (alpha) to a contrast model (Cps) and 4 coefficients per baseline to model the stellar speckle
# Simultaneous fit of all files with ONE ALPHA PER BASELINE (6 alphas per file)


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
from time import time as get_time

####################################################################################################################################################################################
### INPUTS SET UP 
# Output path 
path_output = '/Users/jscigliuto/exoMATISSE/exoMATISSE/data_betaPicb/' #beta Pic b 

# Planet offsets coords mentioned in the OB [mas]
Offset_RA = 279
Offset_Dec = 455

# Grid of coordinates to determine the astrometry of the planet
x      = np.arange(Offset_RA-1, Offset_RA+1, .5) 
y      = np.arange(Offset_Dec-1, Offset_Dec+1, .5) 
xp, yp = np.meshgrid(x, y)  # grid of coords to look for the planet position

# Degree of the polynomial to model the stellar speckle
n_poly = 1 

# Wavelength range to fit
wmin = 3.0e-6
wmax = 4.15e-6

# Number of cores to use for parallelization (chi2 maps)
n_cores = 10

# Path to Cps model 
Cps_model_path = '/Users/jscigliuto/Nextcloud/Py/MATISSE-DataProcessing/Spectrum/computed_spectra/contrast_template_bt-settl_betPic.fits' #beta Pic b

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
    Model with separate polynomials for real and imaginary parts
    """
    # Extract parameters
    alpha = params[0]
    stellar_coeffs = params[1:]
    n_coeffs = stellar_coeffs.size // 2
    stellar_coeffs_real = stellar_coeffs[:n_coeffs]
    stellar_coeffs_imag = stellar_coeffs[n_coeffs:]

    wl_norm = (wl - wl.mean()) / wl.std()
    
    # Polynomial for stellar speckle
    stellar_poly_real = np.polyval(stellar_coeffs_real, wl_norm)
    stellar_poly_imag = np.polyval(stellar_coeffs_imag, wl_norm)
    
    # Compute model 
    cf_model = alpha * Cps_model + stellar_poly_real * np.cos(2 * np.pi * spat_freq) + 1j * stellar_poly_imag * np.sin(2 * np.pi * spat_freq) 
    
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


def chi2_reduced_per_baseline(params, global_data, n_poly, Cps_model, wl, xp, yp, ix, iy, n_files):
    """
    Calculate the reduced chi-square from fitted parameters
    
    CORRECT MODEL: UN ALPHA PAR BASELINE (6 alphas per file)
    
    Parameters structure:
    ---------------------
    params = flat array containing all parameters
    Reshaped as: (n_files, 6 baselines, 5 params per baseline)
    where 5 params = 1 alpha + 4 stellar_coeffs
    
    Total params = n_files × 6 × 5
    """
    
    n_baselines = 6
    n_coeffs_per_baseline = 2 * (n_poly + 1)  # 4 for n_poly=1
    n_params_per_baseline = 1 + n_coeffs_per_baseline  
    
    # Reshape params: (n_files, n_baselines, n_params_per_baseline)
    params_reshaped = params.reshape(n_files, n_baselines, n_params_per_baseline)
    
    # Extract alphas and stellar coeffs
    alphas = params_reshaped[:, :, 0]  # shape: (n_files, 6)
    stellar_coeffs = params_reshaped[:, :, 1:]  # shape: (n_files, 6, 4)
    
    # Position in the grid
    sep = np.sqrt(xp[ix, iy]**2 + yp[ix, iy]**2)
    PA = np.arctan2(xp[ix, iy], yp[ix, iy])
    
    # Load data
    Cps_real = global_data['Cps_real']          # shape: (n_files, 6, n_wl)
    Cps_real_err = global_data['Cps_real_err']
    Cps_imag = global_data['Cps_imag']
    Cps_imag_err = global_data['Cps_imag_err']
    real_err_mask = global_data['real_err_mask']
    imag_err_mask = global_data['imag_err_mask']
    U = global_data['U']                        # shape: (n_files, 6)
    V = global_data['V']
    
    # Compute spatial frequencies
    Bproj = np.sqrt(U**2 + V**2) * mas2rad(sep) * np.cos(np.arctan2(U, V) - PA)
    spat_freq = Bproj[:, :, np.newaxis] / wl[np.newaxis, np.newaxis, :] # shape: (n_files, 6, n_wl)
    
    # Normalize wavelengths
    wl_norm = (wl - wl.mean()) / wl.std()
    
    # Split stellar coeffs into real and imaginary
    n_coeffs = n_coeffs_per_baseline // 2
    stellar_coeffs_real = stellar_coeffs[:, :, :n_coeffs]  # (n_files, 6, 2)
    stellar_coeffs_imag = stellar_coeffs[:, :, n_coeffs:]  # (n_files, 6, 2)
    
    # Compute stellar polynomials for each file and baseline
    stellar_poly_real = np.zeros((n_files, n_baselines, wl.size))
    stellar_poly_imag = np.zeros((n_files, n_baselines, wl.size))
    
    for i_file in range(n_files):
        for i_base in range(n_baselines):
            stellar_poly_real[i_file, i_base] = np.polyval(stellar_coeffs_real[i_file, i_base], wl_norm)
            stellar_poly_imag[i_file, i_base] = np.polyval(stellar_coeffs_imag[i_file, i_base], wl_norm)
    
    # Compute model
    alphas_broadcast = alphas[:, :, np.newaxis]  # (n_files, 6, 1)
    
    cf_model = alphas_broadcast * Cps_model[np.newaxis, np.newaxis, :] + \
               stellar_poly_real * np.cos(2 * np.pi * spat_freq) + \
               1j * stellar_poly_imag * np.sin(2 * np.pi * spat_freq)
    
    # Complex data
    cf_data = Cps_real + 1j * Cps_imag
    
    # Chi-square
    chi2_real = ((np.real(cf_data) - np.real(cf_model)) / Cps_real_err)**2
    chi2_imag = ((np.imag(cf_data) - np.imag(cf_model)) / Cps_imag_err)**2
    
    # Apply masks
    chi2_real_masked = chi2_real * real_err_mask
    chi2_imag_masked = chi2_imag * imag_err_mask
    
    # Total chi-square
    total_chi2 = np.sum(chi2_real_masked) + np.sum(chi2_imag_masked)
    
    # Degrees of freedom
    n_data_points = np.sum(real_err_mask) + np.sum(imag_err_mask)
    n_params = len(params)
    
    # Reduced chi-square
    chi2_red = total_chi2 / (n_data_points - n_params) 
    
    return chi2_red


def process_single_position_per_baseline(idx, xp, yp, global_data, n_poly, Cps_model, wl, n_files, path_output):
    """
    Fit all data with ONE ALPHA PER BASELINE
    
    Total parameters = n_files × 6 baselines × (1 alpha + 4 stellar_coeffs)
                     = n_files × 6 × 5
                     = n_files × 30
    """
    start_time = get_time()
    ix, iy = np.unravel_index(idx, xp.shape)
    
    n_baselines = 6
    n_coeffs_per_baseline = 2 * (n_poly + 1)  # 4 for n_poly=1
    n_params_per_baseline = 1 + n_coeffs_per_baseline  # 1 alpha + 4 coeffs = 5
    n_params_total = n_files * n_baselines * n_params_per_baseline  # n_files × 30
    
    # Initialization
    params_init = np.zeros(n_params_total)
    
    Cps_real = global_data['Cps_real']
    Cps_imag = global_data['Cps_imag']
    wl_norm = (wl - wl.mean()) / wl.std()
    
    param_idx = 0
    for i_file in range(n_files):
        for i_base in range(n_baselines):
            # Initialize alpha
            params_init[param_idx] = 1.0  
            param_idx += 1
            
            # Initialize stellar coeffs
            offset_real_init = np.mean(Cps_real[i_file, i_base])
            offset_imag_init = np.mean(Cps_imag[i_file, i_base])
            slope_real_init = np.polyfit(wl_norm, Cps_real[i_file, i_base], 1)[0]
            slope_imag_init = np.polyfit(wl_norm, Cps_imag[i_file, i_base], 1)[0]
            # offset_real_init = 0.0
            # offset_imag_init = 0.0
            # slope_real_init = 0.0
            # slope_imag_init = 0.0
            
            params_init[param_idx:param_idx+4] = [
                slope_real_init, offset_real_init,
                slope_imag_init, offset_imag_init]
            param_idx += 4
    
    init_time = get_time() - start_time
    
    # Define bounds
    bounds = []
    for i_file in range(n_files):
        for i_base in range(n_baselines):
            bounds.append((0, None))      # alpha >= 0
            bounds.append((None, None))   # slope real
            bounds.append((None, None))   # offset real
            bounds.append((None, None))   # slope imag
            bounds.append((None, None))   # offset imag
    
    optim_start_time = get_time()
    
    result = scipy.optimize.minimize(
        chi2_reduced_per_baseline,
        params_init,
        args=(global_data, n_poly, Cps_model, wl, xp, yp, ix, iy, n_files),
        method='L-BFGS-B',
        bounds=bounds,
    )
    
    optim_time = get_time() - optim_start_time
    total_time = get_time() - start_time
    
    # Extract results
    params_reshaped = result.x.reshape(n_files, n_baselines, n_params_per_baseline)
    alphas_optimal = params_reshaped[:, :, 0]  # shape: (n_files, 6)
    fitted_coeffs = params_reshaped[:, :, 1:]  # shape: (n_files, 6, 4)
    
    # Statistics
    alpha_mean = np.mean(alphas_optimal)
    alpha_std = np.std(alphas_optimal)
    
    chi2_red = result.fun
    
    # Log
    with open(path_output + '/timing_log.txt', 'a') as f:
        f.write(f"pos({xp[ix,iy]:.1f},{yp[ix,iy]:.1f}) | "
                f"init={init_time:.2f}s | optim={optim_time:.2f}s | "
                f"total={total_time:.2f}s | "
                f"alpha_mean={alpha_mean:.4f} ± {alpha_std:.4f} | "
                f"chi2_red={chi2_red:.6f} | "
                f"iter={result.nit} | feval={result.nfev}\n")
    
    return idx, ix, iy, alphas_optimal, fitted_coeffs, chi2_red


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
        f.write(f"MODE: One alpha per baseline (6 alphas per file)\n")
        f.write(f"{'='*80}\n\n")
    
    files_Cps = sorted([file for file in os.listdir(path_output + Cps_path) if '.fits' in file])
    print('Cps path', path_output + Cps_path)
    n_base = 6

    start_time = datetime.now()
    with open(path_output + '/log.txt', 'w') as log_txt:
        log_txt.write(f'Starting the fitting (alpha per baseline mode) ' + str(start_time) + '\n')
        log_txt.write(f'Grid size: {xp.shape[0]} x {xp.shape[1]} = {xp.size} positions\n')
        log_txt.write(f'Using {n_cores} cores for parallelization\n')
    
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
        real_err_mask = (snr_real.T > np.quantile(snr_real, 0.05, axis=1)).T 
        imag_err_mask = (snr_imag.T > np.quantile(snr_imag, 0.05, axis=1)).T
        
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
    alphas_map = np.zeros((xp.shape[0], xp.shape[1], n_files, 6))  # 4D: (nx, ny, n_files, 6)
    alpha_mean_map = np.zeros(xp.shape)
    n_positions = xp.size

    process_func = partial(process_single_position_per_baseline,
        xp=xp, yp=yp, global_data=global_data, n_poly=n_poly, 
        Cps_model=Cps_model, wl=wl, n_files=n_files, path_output=path_output)

    print(f"Processing {n_positions} positions with {n_cores} cores...")

    with mp.Pool(processes=n_cores) as pool:
        results = []
        for i, result in enumerate(pool.imap(process_func, range(n_positions))):
            idx, ix, iy, alphas_opt, fitted_coeffs, chi2 = result
            results.append(result)

            elapsed_time = get_time() - step2_start_time
            avg_time_per_pos = elapsed_time / (i + 1)
            estimated_remaining = avg_time_per_pos * (n_positions - (i + 1))
            
            if (i+1) % 10 == 0 or (i+1) == n_positions:
                alpha_mean = np.mean(alphas_opt)
                print(f"Completed {i+1}/{n_positions} positions | Time: {datetime.now()} | "
                      f"Current: ({xp[ix,iy]:.1f}, {yp[ix,iy]:.1f}) mas, "
                      f"alpha_mean={alpha_mean:.4f}, chi2={chi2:.2f} | "
                      f"Elapsed: {elapsed_time/60:.1f}min | Avg: {avg_time_per_pos:.1f}s/pos | "
                      f"ETA: {estimated_remaining/60:.1f}min")

                with open(path_output + '/timing_log.txt', 'a') as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"CHECKPOINT: {i+1}/{n_positions} positions completed\n")
                    f.write(f"  Total elapsed time: {elapsed_time:.2f}s ({elapsed_time/60:.2f}min)\n")
                    f.write(f"  Average time per position: {avg_time_per_pos:.2f}s\n")
                    f.write(f"  Estimated remaining time: {estimated_remaining:.2f}s ({estimated_remaining/60:.2f}min)\n")
                    f.write(f"  Current position: ({xp[ix,iy]:.1f}, {yp[ix,iy]:.1f}) mas\n")
                    f.write(f"  Current alpha_mean: {alpha_mean:.6f}\n")
                    f.write(f"  Current chi2: {chi2:.6f}\n")
                    f.write(f"{'='*80}\n\n")

    # Fill the alpha and chi2 maps 
    for idx, ix, iy, alphas_optimal, fitted_coeffs, chi2_total in results:
        chi2_map_global[ix, iy] = chi2_total
        alphas_map[ix, iy, :, :] = alphas_optimal  # (n_files, 6)
        alpha_mean_map[ix, iy] = np.mean(alphas_optimal)
    
    # Find best position
    idx_min = np.argmin(chi2_map_global)
    ix_best, iy_best = np.unravel_index(idx_min, xp.shape)
    x_best = xp[ix_best, iy_best]
    y_best = yp[ix_best, iy_best]
    chi2_best = chi2_map_global[ix_best, iy_best]
    alphas_best = alphas_map[ix_best, iy_best, :, :]  # shape: (n_files, 6)
    alpha_mean_best = np.mean(alphas_best)
    sep_best = np.sqrt(x_best**2 + y_best**2)
    PA_best = np.rad2deg(np.arctan2(x_best, y_best))
    
    print("\n" + "=" * 70)
    print("BEST POSITION FOUND:")
    print("=" * 70)
    print(f"RA offset:  {x_best:.2f} mas")
    print(f"Dec offset: {y_best:.2f} mas")
    print(f"Separation: {sep_best:.2f} mas")
    print(f"PA:         {PA_best:.2f} deg")
    print(f"Alpha mean (all baselines): {alpha_mean_best:.6f}")
    print(f"Chi2:       {chi2_best:.6f}")
    print("=" * 70)
    
    with open(path_output + '/log.txt', 'a') as log_txt:
        log_txt.write(f'\nBest position found:\n')
        log_txt.write(f'ix         = {ix_best:.2f}\n')
        log_txt.write(f'iy         = {iy_best:.2f}\n')
        log_txt.write(f'RA offset  = {x_best:.2f} mas\n')
        log_txt.write(f'Dec offset = {y_best:.2f} mas\n')
        log_txt.write(f'Separation = {sep_best:.2f} mas\n')
        log_txt.write(f'PA         = {PA_best:.2f} deg\n')
        log_txt.write(f'Alpha mean = {alpha_mean_best:.6f}\n')
        for i_file in range(n_files):
            log_txt.write(f'File {i_file} alphas: {alphas_best[i_file]}\n')
        log_txt.write(f'Chi2       = {chi2_best:.6f}\n')
    
    # ====================================================================
    # STEP 3: Extract and save coefficients at best position
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Extract fitted coefficients at best position")
    print("=" * 70)

    best_result = None
    for result in results:
        idx, ix, iy, alphas_opt, fitted_coeffs, chi2 = result
        if ix == ix_best and iy == iy_best:
            best_result = result
            break

    # Save normalization parameters
    wl_mean = wl.mean()
    wl_std = wl.std()
    normalization_params = {
        'wl_mean': wl_mean,
        'wl_std': wl_std,
        'wl': wl}
    
    np.save(path_output + '/fitted_params/normalization_params.npy', normalization_params)

    if best_result is not None:
        _, _, _, alphas_best, fitted_coeffs_best, chi2_best = best_result
        # alphas_best shape: (n_files, 6)
        # fitted_coeffs_best shape: (n_files, 6, 4)
        
        for i_file, file_Cps in enumerate(files_Cps):
            # Extract for this file
            alphas_file = alphas_best[i_file]  # shape: (6,)
            stellar_coeffs_file = fitted_coeffs_best[i_file]  # shape: (6, 4)
            
            # Save
            np.save(path_output + f'/fitted_params/alphas_{file_Cps}.npy', alphas_file)
            np.save(path_output + f'/fitted_params/stellar_coeffs_{file_Cps}.npy', stellar_coeffs_file)
            
            # Store in dict
            all_data_dict[file_Cps]['fitted_alphas'] = alphas_file
            all_data_dict[file_Cps]['fitted_stellar_coeffs'] = stellar_coeffs_file
            
            print(f"Saved coefficients for {file_Cps}:")
            print(f"  Alphas (6 baselines): {alphas_file}")
            print(f"  Alpha mean: {np.mean(alphas_file):.6f}")

    # ====================================================================
    # STEP 4: Save results
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Save results")
    print("=" * 70)
    
    # Save maps
    np.save(path_output + '/chi2_maps_fits/chi2_map_global.npy', chi2_map_global)
    np.save(path_output + '/chi2_maps_fits/alphas_map.npy', alphas_map)  # 4D array
    np.save(path_output + '/chi2_maps_fits/alpha_mean_map.npy', alpha_mean_map)
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
        'alpha_mean_best': alpha_mean_best,
        'chi2_best': chi2_best
    }
    np.save(path_output + '/fitted_params/best_position.npy', best_pos_dict)
    
    with open(path_output + '/log.txt', 'a') as log_txt:
        log_txt.write(f'End of fitting at ' + str(datetime.now()) + '\n')
        log_txt.write(f'Total duration: ' + str(datetime.now() - start_time) + '\n')

####################################################################################################################################################################################
### FIGURES
    
    # ====================================================================
    # PLOT: chi2 map & alpha mean map
    # ====================================================================
    
    fig_chi2, ax_chi2 = plt.subplots(1, 1, figsize=(10, 8))
    fig_alpha, ax_alpha = plt.subplots(1, 1, figsize=(10,8))
    
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

    im_alpha = ax_alpha.contourf(xp, yp, alpha_mean_map, levels=50, cmap='viridis')
    ax_alpha.plot(x_best, y_best, 'r*', markersize=20, label=f'Optimized alpha (mean)\n{alpha_mean_best:.6f}')
    ax_alpha.set_xlabel('RA offset [mas]')
    ax_alpha.set_ylabel('Dec offset [mas]')
    ax_alpha.set_title('Alpha mean map')
    ax_alpha.legend()
    ax_alpha.grid(alpha=0.3)
    plt.colorbar(im_alpha, ax=ax_alpha, label=r'$\alpha_{mean}$')

    fig_alpha.savefig(path_output + '/chi2_maps_fits/alpha_mean_map_global.png', dpi=300, bbox_inches='tight')
    plt.close(fig_alpha)

    # ========================================================================
    # PLOT: stellar fits at best position
    # ========================================================================

    # Reload best position
    best_pos_dict = np.load(path_output + '/fitted_params/best_position.npy', allow_pickle=True).item()
    x_best = best_pos_dict['x_best']
    y_best = best_pos_dict['y_best']
    sep_best = best_pos_dict['sep_best']
    PA_best_rad = np.arctan2(x_best, y_best)
    alphas_best = best_pos_dict['alphas_best']  # shape: (n_files, 6)
    
    # Load normalization parameters
    normalization_params = np.load(path_output + '/fitted_params/normalization_params.npy', allow_pickle=True).item()
    wl_mean = normalization_params['wl_mean']
    wl_std = normalization_params['wl_std']
    wl = normalization_params['wl']
    

    for i_file, file_Cps in enumerate(files_Cps):
        print(f"Plot {i_file+1}/{len(files_Cps)}: {file_Cps}")
        
        # Load fitted parameters
        alphas_file = np.load(path_output + f'/fitted_params/alphas_{file_Cps}.npy')  # shape: (6,)
        stellar_coeffs_file = np.load(path_output + f'/fitted_params/stellar_coeffs_{file_Cps}.npy')  # shape: (6, 4)
        
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
        
        for i_base in range(n_base):
            # Get alpha for this baseline
            alpha_optimal = alphas_file[i_base]
            
            # Compute model with fitted parameters
            params = np.concatenate([[alpha_optimal], stellar_coeffs_file[i_base]])
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

            wl_norm = (wl - wl_mean) / wl_std
            stellar_real_poly = np.polyval(stellar_coeffs_file[i_base][:n_poly+1], wl_norm)
            stellar_real_poly_mask = stellar_real_poly[real_err_mask]
            
            ax_Cps_real[i_base].errorbar(wl_real*1e6, Cps_real_cal_mask, 
                                        yerr=Cps_real_cal_err_mask, 
                                        fmt='o', label='Data', color='blue', alpha=0.5)
            ax_Cps_real[i_base].plot(wl_real*1e6, cf_real_model_mask, label='Model', color='red')

            ax_Cps_real[i_base].plot(wl_real*1e6, alpha_optimal * Cps_model_mask + stellar_real_poly_mask, 
                                    label='Envelope', color='purple', linestyle='--')
            ax_Cps_real[i_base].plot(wl_real*1e6, alpha_optimal * Cps_model_mask - stellar_real_poly_mask, 
                                    color='purple', linestyle='--')

            ax_Cps_real[i_base].set_ylabel(f'{base_order_name[i_base]} Cps Real')
            ax_Cps_real[i_base].set_ylim(-3e-3, 3e-3)
            ax_Cps_real[i_base].set_xlim(3.0, 4.15)
            ax_Cps_real[i_base].legend()
            ax_Cps_real[i_base].set_title(f'Alpha = {alpha_optimal:.4f}')

            # Plot imaginary part
            imag_err_mask = data['imag_err_mask'][i_base]
            wl_imag = wl[imag_err_mask]
            Cps_imag_cal_mask = Cps_imag_cal[i_base][imag_err_mask]
            Cps_imag_cal_err_mask = Cps_imag_cal_err[i_base][imag_err_mask]
            cf_imag_model_mask = cf_imag_model[imag_err_mask]
            
            stellar_imag_poly = np.polyval(stellar_coeffs_file[i_base][n_poly+1:], wl_norm)
            stellar_imag_poly_mask = stellar_imag_poly[imag_err_mask]

            ax_Cps_imag[i_base].errorbar(wl_imag*1e6, Cps_imag_cal_mask, 
                                        yerr=Cps_imag_cal_err_mask, 
                                        fmt='o', label='Data', color='green', alpha=0.5)
            ax_Cps_imag[i_base].plot(wl_imag*1e6, cf_imag_model_mask, label='Model', color='orange')

            ax_Cps_imag[i_base].plot(wl_imag*1e6, stellar_imag_poly_mask, 
                                    label='Envelope', color='purple', linestyle='--')
            ax_Cps_imag[i_base].plot(wl_imag*1e6, -stellar_imag_poly_mask, 
                                    color='purple', linestyle='--')
            
            ax_Cps_imag[i_base].set_ylabel(f'{base_order_name[i_base]} Cps Imag')
            ax_Cps_imag[i_base].set_ylim(-4e-3, 4e-3)
            ax_Cps_imag[i_base].set_xlim(3.0, 4.15)
            ax_Cps_imag[i_base].legend()
            ax_Cps_imag[i_base].set_title(f'Alpha = {alpha_optimal:.4f}')
        
        ax_Cps_real[-1].set_xlabel('Wavelength [µm]')
        ax_Cps_imag[-1].set_xlabel('Wavelength [µm]')
        fig_Cps_real.suptitle(f'Fitted Cps Real - {file_Cps}\nBest pos: ({x_best:.1f}, {y_best:.1f}) mas\n(One alpha per baseline)')
        fig_Cps_imag.suptitle(f'Fitted Cps Imag - {file_Cps}\nBest pos: ({x_best:.1f}, {y_best:.1f}) mas\n(One alpha per baseline)')
        fig_Cps_real.savefig(path_output + f'/fitted_params/fitted_Cps_real_{file_Cps[:-5]}.png', 
                            dpi=300, bbox_inches='tight')
        fig_Cps_imag.savefig(path_output + f'/fitted_params/fitted_Cps_imag_{file_Cps[:-5]}.png', 
                            dpi=300, bbox_inches='tight')
        plt.close(fig_Cps_real)
        plt.close(fig_Cps_imag)