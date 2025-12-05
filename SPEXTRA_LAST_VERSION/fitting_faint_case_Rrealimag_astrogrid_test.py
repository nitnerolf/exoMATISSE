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
# x      = np.arange(Offset_RA-5, Offset_RA+5, 1.) 
# y      = np.arange(Offset_Dec-5, Offset_Dec+5, 1.) 
x      = np.arange(Offset_RA-15, Offset_RA+15, 0.4) 
y      = np.arange(Offset_Dec-15, Offset_Dec+15, 0.4) 
# x      = np.arange(Offset_RA-2, Offset_RA+2, 1) 
# y      = np.arange(Offset_Dec-2, Offset_Dec+2, 1)
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
    
    Les modulations sont attachées au contraste, pas au polynôme stellaire
    """
    # Extract parameters
    alpha = params[0]
    stellar_coeffs_real = params[1:]  

    wl_norm = (wl - wl.mean()) / wl.std()
    
    # Polynomial for stellar speckle (REAL only)
    stellar_poly_real = np.polyval(stellar_coeffs_real, wl_norm)
    
    # Compute model: les modulations sont sur le contraste
    cf_model = alpha * Cps_model * (np.cos(2 * np.pi * spat_freq) + 1j * np.sin(2 * np.pi * spat_freq)) + stellar_poly_real
    
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
    
    NOUVEAU MODÈLE:
    params[0] = alpha (only one for all the frames/baselines)
    params[1:] = stellar coeffs REAL only for all files and baselines
               = n_files * 6_baselines * (n_poly+1) coefficients
    """
    alpha = params[0]
    n_coeffs_per_baseline = n_poly + 1  #Only real coeffs
    
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
    stellar_coeffs_all = params[1:].reshape(n_files, 6, n_coeffs_per_baseline)
    
    # Compute all stellar polynomials (REAL only)
    stellar_poly_real = np.zeros((n_files, 6, wl.size))
    
    for i_file in range(n_files):
        for i_base in range(6):
            stellar_poly_real[i_file, i_base] = np.polyval(stellar_coeffs_all[i_file, i_base], wl_norm)

    # Compute the model (vectorized)
    contrast_with_modulation = alpha * Cps_model[np.newaxis, np.newaxis, :] * \
                              (np.cos(2 * np.pi * spat_freq) + 1j * np.sin(2 * np.pi * spat_freq))
    
    cf_model = contrast_with_modulation + stellar_poly_real
    
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
    if n_dof_total > 0:
        chi2_red_global = total_chi2 / n_dof_total
    else:
        chi2_red_global = np.inf

    return chi2_red_global


def process_single_position_global(idx, xp, yp, global_data, n_poly, Cps_model, wl, n_files, path_output):
    """
    Fit all data simultaneously for a given position in the grid
    """
    start_time = get_time()
    ix, iy = np.unravel_index(idx, xp.shape)
    
    # Total number of parameters
    n_coeffs_per_baseline = n_poly + 1  # MODIFIÉ: plus que les coeffs réels
    n_baselines = 6
    n_params = 1 + (n_files * n_baselines * n_coeffs_per_baseline)
    
    # Initialization 
    params_init = np.zeros(n_params)
    params_init[0] = 1.0  # alpha initial = 1

    coeff_idx = 1
    Cps_real = global_data['Cps_real']
    Cps_imag = global_data['Cps_imag']
    
    wl_norm = (wl - wl.mean()) / wl.std()
    
    for i_file in range(n_files):
        for i_base in range(6):
            # Initialiser avec la partie réelle uniquement
            offset_real_init = np.mean(Cps_real[i_file, i_base]) 
            slope_real_init = np.polyfit(wl_norm, Cps_real[i_file, i_base], 1)[0] 
            
            params_init[coeff_idx:coeff_idx+n_coeffs_per_baseline] = [
                slope_real_init, offset_real_init]
            
            coeff_idx += n_coeffs_per_baseline
    
    init_time = get_time() - start_time
    
    ## Define bounds 
    # Alpha bounds
    bounds = [(0.1, 10.0)]  
    
    # Stellar coeffs bounds (REAL only)
    for i in range(n_files * n_baselines):
        bounds.append((-1.0, 1.0))  # slope real 
        bounds.append((-1.0, 1.0))  # offset real 

    optim_start_time = get_time()

    result = scipy.optimize.minimize(
        global_fit_simultaneous,
        params_init,
        args=(global_data, n_poly, Cps_model, wl, xp, yp, ix, iy, n_files),
        method='L-BFGS-B',  
        bounds=bounds,
        # options={
        #     'maxiter': 500,
        #     'ftol': 1e-4,
        #     'disp': False,
        # }
    )
    
    optim_time = get_time() - optim_start_time
    total_time = get_time() - start_time
    
    alpha_optimal = result.x[0]
    chi2_total = result.fun
    fitted_coeffs = result.x[1:]

    # Log
    with open(path_output + '/timing_log.txt', 'a') as f:
        f.write(f"pos({xp[ix,iy]:.1f},{yp[ix,iy]:.1f}) | init={init_time:.2f}s | optim={optim_time:.2f}s | "
                f"total={total_time:.2f}s | alpha={alpha_optimal:.6f} | chi2={chi2_total:.6f} | "
                f"iter={result.nit} | feval={result.nfev}\n")
    
    return idx, ix, iy, alpha_optimal, chi2_total, fitted_coeffs


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
        alpha_map = np.zeros(xp.shape)
        n_positions = xp.size

        process_func = partial(process_single_position_global,
            xp=xp, yp=yp, global_data=global_data, n_poly=n_poly, 
            Cps_model=Cps_model, wl=wl, n_files=n_files, path_output=path_output)

        print(f"Processing {n_positions} positions with {n_cores} cores...")

        with mp.Pool(processes=n_cores) as pool:
            results = []
            for i, result in enumerate(pool.imap(process_func, range(n_positions))):
                idx, ix, iy, alpha_opt, chi2, fitted_coeffs = result
                results.append(result)

                elapsed_time = get_time() - step2_start_time
                avg_time_per_pos = elapsed_time / (i + 1)
                estimated_remaining = avg_time_per_pos * (n_positions - (i + 1))
                
                if (i+1) % 10 == 0 or (i+1) == n_positions:
                    print(f"Completed {i+1}/{n_positions} positions | Time: {datetime.now()} | "
                          f"Current: ({xp[ix,iy]:.1f}, {yp[ix,iy]:.1f}) mas, "
                          f"alpha={alpha_opt:.4f}, chi2={chi2:.2f} | "
                          f"Elapsed: {elapsed_time/60:.1f}min | Avg: {avg_time_per_pos:.1f}s/pos | "
                          f"ETA: {estimated_remaining/60:.1f}min")

                    with open(path_output + '/timing_log.txt', 'a') as f:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"CHECKPOINT: {i+1}/{n_positions} positions completed\n")
                        f.write(f"  Total elapsed time: {elapsed_time:.2f}s ({elapsed_time/60:.2f}min)\n")
                        f.write(f"  Average time per position: {avg_time_per_pos:.2f}s\n")
                        f.write(f"  Estimated remaining time: {estimated_remaining:.2f}s ({estimated_remaining/60:.2f}min)\n")
                        f.write(f"  Current position: ({xp[ix,iy]:.1f}, {yp[ix,iy]:.1f}) mas\n")
                        f.write(f"  Current alpha: {alpha_opt:.6f}\n")
                        f.write(f"  Current chi2: {chi2:.6f}\n")
                        f.write(f"{'='*80}\n\n")

        # Fill the alpha and chi2 maps 
        for idx, ix, iy, alpha_optimal, chi2_total, fitted_coeffs in results:
            chi2_map_global[ix, iy] = chi2_total
            alpha_map[ix, iy] = alpha_optimal
        
        # Find best position
        idx_min = np.argmin(chi2_map_global)
        ix_best, iy_best = np.unravel_index(idx_min, xp.shape)
        x_best = xp[ix_best, iy_best]
        y_best = yp[ix_best, iy_best]
        chi2_best = chi2_map_global[ix_best, iy_best]
        alpha_best = alpha_map[ix_best, iy_best]
        sep_best = np.sqrt(x_best**2 + y_best**2)
        PA_best = np.rad2deg(np.arctan2(x_best, y_best))
        
        print("\n" + "=" * 70)
        print("BEST POSITION FOUND:")
        print("=" * 70)
        print(f"RA offset:  {x_best:.2f} mas")
        print(f"Dec offset: {y_best:.2f} mas")
        print(f"Separation: {sep_best:.2f} mas")
        print(f"PA:         {PA_best:.2f} deg")
        print(f"Alpha:      {alpha_best:.6f}")
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
            log_txt.write(f'Alpha      = {alpha_best:.6f}\n')
            log_txt.write(f'Chi2       = {chi2_best:.6f}\n')
        
        # ====================================================================
        # STEP 3: Extract and save coefficients at best position
        # ====================================================================
        print("\n" + "=" * 70)
        print("STEP 3: Extract fitted coefficients at best position")
        print("=" * 70)
        
        # Find the result corresponding to the best position
        best_result = None
        for result in results:
            idx, ix, iy, alpha_opt, chi2, fitted_coeffs = result
            if ix == ix_best and iy == iy_best:
                best_result = result
                break

        if best_result is not None:
            _, _, _, alpha_best, chi2_best, fitted_coeffs_best = best_result
            
            # Extract and save coefficients for each file
            n_coeffs_per_baseline = n_poly + 1  # MODIFIÉ: plus que real
            coeff_idx = 0
            
            for i_file, file_Cps in enumerate(files_Cps):
                stellar_coeffs_file = np.zeros((6, n_coeffs_per_baseline))
                
                for i_base in range(6):
                    stellar_coeffs_file[i_base] = fitted_coeffs_best[coeff_idx : coeff_idx + n_coeffs_per_baseline]
                    coeff_idx += n_coeffs_per_baseline

                # Store in the main data dictionary for plots
                all_data_dict[file_Cps]['fitted_stellar_coeffs'] = stellar_coeffs_file
                
                # Save to file
                np.save(path_output + f'/fitted_params/stellar_coeffs_{file_Cps}.npy', stellar_coeffs_file)
                
                print(f"Saved coefficients for {file_Cps}")
        
        # ====================================================================
        # STEP 4: Save results
        # ====================================================================
        print("\n" + "=" * 70)
        print("STEP 4: Save results")
        print("=" * 70)
        
        # Save maps
        np.save(path_output + '/chi2_maps_fits/chi2_map_global.npy', chi2_map_global)
        np.save(path_output + '/chi2_maps_fits/alpha_map.npy', alpha_map)
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
            'alpha_best': alpha_best,
            'chi2_best': chi2_best
        }
        np.save(path_output + '/fitted_params/best_position.npy', best_pos_dict)
        np.save(path_output + '/fitted_params/alpha_global.npy', alpha_best) 
        
        with open(path_output + '/log.txt', 'a') as log_txt:
            log_txt.write(f'End of fitting at ' + str(datetime.now()) + '\n')
            log_txt.write(f'Total duration: ' + str(datetime.now() - start_time) + '\n')

####################################################################################################################################################################################
### FIGURES
        
        # ====================================================================
        # PLOT: chi2 map & alpha map
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

        im_alpha = ax_alpha.contourf(xp, yp, alpha_map, levels=50, cmap='viridis')
        ax_alpha.plot(x_best, y_best, 'r*', markersize=20, label=f'Optimized alpha\n{alpha_best:.6f}')
        ax_alpha.set_xlabel('RA offset [mas]')
        ax_alpha.set_ylabel('Dec offset [mas]')
        ax_alpha.set_title('Alpha map')
        ax_alpha.legend()
        ax_alpha.grid(alpha=0.3)
        plt.colorbar(im_alpha, ax=ax_alpha, label=r'$\alpha$')

        fig_alpha.savefig(path_output + '/chi2_maps_fits/alpha_map_global.png', dpi=300, bbox_inches='tight')
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
    alpha_optimal = best_pos_dict['alpha_best']

    for i_file, file_Cps in enumerate(files_Cps):
        print(f"Plot {i_file+1}/{len(files_Cps)}: {file_Cps}")
        
        # Load fitted parameters
        fitted_stellar_coeffs = np.load(path_output + f'/fitted_params/stellar_coeffs_{file_Cps}.npy')
        
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
            params = np.concatenate([[alpha_optimal], fitted_stellar_coeffs[i_base]])
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

            wl_real_norm = (wl_real - wl_real.mean()) / wl_real.std()
            stellar_real_poly = np.polyval(fitted_stellar_coeffs[i_base], wl_real_norm)
            
            ax_Cps_real[i_base].errorbar(wl_real*1e6, Cps_real_cal_mask, 
                                          yerr=Cps_real_cal_err_mask, 
                                          fmt='o', label='Data', color='blue', alpha=0.5)
            ax_Cps_real[i_base].plot(wl_real*1e6, cf_real_model_mask, label='Model', color='red')

            ax_Cps_real[i_base].plot(wl_real*1e6, stellar_real_poly, label='Stellar contamination', color='purple', linestyle='--')

            ax_Cps_real[i_base].set_ylabel(f'{base_order_name[i_base]} Cps Real')
            ax_Cps_real[i_base].set_ylim(-9e-2, 9e-2)
            ax_Cps_real[i_base].set_xlim(3.0, 4.15)
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
            ax_Cps_imag[i_base].set_ylim(-9e-2, 9e-2)
            ax_Cps_imag[i_base].set_xlim(3.0, 4.15)
            ax_Cps_imag[i_base].legend()

            # Modulations real part
            if i_base == 5:
                ax_mod_real[i_base].set_ylabel('Modulations Real')
                ax_mod_real[i_base].set_xlabel('Wavelength [µm]')
                ax_mod_real[i_base].plot(wl_real*1e6, Cps_real_cal_mask - stellar_real_poly, label='Contrast', color='blue', alpha=0.8)
                ax_mod_real[i_base].plot(wl_real*1e6, cf_real_model_mask - stellar_real_poly, label='Model', color='red', alpha=0.8)
            ax_mod_real[i_base].plot(wl_real*1e6, Cps_real_cal_mask - stellar_real_poly, color='blue', alpha=0.8)
            ax_mod_real[i_base].plot(wl_real*1e6, cf_real_model_mask - stellar_real_poly, color='red', alpha=0.8)
            ax_mod_real[i_base].plot(wl_real*1e6, alpha_optimal * Cps_model_mask, label='Cps Model', color='black', linestyle='--', alpha=0.5)
            ax_mod_real[i_base].plot(wl_real*1e6, - alpha_optimal * Cps_model_mask, label='Cps Model', color='black', linestyle='--', alpha=0.5)
            ax_mod_real[i_base].set_xlim(3., 4.1)

            # Modulations imaginary part
            if i_base == 5:
                ax_mod_imag[i_base].set_ylabel('Modulations Imag')
                ax_mod_imag[i_base].set_xlabel('Wavelength [µm]')
                ax_mod_imag[i_base].plot(wl_imag*1e6, Cps_imag_cal_mask, label='Contrast', color='green', alpha=0.8)
                ax_mod_imag[i_base].plot(wl_imag*1e6, cf_imag_model_mask, label='Model', color='orange', alpha=0.8)
            ax_mod_imag[i_base].plot(wl_imag*1e6, Cps_imag_cal_mask, label='Contrast', color='green', alpha=0.8)
            ax_mod_imag[i_base].plot(wl_imag*1e6, cf_imag_model_mask, label='Model', color='orange', alpha=0.8)
            ax_mod_imag[i_base].plot(wl_imag*1e6, alpha_optimal * Cps_model_mask, label='Cps Model', color='black', linestyle='--', alpha=0.5)
            ax_mod_imag[i_base].plot(wl_imag*1e6, - alpha_optimal * Cps_model_mask, label='Cps Model', color='black', linestyle='--', alpha=0.5)
            ax_mod_imag[i_base].set_xlim(3., 4.1)

        ax_Cps_real[-1].set_xlabel('Wavelength [µm]')
        ax_Cps_imag[-1].set_xlabel('Wavelength [µm]')
        fig_Cps_real.suptitle(f'Fitted Cps - {file_Cps}\nBest pos: ({x_best:.1f}, {y_best:.1f}) mas')
        fig_Cps_imag.suptitle(f'Fitted Cps - {file_Cps}\nBest pos: ({x_best:.1f}, {y_best:.1f}) mas')
        fig_Cps_real.savefig(path_output + f'/fitted_params/fitted_Cps_real_{file_Cps[:-5]}.png', 
                             dpi=300, bbox_inches='tight')
        fig_Cps_imag.savefig(path_output + f'/fitted_params/fitted_Cps_imag_{file_Cps[:-5]}.png', 
                             dpi=300, bbox_inches='tight')
        fig_mod_real.savefig(path_output + f'/fitted_params/modulations_real_{file_Cps[:-5]}.png', 
                             dpi=300, bbox_inches='tight')
        fig_mod_imag.savefig(path_output + f'/fitted_params/modulations_imag_{file_Cps[:-5]}.png', 
                             dpi=300, bbox_inches='tight')
