# Importation
import numpy as np
from astropy.io import fits
import os 
import matplotlib.pyplot as plt
import scipy 
import shutil
from datetime import datetime
import multiprocessing as mp
from common_tools import wrap, mas2rad


# Output path 
# path_output = '/Users/jscigliuto/exoMATISSE/exoMATISSE/data_hd72946b/test/' #HD 72946 B
path_output = '/Users/jscigliuto/exoMATISSE/exoMATISSE/data_betaPicb/' #beta Pic b

# Science case
sci_case = 'bright' #'faint'   

# Baseline order and names
base_order_name = ('U3-U4', 'U1-U2', 'U2-U3', 'U2-U4', 'U1-U3', 'U1-U4')


### Initializations 
# Planet offsets coords mentioned in the OB [mas]
Offset_RA = 279
Offset_Dec = 455

# Grid of coordinates to determine the astrometry of the planet
x      = np.arange(Offset_RA-10, Offset_RA+10, 5)  
y      = np.arange(Offset_Dec-10, Offset_Dec+10, 5)  
xp, yp = np.meshgrid(x, y)  # grid of coords to look for the planet position

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
for directory in ['/chi2_maps', '/chi2_maps_fits', '/fitted_params']:
    dir_path = path_output + directory
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


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
    
    # Polynômes for stellar speckle
    stellar_poly_real = np.polyval(stellar_coeffs_real, wl)
    stellar_poly_imag = np.polyval(stellar_coeffs_imag, wl)
    
    # Compute model 
    cf_model = alpha * Cps_model + stellar_poly_real * np.cos(2 * np.pi * spat_freq) + 1j * stellar_poly_imag * np.sin(2 * np.pi * spat_freq)

    # Go back to amplitude/phase space
    amp_model = np.abs(cf_model)
    phi_model = np.angle(cf_model)

    return amp_model, phi_model


def residuals_split(params, spat_freq, wl, Cps_model, Cps_real_cal, 
                    Cps_real_cal_err, Cps_imag_cal, Cps_imag_cal_err, 
                    real_err_mask, imag_err_mask):
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
    chi2_real = chi2_real[real_err_mask]
    chi2_imag = chi2_imag[imag_err_mask]
    chi2      = np.sum(chi2_real + chi2_imag)
    chi2_red  = chi2 / (real_err_mask.sum() + imag_err_mask.sum() - params.size)

    return chi2_red


def residuals_fixed_alpha_split(stellar_coeffs, alpha, spat_freq, wl, Cps_model, 
                                Cps_real_cal, Cps_real_cal_err, 
                                Cps_imag_cal, Cps_imag_cal_err, 
                                real_err_mask, imag_err_mask):
    """
    Residuals with fixed alpha, optimizes only stellar_coeffs (real and imag)
    stellar_coeffs contains [coeffs_real, coeffs_imag]
    """
    params = np.concatenate([[alpha], stellar_coeffs])
    return residuals_split(params, spat_freq, wl, Cps_model, 
                          Cps_real_cal, Cps_real_cal_err, 
                          Cps_imag_cal, Cps_imag_cal_err, 
                          real_err_mask, imag_err_mask)


def global_fit_objective_split(alpha, files_Cps, all_data_dict, n_poly, Cps_model, 
                                ix, iy, U_all, V_all, wl, store_results=False):
    """
    Globally optimizes alpha, while fitting stellar_coeffs (real + imag) independently for each file
    
    ix, iy: indices de la position testée dans la grille
    """
    # Calculer separation et PA pour cette position
    sep = np.sqrt(xp[ix, iy]**2 + yp[ix, iy]**2)  # mas
    PA = np.arctan2(xp[ix, iy], yp[ix, iy])  # radians
    
    total_chi2 = 0
    
    for i_file, file_name in enumerate(files_Cps):
        data = all_data_dict[file_name]
        
        # Calculer spat_freq pour cette position
        U = U_all[i_file]
        V = V_all[i_file]
        Bproj = np.sqrt(U**2 + V**2) * mas2rad(sep) * np.cos(np.arctan2(U, V) - PA)
        spat_freq = np.outer(Bproj, 1/wl)  # shape: (n_base, n_wl)
        
        # Initialization: n_poly+1 coeffs for real and imaginary parts 
        stellar_coeffs_init = [1e-4] * (2 * (n_poly + 1))
        bounds_stellar = [(None, None)] * (2 * (n_poly + 1))
        
        if store_results:
            data['fitted_stellar_coeffs'] = np.zeros((6, 2 * (n_poly + 1)))
            data['chi2_per_baseline'] = np.zeros(6)
        
        for i_base in range(6):
            res = scipy.optimize.minimize(
                residuals_fixed_alpha_split, 
                stellar_coeffs_init,
                args=(alpha, spat_freq[i_base], data['wl'], 
                      Cps_model, data['Cps_real_cal'][i_base], 
                      data['Cps_real_cal_err'][i_base],
                      data['Cps_imag_cal'][i_base], 
                      data['Cps_imag_cal_err'][i_base],
                      data['real_err_mask'][i_base],
                      data['imag_err_mask'][i_base]),
                bounds=bounds_stellar,
                method='L-BFGS-B'
            )
            total_chi2 += res.fun
            
            
            # Stocker les résultats si demandé
            if store_results:
                data['fitted_stellar_coeffs'][i_base] = res.x
                data['chi2_per_baseline'][i_base] = res.fun
    
    return total_chi2


if sci_case == 'bright':
    n_base = 6

    start_time = datetime.now()
    with open(path_output + '/log.txt', 'w') as log_txt:
        log_txt.write(f'Starting the fitting ' + str(start_time) + '\n')
        log_txt.write(f'Grid size: {xp.shape[0]} x {xp.shape[1]} = {xp.size} positions\n')
    
    # ========================================================================
    # STEP 1: Load all the data in one dictionary
    # ========================================================================
    print("=" * 70)
    print("Step 1: Load the data")
    print("=" * 70)
    all_data_dict = {}
    U_all = []  # Stocker U pour chaque fichier
    V_all = []  # Stocker V pour chaque fichier
    
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
        
        # Stock U,V
        U_all.append(U)
        V_all.append(V)

        # Apply wavelength mask (wmin, wmax)
        wl_mask = (wl > wmin) & (wl < wmax)
        wl = wl[wl_mask]
        Cps_real_cal = Cps_real_cal[:, wl_mask]
        Cps_real_cal_err = Cps_real_cal_err[:, wl_mask]
        Cps_imag_cal = Cps_imag_cal[:, wl_mask]
        Cps_imag_cal_err = Cps_imag_cal_err[:, wl_mask]

        # SNR calculations
        snr_real = np.abs(Cps_real_cal / Cps_real_cal_err)
        snr_imag = np.abs(Cps_imag_cal / Cps_imag_cal_err)

        # Filter based on data quality (exclude low SNRs)
        real_err_mask = (snr_real.T > np.quantile(snr_real, 0.05, axis=1)).T 
        imag_err_mask = (snr_imag.T > np.quantile(snr_imag, 0.05, axis=1)).T

        # Store the data in all_data_dict[file_Cps]
        all_data_dict[file_Cps] = {
            'wl': wl,
            'Cps_real_cal': Cps_real_cal,
            'Cps_real_cal_err': Cps_real_cal_err,
            'Cps_imag_cal': Cps_imag_cal,
            'Cps_imag_cal_err': Cps_imag_cal_err,
            'real_err_mask': real_err_mask,
            'imag_err_mask': imag_err_mask
        }
    
    # Load Cps model (une seule fois)
    Cps_model = fits.getdata(Cps_model_path)
    if use_bin_data:
        Cps_model = Cps_model.reshape(-1, 5).mean(axis=1)
    Cps_model = Cps_model[wl_mask]
    
    # ========================================================================
    # STEP 2: Loop over grid positions
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Loop over grid positions")
    print("=" * 70)
    
    # Initialize chi2 map
    chi2_map_global = np.zeros(xp.shape)
    alpha_map = np.zeros(xp.shape)
    
    n_positions = xp.size
    
    for idx in range(n_positions):
        ix, iy = np.unravel_index(idx, xp.shape)
        
        if idx % 10 == 0:
            print(f"Position {idx+1}/{n_positions} - ({xp[ix,iy]:.1f}, {yp[ix,iy]:.1f}) mas")
        
        # ====================================================================
        # ÉTAPE 2a: Global optimization of alpha for this position
        # ====================================================================
        result_alpha = scipy.optimize.minimize_scalar(
            lambda a: global_fit_objective_split(a, files_Cps, all_data_dict, n_poly, 
                                                 Cps_model, ix, iy, U_all, V_all, wl, 
                                                 store_results=False),
            method='brent'
        )
        
        alpha_optimal = result_alpha.x
        chi2_total = result_alpha.fun
        
        # Store results
        chi2_map_global[ix, iy] = chi2_total
        alpha_map[ix, iy] = alpha_optimal
        
        if idx % 10 == 0:
            print(f"  Alpha optimal: {alpha_optimal:.6f}, Chi2: {chi2_total:.6f}")
            with open(path_output + '/log.txt', 'a') as log_txt:
                log_txt.write(f'Position ' + str(idx + 1) + '/' + str(n_positions) + ' end at ' + str(datetime.now()) + '\n')
    
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
    print(f"  RA offset:  {x_best:.2f} mas")
    print(f"  Dec offset: {y_best:.2f} mas")
    print(f"  Separation: {sep_best:.2f} mas")
    print(f"  PA:         {PA_best:.2f} deg")
    print(f"  Alpha:      {alpha_best:.6f}")
    print(f"  Chi2:       {chi2_best:.6f}")
    print("=" * 70)
    
    with open(path_output + '/log.txt', 'a') as log_txt:
        log_txt.write(f'\nBest position found:\n')
        log_txt.write(f'  RA offset  = {x_best:.2f} mas\n')
        log_txt.write(f'  Dec offset = {y_best:.2f} mas\n')
        log_txt.write(f'  Separation = {sep_best:.2f} mas\n')
        log_txt.write(f'  PA         = {PA_best:.2f} deg\n')
        log_txt.write(f'  Alpha      = {alpha_best:.6f}\n')
        log_txt.write(f'  Chi2       = {chi2_best:.6f}\n')
    
    # ========================================================================
    # ÉTAPE 3: Final fit at best position with store_results=True
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Final fit at best position")
    print("=" * 70)
    
    final_chi2 = global_fit_objective_split(
        alpha_best, files_Cps, all_data_dict, n_poly, Cps_model, 
        ix_best, iy_best, U_all, V_all, wl, store_results=True
    )
    
    print(f"\n✓ Final fit completed, chi2: {final_chi2:.6f}")
    
    # ========================================================================
    # ÉTAPE 4: Save results
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 4: Save results")
    print("=" * 70)
    
    # Save chi2 maps
    np.save(path_output + '/chi2_maps_fits/chi2_map_global.npy', chi2_map_global)
    np.save(path_output + '/chi2_maps_fits/alpha_map.npy', alpha_map)
    np.save(path_output + '/chi2_maps_fits/xp.npy', xp)
    np.save(path_output + '/chi2_maps_fits/yp.npy', yp)
    
    # Save best position parameters
    best_pos_dict = {
        'x_best': x_best,
        'y_best': y_best,
        'sep_best': sep_best,
        'PA_best': PA_best,
        'alpha_best': alpha_best,
        'chi2_best': chi2_best
    }
    np.save(path_output + '/fitted_params/best_position.npy', best_pos_dict)
    np.save(path_output + '/fitted_params/alpha_global.npy', alpha_best)
    
    # Save stellar_coeffs for each file
    for i_file, file_Cps in enumerate(files_Cps):
        data = all_data_dict[file_Cps]
        
        np.save(path_output + f'/fitted_params/stellar_coeffs_{file_Cps}.npy', 
                data['fitted_stellar_coeffs'])
        np.save(path_output + f'/chi2_maps/chi2_{file_Cps}.npy', 
                data['chi2_per_baseline'])
        
        print(f"  ✓ File {i_file+1}/{len(files_Cps)}: {file_Cps}")
        print(f"    Mean chi2 per baseline: {np.mean(data['chi2_per_baseline']):.6f}")
    
    with open(path_output + '/log.txt', 'a') as log_txt:
        log_txt.write(f'End of fitting at ' + str(datetime.now()) + '\n')
        log_txt.write(f'Total duration: ' + str(datetime.now() - start_time) + '\n')
    
    # ========================================================================
    # Plot chi2 map
    # ========================================================================
    print("\n" + "=" * 70)
    print("Plotting chi2 map")
    print("=" * 70)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    im = ax.contourf(xp, yp, chi2_map_global, levels=50, cmap='viridis')
    ax.plot(x_best, y_best, 'r*', markersize=20, label=f'Best position\n({x_best:.1f}, {y_best:.1f}) mas')
    ax.set_xlabel('RA offset [mas]')
    ax.set_ylabel('Dec offset [mas]')
    ax.set_title('Chi2 map')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.colorbar(im, ax=ax, label='Chi2')
    
    fig.savefig(path_output + '/chi2_maps_fits/chi2_map_global.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("\n" + "=" * 70)
    print("FINISHED!")
    print("=" * 70)


# ============================================================================
# PLOT stellar fits at best position
# ============================================================================
print("\n" + "=" * 70)
print("Generating plots at best position")
print("=" * 70)

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
    wl = data['wl']
    
    # Recalculate spat_freq at best position
    U = U_all[i_file]
    V = V_all[i_file]
    Bproj = np.sqrt(U**2 + V**2) * mas2rad(sep_best) * np.cos(np.arctan2(U, V) - PA_best_rad)
    spat_freq = np.outer(Bproj, 1/wl)
    
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
        
        # Coherent flux model
        cf_model = amp_model * np.exp(1j * phi_model)
        cf_real_model = np.real(cf_model) 
        cf_imag_model = np.imag(cf_model)
        
        # Plot real part
        real_err_mask         = data['real_err_mask'][i_base]
        wl_real               = wl[real_err_mask]
        Cps_real_cal_mask     = Cps_real_cal[:,real_err_mask]
        Cps_real_cal_err_mask = Cps_real_cal_err[:,real_err_mask]
        cf_real_model_mask    = cf_real_model[real_err_mask]
        Cps_model_mask        = Cps_model[real_err_mask]
        stellar_real_poly     = np.polyval(fitted_stellar_coeffs[i_base][:n_poly+1], wl_real)
        
        ax_Cps_real[i_base].errorbar(wl_real*1e6, Cps_real_cal_mask[i_base], 
                                      yerr=np.sqrt(Cps_real_cal_err_mask[i_base]**2), 
                                      fmt='o', label='Data', color='blue', alpha=0.5)
        ax_Cps_real[i_base].plot(wl_real*1e6, cf_real_model_mask, label='Model', color='red')
        ax_Cps_real[i_base].plot(wl_real*1e6, alpha_optimal * Cps_model_mask + stellar_real_poly, 
                                 label='Envelope', color='purple', linestyle='--')
        ax_Cps_real[i_base].plot(wl_real*1e6, alpha_optimal * Cps_model_mask - stellar_real_poly, 
                                 color='purple', linestyle='--')
        ax_Cps_real[i_base].set_ylabel(f'{base_order_name[i_base]} Cps Real')
        ax_Cps_real[i_base].set_ylim(-3e-3, 3e-3)
        ax_Cps_real[i_base].set_xlim(3.0, 4.15)
        ax_Cps_real[i_base].legend()

        # Plot imaginary part
        imag_err_mask         = data['imag_err_mask'][i_base]
        wl_imag               = wl[imag_err_mask]
        Cps_imag_cal_mask     = Cps_imag_cal[:,imag_err_mask]
        Cps_imag_cal_err_mask = Cps_imag_cal_err[:,imag_err_mask]
        cf_imag_model_mask    = cf_imag_model[imag_err_mask]
        stellar_imag_poly     = np.polyval(fitted_stellar_coeffs[i_base][n_poly+1:], wl_imag)

        ax_Cps_imag[i_base].errorbar(wl_imag*1e6, Cps_imag_cal_mask[i_base], 
                                      yerr=np.sqrt(Cps_imag_cal_err_mask[i_base]**2), 
                                      fmt='o', label='Data', color='green', alpha=0.5)
        ax_Cps_imag[i_base].plot(wl_imag*1e6, cf_imag_model_mask, label='Model', color='orange')
        ax_Cps_imag[i_base].plot(wl_imag*1e6, stellar_imag_poly, 
                                 label='Envelope', color='purple', linestyle='--')
        ax_Cps_imag[i_base].plot(wl_imag*1e6, -stellar_imag_poly, 
                                 color='purple', linestyle='--')
        ax_Cps_imag[i_base].set_ylabel(f'{base_order_name[i_base]} Cps Imag')
        ax_Cps_imag[i_base].set_ylim(-4e-3, 4e-3)
        ax_Cps_imag[i_base].set_xlim(3.0, 4.15)
        ax_Cps_imag[i_base].legend()
    
    ax_Cps_real[-1].set_xlabel('Wavelength [µm]')
    ax_Cps_imag[-1].set_xlabel('Wavelength [µm]')
    fig_Cps_real.suptitle(f'Fitted Cps - {file_Cps}\nBest pos: ({x_best:.1f}, {y_best:.1f}) mas')
    fig_Cps_imag.suptitle(f'Fitted Cps - {file_Cps}\nBest pos: ({x_best:.1f}, {y_best:.1f}) mas')
    fig_Cps_real.savefig(path_output + f'/fitted_params/fitted_Cps_real_{file_Cps[:-5]}.png', 
                         dpi=300, bbox_inches='tight')
    fig_Cps_imag.savefig(path_output + f'/fitted_params/fitted_Cps_imag_{file_Cps[:-5]}.png', 
                         dpi=300, bbox_inches='tight')
    plt.close(fig_Cps_real)
    plt.close(fig_Cps_imag)

