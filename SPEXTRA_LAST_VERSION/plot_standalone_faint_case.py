# Importation
import numpy as np
from astropy.io import fits
import os 
import matplotlib.pyplot as plt
from datetime import datetime

####################################################################################################################################################################################
### INPUTS SET UP 

# Output path 
path_output = '/Users/jscigliuto/exoMATISSE/exoMATISSE/data_hd72946b/' #HD 72946 B

# Degree of the polynomial to model the stellar speckle
n_poly = 1 

# Wavelength range to fit
wmin = 3.0e-6
wmax = 4.15e-6

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

def model_bright(params, wl, spat_freq, Cps_model, wl_mean, wl_std):
    """
    Model with stellar polynomial (REAL only)
    
    cf_model = alpha * Cps_model * (cos(2π·f) + i·sin(2π·f)) + stellar_poly_real
    """
    # Extract parameters
    alpha = params[0]
    stellar_coeffs_real = params[1:]  

    wl_norm = (wl - wl_mean) / wl_std
    
    # Polynomial for stellar speckle (REAL only)
    stellar_poly_real = np.polyval(stellar_coeffs_real, wl_norm)
    
    # Compute model
    cf_model = alpha * Cps_model * (np.cos(2 * np.pi * spat_freq) + 1j * np.sin(2 * np.pi * spat_freq)) + stellar_poly_real
    
    # Go back to amplitude/phase space
    amp_model = np.abs(cf_model)
    phi_model = np.angle(cf_model)

    return amp_model, phi_model


####################################################################################################################################################################################
### SCRIPT

print("="*70)
print("REGENERATING FIGURES FROM SAVED RESULTS")
print("="*70)
print()

# Create figures directory if needed
if not os.path.isdir(path_output + '/figures_replot'):
    os.makedirs(path_output + '/figures_replot')

# Load Cps files list
files_Cps = sorted([file for file in os.listdir(path_output + Cps_path) if '.fits' in file])
print(f"Found {len(files_Cps)} Cps files")
print()

# Load wavelengths from first file
hdu = fits.open(path_output + Cps_path + files_Cps[0])
wl = hdu['WAVELENGTH'].data
hdu.close()

# Load Cps model
Cps_model = fits.getdata(Cps_model_path)
if use_bin_data:
    Cps_model = Cps_model.reshape(-1, 5).mean(axis=1)

wl_mask = (wl > wmin) & (wl < wmax)
Cps_model = Cps_model[wl_mask]  
wl = wl[wl_mask]

# Load best position
best_pos_dict = np.load(path_output + '/fitted_params/best_position.npy', allow_pickle=True).item()
x_best = best_pos_dict['x_best']
y_best = best_pos_dict['y_best']
sep_best = best_pos_dict['sep_best']
PA_best_rad = np.arctan2(x_best, y_best)
alphas_best = best_pos_dict['alphas_best']

print(f"Best position: ({x_best:.2f}, {y_best:.2f}) mas")
print(f"Separation: {sep_best:.2f} mas")
print(f"PA: {np.rad2deg(PA_best_rad):.2f} deg")
print()

# Load normalization parameters
normalization_params = np.load(path_output + '/fitted_params/normalization_params.npy', allow_pickle=True).item()
wl_mean = normalization_params['wl_mean']
wl_std = normalization_params['wl_std']


# Load grid coordinates
xp = np.load(path_output + '/chi2_maps_fits/xp.npy')
yp = np.load(path_output + '/chi2_maps_fits/yp.npy')

# ====================================================================
# PLOT 1: Chi2 map
# ====================================================================
print("Generating Chi2 map...")

chi2_map_global = np.load(path_output + '/chi2_maps_fits/chi2_map_global.npy')

fig_chi2, ax_chi2 = plt.subplots(1, 1, figsize=(10, 8))

im_chi2 = ax_chi2.contourf(xp, yp, chi2_map_global, levels=50, cmap='viridis')
ax_chi2.plot(x_best, y_best, 'r*', markersize=20, label=f'Best position\n({x_best:.1f}, {y_best:.1f}) mas')
ax_chi2.set_xlabel('RA offset [mas]')
ax_chi2.set_ylabel('Dec offset [mas]')
ax_chi2.set_title('Chi2 map')
ax_chi2.legend()
ax_chi2.grid(alpha=0.3)
plt.colorbar(im_chi2, ax=ax_chi2, label=r'$\chi^2$')

fig_chi2.savefig(path_output + '/figures_replot/chi2_map_global.png', dpi=300, bbox_inches='tight')
plt.close(fig_chi2)
print("✓ Saved: chi2_map_global.png")



# ====================================================================
# PLOT 3: Fitted Cps for each file at best position
# ====================================================================
plot_all_file = True
if plot_all_file == True : 
    print("\nGenerating fitted Cps plots...")

    for i_file, file_Cps in enumerate(files_Cps):
        print(f"  File {i_file+1}/{len(files_Cps)}: {file_Cps}")
        
        # Load Cps data
        hdu = fits.open(path_output + Cps_path + file_Cps)
        wl_file = hdu['WAVELENGTH'].data
        Cps_real_cal = hdu['CPS_REAL'].data
        Cps_real_cal_err = hdu['CPS_REAL_ERR'].data
        Cps_imag_cal = hdu['CPS_IMAG'].data
        Cps_imag_cal_err = hdu['CPS_IMAG_ERR'].data
        U = hdu['U'].data
        V = hdu['V'].data
        hdu.close()

        # Mask wavelength 
        wl_mask_file = (wl_file > wmin) & (wl_file < wmax)
        wl_file = wl_file[wl_mask_file]
        Cps_real_cal = Cps_real_cal[:, wl_mask_file]
        Cps_real_cal_err = Cps_real_cal_err[:, wl_mask_file]
        Cps_imag_cal = Cps_imag_cal[:, wl_mask_file]
        Cps_imag_cal_err = Cps_imag_cal_err[:, wl_mask_file]
        
        # Mask based on SNR
        snr_real = np.abs(Cps_real_cal / Cps_real_cal_err) 
        snr_imag = np.abs(Cps_imag_cal / Cps_imag_cal_err)
        real_err_mask = (snr_real.T > np.quantile(snr_real, 0.05, axis=1)).T & (snr_real.T < np.quantile(snr_real, 0.95, axis=1)).T
        imag_err_mask = (snr_imag.T > np.quantile(snr_imag, 0.05, axis=1)).T & (snr_imag.T < np.quantile(snr_imag, 0.95, axis=1)).T
        
        # Load fitted parameters
        fitted_stellar_coeffs = np.load(path_output + f'/fitted_params/stellar_coeffs_{file_Cps}.npy')
        alphas_optimal = np.load(path_output + f'/fitted_params/alphas_{file_Cps}.npy')
        
        # Recalculate spatial frequency at best position
        Bproj = np.sqrt(U**2 + V**2) * mas2rad(sep_best) * np.cos(np.arctan2(U, V) - PA_best_rad)
        spat_freq = np.outer(Bproj, 1/wl)
        
        n_base = spat_freq.shape[0]

        # Create figures
        fig_Cps_real, ax_Cps_real = plt.subplots(n_base, 1, figsize=(8, 3*n_base), sharex=True)
        fig_Cps_imag, ax_Cps_imag = plt.subplots(n_base, 1, figsize=(8, 3*n_base), sharex=True)
        fig_mod_real, ax_mod_real = plt.subplots(n_base, 1, figsize=(8, 3*n_base), sharex=True)
        fig_mod_imag, ax_mod_imag = plt.subplots(n_base, 1, figsize=(8, 3*n_base), sharex=True)
        
        for i_base in range(n_base):
            # Compute model with fitted parameters
            params = np.concatenate([[alphas_optimal[i_base]], fitted_stellar_coeffs[i_base]])
            amp_model, phi_model = model_bright(params, wl, spat_freq[i_base], Cps_model, wl_mean, wl_std)
            
            # Coherent flux model
            cf_model = amp_model * np.exp(1j * phi_model)
            cf_real_model = np.real(cf_model) 
            cf_imag_model = np.imag(cf_model)
            
            # Plot real part
            wl_real = wl[real_err_mask[i_base]]
            Cps_real_cal_mask = Cps_real_cal[i_base][real_err_mask[i_base]]
            Cps_real_cal_err_mask = Cps_real_cal_err[i_base][real_err_mask[i_base]]
            cf_real_model_mask = cf_real_model[real_err_mask[i_base]]
            Cps_model_mask = Cps_model[real_err_mask[i_base]]

            wl_real_norm = (wl_real - wl_mean) / wl_std
            stellar_real_poly = np.polyval(fitted_stellar_coeffs[i_base], wl_real_norm)
            
            ax_Cps_real[i_base].errorbar(wl_real*1e6, Cps_real_cal_mask, 
                                        yerr=Cps_real_cal_err_mask, 
                                        fmt='o', label='Data', color='blue', alpha=0.5)
            ax_Cps_real[i_base].plot(wl_real*1e6, cf_real_model_mask, label='Model', color='red')
            ax_Cps_real[i_base].plot(wl_real*1e6, stellar_real_poly, label='Stellar contamination', 
                                    color='purple', linestyle='--')

            ax_Cps_real[i_base].set_ylabel(f'{base_order_name[i_base]} Cps Real')
            ax_Cps_real[i_base].set_ylim(-3e-2, 4e-2)
            ax_Cps_real[i_base].set_xlim(3.0, 4.15)
            ax_Cps_real[i_base].set_title(f'α = {alphas_optimal[i_base]:.4f}')
            ax_Cps_real[i_base].legend()

            # Plot imaginary part
            wl_imag = wl[imag_err_mask[i_base]]
            Cps_imag_cal_mask = Cps_imag_cal[i_base][imag_err_mask[i_base]]
            Cps_imag_cal_err_mask = Cps_imag_cal_err[i_base][imag_err_mask[i_base]]
            cf_imag_model_mask = cf_imag_model[imag_err_mask[i_base]]

            ax_Cps_imag[i_base].errorbar(wl_imag*1e6, Cps_imag_cal_mask, 
                                        yerr=Cps_imag_cal_err_mask, 
                                        fmt='o', label='Data', color='green', alpha=0.5)
            ax_Cps_imag[i_base].plot(wl_imag*1e6, cf_imag_model_mask, label='Model', color='orange')
            
            ax_Cps_imag[i_base].set_ylabel(f'{base_order_name[i_base]} Cps Imag')
            ax_Cps_imag[i_base].set_ylim(-3e-2, 3e-2)
            ax_Cps_imag[i_base].set_xlim(3.0, 4.15)
            ax_Cps_imag[i_base].set_title(f'α = {alphas_optimal[i_base]:.4f}')
            ax_Cps_imag[i_base].legend()

            # Modulations real part
            if i_base == 5:
                ax_mod_real[i_base].set_ylabel('Modulations Real')
                ax_mod_real[i_base].set_xlabel('Wavelength [µm]')
                ax_mod_real[i_base].errorbar(wl_real*1e6, Cps_real_cal_mask - stellar_real_poly, yerr= Cps_real_cal_err_mask,
                                        label='Contrast', color='blue', fmt='+', alpha=0.8)
                ax_mod_real[i_base].plot(wl_real*1e6, cf_real_model_mask - stellar_real_poly, 
                                        label='Model', color='red', alpha=0.8)
            #data
            ax_mod_real[i_base].errorbar(wl_real*1e6, Cps_real_cal_mask - stellar_real_poly, yerr= Cps_real_cal_err_mask,
                                    color='blue', fmt='+', alpha=0.8)
            #model
            ax_mod_real[i_base].plot(wl_real*1e6, cf_real_model_mask - stellar_real_poly, 
                                    color='red', alpha=0.8)
            #envelope
            ax_mod_real[i_base].plot(wl_real*1e6, alphas_optimal[i_base] * Cps_model_mask, 
                                    label='Cps Model', color='black', linestyle='--', alpha=0.5)
            ax_mod_real[i_base].plot(wl_real*1e6, -alphas_optimal[i_base] * Cps_model_mask, 
                                    color='black', linestyle='--', alpha=0.5)
            ax_mod_real[i_base].set_xlim(3., 4.1)
            ax_mod_real[i_base].set_ylim(-5e-4, 5e-4)
            ax_mod_real[i_base].set_title(f'α = {alphas_optimal[i_base]:.4f}')

            # Modulations imaginary part
            if i_base == 5:
                ax_mod_imag[i_base].set_ylabel('Modulations Imag')
                ax_mod_imag[i_base].set_xlabel('Wavelength [µm]')
                ax_mod_imag[i_base].errorbar(wl_imag*1e6, Cps_imag_cal_mask, yerr=Cps_imag_cal_err_mask,
                                        label='Contrast', color='green', fmt='+', alpha=0.8)
                ax_mod_imag[i_base].plot(wl_imag*1e6, cf_imag_model_mask, 
                                        label='Model', color='orange', alpha=0.8)
            #data
            ax_mod_imag[i_base].errorbar(wl_imag*1e6, Cps_imag_cal_mask, yerr=Cps_imag_cal_err_mask, 
                                        color='green', fmt='+', alpha=0.8)
            #model
            ax_mod_imag[i_base].plot(wl_imag*1e6, cf_imag_model_mask, color='orange', alpha=0.8)
            #envelope
            ax_mod_imag[i_base].plot(wl_imag*1e6, alphas_optimal[i_base] * Cps_model_mask, 
                                    label='Cps Model', color='black', linestyle='--', alpha=0.5)
            ax_mod_imag[i_base].plot(wl_imag*1e6, -alphas_optimal[i_base] * Cps_model_mask, 
                                    color='black', linestyle='--', alpha=0.5)
            ax_mod_imag[i_base].set_xlim(3., 4.1)
            ax_mod_imag[i_base].set_ylim(-5e-4, 5e-4)
            ax_mod_imag[i_base].set_title(f'α = {alphas_optimal[i_base]:.4f}')

        ax_Cps_real[-1].set_xlabel('Wavelength [µm]')
        ax_Cps_imag[-1].set_xlabel('Wavelength [µm]')
        fig_Cps_real.suptitle(f'Fitted Cps - {file_Cps}\nBest pos: ({x_best:.1f}, {y_best:.1f}) mas')
        fig_Cps_imag.suptitle(f'Fitted Cps - {file_Cps}\nBest pos: ({x_best:.1f}, {y_best:.1f}) mas')
        fig_Cps_real.tight_layout()
        fig_Cps_imag.tight_layout()
        fig_mod_real.tight_layout()
        fig_mod_imag.tight_layout()
        
        fig_Cps_real.savefig(path_output + f'/figures_replot/fitted_Cps_real_{file_Cps[:-5]}.png', 
                            dpi=300, bbox_inches='tight')
        fig_Cps_imag.savefig(path_output + f'/figures_replot/fitted_Cps_imag_{file_Cps[:-5]}.png', 
                            dpi=300, bbox_inches='tight')
        fig_mod_real.savefig(path_output + f'/figures_replot/modulations_real_{file_Cps[:-5]}.png', 
                            dpi=300, bbox_inches='tight')
        fig_mod_imag.savefig(path_output + f'/figures_replot/modulations_imag_{file_Cps[:-5]}.png', 
                            dpi=300, bbox_inches='tight')
        
        plt.close('all')

    print("✓ All fitted Cps plots saved")

# ====================================================================
# PLOT 4: Stacked modulations (if available)
# ====================================================================
print("\nGenerating stacked modulations plots...")

try:
    all_stacked_data = np.load(path_output + '/fitted_params/all_stacked_data.npy', allow_pickle=True)
    n_stacks = len(all_stacked_data)
    
    print(f"Found {n_stacks} stacked groups")
    
    for i_stack in range(n_stacks):
        print(f"  Group {i_stack+1}/{n_stacks}")
        
        stacked_data = all_stacked_data[i_stack]
        
        wl = stacked_data['wl']
        stacked_modulations_real = stacked_data['modulations_real']
        stacked_modulations_imag = stacked_data['modulations_imag']
        stacked_modulations_real_err = stacked_data['modulations_real_err']
        stacked_modulations_imag_err = stacked_data['modulations_imag_err']
        stacked_model_real = stacked_data['model_real']
        stacked_model_imag = stacked_data['model_imag']
        stacked_stellar_real = stacked_data['stellar_real']
        stacked_masks_real = stacked_data['masks_real']
        stacked_masks_imag = stacked_data['masks_imag']
        alphas_mean = stacked_data['alphas_mean']
        start_idx = stacked_data['start_idx']
        end_idx = stacked_data['end_idx']
        n_frames_in_stack = stacked_data['n_frames']
        
        # Plot: Stacked modulations real
        fig_mod_stacked_real, ax_mod_stacked_real = plt.subplots(6, 1, figsize=(10, 18), sharex=True)

        fac_mod = 10#np.sqrt(20)
        print(fac_mod)
        
        for i_base in range(6):
            mask = stacked_masks_real[i_base]
            wl_masked = wl[mask]
            Cps_model_masked = Cps_model[mask]
            
            modulations = stacked_modulations_real[i_base, mask]
            modulations_err = stacked_modulations_real_err[i_base, mask]
            model_modulations = stacked_model_real[i_base, mask] - stacked_stellar_real[i_base, mask]
            
            # Plot with errorbars
            ax_mod_stacked_real[i_base].errorbar(
                wl_masked*1e6, modulations, yerr=modulations_err,
                fmt='+', color='blue', alpha=0.7, label='Data ± σ')
            # ax_mod_stacked_real[i_base].plot(wl_masked*1e6, modulations,
            #                                   label='Data - Stellar', color='blue', linewidth=2)
            ax_mod_stacked_real[i_base].plot(wl_masked*1e6, fac_mod*model_modulations,
                                              label='Model - Stellar', color='red', linewidth=2, alpha=0.8)
            
            # Theoretical contrast envelope
            ax_mod_stacked_real[i_base].plot(wl_masked*1e6, fac_mod * alphas_mean[i_base] * Cps_model_masked,
                                              label=f'α={alphas_mean[i_base]:.3f} × Cps_model', 
                                              color='black', linestyle='--', linewidth=1.5, alpha=0.6)
            ax_mod_stacked_real[i_base].plot(wl_masked*1e6, -fac_mod * alphas_mean[i_base] * Cps_model_masked,
                                              color='black', linestyle='--', linewidth=1.5, alpha=0.6)
            
            ax_mod_stacked_real[i_base].set_ylabel(f'{base_order_name[i_base]}\nModulations Real')
            ax_mod_stacked_real[i_base].axhline(0, color='gray', linestyle=':', linewidth=0.5)
            ax_mod_stacked_real[i_base].legend(loc='upper right', fontsize=7)
            ax_mod_stacked_real[i_base].grid(alpha=0.3)
            ax_mod_stacked_real[i_base].set_ylim(-5e-3, 5e-3)

        ax_mod_stacked_real[-1].set_xlabel('Wavelength [µm]')
        ax_mod_stacked_real[-1].set_xlim(3.0, 4.15)
        fig_mod_stacked_real.suptitle(
            f'Stacked Modulations Real - Group {i_stack} (frames {start_idx}-{end_idx-1}, n={n_frames_in_stack})\n'
            f'Best pos: ({x_best:.1f}, {y_best:.1f}) mas', 
            fontsize=14
        )
        fig_mod_stacked_real.tight_layout()
        fig_mod_stacked_real.savefig(
            path_output + f'/figures_replot/stacked_modulations_real_group{i_stack:03d}.png', 
            dpi=300, bbox_inches='tight'
        )
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
            
            # Plot with errorbars
            ax_mod_stacked_imag[i_base].errorbar(
                wl_masked*1e6, modulations, yerr=modulations_err,
                fmt='+', color='green', alpha=0.7, label='Data ± σ')
            # ax_mod_stacked_imag[i_base].plot(wl_masked*1e6, modulations,
            #                                   label='Data', color='green', linewidth=2)
            ax_mod_stacked_imag[i_base].plot(wl_masked*1e6, fac_mod*model_modulations,
                                              label='Model', color='orange', linewidth=2, alpha=0.8)
            
            # Theoretical contrast envelope
            ax_mod_stacked_imag[i_base].plot(wl_masked*1e6, fac_mod * alphas_mean[i_base] * Cps_model_masked,
                                              label=f'α={alphas_mean[i_base]:.3f} × Cps_model', 
                                              color='black', linestyle='--', linewidth=1.5, alpha=0.6)
            ax_mod_stacked_imag[i_base].plot(wl_masked*1e6, -fac_mod * alphas_mean[i_base] * Cps_model_masked,
                                              color='black', linestyle='--', linewidth=1.5, alpha=0.6)
            
            ax_mod_stacked_imag[i_base].set_ylabel(f'{base_order_name[i_base]}\nModulations Imag')
            ax_mod_stacked_imag[i_base].axhline(0, color='gray', linestyle=':', linewidth=0.5)
            ax_mod_stacked_imag[i_base].legend(loc='upper right', fontsize=7)
            ax_mod_stacked_imag[i_base].grid(alpha=0.3)
            ax_mod_stacked_imag[i_base].set_ylim(-5e-3, 5e-3)
        
        ax_mod_stacked_imag[-1].set_xlabel('Wavelength [µm]')
        ax_mod_stacked_imag[-1].set_xlim(3.0, 4.15)
        fig_mod_stacked_imag.suptitle(
            f'Stacked Modulations Imag - Group {i_stack} (frames {start_idx}-{end_idx-1}, n={n_frames_in_stack})\n'
            f'Best pos: ({x_best:.1f}, {y_best:.1f}) mas', 
            fontsize=14
        )
        fig_mod_stacked_imag.tight_layout()
        fig_mod_stacked_imag.savefig(
            path_output + f'/figures_replot/stacked_modulations_imag_group{i_stack:03d}.png', 
            dpi=300, bbox_inches='tight'
        )
        plt.close(fig_mod_stacked_imag)
    
    print("✓ All stacked modulations plots saved")
    
except FileNotFoundError:
    print("⚠ Stacked data not found - skipping stacked modulations plots")

print()
print("="*70)
print("ALL FIGURES REGENERATED SUCCESSFULLY!")
print("="*70)
print(f"Figures saved in: {path_output}/figures_replot/")
print()