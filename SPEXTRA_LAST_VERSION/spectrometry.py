# Importation
import numpy as np
from astropy.io import fits
import os 
import matplotlib.pyplot as plt
import astropy.constants as cst
import spectres
import scipy
from common_tools import wrap, mas2rad

####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################
### INPUTS SET UP 

# Output path 
# path_output = '/Users/jscigliuto/exoMATISSE/exoMATISSE/data_hd72946b/test/' #HD 72946 B
path_output = '/Users/jscigliuto/exoMATISSE/exoMATISSE/data_betaPicb/' #beta Pic b

path_oifits = '/Users/jscigliuto/Desktop/Licallo_backup/Pipeline/betaPicb/corrPhaseMathis_MACAO/corrected_data_bin/' #beta Pic b

# Science case
sci_case = 'bright' #'faint'   

# Planet offsets coords mentioned in the OB [mas]
Offset_RA = 279
Offset_Dec = 455

# Grid of coordinates to determine the astrometry of the planet
# x      = np.arange(Offset_RA-10, Offset_RA+10, 0.4) 
# y      = np.arange(Offset_Dec-10, Offset_Dec+10, 0.4)  
x      = np.arange(Offset_RA-5, Offset_RA+5, 1.0) 
y      = np.arange(Offset_Dec-5, Offset_Dec+5, 1.0) 
xp, yp = np.meshgrid(x, y)  # grid of coords to look for the planet position

# Degree of the polynomial to model the stellar speckle
n_poly = 1 

# Wavelength range to fit
wmin = 3.0e-6
wmax = 4.15e-6


# Number of cores to use for parallelization (chi2 maps)
n_cores = 10

# Path to Cps and star model 
# Cps_model_path = '/Users/jscigliuto/Nextcloud/Py/MATISSE-DataProcessing/Spectrum/computed_spectra/contrast_template_bt-settl_hd72946_ph1ld.fits' #HD 72946 B

Cps_model_path = '/Users/jscigliuto/Nextcloud/Py/MATISSE-DataProcessing/Spectrum/computed_spectra/contrast_template_bt-settl_betPic.fits' #beta Pic b
star_model_path = '/Users/jscigliuto/Nextcloud/Py/MATISSE-DataProcessing/Spectrum/computed_spectra/BT-NextGen_T7890K_lg3.8_M0.0_R15.5_res300.800.txt' #beta Pic
planet_model_path = '/Users/jscigliuto/Nextcloud/Py/MATISSE-DataProcessing/Spectrum/computed_spectra/planet_spectrum_template_bt-settl_betPicb.fits' #beta Pic b

# Flag for binned data
use_bin_data = True

### INITIALIZATION 
# Baseline order and names
base_order_name = ('U3-U4', 'U1-U2', 'U2-U3', 'U2-U4', 'U1-U3', 'U1-U4')
# Cps files path 
Cps_path = 'Contrast_fits/'

####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################
### FUNCTIONS

####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################
### SCRIPT

# Create output directories if they do not exist
if not os.path.isdir(path_output + '/spectrometry'):
    os.makedirs(path_output + '/spectrometry')

files_Cps = sorted([file for file in os.listdir(path_output + Cps_path) if '.fits' in file])
files_oifits = sorted([file for file in os.listdir(path_oifits) if '.fits' in file])

# Pre-reading to get the wavelengths
hdul = fits.open(path_oifits + files_oifits[0])
wl = hdul['OI_WAVELENGTH'].data['EFF_WAVE']
n_wave = wl.size
hdul.close()

Cps_real_cal_init     = np.zeros((1, n_wave))
Cps_real_cal_err_init = np.zeros((1, n_wave))
Cps_imag_cal_init     = np.zeros((1, n_wave))
Cps_imag_cal_err_init = np.zeros((1, n_wave))
stellar_coeffs_all = np.zeros((1, 2*(n_poly+1)))
U_init = np.zeros(1)
V_init = np.zeros(1)

for i_file, file_Cps in enumerate(files_Cps):

    print(f'Processing {file_Cps}...')

    # Load Cps data per file
    hdu = fits.open(path_output + Cps_path + file_Cps)
    wl = hdu['WAVELENGTH'].data
    Cps_real_cal = hdu['CPS_REAL'].data
    Cps_real_cal_err = hdu['CPS_REAL_ERR'].data
    Cps_imag_cal = hdu['CPS_IMAG'].data
    Cps_imag_cal_err = hdu['CPS_IMAG_ERR'].data
    U = hdu['U'].data
    V = hdu['V'].data

    fitted_stellar_coeffs = np.load(path_output + f'/fitted_params/stellar_coeffs_{file_Cps}.npy')

    # # SNR
    # snr_real = np.abs(Cps_real_cal / Cps_real_cal_err)
    # snr_imag = np.abs(Cps_imag_cal / Cps_imag_cal_err)

    # # Filter based on data quality (exclude low SNRs)
    # real_err_mask = (snr_real.T > np.quantile(snr_real, 0.05, axis=1)).T 
    # imag_err_mask = (snr_imag.T > np.quantile(snr_imag, 0.05, axis=1)).T

    # # Apply the mask
    # Cps_real_cal     = Cps_real_cal[real_err_mask]
    # Cps_real_cal_err = Cps_real_cal_err[real_err_mask]
    # Cps_imag_cal     = Cps_imag_cal[imag_err_mask]
    # Cps_imag_cal_err = Cps_imag_cal_err[imag_err_mask]

    # Concatenate the all the data
    Cps_real_cal_all     = np.concatenate((Cps_real_cal_init, Cps_real_cal))
    Cps_real_cal_err_all = np.concatenate((Cps_real_cal_err_init, Cps_real_cal_err))
    Cps_imag_cal_all     = np.concatenate((Cps_imag_cal_init, Cps_imag_cal))
    Cps_imag_cal_err_all = np.concatenate((Cps_imag_cal_err_init, Cps_imag_cal_err))
    stellar_coeffs_all   = np.concatenate((stellar_coeffs_all, fitted_stellar_coeffs))
    U_all = np.concatenate((U_init, U))
    V_all = np.concatenate((V_init, V))

# Delete the null first element of the concatenation
Cps_real_cal_all     = np.delete(Cps_real_cal_all, 0, 0)
Cps_real_cal_err_all = np.delete(Cps_real_cal_err_all, 0, 0)
Cps_imag_cal_all     = np.delete(Cps_imag_cal_all, 0, 0)
Cps_imag_cal_err_all = np.delete(Cps_imag_cal_err_all, 0, 0)
stellar_coeffs_all   = np.delete(stellar_coeffs_all, 0, 0)
U_all = np.delete(U_all, 0, 0)
V_all = np.delete(V_all, 0, 0)


## Build the a contrast file with all the frame
# Get the alpha
alpha_best = np.load(path_output + f'/fitted_params/alpha_global.npy')

# Complexify 
Cps_all = Cps_real_cal_all + 1j * Cps_imag_cal_all
Cps_amp_all = np.abs(Cps_all)
Cps_phi_all = np.angle(Cps_all)

plot=False
if plot==True:
    plt.figure(figsize=(15, 5))
    plt.plot(wl*1e6, Cps_amp_all[4], label='Cps Amp - First baseline')
    plt.xlabel(r'Wavelength [$\mu$m]')
    plt.ylabel('Cps Amplitude')
    plt.ylim(0, 3e-3)
    plt.legend()
    plt.tight_layout()

    # plt.figure(figsize=(15, 5))
    # plt.plot(wl*1e6, Cps_phi_all[0], label='Cps Phase - First baseline')
    # plt.xlabel(r'Wavelength [$\mu$m]')
    # plt.ylabel('Cps Phase [rad]')
    # plt.legend()
    # plt.tight_layout()

    plt.figure(figsize=(15, 5))
    plt.plot(wl*1e6, Cps_real_cal_all[0], label='Cps Real - First baseline')
    plt.xlabel(r'Wavelength [$\mu$m]')
    plt.ylabel('Cps Real part')
    plt.ylim(-1e-3, 2e-3)
    plt.ylim
    plt.legend()
    plt.tight_layout()

    # plt.figure(figsize=(15, 5))
    # plt.plot(wl*1e6, Cps_imag_cal_all[0], label='Cps Imag - First baseline')
    # plt.xlabel(r'Wavelength [$\mu$m]')
    # plt.ylabel('Cps Imag part')
    # plt.legend()
    # plt.tight_layout()

    plt.show()


## Build the speckle flux and the modulation term
# Compute PA and separations
PA = np.arctan2(xp, yp)
sep = np.sqrt(xp**2+yp**2)

best_pos_dict = np.load(path_output + '/fitted_params/best_position.npy', allow_pickle=True).item()
x_best = best_pos_dict['x_best']
y_best = best_pos_dict['y_best']
sep_best = best_pos_dict['sep_best']
PA_best_rad = np.arctan2(x_best, y_best)
alpha_optimal = best_pos_dict['alpha_best']

# Compute the spatial frequencies projected on these coordinates
Bproj = np.sqrt(U_all**2 + V_all**2) * mas2rad(sep_best) * np.cos(np.arctan2(U_all, V_all) - PA_best_rad) 
spat_freq = np.outer(Bproj, 1/wl)

# Build the speckle coherent flux with the quantities fitted previously 
stellar_part = np.zeros_like(Cps_real_cal_all, dtype=complex)
stellar_poly_real = np.zeros_like(Cps_real_cal_all, dtype=complex)
stellar_poly_imag = np.zeros_like(Cps_real_cal_all, dtype=complex)

for i_base in range(6):
    # Extract coeffs (real+imag)
    stellar_poly_real[i_base] = np.polyval(stellar_coeffs_all[i_base, :n_poly+1], wl)
    stellar_poly_imag[i_base] = np.polyval(stellar_coeffs_all[i_base, n_poly+1:], wl)

    # Complexify
    stellar_part[i_base] = stellar_poly_real[i_base] * np.cos(2 * np.pi * spat_freq[i_base]) + 1j * stellar_poly_imag[i_base] * np.sin(2 * np.pi * spat_freq[i_base])

    # Amp/phase space
    amp_stellar_part = np.abs(stellar_part[i_base])
    phi_stellar_part = np.angle(stellar_part[i_base])

    # Save
    stellar_part[i_base] = amp_stellar_part * np.exp(1j * phi_stellar_part)

## Remove the stellar part to get the contrast only
Cps_only_all = Cps_all - stellar_part

# Cps_only_all_amp = np.abs(Cps_only_all)
# Cps_only_all_amp_err = np.sqrt((Cps_real_cal_all*Cps_real_cal_err_all)**2 + (Cps_imag_cal_all*Cps_imag_cal_err_all)**2) / Cps_amp_all
# Cps_only_all_phi_err = np.sqrt((Cps_imag_cal_all*Cps_real_cal_err_all)**2 + (Cps_real_cal_all*Cps_imag_cal_err_all)**2) / (Cps_amp_all**2)

# Cps_only_all_real_err = np.sqrt((np.cos(Cps_phi_all) * Cps_only_all_amp_err)**2 + (Cps_amp_all * np.sin(Cps_phi_all) * Cps_only_all_phi_err)**2)
# Cps_only_all_real_err = np.sqrt((Cps_real_cal_err_all / Cps_real_cal_all)**2) # Stellar part contribute to the error budget ? 
Cps_only_all_real_err = Cps_real_cal_err_all  

Cps_only_all_real = np.real(Cps_only_all)



C = np.zeros_like(wl)
C_err = np.zeros_like(wl)
# for iw in range(wl.size):
#     low_limit = np.quantile(Cps_only_all[:, iw], 0.05)
#     high_limit = np.quantile(Cps_only_all[:, iw], 0.95)
#     Cps_valid = (Cps_only_all[:, iw] > low_limit) & (Cps_only_all[:, iw] < high_limit)
#     C[iw] = np.median(Cps_only_all[Cps_valid, iw])
#     print(np.std(Cps_only_all[Cps_valid, iw]))
#     C_err[iw] = np.std(Cps_only_all[Cps_valid, iw], ddof=1) / np.sqrt(Cps_valid.sum())
#     # C_err[iw] = np.median(Cps_only_all_real_err[Cps_valid, iw])

# for iw in range(wl.size):
#     median_Cps = np.median(Cps_only_all[:, iw])
#     std_Cps = 3 * 1.4826 * np.median(np.abs(Cps_only_all[:, iw] - median_Cps))
#     Cps_valid = (Cps_only_all[:, iw] > median_Cps - std_Cps) & (Cps_only_all[:, iw] < median_Cps + std_Cps)
#     C[iw] = np.mean(Cps_only_all[Cps_valid, iw])
#     # C_err[iw] = np.std(Cps_only_all[Cps_valid, iw], ddof=1) / np.sqrt(Cps_valid.sum())
#     C_err[iw] = np.mean(Cps_only_all_real_err[Cps_valid, iw])

for iw in range(wl.size):
    # Filter the outliers
    low_limit = np.quantile(Cps_only_all_real[:, iw], 0.05)
    high_limit = np.quantile(Cps_only_all_real[:, iw], 0.95)
    Cps_valid = (Cps_only_all_real[:, iw] > low_limit) & (Cps_only_all_real[:, iw] < high_limit)
    
    values_valid = Cps_only_all_real[Cps_valid, iw]
    errors_valid = Cps_only_all_real_err[Cps_valid, iw]
    n_valid = Cps_valid.sum()

    # Moyenne simple
    C[iw] = np.mean(values_valid)
    C_err[iw] = np.sqrt(np.sum(errors_valid**2)) / n_valid

    # Moyenne pondérée 
    # weights = 1.0 / (errors_valid**2)
    # C[iw] = np.sum(weights * values_valid) / np.sum(weights)
    # C_err[iw] = 1.0 / np.sqrt(np.sum(weights))

####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################
### FIGURES
star_model_spectrum = np.loadtxt(star_model_path)
# wl_star, spec_star_conv_SI = star_model_spectrum[:, 0], star_model_spectrum[:, 1]
# spec_star_conv_SI = spectres.spectres(wl[::-1], wl_star, spec_star_conv_SI)
# spec_star_conv = spec_star_conv_SI * 1e6
# spec_star_conv = spec_star_conv * (wl[::-1])**2 / cst.c.value
# spec_star_conv *= 1e26 # Jy
# spec_star_conv = spec_star_conv_SI[::-1]
wl_star, spec_star = star_model_spectrum[:, 0], star_model_spectrum[:, 1] #µm, W/m2/µm
wl_star = wl_star * 1e-6 #m
f_interp = scipy.interpolate.interp1d(wl_star, spec_star, kind='linear', bounds_error=False, fill_value="extrapolate")
spec_star_interp = f_interp(wl)

spec_planet = C * spec_star_interp 
spec_planet_err = C_err * spec_star_interp

# Back to SI 
# spec_planet_SI = spec_planet * 1e-26 * (cst.c.value / wl**2) * 1e-6
# spec_planet_err_SI = spec_planet_err * 1e-26 * (cst.c.value / (wl*1e-6)**2) * 1e-6 


## Contrast C
plt.figure(figsize=(15, 5))
plt.errorbar(wl*1e6, C, yerr=C_err, fmt='o', label='Planet Spectrum')
plt.xlabel(r'Wavelength [$\mu$m]')
plt.ylabel('Contrast')
plt.legend(loc='upper left')
plt.ylim(-1e-3, 2e-3)
plt.xlim(2.76, 5.0)
plt.tight_layout()
plt.savefig(path_output + '/spectrometry/Contrast_Spectrum.png')

## SNR
snr = C / C_err
plt.figure(figsize=(15, 5))
plt.plot(wl*1e6, snr, 'o-', label='SNR Spectrum')
plt.xlabel(r'Wavelength [$\mu$m]')
plt.ylabel('SNR')
plt.axhline(y=0, color='r', linestyle='--', label='SNR=3')
plt.xlim(2.76, 5.0)
plt.tight_layout()
plt.savefig(path_output + '/spectrometry/SNR.png')

plt.figure(figsize=(15, 5))
plt.plot(wl*1e6, C_err, 'o-', label='SNR Spectrum')
plt.axhline(y=np.median(Cps_only_all_real_err), color='r', linestyle='--', label=f'Median Err {np.median(C_err[(wl>3e-6) | (wl<4e-6)]):.2e}')
plt.xlabel(r'Wavelength [$\mu$m]')
plt.ylabel('Err')
plt.xlim(2.76, 5.0)
# plt.ylim(0, 2)
plt.ylim(0, 8e-4)
plt.tight_layout()
plt.legend()
plt.savefig(path_output + '/spectrometry/Contrast_Error.png')

## Planetary spectrum
plt.figure(figsize=(15, 5))
plt.errorbar(wl*1e6, spec_planet, yerr=np.sqrt(spec_planet_err**2), fmt='o', label='Planet Spectrum')
plt.xlabel(r'Wavelength [$\mu$m]')
plt.ylabel('Flux')
plt.legend(loc='upper left')
plt.ylim(-2e-15, 9e-15)
plt.xlim(2.76, 5.0)
plt.tight_layout()
plt.savefig(path_output + '/spectrometry/Planet_Spectrum.png')

plt.figure(figsize=(15, 5))
plt.fill_between(wl*1e6, spec_planet-spec_planet_err, spec_planet+spec_planet_err, alpha=0.3, color='plum', label='Planet Spectrum')
plt.plot(wl*1e6, spec_planet, marker='+')
plt.xlabel(r'Wavelength [$\mu$m]')
plt.ylabel(r'Flux [W/m$^2$/µm]')
plt.legend(loc='upper left')
plt.ylim(-2e-15, 9e-15)
plt.xlim(2.76, 5.0)
plt.tight_layout()
plt.savefig(path_output + '/spectrometry/Planet_Spectrum_filled.png')

spec_model_planet = fits.getdata(planet_model_path)
if use_bin_data == True:
    spec_model_planet = spec_model_planet.reshape(-1, 5).mean(axis=1)
wl_mask = (wl >= wmin) & (wl <= wmax) 
alpha = 1.357996097634616

plt.figure(figsize=(15, 5))
plt.plot(wl[wl_mask]*1e6, alpha*spec_model_planet[wl_mask]-spec_planet[wl_mask], marker='+')
plt.xlabel(r'Wavelength [$\mu$m]')
plt.ylabel('Flux Residuals (Model - Measured)')
# plt.ylim(-2e-15, 9e-15)
plt.xlim(3., 4.2)
plt.tight_layout()
plt.savefig(path_output + '/spectrometry/Planet_Spectrum_Residual.png')