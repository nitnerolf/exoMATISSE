# Importation
import numpy as np
from astropy.io import fits
import os 
import matplotlib.pyplot as plt
import astropy.constants as cst
import scipy

####################################################################################################################################################################################
### INPUTS SET UP 

# Output path 
path_output = '/Users/jscigliuto/exoMATISSE/exoMATISSE/data_hd72946b/' #HD 72946 B

# Oifits path
path_oifits = '/Users/jscigliuto/Desktop/Licallo_backup/Pipeline/hd72946b/corrPhaseMathis_MACAO/corrected_data_bin/'

# Science case
sci_case = 'faint'   

# Planet offsets coords mentioned in the OB (mas)
Offset_RA = 106
Offset_Dec = -145

# Grid of coordinates to determine the astrometry of the planet
x = np.arange(Offset_RA-2, Offset_RA+2, 1) 
y = np.arange(Offset_Dec-2, Offset_Dec+2, 1)
xp, yp = np.meshgrid(x, y)

# Degree of the polynomial to model the stellar speckle
n_poly = 1 

# Wavelength range to fit
wmin = 3.0e-6
wmax = 4.15e-6

# Number of cores to use for parallelization (chi2 maps)
n_cores = 10

# Path to Cps and star model 
Cps_model_path = '/Users/jscigliuto/Nextcloud/Py/MATISSE-DataProcessing/Spectrum/computed_spectra/contrast_template_bt-settl_hd72946_ph1ld.fits'
star_model_path = '/Users/jscigliuto/Nextcloud/Py/MATISSE-DataProcessing/Spectrum/computed_spectra/BT-NextGen_T5638K_lg4.51_M0.0_R9.09_res300.800.txt'
planet_model_path = '/Users/jscigliuto/Nextcloud/Py/MATISSE-DataProcessing/Spectrum/computed_spectra/planet_spectrum_template_bt-settl_hd72946_ph1ld.fits'

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


####################################################################################################################################################################################
### SCRIPT

# Create output directories if they do not exist
if not os.path.isdir(path_output + '/spectrometry'):
    os.makedirs(path_output + '/spectrometry')

files_Cps = sorted([file for file in os.listdir(path_output + Cps_path) if '.fits' in file])

print("="*70)
print("LOADING DATA FROM ALL FILES")
print("="*70)
print(f"Found {len(files_Cps)} Cps files to process")
print()

# Pre-reading to get the wavelengths
hdu = fits.open(path_output + Cps_path + files_Cps[0])
wl = hdu['WAVELENGTH'].data
n_wave = wl.size
hdu.close()

# Initialize lists for concatenation
Cps_real_cal_list = []
Cps_real_cal_err_list = []
Cps_imag_cal_list = []
Cps_imag_cal_err_list = []
stellar_coeffs_list = []
alphas_list = []
U_list = []
V_list = []

# Load all files
for i_file, file_Cps in enumerate(files_Cps):
    print(f'Processing {i_file+1}/{len(files_Cps)}: {file_Cps}')

    # Load Cps data per file
    hdu = fits.open(path_output + Cps_path + file_Cps)
    wl = hdu['WAVELENGTH'].data
    Cps_real_cal = hdu['CPS_REAL'].data
    Cps_real_cal_err = hdu['CPS_REAL_ERR'].data
    Cps_imag_cal = hdu['CPS_IMAG'].data
    Cps_imag_cal_err = hdu['CPS_IMAG_ERR'].data
    U = hdu['U'].data
    V = hdu['V'].data
    hdu.close()

    # Load fitted stellar coefficients (real part only)
    fitted_stellar_coeffs = np.load(path_output + f'/fitted_params/stellar_coeffs_{file_Cps}.npy')
    fitted_alphas = np.load(path_output + f'/fitted_params/alphas_{file_Cps}.npy')
    
    print(f"  Data shape: {Cps_real_cal.shape}, Coeffs shape: {fitted_stellar_coeffs.shape}, Alphas shape: {fitted_alphas.shape}")
    
    # Add to lists
    Cps_real_cal_list.append(Cps_real_cal)
    Cps_real_cal_err_list.append(Cps_real_cal_err)
    Cps_imag_cal_list.append(Cps_imag_cal)
    Cps_imag_cal_err_list.append(Cps_imag_cal_err)
    stellar_coeffs_list.append(fitted_stellar_coeffs)
    alphas_list.append(fitted_alphas)
    U_list.append(U)
    V_list.append(V)

print("Concatenating all data...")

# Concatenate all at once
Cps_real_cal_all = np.vstack(Cps_real_cal_list)
Cps_real_cal_err_all = np.vstack(Cps_real_cal_err_list)
Cps_imag_cal_all = np.vstack(Cps_imag_cal_list)
Cps_imag_cal_err_all = np.vstack(Cps_imag_cal_err_list)
stellar_coeffs_all = np.vstack(stellar_coeffs_list)
alphas_all = np.concatenate(alphas_list)
U_all = np.concatenate(U_list)
V_all = np.concatenate(V_list)

# Verify final shapes
print("="*70)
print("FINAL SHAPES AFTER CONCATENATION:")
print("="*70)
print(f"Number of files processed:      {len(files_Cps)}")
print(f"Expected baselines (6 per file): {len(files_Cps) * 6}")
print()
print(f"Cps_real_cal_all.shape:         {Cps_real_cal_all.shape}")
print(f"Cps_real_cal_err_all.shape:     {Cps_real_cal_err_all.shape}")
print(f"Cps_imag_cal_all.shape:         {Cps_imag_cal_all.shape}")
print(f"Cps_imag_cal_err_all.shape:     {Cps_imag_cal_err_all.shape}")
print(f"stellar_coeffs_all.shape:       {stellar_coeffs_all.shape}")
print(f"alphas_all.shape:               {alphas_all.shape}")
print(f"U_all.shape:                    {U_all.shape}")
print(f"V_all.shape:                    {V_all.shape}")

# Verify consistency
# expected_n_baselines = len(files_Cps) * 6
# assert Cps_real_cal_all.shape[0] == stellar_coeffs_all.shape[0], \
#     f"Mismatch: {stellar_coeffs_all.shape[0]} coeffs vs {Cps_real_cal_all.shape[0]} data points"

# assert Cps_real_cal_all.shape[0] == expected_n_baselines, \
#     f"Expected {expected_n_baselines} baselines but got {Cps_real_cal_all.shape[0]}"

# print()
# print("✓ All shapes are consistent!")
# print("="*70)
# print()


# Complexify Cps data
Cps_all = Cps_real_cal_all + 1j * Cps_imag_cal_all
Cps_amp_all = np.abs(Cps_all)
Cps_phi_all = np.angle(Cps_all)

# Optional plotting
plot = False
if plot:
    plt.figure(figsize=(15, 5))
    plt.plot(wl*1e6, Cps_amp_all[4], label='Cps Amp - Baseline 4')
    plt.xlabel(r'Wavelength [$\mu$m]')
    plt.ylabel('Cps Amplitude')
    plt.ylim(0, 3e-3)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(15, 5))
    plt.plot(wl*1e6, Cps_real_cal_all[0], label='Cps Real - Baseline 0')
    plt.xlabel(r'Wavelength [$\mu$m]')
    plt.ylabel('Cps Real part')
    plt.ylim(-1e-3, 2e-3)
    plt.legend()
    plt.tight_layout()

    plt.show()


##############################################################################
# Build the speckle flux and the modulation term
##############################################################################

print("="*70)
print("COMPUTING STELLAR CONTAMINATION MODEL")
print("="*70)

# Load best position
best_pos_dict = np.load(path_output + '/fitted_params/best_position.npy', allow_pickle=True).item()
# x_best = best_pos_dict['x_best']
# y_best = best_pos_dict['y_best']
x_best = 106
y_best = - 145
sep_best = best_pos_dict['sep_best']
PA_best_rad = np.arctan2(x_best, y_best)



print(f"Best position: ({x_best:.2f}, {y_best:.2f}) mas")
print(f"Separation: {sep_best:.2f} mas")
print(f"PA: {np.rad2deg(PA_best_rad):.2f} deg")

# Load normalization parameters
normalization_params = np.load(path_output + '/fitted_params/normalization_params.npy', allow_pickle=True).item()
wl_mean = normalization_params['wl_mean']
wl_std = normalization_params['wl_std']

# Normalize the wavelength grid using THE SAME normalization as during fitting
wl_norm = (wl - wl_mean) / wl_std

# Compute the spatial frequencies projected on best position
Bproj = np.sqrt(U_all**2 + V_all**2) * mas2rad(sep_best) * np.cos(np.arctan2(U_all, V_all) - PA_best_rad) 
spat_freq = np.outer(Bproj, 1/wl)
modulation_part = (np.cos(2*np.pi * spat_freq) + 1j * np.sin(2*np.pi * spat_freq))
# print(f"spat_freq.shape: {spat_freq.shape}")

# Build the stellar contamination arrays
n_baselines_total = stellar_coeffs_all.shape[0]
n_wavelengths = wl.size
print(f"Building stellar contamination for {n_baselines_total} baselines × {n_wavelengths} wavelengths")

# Initialize with correct shapes
stellar_part = np.zeros((n_baselines_total, n_wavelengths), dtype=complex)
stellar_poly_real = np.zeros((n_baselines_total, n_wavelengths))

# Compute stellar polynomials for each baseline
for i_base in range(n_baselines_total):

    ## Stellar part
    # Extract polynomial coefficients for this baseline (real only)
    coeffs_real = stellar_coeffs_all[i_base, :]
    
    # Evaluate polynomial on normalized wavelengths
    stellar_poly_real[i_base, :] = np.polyval(coeffs_real, wl_norm)

    # Stellar contamination is real only 
    stellar_part[i_base, :] = stellar_poly_real[i_base, :] 
    
    # Progress indicator
    # if (i_base + 1) % 100 == 0 or (i_base + 1) == n_baselines_total:
    #     print(f"  Processed {i_base + 1}/{n_baselines_total} baselines...")

print("✓ Stellar contamination model complete")

# Diagnostic: Check magnitude of stellar contamination vs data
# stellar_real_mean = np.mean(np.real(stellar_part))
# stellar_real_std = np.std(np.real(stellar_part))
# Cps_real_mean = np.mean(Cps_real_cal_all)
# Cps_real_std = np.std(Cps_real_cal_all)

# ratio_real = stellar_real_std / np.abs(Cps_real_mean)

# print()
# print("DIAGNOSTIC - Stellar Contamination Magnitude:")
# print(f"  Stellar Real mean: {stellar_real_mean:.6e}")
# print(f"  Stellar Real std:  {stellar_real_std:.6e}")
# print(f"  Cps Real mean:     {Cps_real_mean:.6e}")
# print(f"  Cps Real std:      {Cps_real_std:.6e}")

# Verify shapes before subtraction
# print("VERIFICATION BEFORE SUBTRACTION:")
# print(f"Cps_all.shape:      {Cps_all.shape}")
# print(f"stellar_part.shape: {stellar_part.shape}")

# assert Cps_all.shape == stellar_part.shape, \
#     f"Shape mismatch! Cps_all: {Cps_all.shape}, stellar_part: {stellar_part.shape}"

# print("✓ Shapes match!")

##############################################################################
# Remove stellar contamination to extract planet signal
##############################################################################

print("="*70)
print("EXTRACTING PLANET SIGNAL")
print("="*70)

# Subtract stellar contamination and remove modulations - C*mod + R
Cps_mod_all = Cps_all - stellar_part # - C*mod
Cps_only_all = Cps_mod_all / modulation_part # - C

print(f"✓ Subtraction and division successful")

# Extract real part and propagate errors
Cps_only_all_real = np.real(Cps_only_all)
Cps_only_all_imag = np.imag(Cps_only_all)
Cps_only_all_real_err = Cps_real_cal_err_all
Cps_only_all_imag_err = Cps_imag_cal_err_all

# Compute contrast spectrum by averaging over all baselines
# C_real = np.zeros_like(wl)
# C_real_err = np.zeros_like(wl)
# C_imag = np.zeros_like(wl)
# C_imag_err = np.zeros_like(wl)
C = np.zeros_like(wl)
C_err = np.zeros_like(wl)

C_all_points = np.zeros_like(wl)  
C_all_points_err = np.zeros_like(wl)
n_points_rejected = np.zeros_like(wl)

print("Computing contrast spectrum...")

# for iw in range(wl.size):

#     # Filter outliers at this wavelength (real)
#     low_limit = np.quantile(Cps_only_all_real[:, iw], 0.05)
#     high_limit= np.quantile(Cps_only_all_real[:, iw], 0.95)
#     Cps_valid = (Cps_only_all_real[:, iw] > low_limit) & (Cps_only_all_real[:, iw] < high_limit)

#     values_valid = Cps_only_all_real[Cps_valid, iw]
#     errors_valid = Cps_only_all_real_err[Cps_valid, iw]
#     n_valid = Cps_valid.sum()

#     # Weighted mean
#     weights = 1.0 / (errors_valid**2)
#     C[iw] = np.sum(weights * values_valid) / np.sum(weights)
#     C_err[iw] = 1.0 / np.sqrt(np.sum(weights))

    # # Filter outliers at this wavelength (real)
    # low_limit_real = np.quantile(Cps_only_all_real[:, iw], 0.05)
    # high_limit_real = np.quantile(Cps_only_all_real[:, iw], 0.95)
    # Cps_valid_real = (Cps_only_all_real[:, iw] > low_limit_real) & (Cps_only_all_real[:, iw] < high_limit_real)
    
    # values_valid_real = Cps_only_all_real[Cps_valid_real, iw]
    # errors_valid_real = Cps_only_all_real_err[Cps_valid_real, iw]

    # # Weighted mean (real)
    # weights_real = 1.0 / (errors_valid_real**2)
    # C_real[iw] = np.sum(weights_real * values_valid_real) / np.sum(weights_real)
    # C_real_err[iw] = 1.0 / np.sqrt(np.sum(weights_real))

    # # Filter outliers at this wavelength (imag)
    # low_limit_imag = np.quantile(Cps_only_all_imag[:, iw], 0.05)
    # high_limit_imag = np.quantile(Cps_only_all_imag[:, iw], 0.95)
    # Cps_valid_imag = (Cps_only_all_imag[:, iw] > low_limit_imag) & (Cps_only_all_imag[:, iw] < high_limit_imag)
    
    # values_valid_imag = Cps_only_all_imag[Cps_valid_imag, iw]
    # errors_valid_imag = Cps_only_all_imag_err[Cps_valid_imag, iw]

    # # Weighted mean (imag)
    # weights_imag = 1.0 / (errors_valid_imag**2)
    # C_imag[iw] = np.sum(weights_imag * values_valid_imag) / np.sum(weights_imag)
    # C_imag_err[iw] = 1.0 / np.sqrt(np.sum(weights_imag))

for iw in range(wl.size):

    # Before the filter  
    weights_all = 1.0 / (Cps_only_all_real_err[:, iw]**2)
    C_all_points[iw] = np.sum(weights_all * Cps_only_all_real[:, iw]) / np.sum(weights_all)
    C_all_points_err[iw] = 1.0 / np.sqrt(np.sum(weights_all))

    # Filter outliers at this wavelength
    low_limit = np.quantile(Cps_only_all_real[:, iw], 0.05)
    high_limit= np.quantile(Cps_only_all_real[:, iw], 0.95)
    Cps_valid = (Cps_only_all_real[:, iw] > low_limit) & (Cps_only_all_real[:, iw] < high_limit)

    # Count rejected points
    n_points_rejected[iw] = (~Cps_valid).sum()

    values_valid = Cps_only_all_real[Cps_valid, iw]
    errors_valid = Cps_only_all_real_err[Cps_valid, iw]
    n_valid = Cps_valid.sum()

    # Weighted mean - after the filter 
    weights = 1.0 / (errors_valid**2)
    C[iw] = np.sum(weights * values_valid) / np.sum(weights)
    C_err[iw] = 1.0 / np.sqrt(np.sum(weights))

# Compute amplitude and phase from complex contrast
# C = C_real + 1j * C_imag
# C_err = np.sqrt(C_imag_err**2 + C_real_err**2)
C_real = np.real(C)
C_real_err = np.real(C_err)
C_imag = np.imag(C)
C_imag_err = np.imag(C_err)
C_amp = np.abs(C)
C_phi = np.angle(C)

# Error propagation for amplitude
# C_amp_err = np.sqrt((C_real * C_real_err)**2 + (C_imag * C_imag_err)**2) / C_amp

print(f"✓ Contrast spectrum computed")
# print(f"  Median contrast (real): {np.median(C_real[(wl > wmin) & (wl < wmax)]):.6e}")
# print(f"  Median contrast (imag): {np.median(C_imag[(wl > wmin) & (wl < wmax)]):.6e}")
# print(f"  Median contrast (amp):  {np.median(C_amp[(wl > wmin) & (wl < wmax)]):.6e}")
# print(f"  Median error (amp):     {np.median(C_amp_err[(wl > wmin) & (wl < wmax)]):.6e}")
print("="*70)



##############################################################################
# Convert contrast to planet flux spectrum
##############################################################################

print("="*70)
print("CONVERTING TO PLANET FLUX SPECTRUM")
print("="*70)

# Load star model spectrum
star_model_spectrum = np.loadtxt(star_model_path)
wl_star, spec_star = star_model_spectrum[:, 0], star_model_spectrum[:, 1]  # µm, W/m²/µm
wl_star = wl_star * 1e-6  # Convert to meters

# Interpolate stellar spectrum to data wavelengths
f_interp = scipy.interpolate.interp1d(wl_star, spec_star, kind='linear', bounds_error=False, fill_value="extrapolate")
spec_star_interp = f_interp(wl)

## Compute planet spectrum
# Compute planet spectrum after the filtering
spec_planet = C_real * spec_star_interp 
spec_planet_err = C_real_err * spec_star_interp
# Compute planet spectrum before the filtering
spec_planet_all_points = C_all_points * spec_star_interp
spec_planet_all_points_err = C_all_points_err * spec_star_interp

print(f"✓ Planet spectrum computed")
L_wl = 3.5e-6 
L_idx = np.argmin(np.abs(wl - L_wl))
print(f"Flux at 3.5 µm: {np.median(spec_planet[L_idx]):.6e} W/m²/µm")

# Save planetary spectrum to fits file
out_path = os.path.join(path_output, 'spectrometry', 'planet_spectrum.fits')
fits.writeto(out_path, np.array([wl, spec_planet, spec_planet_err]), overwrite=True)

print(f"✓ Saved to: {out_path}")
print("="*70)


##############################################################################
# FIGURES
##############################################################################

print("="*70)
print("GENERATING FIGURES")
print("="*70)

# Contrast spectrum 
plt.figure(figsize=(15, 5))
plt.errorbar(wl*1e6, C_real, yerr=C_real_err, fmt='o', label='Contrast Spectrum', markersize=4)
plt.xlabel(r'Wavelength [$\mu$m]', fontsize=12)
plt.ylabel('Contrast', fontsize=12)
plt.legend(loc='upper left')
plt.ylim(-6e-3, 4e-3)
plt.xlim(2.76, 5.0)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(path_output + '/spectrometry/Contrast_Spectrum.png', dpi=300)
plt.close()
print("✓ Saved: Contrast_Spectrum.png")

# Contrast spectrum (real and imaginary parts)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

ax1.errorbar(wl*1e6, C_real, yerr=C_real_err, fmt='o', label='Contrast Real', markersize=4, color='blue')
ax1.set_ylabel('Contrast Real', fontsize=12)
ax1.legend(loc='upper left')
ax1.grid(alpha=0.3)
ax1.set_ylim(-4e-3, 4e-3)
ax1.axhline(0, color='k', linestyle='--', alpha=0.3)

ax2.errorbar(wl*1e6, C_imag, yerr=C_imag_err, fmt='o', label='Contrast Imag', markersize=4, color='green')
ax2.set_ylabel('Contrast Imag', fontsize=12)
ax2.set_xlabel(r'Wavelength [$\mu$m]', fontsize=12)
ax2.legend(loc='upper left')
ax2.grid(alpha=0.3)
ax2.set_ylim(-1e-3, 2e-3)
ax2.axhline(0, color='k', linestyle='--', alpha=0.3)

ax1.set_xlim(2.76, 5.0)
fig.tight_layout()
fig.savefig(path_output + '/spectrometry/Contrast_Spectrum_RealImag.png', dpi=300)
plt.close(fig)
print("✓ Saved: Contrast_Spectrum_RealImag.png")

# SNR
snr = C / C_err
wl_mask_snr = (wl >= 3.0e-6) & (wl <= 4.0e-6)
snr_median = np.median(snr[wl_mask_snr])
plt.figure(figsize=(8, 6))
plt.plot(wl*1e6, snr, 'o-', label='SNR', markersize=4)
plt.xlabel(r'Wavelength [$\mu$m]', fontsize=12)
plt.ylabel('SNR', fontsize=12)
plt.axhline(y=snr_median, color='r', linestyle='--', label=f'Median SNR (3-4 µm) = {snr_median:.2f}')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.xlim(2.76, 5.0)
plt.ylim(-1, 4)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(path_output + '/spectrometry/SNR.png', dpi=300)
plt.close()
print(f"✓ Saved: SNR.png (Median SNR 3-4 µm: {snr_median:.2f})")

# Contrast error
plt.figure(figsize=(15, 5))
plt.plot(wl*1e6, C_real_err, 'o-', markersize=4)
median_err = np.median(C_real_err[(wl > wmin) & (wl < wmax)])
plt.axhline(y=median_err, color='r', linestyle='--', label=f'Median Error = {median_err:.2e}')
plt.xlabel(r'Wavelength [$\mu$m]', fontsize=12)
plt.ylabel('Contrast Error', fontsize=12)
plt.xlim(2.76, 5.0)
plt.ylim(0, 1e-3)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(path_output + '/spectrometry/Contrast_Error.png', dpi=300)
plt.close()
print("✓ Saved: Contrast_Error.png")

# Planet spectrum
plt.figure(figsize=(15, 5))
plt.errorbar(wl*1e6, spec_planet, yerr=spec_planet_err, fmt='+', label='Planet Spectrum', markersize=4)
plt.xlabel(r'Wavelength [$\mu$m]', fontsize=12)
plt.ylabel(r'Flux [W/m$^2$/µm]', fontsize=12)
plt.legend(loc='upper left')
plt.ylim(-2e-14, 2e-14)
plt.xlim(2.76, 5.0)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(path_output + '/spectrometry/Planet_Spectrum.png', dpi=300)
plt.close()
print("✓ Saved: Planet_Spectrum.png")

# Planet spectrum (filled)
plt.figure(figsize=(15, 5))
plt.fill_between(wl*1e6, spec_planet-spec_planet_err, spec_planet+spec_planet_err, 
                 alpha=0.3, color='plum', label='1σ uncertainty')
plt.plot(wl*1e6, spec_planet, marker='+', color='purple', markersize=6, label='Planet Spectrum')
plt.xlabel(r'Wavelength [$\mu$m]', fontsize=12)
plt.ylabel(r'Flux [W/m$^2$/µm]', fontsize=12)
plt.legend(loc='upper left')
plt.ylim(-1e-14, 1e-14)
# plt.axhline(y=0)
# plt.axhline(y=0.275e-15, label='expected flux level at 3.5µm', color='green')
plt.xlim(2.76, 5.0)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(path_output + '/spectrometry/Planet_Spectrum_filled.png', dpi=300)
plt.close()
print("✓ Saved: Planet_Spectrum_filled.png")

# Diagnostic plot: Stellar subtraction quality
fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

# Plot 1: Raw data (real part)
axes[0].plot(wl*1e6, Cps_real_cal_all[0], label='Raw Cps Real (baseline 0)', alpha=0.7)
axes[0].plot(wl*1e6, stellar_poly_real[0], label='Stellar contamination', linestyle='--', color='red')
axes[0].set_ylabel('Flux', fontsize=12)
axes[0].set_ylim(-2e-2, 5e-2)
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_title('Stellar Subtraction Diagnostic')

# Plot 2: After subtraction
axes[1].plot(wl*1e6, Cps_only_all_real[0], label='After stellar subtraction (baseline 0)', color='blue', alpha=0.7)
axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[1].set_ylabel('Flux (after subtraction)', fontsize=12)
axes[1].set_ylim(-2e-2, 3e-2)
axes[1].legend()
axes[1].grid(alpha=0.3)

# Plot 3: Ratio |Stellar| / |Data|
ratio_diagnostic = np.abs(stellar_poly_real) / (np.abs(Cps_only_all_real) + 1e-10) 
axes[2].plot(wl*1e6, ratio_diagnostic[0], label='|Stellar| / |Data| (baseline 0)', color='purple', alpha=0.7)
axes[2].axhline(1.0, color='r', linestyle='--', alpha=0.5, label='Ratio = 1')
axes[2].set_ylabel('Ratio', fontsize=12)
axes[2].set_xlabel(r'Wavelength [$\mu$m]', fontsize=12)
axes[2].legend()
axes[2].grid(alpha=0.3)
axes[2].set_ylim(0, 20)

axes[2].set_xlim(2.76, 5.0)
fig.tight_layout()
fig.savefig(path_output + '/spectrometry/Diagnostic_Stellar_Subtraction.png', dpi=300)
plt.close(fig)
print("✓ Saved: Diagnostic_Stellar_Subtraction.png")

print()
print("="*70)
print("EXTRACTION COMPLETE!")
print("="*70)