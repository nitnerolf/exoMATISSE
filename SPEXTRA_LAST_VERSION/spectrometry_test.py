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
path_output = '/Users/jscigliuto/exoMATISSE/exoMATISSE/data_betaPicb/' #beta Pic b

path_oifits = '/Users/jscigliuto/Desktop/Licallo_backup/Pipeline/betaPicb/corrPhaseMathis_MACAO/corrected_data_bin/' #beta Pic b

# Science case
sci_case = 'bright' #'faint'   

# Planet offsets coords mentioned in the OB [mas]
Offset_RA = 279
Offset_Dec = 455

# Grid of coordinates to determine the astrometry of the planet
x = np.arange(Offset_RA-5, Offset_RA+5, 1.0) 
y = np.arange(Offset_Dec-5, Offset_Dec+5, 1.0) 
xp, yp = np.meshgrid(x, y)

# Degree of the polynomial to model the stellar speckle
n_poly = 1 

# Wavelength range to fit
wmin = 3.0e-6
wmax = 4.15e-6

# Number of cores to use for parallelization (chi2 maps)
n_cores = 10

# Path to Cps and star model 
Cps_model_path = '/Users/jscigliuto/Nextcloud/Py/MATISSE-DataProcessing/Spectrum/computed_spectra/contrast_template_bt-settl_betPic.fits'
star_model_path = '/Users/jscigliuto/Nextcloud/Py/MATISSE-DataProcessing/Spectrum/computed_spectra/BT-NextGen_T7890K_lg3.8_M0.0_R15.5_res300.800.txt'
planet_model_path = '/Users/jscigliuto/Nextcloud/Py/MATISSE-DataProcessing/Spectrum/computed_spectra/planet_spectrum_template_bt-settl_betPicb.fits'

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
files_oifits = sorted([file for file in os.listdir(path_oifits) if '.fits' in file])

print("="*70)
print("LOADING DATA FROM ALL FILES")
print("="*70)
print(f"Found {len(files_Cps)} Cps files to process")
print()

# Pre-reading to get the wavelengths
hdul = fits.open(path_oifits + files_oifits[0])
wl = hdul['OI_WAVELENGTH'].data['EFF_WAVE']
n_wave = wl.size
hdul.close()

# Initialize LISTS for dynamic concatenation
Cps_real_cal_list = []
Cps_real_cal_err_list = []
Cps_imag_cal_list = []
Cps_imag_cal_err_list = []
stellar_coeffs_list = []
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

    # Load fitted stellar coefficients
    fitted_stellar_coeffs = np.load(path_output + f'/fitted_params/stellar_coeffs_{file_Cps}.npy')
    
    print(f"  Data shape: {Cps_real_cal.shape}, Coeffs shape: {fitted_stellar_coeffs.shape}")
    
    # Append to lists
    Cps_real_cal_list.append(Cps_real_cal)
    Cps_real_cal_err_list.append(Cps_real_cal_err)
    Cps_imag_cal_list.append(Cps_imag_cal)
    Cps_imag_cal_err_list.append(Cps_imag_cal_err)
    stellar_coeffs_list.append(fitted_stellar_coeffs)
    U_list.append(U)
    V_list.append(V)

print("Concatenating all data...")

# Concatenate all at once
Cps_real_cal_all = np.vstack(Cps_real_cal_list)
Cps_real_cal_err_all = np.vstack(Cps_real_cal_err_list)
Cps_imag_cal_all = np.vstack(Cps_imag_cal_list)
Cps_imag_cal_err_all = np.vstack(Cps_imag_cal_err_list)
stellar_coeffs_all = np.vstack(stellar_coeffs_list)
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
print(f"U_all.shape:                    {U_all.shape}")
print(f"V_all.shape:                    {V_all.shape}")

# Verify consistency
expected_n_baselines = len(files_Cps) * 6
assert Cps_real_cal_all.shape[0] == stellar_coeffs_all.shape[0], \
    f"Mismatch: {stellar_coeffs_all.shape[0]} coeffs vs {Cps_real_cal_all.shape[0]} data points"

assert Cps_real_cal_all.shape[0] == expected_n_baselines, \
    f"Expected {expected_n_baselines} baselines but got {Cps_real_cal_all.shape[0]}"

print()
print("✓ All shapes are consistent!")
print("="*70)
print()


## Complexify Cps data
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
x_best = best_pos_dict['x_best']
y_best = best_pos_dict['y_best']
sep_best = best_pos_dict['sep_best']
PA_best_rad = np.arctan2(x_best, y_best)

print(f"Best position: ({x_best:.2f}, {y_best:.2f}) mas")
print(f"Separation: {sep_best:.2f} mas")
print(f"PA: {np.rad2deg(PA_best_rad):.2f} deg")
print()

# Load normalization parameters
normalization_params = np.load(path_output + '/fitted_params/normalization_params.npy', allow_pickle=True).item()
wl_mean = normalization_params['wl_mean']
wl_std = normalization_params['wl_std']

# Normalize the wavelength grid
wl_norm = (wl - wl_mean) / wl_std

# Compute the spatial frequencies projected on best position
Bproj = np.sqrt(U_all**2 + V_all**2) * mas2rad(sep_best) * np.cos(np.arctan2(U_all, V_all) - PA_best_rad) 
spat_freq = np.outer(Bproj, 1/wl)

print(f"spat_freq.shape: {spat_freq.shape}")
print()

# Build the stellar contamination arrays
n_baselines_total = stellar_coeffs_all.shape[0]
n_wavelengths = wl.size

print(f"Building stellar contamination for {n_baselines_total} baselines × {n_wavelengths} wavelengths")
print()

# Initialize with correct shapes
stellar_part = np.zeros((n_baselines_total, n_wavelengths), dtype=complex)
stellar_poly_real = np.zeros((n_baselines_total, n_wavelengths))
stellar_poly_imag = np.zeros((n_baselines_total, n_wavelengths))

# Compute stellar polynomials for each baseline
for i_base in range(n_baselines_total):
    # Extract polynomial coefficients for this baseline
    coeffs_real = stellar_coeffs_all[i_base, :n_poly+1]
    coeffs_imag = stellar_coeffs_all[i_base, n_poly+1:]
    
    # Evaluate polynomials
    stellar_poly_real[i_base, :] = np.polyval(coeffs_real, wl_norm)
    stellar_poly_imag[i_base, :] = np.polyval(coeffs_imag, wl_norm)

    # Compute complex stellar contamination with spatial modulation
    stellar_part[i_base, :] = (
        stellar_poly_real[i_base, :] * np.cos(2 * np.pi * spat_freq[i_base, :]) + 
        1j * stellar_poly_imag[i_base, :] * np.sin(2 * np.pi * spat_freq[i_base, :])
    )
    
    # Progress indicator
    if (i_base + 1) % 100 == 0 or (i_base + 1) == n_baselines_total:
        print(f"  Processed {i_base + 1}/{n_baselines_total} baselines...")

print()
print("✓ Stellar contamination model complete")
print("="*70)
print()

# Verify shapes before subtraction
print("VERIFICATION BEFORE SUBTRACTION:")
print(f"Cps_all.shape:      {Cps_all.shape}")
print(f"stellar_part.shape: {stellar_part.shape}")

assert Cps_all.shape == stellar_part.shape, \
    f"Shape mismatch! Cps_all: {Cps_all.shape}, stellar_part: {stellar_part.shape}"

print("✓ Shapes match!")
print()

##############################################################################
# Remove stellar contamination to extract planet signal
##############################################################################

print("="*70)
print("EXTRACTING PLANET SIGNAL")
print("="*70)

# Subtract stellar contamination
Cps_only_all = Cps_all - stellar_part

print(f"✓ Subtraction successful")
print(f"Cps_only_all.shape: {Cps_only_all.shape}")
print()

# Extract real part and propagate errors
Cps_only_all_real = np.real(Cps_only_all)
Cps_only_all_real_err = Cps_real_cal_err_all

# Compute contrast spectrum by averaging over all baselines
C = np.zeros_like(wl)
C_err = np.zeros_like(wl)

print("Computing contrast spectrum...")
print("  Method: Mean with outlier rejection (5% and 95% quantiles)")
print()

for iw in range(wl.size):
    # Filter outliers at this wavelength
    low_limit = np.quantile(Cps_only_all_real[:, iw], 0.05)
    high_limit = np.quantile(Cps_only_all_real[:, iw], 0.95)
    Cps_valid = (Cps_only_all_real[:, iw] > low_limit) & (Cps_only_all_real[:, iw] < high_limit)
    
    values_valid = Cps_only_all_real[Cps_valid, iw]
    errors_valid = Cps_only_all_real_err[Cps_valid, iw]
    n_valid = Cps_valid.sum()

    # Simple mean
    # C[iw] = np.mean(values_valid)
    # C_err[iw] = np.sqrt(np.sum(errors_valid**2)) / n_valid

    # Weighted mean
    weights = 1.0 / (errors_valid**2)
    C[iw] = np.sum(weights * values_valid) / np.sum(weights)
    C_err[iw] = 1.0 / np.sqrt(np.sum(weights))

print(f"✓ Contrast spectrum computed")
print(f"  Median contrast: {np.median(C[(wl > wmin) & (wl < wmax)]):.6e}")
print(f"  Median error: {np.median(C_err[(wl > wmin) & (wl < wmax)]):.6e}")
print("="*70)
print()


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

# Compute planet spectrum
spec_planet = C * spec_star_interp 
spec_planet_err = C_err * spec_star_interp

print(f"✓ Planet spectrum computed")
print(f"  Median flux: {np.median(spec_planet[(wl > wmin) & (wl < wmax)]):.6e} W/m²/µm")
print()

# Save planetary spectrum to FITS file
out_path = os.path.join(path_output, 'spectrometry', 'planet_spectrum.fits')
fits.writeto(out_path, np.array([wl, spec_planet, spec_planet_err]), overwrite=True)

print(f"✓ Saved to: {out_path}")
print("="*70)
print()


##############################################################################
# FIGURES
##############################################################################

print("="*70)
print("GENERATING FIGURES")
print("="*70)

# Contrast spectrum
plt.figure(figsize=(15, 5))
plt.errorbar(wl*1e6, C, yerr=C_err, fmt='o', label='Contrast Spectrum', markersize=4)
plt.xlabel(r'Wavelength [$\mu$m]', fontsize=12)
plt.ylabel('Contrast', fontsize=12)
plt.legend(loc='upper left')
plt.ylim(0e-3, 1.5e-3)
plt.xlim(2.76, 5.0)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(path_output + '/spectrometry/Contrast_Spectrum.png', dpi=300)
plt.close()
print("✓ Saved: Contrast_Spectrum.png")

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
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(path_output + '/spectrometry/SNR.png', dpi=300)
plt.close()
print(f"✓ Saved: SNR.png (Median SNR 3-4 µm: {snr_median:.2f})")

# Contrast error
plt.figure(figsize=(15, 5))
plt.plot(wl*1e6, C_err, 'o-', markersize=4)
median_err = np.median(C_err[(wl > wmin) & (wl < wmax)])
plt.axhline(y=median_err, color='r', linestyle='--', label=f'Median Error = {median_err:.2e}')
plt.xlabel(r'Wavelength [$\mu$m]', fontsize=12)
plt.ylabel('Contrast Error', fontsize=12)
plt.xlim(2.76, 5.0)
plt.ylim(0, 1e-4)
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
plt.ylim(0e-15, 4.5e-15)
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
plt.ylim(0e-15, 4.5e-15)
plt.xlim(2.76, 5.0)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(path_output + '/spectrometry/Planet_Spectrum_filled.png', dpi=300)
plt.close()
print("✓ Saved: Planet_Spectrum_filled.png")

# Comparison with Houllé+2025
mhoulle = fits.getdata('/Users/jscigliuto/exoMATISSE/exoMATISSE/SPEXTRA_LAST_VERSION/spectrum_BetaPicb_formosa_MATISSE-LM_cov.fits', ext=1)

# Extraire wavelength et flux
wl_houlle = mhoulle['WAV']
flux_houlle = mhoulle['FLX']
errors_houlle = np.sqrt(np.diag(mhoulle['COV']))
plt.figure(figsize=(15, 5))

# Houllé+2025
plt.plot(wl_houlle, flux_houlle, marker='+', markersize=6, label='Houllé+2025 Spectrum', color='darkorange')
plt.fill_between(wl_houlle, flux_houlle-errors_houlle, flux_houlle+errors_houlle, alpha=0.3, color='orange', label='Houllé+2025 1σ uncertainty')
# Our data
plt.fill_between(wl*1e6, spec_planet-spec_planet_err, spec_planet+spec_planet_err, 
                 alpha=0.3, color='plum', label='1σ uncertainty')
plt.plot(wl*1e6, spec_planet, marker='+', color='purple', markersize=6, label='Planet Spectrum')

plt.xlabel(r'Wavelength [$\mu$m]', fontsize=12)
plt.ylabel(r'Flux [W/m$^2$/µm]', fontsize=12)
plt.legend(loc='upper right')
plt.ylim(0e-15, 4.5e-15)
plt.xlim(2.76, 5.0)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(path_output + '/spectrometry/Planet_Spectrum_compared.png', dpi=300)
plt.close()
print("✓ Saved: Planet_Spectrum_compared.png")

# Residuals with model
spec_model_planet = fits.getdata(planet_model_path)
if use_bin_data:
    spec_model_planet = spec_model_planet.reshape(-1, 5).mean(axis=1)

wl_mask = (wl >= wmin) & (wl <= wmax)
alpha_model = 1.357996097634616  # Example value, adjust as needed

plt.figure(figsize=(15, 5))
plt.plot(wl[wl_mask]*1e6, alpha_model*spec_model_planet[wl_mask] - spec_planet[wl_mask], 
         marker='+', markersize=6)
plt.xlabel(r'Wavelength [$\mu$m]', fontsize=12)
plt.ylabel('Flux Residuals (Model - Measured)', fontsize=12)
plt.xlim(3.0, 4.2)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(path_output + '/spectrometry/Planet_Spectrum_Residual.png', dpi=300)
plt.close()
print("✓ Saved: Planet_Spectrum_Residual.png")