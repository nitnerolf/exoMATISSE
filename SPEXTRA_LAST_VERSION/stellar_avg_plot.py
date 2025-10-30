# Importation
import numpy as np
from astropy.io import fits
import os 
import matplotlib.pyplot as plt
import scipy 
import shutil
import multiprocessing as mp
from common_tools import reorder_baselines, wrap, mas2rad

# Output path 
path_output = '/Users/jscigliuto/exoMATISSE/exoMATISSE/data_hd72946b/test/'

# Path of the OiFits files (outputs of the pipeline, phase corrected)
path_oifits = '/Users/jscigliuto/Nextcloud/DATA/HD72946B/corrected_data_wo_rmnrec/'

files_star   = sorted([file for file in os.listdir(path_oifits) if '.fits' in file and '_star' in file])

# Extract the wavelength grid 
hdul = fits.open(path_oifits + files_star[0])
wl = hdul['OI_WAVELENGTH'].data['EFF_WAVE']

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))

OB_star = [1] 
# Loop to interpolate the average star quantities between the OBs
for i_OB in OB_star:
    print(f'Processing OB {i_OB}...')
    with fits.open(path_output + f'/stellar_OB_averages/star_avg_visphi_OB{i_OB}.fits') as hdul_phi:
        visphi_data = hdul_phi[0].data
        cf_phi_star = visphi_data[0]  
        cf_phi_star_err = visphi_data[1]
    with fits.open(path_output + f'/stellar_OB_averages/star_avg_visamp_OB{i_OB}.fits') as hdul_amp:
        visamp_data = hdul_amp[0].data
        cf_amp_star = visamp_data[0]
        cf_amp_star_err = visamp_data[1]

    # Complexify
    cf_star = cf_amp_star * np.exp(1j * cf_phi_star)

    cf_real_star_err = np.sqrt((np.cos(cf_phi_star) * cf_amp_star_err) ** 2 \
                                + (cf_amp_star * np.sin(cf_phi_star) * cf_phi_star_err) ** 2)
    cf_imag_star_err = np.sqrt((np.sin(cf_phi_star) * cf_amp_star_err) ** 2 \
                                + (cf_amp_star * np.cos(cf_phi_star) * cf_phi_star_err) ** 2)
    
    # Plot
    for i_base in range(6):
        ax.errorbar(wl*1e6, np.real(cf_star[i_base]), yerr=cf_real_star_err[i_base], label=f'OB {i_OB} - Baseline {i_base+1}', alpha=0.7)
        ax.set_xlabel('Wavelength (µm)', fontsize=14)
        # ax.set_ylim(-1, 1)
        ax.set_ylabel('Real part', fontsize=14)
        ax.legend(loc='upper right', fontsize=10)

    for i_base in range(6):
        ax1.errorbar(wl*1e6, np.imag(cf_star[i_base]), yerr=cf_imag_star_err[i_base], label=f'OB {i_OB} - Baseline {i_base+1}', alpha=0.7)
        ax1.set_xlabel('Wavelength (µm)', fontsize=14)
        ax1.set_ylabel('Imaginary part', fontsize=14)
        ax1.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.show()


