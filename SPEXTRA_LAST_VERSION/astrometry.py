### ASTROMETRY DETERMINATION 


# Importation
import numpy as np
from astropy.io import fits
import os 
import matplotlib.pyplot as plt
from common_tools import reorder_baselines, wrap


# Path of the OiFits files (outputs of the pipeline, phase corrected)
path_oifits = '/Users/jscigliuto/Nextcloud/DATA/HD72946B/corrected_data_wo_rmnrec/'

# Output path 
path_output = '/Users/jscigliuto/exoMATISSE/exoMATISSE/data_hd72946b/test/'

# Models path
model_star = ''
model_planet = ''

# Flags
plot     = True
sci_case = 'bright' #'faint' 

# Baseline order and names
base_order_name = ('U3-U4', 'U1-U2', 'U2-U3', 'U2-U4', 'U1-U3', 'U1-U4')

### Functions

def compute_average_star_quantities(path_oifits, filelist_star):
    n_files = len(filelist_star)
    for i_file, file in enumerate(filelist_star):
        hdul = fits.open(path_oifits + file)
        hdul = reorder_baselines(hdul)

        if i_file == 0:
            # Initialize the concatenations
            n_wave = hdul['OI_WAVELENGTH'].data['EFF_WAVE'].size
            cf_amp_star_all     = np.zeros((n_files, 6, n_wave))
            cf_amp_star_err_all = np.zeros((n_files, 6, n_wave))
            cf_phi_star_all     = np.zeros((n_files, 6, n_wave))
            cf_phi_star_err_all = np.zeros((n_files, 6, n_wave))
            mjd                 = np.zeros((n_files))

        cf_amp_star_all[i_file]     = hdul['OI_VIS'].data['VISAMP']
        cf_amp_star_err_all[i_file] = hdul['OI_VIS'].data['VISAMPERR']
        cf_phi_star_all[i_file]     = np.deg2rad(hdul['OI_VIS'].data['VISPHI'])
        cf_phi_star_err_all[i_file] = np.deg2rad(hdul['OI_VIS'].data['VISPHIERR'])
        mjd                         = hdul['OI_VIS'].data['MJD']

        hdul.close()

    # Complexify
    cf_star_all          = cf_amp_star_all * np.exp(1j * cf_phi_star_all)
    cf_real_star_err_all = np.sqrt((np.cos(cf_phi_star_all) * cf_amp_star_err_all) ** 2 \
                                + (cf_amp_star_all * np.sin(cf_phi_star_all) * cf_phi_star_err_all) ** 2)
    cf_imag_star_err_all = np.sqrt((np.sin(cf_phi_star_all) * cf_amp_star_err_all) ** 2 \
                                + (cf_amp_star_all * np.cos(cf_phi_star_all) * cf_phi_star_err_all) ** 2)
    # Average
    mean_cf_star          = np.mean(np.real(cf_star_all), axis=0) + 1j * np.mean(np.imag(cf_star_all), axis=0)
    mean_cf_real_star_err = np.sqrt(np.sum(cf_real_star_err_all**2, axis=0)) / n_files
    mean_cf_imag_star_err = np.sqrt(np.sum(cf_imag_star_err_all**2, axis=0)) / n_files

    # Go back to amplitude/phase space
    mean_cf_amp_star = np.abs(mean_cf_star)
    mean_cf_phi_star = wrap(np.angle(mean_cf_star))
    mean_cf_amp_star_err = np.sqrt((np.real(mean_cf_star) * mean_cf_real_star_err) ** 2 \
                                    + (np.imag(mean_cf_star) * mean_cf_imag_star_err) ** 2) / mean_cf_amp_star
    mean_cf_phi_star_err = np.sqrt((np.imag(mean_cf_star) * mean_cf_real_star_err) ** 2 \
                                     + (np.real(mean_cf_star) * mean_cf_imag_star_err) ** 2) / mean_cf_amp_star**2
    
    # Avg the MJDs over the OB
    mean_mjd = np.mean(mjd)

    return mean_cf_amp_star, mean_cf_amp_star_err, \
           mean_cf_phi_star, mean_cf_phi_star_err, mean_mjd

###

### Initializations
# Create a dictionary to store the mjd for each OB
mjd_dict = {}

# Create necessary directories
if not os.path.isdir(path_output + '/stellar_OB_averages'):
    os.makedirs(path_output + '/stellar_OB_averages')
if not os.path.isdir(path_output + '/Contrast'):
        os.makedirs(path_output + '/Contrast')
if not os.path.isdir(path_output + '/Contrast_fits'):
        os.makedirs(path_output + '/Contrast_fits')
if not os.path.isdir(path_output + '/SNR'):
        os.makedirs(path_output + '/SNR')
                

# List the input OiFits files
files        = sorted([file for file in os.listdir(path_oifits) if '.fits' in file])
files_star   = sorted([file for file in os.listdir(path_oifits) if '.fits' in file and '_star' in file])
files_planet = sorted([file for file in os.listdir(path_oifits) if '.fits' in file and '_planet' in file])
n_file_planet = len(files_planet)
OBs_star     = list(set([int(file[file.find('OB')+2:file.find('_exp')]) for file in files_star]))
OBs_planet   = list(set([int(file[file.find('OB')+2:file.find('_exp')]) for file in files_planet]))


### Compute the Cps for each planet frame
# Loop to average the star quantities for each OB
for i_OB in OBs_star:

    # Select the files corresponding to the current OB
    filelist_star = [file for file in files_star if f'OB{i_OB}_' in file]

    # Extract the wavelength grid 
    hdul = fits.open(path_oifits + filelist_star[0])
    wl = hdul['OI_WAVELENGTH'].data['EFF_WAVE']

    # Compute the average star quantities
    mean_cf_amp_star, mean_cf_amp_star_err, \
    mean_cf_phi_star, mean_cf_phi_star_err, mean_mjd = compute_average_star_quantities(path_oifits, filelist_star)

    visamp_data = np.array([mean_cf_amp_star, mean_cf_amp_star_err])
    visphi_data = np.array([mean_cf_phi_star, mean_cf_phi_star_err])

    # Store mean_mjd for each i_OB
    mjd_dict[i_OB] = mean_mjd

    # Save the amplitude, phase and associated errors in fits files
    fits.writeto(path_output + f'/stellar_OB_averages/star_avg_visamp_OB{i_OB}.fits', visamp_data, overwrite=True)
    fits.writeto(path_output + f'/stellar_OB_averages/star_avg_visphi_OB{i_OB}.fits', visphi_data, overwrite=True)

# Initialize the contrast (p/s) array
n_wave           = wl.size
Cps_all          = np.zeros((n_file_planet, 6, n_wave), dtype=complex)
Cps_real_err_all = np.zeros((n_file_planet, 6, n_wave))
Cps_imag_err_all = np.zeros((n_file_planet, 6, n_wave))
snr_real_all     = np.zeros((n_file_planet, 6, n_wave))
snr_imag_all     = np.zeros((n_file_planet, 6, n_wave))

# Loop to interpolate the average star quantities between the OBs
for n_file, file_planet in enumerate(files_planet):

    # Identify the OB numbers for the planet and star (for interpolation)
    num_OB_planet    = int(file_planet[file_planet.find('OB')+2:file_planet.find('_exp')])
    num_exp_planet   = int(file_planet[file_planet.find('_exp')+4:file_planet.find('_frame')])
    num_frame_planet = int(file_planet[file_planet.find('_frame')+6:file_planet.find('_planet')])
    num_OB_star = [int(num_OB_planet)-1, int(num_OB_planet)+1] # Assuming star OBs are just before and after planet OB

    # Extract planet quantities
    hdul_planet = fits.open(path_oifits + file_planet)
    hdul_planet = reorder_baselines(hdul_planet)
    cf_amp_planet     = hdul_planet['OI_VIS'].data['VISAMP']
    cf_amp_planet_err = hdul_planet['OI_VIS'].data['VISAMPERR']
    cf_phi_planet     = np.deg2rad(hdul_planet['OI_VIS'].data['VISPHI'])
    cf_phi_planet_err = np.deg2rad(hdul_planet['OI_VIS'].data['VISPHIERR'])
    mjd_planet        = hdul_planet['OI_VIS'].data['MJD'][0]
    U_planet          = hdul_planet['OI_VIS'].data['UCOORD']
    V_planet          = hdul_planet['OI_VIS'].data['VCOORD']

    # Complexify planet quantities
    cf_planet          = cf_amp_planet * np.exp(1j * cf_phi_planet)
    cf_real_planet_err = np.sqrt((np.cos(cf_phi_planet) * cf_amp_planet_err) ** 2 \
                                + (cf_amp_planet * np.sin(cf_phi_planet) * cf_phi_planet_err) ** 2)
    cf_imag_planet_err = np.sqrt((np.sin(cf_phi_planet) * cf_amp_planet_err) ** 2 \
                                + (cf_amp_planet * np.cos(cf_phi_planet) * cf_phi_planet_err) ** 2)

    # 1 st star OB
    with fits.open(path_output + f'/stellar_OB_averages/star_avg_visphi_OB{num_OB_star[0]}.fits') as hdul_phi1:
        visphi_data = hdul_phi1[0].data
        cf_phi_star_1 = visphi_data[0]  
        cf_phi_star_err_1 = visphi_data[1]  
    with fits.open(path_output + f'/stellar_OB_averages/star_avg_visamp_OB{num_OB_star[0]}.fits') as hdul_amp1:
        visamp_data = hdul_amp1[0].data
        cf_amp_star_1 = visamp_data[0]
        cf_amp_star_err_1 = visamp_data[1]

    # Complexify 
    cf_star_1 = cf_amp_star_1 * np.exp(1j * cf_phi_star_1)
    cf_real_star_1 = np.real(cf_star_1)
    cf_imag_star_1 = np.imag(cf_star_1)
    cf_real_star_err_1 = np.sqrt((np.cos(cf_phi_star_1) * cf_amp_star_err_1) ** 2 \
                                + (cf_amp_star_1 * np.sin(cf_phi_star_1) * cf_phi_star_err_1) ** 2)
    cf_imag_star_err_1 = np.sqrt((np.sin(cf_phi_star_1) * cf_amp_star_err_1) ** 2 \
                                + (cf_amp_star_1 * np.cos(cf_phi_star_1) * cf_phi_star_err_1) ** 2)

    # Next star OB
    with fits.open(path_output + f'/stellar_OB_averages/star_avg_visphi_OB{num_OB_star[1]}.fits') as hdul_phi2:
        visphi_data = hdul_phi2[0].data
        cf_phi_star_2 = visphi_data[0]  
        cf_phi_star_err_2 = visphi_data[1]  
    with fits.open(path_output + f'/stellar_OB_averages/star_avg_visamp_OB{num_OB_star[1]}.fits') as hdul_amp2:
        visamp_data = hdul_amp2[0].data
        cf_amp_star_2 = visamp_data[0]
        cf_amp_star_err_2 = visamp_data[1]
    
    # Complexify
    cf_star_2 = cf_amp_star_2 * np.exp(1j * cf_phi_star_2)
    cf_real_star_2 = np.real(cf_star_2)
    cf_imag_star_2 = np.imag(cf_star_2)
    cf_real_star_err_2 = np.sqrt((np.cos(cf_phi_star_2) * cf_amp_star_err_2) ** 2 \
                                + (cf_amp_star_2 * np.sin(cf_phi_star_2) * cf_phi_star_err_2) ** 2)
    cf_imag_star_err_2 = np.sqrt((np.sin(cf_phi_star_2) * cf_amp_star_err_2) ** 2 \
                                + (cf_amp_star_2 * np.cos(cf_phi_star_2) * cf_phi_star_err_2) ** 2)
    

    # Extract the MJD values for the two OBs
    mjd_1 = mjd_dict[num_OB_star[0]]
    mjd_2 = mjd_dict[num_OB_star[1]]
    u = (mjd_planet - mjd_1) / (mjd_2 - mjd_1)

    # Interpolate at the planet mjd
    cf_real_star_interp = (1 - u) * cf_real_star_1 + u * cf_real_star_2
    cf_imag_star_interp = (1 - u) * cf_imag_star_1 + u * cf_imag_star_2
    cf_real_star_err = np.sqrt(((1 - u) * cf_real_star_err_1) ** 2 + (u * cf_real_star_err_2) ** 2)
    cf_imag_star_err = np.sqrt(((1 - u) * cf_imag_star_err_1) ** 2 + (u * cf_imag_star_err_2) ** 2)

    # Complexify 
    cf_star_interp = cf_real_star_interp + 1j * cf_imag_star_interp

    # Compute the planet-to-star flux ratio and associated errors
    Cps = cf_planet / cf_star_interp
    Cps_real_err = Cps * np.sqrt((cf_real_planet_err / np.real(cf_planet)) ** 2 \
                           + (cf_real_star_err / np.real(cf_star_interp)) ** 2) 
    Cps_imag_err = Cps * np.sqrt((cf_imag_planet_err / np.imag(cf_planet)) ** 2 \
                           + (cf_imag_star_err / np.imag(cf_star_interp)) ** 2) 
    
    ##### ENREGISTRER FICHIER CPS + ERREURS
    
    Cps_all[n_file]          = Cps
    Cps_real_err_all[n_file] = Cps_real_err
    Cps_imag_err_all[n_file] = Cps_imag_err

    # Compute SNR
    snr_real = np.abs(np.real(Cps) / Cps_real_err)
    snr_imag = np.abs(np.imag(Cps) / Cps_imag_err)
    snr_real_all[n_file] = snr_real
    snr_imag_all[n_file] = snr_imag

    # Save contrast and associated real/imag errors to a FITS file
    cps_real = np.real(Cps)
    cps_imag = np.imag(Cps)
    cps_real_err = np.real(Cps_real_err)
    cps_imag_err = np.real(Cps_imag_err)

    hdul = fits.HDUList()
    hdul.append(fits.PrimaryHDU())
    hdul.append(fits.ImageHDU(cps_real.astype(np.float32), name='CPS_REAL'))
    hdul.append(fits.ImageHDU(cps_imag.astype(np.float32), name='CPS_IMAG'))
    hdul.append(fits.ImageHDU(cps_real_err.astype(np.float32), name='CPS_REAL_ERR'))
    hdul.append(fits.ImageHDU(cps_imag_err.astype(np.float32), name='CPS_IMAG_ERR'))
    hdul.append(fits.ImageHDU(U_planet, name='U'))
    hdul.append(fits.ImageHDU(V_planet, name='V'))

    # Add simple metadata
    hdr = hdul[0].header
    hdul.append(fits.ImageHDU(wl.astype(np.float32), name='WAVELENGTH'))
    hdr['OB'] = num_OB_planet
    hdr['EXP'] = num_exp_planet
    hdr['FRAME'] = num_frame_planet
    hdr['MJD'] = mjd_planet

    outfile = path_output + f'/Contrast_fits/Cps_OB{num_OB_planet}_exp{num_exp_planet}_frame{num_frame_planet}.fits'
    hdul.writeto(outfile, overwrite=True)
    

    ## Plots
    if plot == True:
        fig_Cps, ax_Cps = plt.subplots(3, 2, figsize=(12, 8))
        ax_Cps = ax_Cps.flatten()

        fig_snr, ax_snr = plt.subplots(3, 2, figsize=(12, 8))
        ax_snr = ax_snr.flatten()

        for i_base in range(6): 
            
            #Cps
            fig_Cps.text(0.5, 0.001, 'Wavelength (µm)', ha='center', fontsize=12)
            fig_Cps.text(0.001, 0.5, 'Real Contrast (p/s)', va='center', rotation='vertical', fontsize=12)
            fig_Cps.suptitle('Planet-to-Star Contrast', fontsize=16)
            ax_Cps[i_base].plot(wl*1e6, np.real(Cps_all[n_file,i_base]), color='crimson', alpha=0.8)
            ax_Cps[i_base].fill_between(wl*1e6, np.real(Cps_all[n_file,i_base])-Cps_real_err_all[n_file, i_base], 
                                    np.real(Cps_all[n_file,i_base])+Cps_real_err_all[n_file, i_base], alpha=0.2, color='hotpink')
            ax_Cps[i_base].set_ylim(-0.5e-1, 0.75e-1)
            ax_Cps[i_base].set_title(f'Baseline {base_order_name[i_base]}')
            # ax[i_base].set_ylim(-1.5e-1, 2e-1)
            # ax[i_base].legend(loc='upper right')

            #SNR
            fig_snr.text(0.5, 0.001, 'Wavelength (µm)', ha='center', fontsize=12)
            fig_snr.text(0.001, 0.5, 'SNR Real Contrast', va='center', rotation='vertical', fontsize=12)
            fig_snr.suptitle('SNR on Planet-to-Star Contrast', fontsize=16)
            ax_snr[i_base].plot(wl*1e6, snr_real_all[n_file,i_base], color='navy', alpha=0.8)
            ax_snr[i_base].set_ylim(0, 6)
            ax_snr[i_base].set_title(f'Baseline {base_order_name[i_base]}')

        plt.tight_layout()

        # Save the figure
        fig_Cps.savefig(path_output + f'/Contrast/Cps_OB{num_OB_planet}_exp{num_exp_planet}_frame{num_frame_planet}.png', dpi=300)
        fig_snr.savefig(path_output + f'/SNR/snr_Cps_OB{num_OB_planet}_exp{num_exp_planet}_frame{num_frame_planet}.png', dpi=300)



##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

### Initializations 
# Planet offsets coords mentioned in the OB [mas]
Offset_RA = 106
Offset_Dec = -145

# Grid of coordinates to determine the astrometry of the planet
x      = np.linspace(Offset_RA-50, Offset_RA+50, 100)
y      = np.linspace(Offset_Dec-50, Offset_Dec+50, 100)
xp, yp = np.meshgrid(x, y) #grid of coords to look for the planet position

# 
n_poly = 1 # Degree of the polynomial to model the stellar speckle
stellar_coeffs_init = 1.e-4 * (n_poly + 1)  # Initial coeffs for the stellar speckle polynomial
alpha = 1. # Multiplicative factor to scale the Cps
params_init = np.array([alpha, *stellar_coeffs_init]) 
n_params = params_init.size
bounds_params = [(0., None), (None, None)* (n_poly + 1)]

# List the Cps files
Cps_path = path_output + '/test/Contrast_fits/'
files_Cps = sorted([file for file in os.listdir(path_output + Cps_path) if '.fits' in file and '_planet' in file])

# Path to Cps model 
Cps_model_path = ''

### Determine the astrometry 

for i_file, file_Cps in enumerate(files_Cps):
    
    num_OB_planet    = int(file_Cps[file_Cps.find('OB')+2:file_Cps.find('_exp')])
    num_exp_planet   = int(file_Cps[file_Cps.find('_exp')+4:file_Cps.find('_frame')])
    num_frame_planet = int(file_Cps[file_Cps.find('_frame')+6:file_Cps.find('.fits')])

    # Extract Cps quantities
    hdul_Cps = fits.open(path_output + Cps_path + file_Cps)
    U = hdul_planet['OI_VIS'].data['U']
    V = hdul_planet['OI_VIS'].data['V']

    if sci_case == 'faint':
        
        # Position angle (PA) and separation grid
        PA = np.arctan2(yp, xp) 
        sep = np.sqrt(xp**2 + yp**2)

        # Baseline-PA coverage in the UV-space
        PAcov = np.arctan2(U,V)
        Bcov = np.sqrt(U**2 + V**2)
        
        


    # elif sci_case == 'bright':
    #     # Fitter directement les oscillations (via current model)
