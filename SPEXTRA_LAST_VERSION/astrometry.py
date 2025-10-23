### ASTROMETRY DETERMINATION 


# Importation
import numpy as np
from astropy.io import fits
import os 
import matplotlib.pyplot as plt
from common_tools import reorder_baselines, wrap


# Path of the OiFits files (outputs of the pipeline, phase corrected)
path_oifits = '/Users/jscigliuto/exoMATISSE/exoMATISSE/data_hd72946b/corrected_data/'

# Output path 
path_output = '/Users/jscigliuto/exoMATISSE/exoMATISSE/data_hd72946b/test/'

# Plotting flag (create directories with plots)
plot = True

# Baseline order and names
base_order_name = ('U3-U4', 'U1-U2', 'U2-U3', 'U2-U4', 'U1-U3', 'U1-U4')

###

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

# Create a dictionary to store the mjd for each OB
mjd_dict = {}

# Create necessary directories
if not os.path.isdir(path_output + '/stellar_OB_averages'):
    os.makedirs(path_output + '/stellar_OB_averages')
if not os.path.isdir(path_output + '/Contrast'):
        os.makedirs(path_output + '/Contrast')

# List the input OiFits files
files        = sorted([file for file in os.listdir(path_oifits) if '.fits' in file])
files_star   = sorted([file for file in os.listdir(path_oifits) if '.fits' in file and '_star' in file])
files_planet = sorted([file for file in os.listdir(path_oifits) if '.fits' in file and '_planet' in file])
n_file_planet = len(files_planet)
OBs_star     = list(set([int(file[file.find('OB')+2:file.find('_exp')]) for file in files_star]))
OBs_planet   = list(set([int(file[file.find('OB')+2:file.find('_exp')]) for file in files_planet]))



# Loop to average the star quantities for each OB
for i_OB in OBs_star:

    # Select the files corresponding to the current OB
    filelist_star = [file for file in files_star if f'OB{i_OB}_' in file]
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
    
    Cps_all[n_file]          = Cps
    Cps_real_err_all[n_file] = Cps_real_err
    Cps_imag_err_all[n_file] = Cps_imag_err

    ### Plots
    if plot == True:
        fig, ax = plt.subplots(3, 2, figsize=(12, 8))
        ax = ax.flatten()

        for i_base in range(6): 

            fig.text(0.5, 0.001, 'Wavelength (µm)', ha='center', fontsize=12)
            fig.text(0.001, 0.5, 'Real Contrast (p/s)', va='center', rotation='vertical', fontsize=12)
            fig.suptitle('Planet-to-Star Contrast', fontsize=16)
            ax[i_base].plot(wl*1e6, np.real(Cps_all[n_file,i_base]), color='crimson', alpha=0.8)
            ax[i_base].fill_between(wl*1e6, np.real(Cps_all[n_file,i_base])-Cps_real_err_all[n_file, i_base], 
                                    np.real(Cps_all[n_file,i_base])+Cps_real_err_all[n_file, i_base], alpha=0.2, color='hotpink')
            ax[i_base].set_ylim(-0.5e-1, 0.75e-1)
            ax[i_base].set_title(f'Baseline {base_order_name[i_base]}')
            # ax[i_base].set_ylim(-1.5e-1, 2e-1)
            # ax[i_base].legend(loc='upper right')
        plt.tight_layout()

        # Save the figure
        fig.savefig(path_output + f'/Contrast/Cps_OB{num_OB_planet}_exp{num_exp_planet}_frame{num_frame_planet}.png', dpi=300)


