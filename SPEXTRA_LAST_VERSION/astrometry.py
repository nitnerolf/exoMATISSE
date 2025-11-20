### ASTROMETRY DETERMINATION 


# Importation
import numpy as np
from astropy.io import fits
import os 
import shutil
import matplotlib.pyplot as plt
import scipy 
from common_tools import reorder_baselines, wrap, mas2rad


# Path of the OiFits files (outputs of the pipeline, phase corrected)
# path_oifits = '/Users/jscigliuto/Nextcloud/DATA/HD72946B/corrected_data_wo_rmnrec/' #HD 72946 B
path_oifits = '/Users/jscigliuto/Desktop/Licallo_backup/Pipeline/betaPicb/corrPhaseMathis_MACAO/corrected_data_bin/' #beta Pic b

# Output path 
# path_output = '/Users/jscigliuto/exoMATISSE/exoMATISSE/data_hd72946b/' #HD 72946 B
path_output = '/Users/jscigliuto/exoMATISSE/exoMATISSE/data_betaPicb/' #beta Pic b


# Flags
plot = True
target = 'betaPicb' #'Other' 

# Baseline order and names
base_order_name = ('U3-U4', 'U1-U2', 'U2-U3', 'U2-U4', 'U1-U3', 'U1-U4')

### Functions

def compute_average_star_quantities(path_oifits, filelist_star):
    n_files = len(filelist_star)
    for i_file, file in enumerate(filelist_star):
        hdul = fits.open(path_oifits + file)
        # hdul = reorder_baselines(hdul)

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

# # Create necessary directories
# if not os.path.isdir(path_output + '/stellar_OB_averages'):
#     os.makedirs(path_output + '/stellar_OB_averages')
# if os.path.isdir(path_output + '/stellar_OB_averages'):
#     shutil.rmtree(path_output + '/stellar_OB_averages')
#     os.makedirs(path_output + '/stellar_OB_averages')
# if not os.path.isdir(path_output + '/Contrast'):
#     os.makedirs(path_output + '/Contrast')
# if os.path.isdir(path_output + '/Contrast'):
#     shutil.rmtree(path_output + '/Contrast')
#     os.makedirs(path_output + '/Contrast')
# if not os.path.isdir(path_output + '/Contrast_fits'):
#     os.makedirs(path_output + '/Contrast_fits')
# if os.path.isdir(path_output + '/Contrast_fits'):
#     shutil.rmtree(path_output + '/Contrast_fits')
#     os.makedirs(path_output + '/Contrast_fits')
# if not os.path.isdir(path_output + '/SNR'):
#     os.makedirs(path_output + '/SNR')
# if os.path.isdir(path_output + '/SNR'):
#     shutil.rmtree(path_output + '/SNR')
#     os.makedirs(path_output + '/SNR')

for directory in ['/Stellar_OB_averages', '/Contrast', '/Contrast_fits', '/Plots', '/SNR']:
        dir_path = path_output + directory
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

for directory in [ '/Amplitude', '/Phase', '/Real_part', '/Imag_part']:
        dir_path = path_output + '/Plots' + directory
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)


# List the input OiFits files
files        = sorted([file for file in os.listdir(path_oifits) if '.fits' in file])
files_star   = sorted([file for file in os.listdir(path_oifits) if '.fits' in file and '_star' in file])
files_planet = sorted([file for file in os.listdir(path_oifits) if '.fits' in file and '_planet' in file])
n_file_planet = len(files_planet)
OBs_star     = list(set([int(file[file.find('OB')+2:file.find('_exp')]) for file in files_star]))
OBs_planet   = list(set([int(file[file.find('OB')+2:file.find('_exp')]) for file in files_planet]))

# print(files_star)
# print(files_planet)


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

print(f"target = '{target}'")

if target == 'betaPicb':
    print("Entering betaPicb branch")
    for n_file, file_planet in enumerate(files_planet):

        # Identify the OB numbers for the planet and star (for interpolation)
        num_OB_planet    = int(file_planet[file_planet.find('OB')+2:file_planet.find('_exp')])
        num_exp_planet   = int(file_planet[file_planet.find('_exp')+4:file_planet.find('_frame')])
        num_frame_planet = int(file_planet[file_planet.find('_frame')+6:file_planet.find('_planet')])

        # Extract planet quantities
        hdul_planet = fits.open(path_oifits + file_planet)
        # hdul_planet = reorder_baselines(hdul_planet)
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

        # The star OB correspond to the planet one, so no interpolation here
        with fits.open(path_output + f'/stellar_OB_averages/star_avg_visphi_OB{num_OB_planet}.fits') as hdul_phi:
            visphi_data = hdul_phi[0].data
            cf_phi_star = visphi_data[0]  
            cf_phi_star_err = visphi_data[1]  
        with fits.open(path_output + f'/stellar_OB_averages/star_avg_visamp_OB{num_OB_planet}.fits') as hdul_amp:
            visamp_data = hdul_amp[0].data
            cf_amp_star = visamp_data[0]
            cf_amp_star_err = visamp_data[1]

        # Complexify 
        cf_star = cf_amp_star * np.exp(1j * cf_phi_star)
        cf_real_star = np.real(cf_star)
        cf_imag_star = np.imag(cf_star)
        cf_real_star_err = np.sqrt((np.cos(cf_phi_star) * cf_amp_star_err) ** 2 \
                                    + (cf_amp_star * np.sin(cf_phi_star) * cf_phi_star_err) ** 2)
        cf_imag_star_err = np.sqrt((np.sin(cf_phi_star) * cf_amp_star_err) ** 2 \
                                    + (cf_amp_star * np.cos(cf_phi_star) * cf_phi_star_err) ** 2)

        
        # Complexify
        cf_star = cf_real_star + 1j * cf_imag_star
        cf_real_star_err = cf_real_star_err
        cf_imag_star_err = cf_imag_star_err

        # Compute the planet-to-star flux ratio and associated errors
        Cps = cf_planet / cf_star

        cf_planet_real = np.real(cf_planet)
        cf_planet_imag = np.imag(cf_planet)
        cf_star_interp_real = np.real(cf_star)
        cf_star_interp_imag = np.imag(cf_star)

        norm = cf_star_interp_real**2 + cf_star_interp_imag**2

        #dRe(Cps)/d... (real)
        dRe_planet = cf_star_interp_real / norm
        dIm_planet = cf_star_interp_imag / norm
        dRe_star = (cf_planet_imag * cf_star_interp_imag - cf_planet_real * cf_star_interp_real) * 2 * cf_star_interp_real / (norm**2) 
        dIm_star = (cf_planet_real * cf_star_interp_imag - cf_planet_imag * cf_star_interp_real) * 2 * cf_star_interp_imag / (norm**2)

        Cps_real_err = np.sqrt((dRe_planet * cf_real_planet_err)**2 +
            (dIm_planet * cf_imag_planet_err)**2 +
            (dRe_star * cf_real_star_err)**2 +
            (dIm_star * cf_imag_star_err)**2)
        
        #dIm(Cps)/d... (imag)
        dRe_planet = -cf_star_interp_imag / norm
        dIm_planet = cf_star_interp_real / norm
        dRe_star = (cf_planet_real * cf_star_interp_real + cf_planet_imag * cf_star_interp_imag) * 2 * cf_star_interp_real / (norm**2) 
        dIm_star = -(cf_planet_real * cf_star_interp_imag + cf_planet_imag * cf_star_interp_real) * 2 * cf_star_interp_imag / (norm**2)

        Cps_imag_err = np.sqrt((dRe_planet * cf_real_planet_err)**2 +
            (dIm_planet * cf_imag_planet_err)**2 +
            (dRe_star * cf_real_star_err)**2 +
            (dIm_star * cf_imag_star_err)**2)

        # Cps_real_err = Cps * np.sqrt((cf_real_planet_err / np.real(cf_planet)) ** 2 \
        #                     + (cf_real_star_err / np.real(cf_star_interp)) ** 2) 
        # Cps_imag_err = Cps * np.sqrt((cf_imag_planet_err / np.imag(cf_planet)) ** 2 \
        #                     + (cf_imag_star_err / np.imag(cf_star_interp)) ** 2) 
        
        
        Cps_all[n_file]          = Cps
        Cps_real_err_all[n_file] = Cps_real_err
        Cps_imag_err_all[n_file] = Cps_imag_err

        # Compute SNR
        snr_real = np.abs(np.real(Cps) / Cps_real_err)
        snr_imag = np.abs(np.imag(Cps) / Cps_imag_err)
        snr_real_all[n_file] = snr_real
        snr_imag_all[n_file] = snr_imag

        # Save contrast and associated real/imag errors to a FITS file
        Cps_real = np.real(Cps)
        Cps_imag = np.imag(Cps)

        hdul = fits.HDUList()
        hdul.append(fits.PrimaryHDU())
        hdul.append(fits.ImageHDU(Cps_real.astype(np.float32), name='CPS_REAL'))
        hdul.append(fits.ImageHDU(Cps_imag.astype(np.float32), name='CPS_IMAG'))
        hdul.append(fits.ImageHDU(Cps_real_err.astype(np.float32), name='CPS_REAL_ERR'))
        hdul.append(fits.ImageHDU(Cps_imag_err.astype(np.float32), name='CPS_IMAG_ERR'))
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
            fig_Cps, ax_Cps = plt.subplots(6, 1, figsize=(12, 8), sharex=True)
            ax_Cps = ax_Cps.flatten()

            fig_snr, ax_snr = plt.subplots(6, 1, figsize=(12, 8), sharex=True)
            ax_snr = ax_snr.flatten()

            fig_amp, ax_amp = plt.subplots(6, 1, figsize=(12, 8), sharex=True)
            ax_amp = ax_amp.flatten()

            fig_phi, ax_phi = plt.subplots(6, 1, figsize=(12, 8), sharex=True)
            ax_phi = ax_phi.flatten()

            fig_real, ax_real = plt.subplots(6, 1, figsize=(12, 8), sharex=True)
            ax_real = ax_real.flatten()

            fig_imag, ax_imag = plt.subplots(6, 1, figsize=(12, 8), sharex=True)
            ax_imag = ax_imag.flatten()


            for i_base in range(6): 
                
                #Cps
                fig_Cps.text(0.5, 0.04, 'Wavelength (µm)', ha='center', va='bottom', fontsize=12)
                fig_Cps.text(0.06, 0.5, 'Real Contrast (p/s)', va='center', ha='right', rotation='vertical', fontsize=12)
                fig_Cps.suptitle('Planet-to-Star Contrast', fontsize=16)
                if i_base < 5:
                    # ax_Cps[i_base].set_xlabel('')      
                    ax_Cps[i_base].tick_params(labelbottom=False)  

                ax_Cps[i_base].plot(wl*1e6, np.real(Cps_all[n_file,i_base]), color='crimson', alpha=0.8)
                ax_Cps[i_base].fill_between(wl*1e6, np.real(Cps_all[n_file,i_base])-Cps_real_err_all[n_file, i_base], 
                                        np.real(Cps_all[n_file,i_base])+Cps_real_err_all[n_file, i_base], alpha=0.2, color='hotpink')
                ax_Cps[i_base].set_ylim(-3e-3, 5e-3)
                # ax_Cps[i_base].set_title(f'Baseline {base_order_name[i_base]}', loc='left')
                ax_Cps[i_base].set_ylabel(f'{base_order_name[i_base]}')
                # ax[i_base].set_xlim
                # ax[i_base].set_ylim(-1.5e-1, 2e-1)
                # ax[i_base].legend(loc='upper right')

                #Amplitude
                fig_amp.text(0.5, 0.04, 'Wavelength (µm)', ha='center', va='bottom', fontsize=12)
                fig_amp.text(0.06, 0.5, 'Amplitude Contrast (p/s)', va='center', ha='right', rotation='vertical', fontsize=12)
                fig_amp.suptitle('Planet-to-Star Contrast Amplitude', fontsize=16)
                if i_base < 5:   
                    ax_amp[i_base].tick_params(labelbottom=False)   
                ax_amp[i_base].plot(wl*1e6, np.abs(Cps_all[n_file,i_base]), color='darkorange', alpha=0.8)
                ax_amp[i_base].fill_between(wl*1e6, np.abs(Cps_all[n_file,i_base])-Cps_real_err_all[n_file, i_base], 
                                        np.abs(Cps_all[n_file,i_base])+Cps_real_err_all[n_file, i_base], alpha=0.2, color='gold')
                ax_amp[i_base].set_ylim(0, 5e-3)
                ax_amp[i_base].set_title(f'Baseline {base_order_name[i_base]}')

                #Phase
                fig_phi.text(0.5, 0.04, 'Wavelength (µm)', ha='center', va='bottom', fontsize=12)
                fig_phi.text(0.06, 0.5, 'Phase Contrast (rad)', va='center', ha='right', rotation='vertical', fontsize=12)
                fig_phi.suptitle('Planet-to-Star Contrast Phase', fontsize=16)
                if i_base < 5:   
                    ax_phi[i_base].tick_params(labelbottom=False)
                ax_phi[i_base].plot(wl*1e6, np.angle(Cps_all[n_file,i_base]), color='green', alpha=0.8)
                ax_phi[i_base].set_ylim(-np.pi, np.pi)
                ax_phi[i_base].set_title(f'Baseline {base_order_name[i_base]}')

                #Real part
                fig_real.text(0.5, 0.04, 'Wavelength (µm)', ha='center', va='bottom', fontsize=12)
                fig_real.text(0.06, 0.5, 'Real Contrast (p/s)', va='center', ha='right', rotation='vertical', fontsize=12)
                fig_real.suptitle('Planet-to-Star Contrast Real Part', fontsize=16)
                if i_base < 5:   
                    ax_real[i_base].tick_params(labelbottom=False)
                ax_real[i_base].plot(wl*1e6, np.real(Cps_all[n_file,i_base]), color='blue', alpha=0.8)
                ax_real[i_base].fill_between(wl*1e6, np.real(Cps_all[n_file,i_base])-Cps_real_err_all[n_file, i_base], 
                                        np.real(Cps_all[n_file,i_base])+Cps_real_err_all[n_file, i_base], alpha=0.2, color='lightblue')
                ax_real[i_base].set_ylim(-3e-3, 5e-3)
                ax_real[i_base].set_title(f'Baseline {base_order_name[i_base]}')

                #Imaginary part
                fig_imag.text(0.5, 0.04, 'Wavelength (µm)', ha='center', va='bottom', fontsize=12)
                fig_imag.text(0.06, 0.5, 'Imaginary Contrast (p/s)', va='center', ha='right', rotation='vertical', fontsize=12)
                fig_imag.suptitle('Planet-to-Star Contrast Imaginary Part', fontsize=16)
                if i_base < 5:   
                    ax_imag[i_base].tick_params(labelbottom=False)
                ax_imag[i_base].plot(wl*1e6, np.imag(Cps_all[n_file,i_base]), color='purple', alpha=0.8)
                ax_imag[i_base].fill_between(wl*1e6, np.imag(Cps_all[n_file,i_base])-Cps_imag_err_all[n_file, i_base], 
                                        np.imag(Cps_all[n_file,i_base])+Cps_imag_err_all[n_file, i_base], alpha=0.2, color='violet')
                ax_imag[i_base].set_ylim(-3e-3, 5e-3)
                ax_imag[i_base].set_title(f'Baseline {base_order_name[i_base]}')

                #SNR
                fig_snr.text(0.5, 0.001, 'Wavelength (µm)', ha='center', fontsize=12)
                fig_snr.text(0.001, 0.5, 'SNR Real Contrast', va='center', rotation='vertical', fontsize=12)
                fig_snr.suptitle('SNR on Planet-to-Star Contrast', fontsize=16)
                ax_snr[i_base].plot(wl*1e6, snr_real_all[n_file,i_base], color='navy', alpha=0.8)
                ax_snr[i_base].set_ylim(0, 6)
                ax_snr[i_base].set_title(f'Baseline {base_order_name[i_base]}')

            plt.tight_layout()

            print(f"Saved plots for OB{num_OB_planet}_exp{num_exp_planet}_frame{num_frame_planet}")
            print(f"File processed: {n_file+1} / {n_file_planet}")

            # Save the figure
            fig_Cps.savefig(path_output + f'/Contrast/Cps_OB{num_OB_planet}_exp{num_exp_planet}_frame{num_frame_planet}.png', dpi=300)
            fig_snr.savefig(path_output + f'/SNR/snr_Cps_OB{num_OB_planet}_exp{num_exp_planet}_frame{num_frame_planet}.png', dpi=300)
            fig_amp.savefig(path_output + f'/Plots/Amplitude/Cps_amplitude_OB{num_OB_planet}_exp{num_exp_planet}_frame{num_frame_planet}.png', dpi=300)
            fig_phi.savefig(path_output + f'/Plots/Phase/Cps_phase_OB{num_OB_planet}_exp{num_exp_planet}_frame{num_frame_planet}.png', dpi=300)
            fig_real.savefig(path_output + f'/Plots/Real_part/Cps_real_OB{num_OB_planet}_exp{num_exp_planet}_frame{num_frame_planet}.png', dpi=300)
            fig_imag.savefig(path_output + f'/Plots/Imag_part/Cps_imag_OB{num_OB_planet}_exp{num_exp_planet}_frame{num_frame_planet}.png', dpi=300)
            plt.close(fig_Cps)
            plt.close(fig_snr)

######################################################################################################################################################################################################
######################################################################################################################################################################################################
######################################################################################################################################################################################################
######################################################################################################################################################################################################
######################################################################################################################################################################################################
######################################################################################################################################################################################################
######################################################################################################################################################################################################
######################################################################################################################################################################################################
######################################################################################################################################################################################################
######################################################################################################################################################################################################
######################################################################################################################################################################################################
######################################################################################################################################################################################################



else:
    print("Entering Other target branch")
    # Loop to interpolate the average star quantities between the OBs
    for n_file, file_planet in enumerate(files_planet):

        # Identify the OB numbers for the planet and star (for interpolation)
        num_OB_planet    = int(file_planet[file_planet.find('OB')+2:file_planet.find('_exp')])
        num_exp_planet   = int(file_planet[file_planet.find('_exp')+4:file_planet.find('_frame')])
        num_frame_planet = int(file_planet[file_planet.find('_frame')+6:file_planet.find('_planet')])
        num_OB_star      = [int(num_OB_planet)-1, int(num_OB_planet)+1] # Assuming star OBs are just before and after planet OB

        # Extract planet quantities
        hdul_planet = fits.open(path_oifits + file_planet)
        # hdul_planet = reorder_baselines(hdul_planet)
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

        cf_planet_real = np.real(cf_planet)
        cf_planet_imag = np.imag(cf_planet)
        cf_star_interp_real = np.real(cf_star_interp)
        cf_star_interp_imag = np.imag(cf_star_interp)

        norm = cf_star_interp_real**2 + cf_star_interp_imag**2

        #dRe(Cps)/d... (real)
        dRe_planet = cf_star_interp_real / norm
        dIm_planet = cf_star_interp_imag / norm
        dRe_star = (cf_planet_imag * cf_star_interp_imag - cf_planet_real * cf_star_interp_real) * 2 * cf_star_interp_real / (norm**2) 
        dIm_star = (cf_planet_real * cf_star_interp_imag - cf_planet_imag * cf_star_interp_real) * 2 * cf_star_interp_imag / (norm**2)

        Cps_real_err = np.sqrt((dRe_planet * cf_real_planet_err)**2 +
            (dIm_planet * cf_imag_planet_err)**2 +
            (dRe_star * cf_real_star_err)**2 +
            (dIm_star * cf_imag_star_err)**2)
        
        #dIm(Cps)/d... (imag)
        dRe_planet = -cf_star_interp_imag / norm
        dIm_planet = cf_star_interp_real / norm
        dRe_star = (cf_planet_real * cf_star_interp_real + cf_planet_imag * cf_star_interp_imag) * 2 * cf_star_interp_real / (norm**2) 
        dIm_star = -(cf_planet_real * cf_star_interp_imag + cf_planet_imag * cf_star_interp_real) * 2 * cf_star_interp_imag / (norm**2)

        Cps_imag_err = np.sqrt((dRe_planet * cf_real_planet_err)**2 +
            (dIm_planet * cf_imag_planet_err)**2 +
            (dRe_star * cf_real_star_err)**2 +
            (dIm_star * cf_imag_star_err)**2)


        # Cps_real_err = Cps * np.sqrt((cf_real_planet_err / np.real(cf_planet)) ** 2 \
        #                     + (cf_real_star_err / np.real(cf_star_interp)) ** 2) 
        # Cps_imag_err = Cps * np.sqrt((cf_imag_planet_err / np.imag(cf_planet)) ** 2 \
        #                     + (cf_imag_star_err / np.imag(cf_star_interp)) ** 2) 
        
        
        Cps_all[n_file]          = Cps
        Cps_real_err_all[n_file] = Cps_real_err
        Cps_imag_err_all[n_file] = Cps_imag_err

        # Compute SNR
        snr_real = np.abs(np.real(Cps) / Cps_real_err)
        snr_imag = np.abs(np.imag(Cps) / Cps_imag_err)
        snr_real_all[n_file] = snr_real
        snr_imag_all[n_file] = snr_imag

        # Save contrast and associated real/imag errors to a FITS file
        Cps_real = np.real(Cps)
        Cps_imag = np.imag(Cps)
        Cps_real_err = np.real(Cps_real_err)
        Cps_imag_err = np.real(Cps_imag_err)

        hdul = fits.HDUList()
        hdul.append(fits.PrimaryHDU())
        hdul.append(fits.ImageHDU(Cps_real.astype(np.float32), name='CPS_REAL'))
        hdul.append(fits.ImageHDU(Cps_imag.astype(np.float32), name='CPS_IMAG'))
        hdul.append(fits.ImageHDU(Cps_real_err.astype(np.float32), name='CPS_REAL_ERR'))
        hdul.append(fits.ImageHDU(Cps_imag_err.astype(np.float32), name='CPS_IMAG_ERR'))
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
            fig_Cps, ax_Cps = plt.subplots(6, 1, figsize=(12, 8), sharex=True)
            ax_Cps = ax_Cps.flatten()

            fig_snr, ax_snr = plt.subplots(6, 1, figsize=(12, 8))
            ax_snr = ax_snr.flatten()

            for i_base in range(6): 
                
                #Cps
                fig_Cps.text(0.5, 0.04, 'Wavelength (µm)', ha='center', va='bottom', fontsize=12)
                fig_Cps.text(0.06, 0.5, 'Real Contrast (p/s)', va='center', ha='right', rotation='vertical', fontsize=12)
                fig_Cps.suptitle('Planet-to-Star Contrast', fontsize=16)
                if i_base < 5:
                    # ax_Cps[i_base].set_xlabel('')      
                    ax_Cps[i_base].tick_params(labelbottom=False)  

                ax_Cps[i_base].plot(wl*1e6, np.real(Cps_all[n_file,i_base]), color='crimson', alpha=0.8)
                ax_Cps[i_base].fill_between(wl*1e6, np.real(Cps_all[n_file,i_base])-Cps_real_err_all[n_file, i_base], 
                                        np.real(Cps_all[n_file,i_base])+Cps_real_err_all[n_file, i_base], alpha=0.2, color='hotpink')
                ax_Cps[i_base].set_ylim(-0.5e-1, 1e-1)
                # ax_Cps[i_base].set_title(f'Baseline {base_order_name[i_base]}', loc='left')
                ax_Cps[i_base].set_ylabel(f'{base_order_name[i_base]}')
                # ax[i_base].set_xlim
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
            plt.close(fig_Cps)
            plt.close(fig_snr)