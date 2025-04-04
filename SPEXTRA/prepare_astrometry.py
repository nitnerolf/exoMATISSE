#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Compute the stellar average per OB and the errors associated to each exposure

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from astropy.io import fits
from astropy.stats.circstats import circstd
from common_tools import reorder_baselines, wrap, get_DIT

plot_results = False
compute_covariance = False

DIT_planet = 10 #s

def compute_average_star_quantities(path_oifits, filelist_star):
    n_files = len(filelist_star)
    for i_file, file in enumerate(filelist_star):
        hdul = fits.open(path_oifits + file)
        hdul = reorder_baselines(hdul)
        DIT_star = get_DIT(hdul)
        if i_file == 0:
            # Initialize the concatenations
            n_wave = hdul['OI_WAVELENGTH'].data['EFF_WAVE'].size
            cf_amp_star_all     = np.zeros((n_files, 6, n_wave))
            cf_amp_star_err_all = np.zeros((n_files, 6, n_wave))
            cf_phi_star_all     = np.zeros((n_files, 6, n_wave))
            cf_phi_star_err_all = np.zeros((n_files, 6, n_wave))
            #t3_phi_star_all     = np.zeros((n_files, 4, n_wave))
            #t3_phi_star_err_all = np.zeros((n_files, 4, n_wave))
        cf_amp_star_all[i_file]     = hdul['OI_VIS'].data['VISAMP']
        cf_amp_star_err_all[i_file] = hdul['OI_VIS'].data['VISAMPERR']
        cf_phi_star_all[i_file]     = np.deg2rad(hdul['OI_VIS'].data['VISPHI'])
        cf_phi_star_err_all[i_file] = np.deg2rad(hdul['OI_VIS'].data['VISPHIERR'])
        #t3_phi_star_all[i_file]     = np.deg2rad(hdul['OI_T3'].data['T3PHI'])
        #t3_phi_star_err_all[i_file] = np.deg2rad(hdul['OI_T3'].data['T3PHIERR'])
        hdul.close()
    # Complexify
    cf_star_all          = cf_amp_star_all * np.exp(1j * cf_phi_star_all)
    print('Star coherent flux red factor:', (DIT_planet/DIT_star[0]))
    cf_star_all          = cf_star_all * (DIT_planet/DIT_star[0])
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
    # mean_t3_phi_star     = wrap(np.angle(np.mean(np.cos(t3_phi_star_all), axis=0) + 1j * np.mean(np.sin(t3_phi_star_all), axis=0)))
    # mean_t3_phi_star_err = np.sqrt(np.mean(t3_phi_star_err_all**2, axis=0))

    return mean_cf_amp_star, mean_cf_amp_star_err, \
           mean_cf_phi_star, mean_cf_phi_star_err, \
           #mean_t3_phi_star, mean_t3_phi_star_err


if __name__ == '__main__':

    path_oifits = sys.argv[1]
    path_output = sys.argv[2]

    # Create necessary directories
    if not os.path.isdir(path_output + '/stellar_OB_averages'):
        os.makedirs(path_output + '/stellar_OB_averages')
    if not os.path.isdir(path_output + '/stellar_OB_averages_plots'):
        os.makedirs(path_output + '/stellar_OB_averages_plots')
    if not os.path.isdir(path_output + '/error_estimates'):
        os.makedirs(path_output + '/error_estimates')
    if not os.path.isdir(path_output + '/error_estimates_plots'):
        os.makedirs(path_output + '/error_estimates_plots')

    # List the input OIFITS files
    
    files = sorted([file for file in os.listdir(path_oifits) if '.fits' in file and 'flagged' not in file])

    ### Compute the average stellar data per OB ###

    # Find all the available OB numbers
    #OB_list = set([int(file[file.find('OB')+2:file.find('_exp')]) for file in files])
    # OB_list_planet = set([int(file[file.find('OB')+2:file.find('_exp')]) for file in files if '_unknown' in file])
    OB_list_planet = set([int(file[file.find('OB')+2:file.find('_exp')]) for file in files if '_planet' in file])
    OB_list_star = set([int(file[file.find('OB')+2:file.find('_exp')]) for file in files if '_star' in file])
    n_OBs = len(OB_list_planet)+len(OB_list_star)

    OB_association = {key: value for key, value in zip(OB_list_planet, OB_list_star)}
    print(OB_list_planet, OB_list_star, OB_association)


    # Loop over the OBs
    for i_OB in OB_list_star:
        # Find the star OIFITS files of the given OB
        files_star_OB = sorted([file for file in files if f'OB{i_OB}' in file and 'star' in file and 'flagged' not in file])
        if files_star_OB == []:
            continue
        # Compute the average quantities
        # mean_cf_amp_star, mean_cf_amp_star_err, \
        # mean_cf_phi_star, mean_cf_phi_star_err \
        # mean_t3_phi_star, mean_t3_phi_star_err \
        #     = compute_average_star_quantities(path_oifits, files_star_OB)
        mean_cf_amp_star, mean_cf_amp_star_err, \
        mean_cf_phi_star, mean_cf_phi_star_err \
            = compute_average_star_quantities(path_oifits, files_star_OB)
        # Save them
        visamp_data = np.array([mean_cf_amp_star, mean_cf_amp_star_err])
        visphi_data = np.array([mean_cf_phi_star, mean_cf_phi_star_err])
        #t3phi_data = np.array([mean_t3_phi_star, mean_t3_phi_star_err])
        fits.writeto(path_output + f'/stellar_OB_averages/star_avg_visamp_OB{i_OB}.fits', visamp_data, overwrite=True)
        fits.writeto(path_output + f'/stellar_OB_averages/star_avg_visphi_OB{i_OB}.fits', visphi_data, overwrite=True)
        #fits.writeto(path_output + f'/stellar_OB_averages/star_avg_t3phi_OB{i_OB}.fits', t3phi_data, overwrite=True)
       
       # Plots
        if plot_results:

            with fits.open(path_oifits + files_star_OB[0]) as hdul:
                wl = hdul['OI_WAVELENGTH'].data['EFF_WAVE']

            fig_amp, axes_amp = plt.subplots(6, 1, figsize=(10, 8))
            axes_amp = axes_amp.flat
            fig_phi, axes_phi = plt.subplots(6, 1, figsize=(10, 8))
            axes_phi = axes_phi.flat

            for i in range(6):
                axes_amp[i].errorbar(wl, mean_cf_amp_star[i], yerr=mean_cf_amp_star_err[i])
                axes_phi[i].errorbar(wl, np.rad2deg(mean_cf_phi_star[i]), yerr=np.rad2deg(mean_cf_phi_star_err[i]))
                axes_amp[i].set_ylabel('Amplitude')
                axes_phi[i].set_ylabel('Phase [deg]')
                if i == 5:
                    axes_amp[i].set_xlabel('Wavelength [$\mu$m]')
                    axes_phi[i].set_xlabel('Wavelength [$\mu$m]')
                else:
                    axes_amp[i].set_xticklabels([])
                    axes_phi[i].set_xticklabels([])
                
            fig_amp.tight_layout()
            fig_phi.tight_layout()
            fig_amp.savefig(path_output + f'/stellar_OB_averages_plots/star_avg_visamp_OB{i_OB}.png')
            fig_phi.savefig(path_output + f'/stellar_OB_averages_plots/star_avg_visphi_OB{i_OB}.png')

            # fig_t3, axes_t3 = plt.subplots(4, 1, figsize=(8, 8))
            # axes_t3 = axes_t3.flat

            # for i in range(4):
            #     axes_t3[i].errorbar(wl, np.rad2deg(mean_t3_phi_star[i]), yerr=np.rad2deg(mean_t3_phi_star_err[i]))
            #     axes_t3[i].set_ylabel('Closure phase [deg]')
            #     if i == 3:
            #         axes_t3[i].set_xlabel('Wavelength [$\mu$m]')
            #     else:
            #         axes_t3[i].set_xticklabels([])

            # fig_t3.tight_layout()
            # fig_t3.savefig(path_output + f'/stellar_OB_averages_plots/star_avg_t3phi_OB{i_OB}.png')

            # plt.close('all')

    
    ### Compute the uncertainty estimates per exposure ###

    # Loop over the OBs
    for i_OB in OB_list_planet:
        # Search planet files
        # files_planet = sorted([file for file in os.listdir(path_oifits) if 'unknown' in file and 'flagged' not in file])
        files_planet = sorted([file for file in os.listdir(path_oifits) if 'planet' in file and 'flagged' not in file])
        # Look for the 1st frames of each exposure
        # files_planet_frame1 = sorted([file for file in os.listdir(path_oifits) if 'frame1_unknown' in file and 'flagged' not in file])
        files_planet_frame1 = sorted([file for file in os.listdir(path_oifits) if 'frame1_planet' in file and 'flagged' not in file]) 

        # Load the stellar average quantities of the OB
        cf_amp_star_data = fits.getdata(path_output + f'/stellar_OB_averages/star_avg_visamp_OB{OB_association[i_OB]}.fits')
        cf_amp_star = cf_amp_star_data[0]
        cf_phi_star_data = fits.getdata(path_output + f'/stellar_OB_averages/star_avg_visphi_OB{OB_association[i_OB]}.fits')
        cf_phi_star = cf_phi_star_data[0]
        # t3_phi_star_data = fits.getdata(path_output + f'/stellar_OB_averages/star_avg_t3phi_OB{i_OB}.fits')
        # t3_phi_star = t3_phi_star_data[0]

        # Loop over exposures
        for frame1 in files_planet_frame1:
            exposure_name = frame1[:frame1.find('_frame1')]
            # Search all the frames of the given exposure
            frame_files = [frame for frame in files if frame1[:frame1.find('_frame1')] in frame]

            # Loop over frames
            for i_frame, frame_file in enumerate(frame_files):
                hdul_frame = fits.open(path_oifits + frame_file)
                hdul_frame = reorder_baselines(hdul_frame)
                if i_frame == 0:
                    # Initialize the concatenations
                    n_wave = hdul_frame['OI_WAVELENGTH'].data['EFF_WAVE'].size
                    frame_cf_amps = np.zeros((len(frame_files), 6, n_wave))
                    frame_cf_phis = np.zeros((len(frame_files), 6, n_wave))
                    # frame_t3_phis = np.zeros((len(frame_files), 4, n_wave))
                frame_cf_amps[i_frame] = hdul_frame['OI_VIS'].data['VISAMP']
                frame_cf_phis[i_frame] = np.deg2rad(hdul_frame['OI_VIS'].data['VISPHI'])
                # frame_t3_phis[i_frame] = np.deg2rad(hdul_frame['OI_T3'].data['T3PHI'])
                hdul_frame.close()

            # Take the planet/star ratios and compute the standard deviation over the frames of the exposure
            cf_amp_cal_err = np.std(frame_cf_amps / cf_amp_star, axis=0, ddof=1)
            cf_phi_cal_err = np.std(wrap(frame_cf_phis - cf_phi_star), axis=0, ddof=1)
            # t3_phi_cal_err = np.std(wrap(frame_t3_phis - t3_phi_star), axis=0, ddof=1)

            cf_phi_cal_err_2 = circstd(frame_cf_phis - cf_phi_star, axis=0) * np.sqrt(frame_cf_phis.shape[0]/(frame_cf_phis.shape[0]-1))
            # t3_phi_cal_err_2 = circstd(frame_t3_phis - t3_phi_star, axis=0) * np.sqrt(frame_t3_phis.shape[0]/(frame_t3_phis.shape[0]-1))

            fits.writeto(path_output + f"/error_estimates/{exposure_name}_cf_amp_cal_err.fits", cf_amp_cal_err, overwrite=True)
            fits.writeto(path_output + f"/error_estimates/{exposure_name}_cf_phi_cal_err.fits", cf_phi_cal_err_2, overwrite=True)
            # fits.writeto(path_output + f"/error_estimates/{exposure_name}_t3_phi_cal_err.fits", t3_phi_cal_err_2, overwrite=True)

            # Compute the covariance
            if compute_covariance:
                cf_amp_cal_cov = np.zeros((6, n_wave, n_wave))
                cf_phi_cal_cov = np.zeros((6, n_wave, n_wave))
                # t3_phi_cal_cov = np.zeros((4, n_wave, n_wave))
                for i in range(6):
                    cf_amp_cal_cov[i] = np.cov((frame_cf_amps[:, i, :] / cf_amp_star[i]).T, ddof=1)
                    cf_phi_cal_cov[i] = np.cov((wrap(frame_cf_phis[:, i, :] - cf_phi_star[i])).T, ddof=1)
                # for i in range(4):
                    # t3_phi_cal_cov[i] = np.cov((wrap(frame_t3_phis[:, i, :] - t3_phi_star[i])).T, ddof=1)
                fits.writeto(path_output + f"/error_estimates/{exposure_name}_cf_amp_cal_cov.fits", cf_amp_cal_cov, overwrite=True)
                fits.writeto(path_output + f"/error_estimates/{exposure_name}_cf_phi_cal_cov.fits", cf_phi_cal_cov, overwrite=True)
                # fits.writeto(path_output + f"/error_estimates/{exposure_name}_t3_phi_cal_cov.fits", t3_phi_cal_cov, overwrite=True)    

            # Plots
            if plot_results:
                
                fig_amp, axes_amp = plt.subplots(6, 1, figsize=(10, 8))
                axes_amp = axes_amp.flat
                fig_phi, axes_phi = plt.subplots(6, 1, figsize=(10, 8))
                axes_phi = axes_phi.flat
                fig_amp_cov, axes_amp_cov = plt.subplots(3, 2, figsize=(10, 8))
                axes_amp_cov = axes_amp_cov.flat
                fig_phi_cov, axes_phi_cov = plt.subplots(3, 2, figsize=(10, 8))
                axes_phi_cov = axes_phi_cov.flat

                for i in range(6):
                    axes_amp[i].semilogy(wl, cf_amp_cal_err[i])
                    axes_phi[i].plot(wl, np.rad2deg(cf_phi_cal_err[i]))
                    axes_phi[i].plot(wl, np.rad2deg(cf_phi_cal_err_2[i]), c='r')
                    axes_amp[i].set_ylabel('Amplitude')
                    axes_phi[i].set_ylabel('Phase [deg]')
                    if i == 5:
                        axes_amp[i].set_xlabel('Wavelength [$\mu$m]')
                        axes_phi[i].set_xlabel('Wavelength [$\mu$m]')
                    else:
                        axes_amp[i].set_xticklabels([])
                        axes_phi[i].set_xticklabels([])

                fig_amp.tight_layout()
                fig_phi.tight_layout()
                fig_amp.savefig(path_output + f'/error_estimates_plots/{exposure_name}_cf_amp_cal_err.png')
                fig_phi.savefig(path_output + f'/error_estimates_plots/{exposure_name}_cf_phi_cal_err.png')

                if compute_covariance:
                    for i in range(6):
                        axes_amp_cov[i].imshow(cf_amp_cal_cov[i])
                        axes_phi_cov[i].imshow(cf_phi_cal_cov[i])
                    fig_amp_cov.tight_layout()
                    fig_phi_cov.tight_layout()
                    fig_amp_cov.savefig(path_output + f'/error_estimates_plots/{exposure_name}_cf_amp_cal_cov.png')
                    fig_phi_cov.savefig(path_output + f'/error_estimates_plots/{exposure_name}_cf_phi_cal_cov.png')

                # fig_t3, axes_t3 = plt.subplots(4, 1, figsize=(8, 8))
                # axes_t3 = axes_t3.flat
                # fig_t3_cov, axes_t3_cov = plt.subplots(2, 2, figsize=(8, 8))
                # axes_t3_cov = axes_t3_cov.flat

                # for i in range(4):
                #     axes_t3[i].plot(wl, np.rad2deg(t3_phi_cal_err[i]))
                #     axes_t3[i].plot(wl, np.rad2deg(t3_phi_cal_err_2[i]))
                #     axes_t3[i].set_ylabel('Closure phase [deg]')
                #     if i == 3:
                #         axes_t3[i].set_xlabel('Wavelength [$\mu$m]')
                #     else:
                #         axes_t3[i].set_xticklabels([])

                # fig_t3.tight_layout()
                # fig_t3.savefig(path_output + f'/error_estimates_plots/{exposure_name}_t3_phi_cal_err.png')

                # if compute_covariance:
                #     for i in range(4):
                #         axes_t3_cov[i].imshow(t3_phi_cal_cov[i])
                #     fig_t3_cov.tight_layout()
                #     fig_t3_cov.savefig(path_output + f'/error_estimates_plots/{exposure_name}_t3_phi_cal_cov.png')
                    
                # plt.close('all')