#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################################
#
# Example script to process MATISSE data of beta Pic b with GPAO
# Author: fmillour
# Date: 18/11/2024
# Project: OPTRA
# 
################################################################################

from op_pipeline   import *
from op_corrflux   import *
from op_rawdata    import *
from op_flux       import *
from op_vis        import *
from op_oifits     import *
import numpy as np
import matplotlib.pyplot as plt
from astropy.io    import fits
import os
from scipy.ndimage import median_filter
from scipy         import *
from scipy         import stats
from tqdm import tqdm

#plt.ion()
plot = True
verbose = False


bbasedir = os.path.expanduser('~/SynologyDrive/driveFlorentin/DATA/')
#bbasedir = os.path.expanduser('E:\SynologyDrive\dataExt4Tb/')
### MACAO data
basedir  = bbasedir+'beta_Pic_b/2022-11-08_Raw_MATISSE-LM_betaPic_b/'
#basedir  = bbasedir+'2023-02-03/'
### GPAO data
#basedir  = bbasedir+'2024-11-17_MATISSE_betaPic_b/'

outdir    = os.path.expanduser('~/beta_pic_b/')

########################################################
# Calibration files
#caldir = os.path.expanduser('~/driveFlorentin/DATA/CALIB2024/')
caldir = bbasedir+'/CALIB2024/'

ext = '.fits.gz'
kappafile = caldir+'KAPPA_MATRIX_L_MED'+ext
shiftfile = caldir+'SHIFT_L_MED'+ext
flatfile  = caldir+'FLATFIELD_L_SLOW'+ext
badfile   = caldir+'BADPIX_L_SLOW'+ext

colors = ['#7a0e04', '#7a4f04', '#6a7a04', '#317a04', '#047d6f', '#04477d', '#45077a']

if 1:
    # List all obs files and sky files in directory
    datfiles = op_sort_files(basedir)

    # Assign to each obs file a sky file
    data = op_assign_sky(datfiles)
        
    skyfiles  = data['matched_sky']
    starfiles = data['obs']
if 0:
    #starfiles = ["MATISSE_OBS_SIPHOT_LM_OBJECT_323_0001.fits.gz"]
    starfiles = ["MATISSE_OBS_SIPHOT_LM_STD_323_0001.fits.gz"]
    skyfiles  = ["MATISSE_OBS_SIPHOT_LM_SKY_323_0004.fits.gz"]
#starfiles = [f for f in starfiles if 'STD' in f]

#for i in range(len(starfiles)):
#    print('Starfile:', starfiles[i],'Skyfile:', skyfiles[i])

uCoord = []; vCoord = []

for ifile, obsfile in enumerate(starfiles):
    starfile = basedir + obsfile
    print('Processing file:', obsfile, 'number:', ifile+1, 'of', len(starfiles), '/ Associated sky', skyfiles[ifile])
    if '_N_' in obsfile:
        continue # skip N band files
    skyfile = basedir + skyfiles[ifile]
    #print('Star file:', os.path.basename(starfiles[ifile]), ' Sky file:', os.path.basename(skyfile))
    u=[];v=[]
    ##########################################################
    op_compute_oifits(starfile, skyfile, badfile, flatfile, shiftfile, plot=plot)

    # bdata = op_loadAndCal_rawdata(basedir + obsfile, skyfile, badfile, flatfile, verbose=verbose, plot=plotRaw)
    
    # ##########################################################

    # cfdata = op_get_corrflux(bdata, shiftfile, plot=plotCorr, verbose=verbose)

    # print('Shape of bdata:', bdata['INTERF']['data'].shape)

    # if plotFringes:
    #     fig, ax1 = plt.subplots(1, 1, figsize=(8, 4), sharex=True, sharey=True)
    #     # Plot the first frame of intf
    #     ax1.imshow(np.mean(bdata['INTERF']['data'], axis=0), cmap='viridis')
    #     ax1.set_title('average intf')

    #     plt.show()
        
    # wlen = cfdata['OI_WAVELENGTH']['EFF_WAVE']
    # #print(wlen)

    # #########################################################

    # basename = os.path.basename(obsfile)
    # basen    = os.path.splitext(basename)[0]
    # directory = cfdata['hdr']['DATE-OBS'].split('T')[0]+'_OIFITS/'
    # if not os.path.exists(outdir+directory):
    #     os.makedirs(outdir+directory)
    # chip = cfdata['hdr']['ESO DET CHIP NAME']
    # if 'HAWAII' in chip:
    #     band = 'L'
    # elif 'AQUARIUS' in chip:
    #     band = 'N'
    # basen = directory+cfdata['hdr']['INSTRUME'][0:3]    + '_' +\
    # cfdata['hdr']['DATE-OBS'].replace(':','-')          + '_' +\
    # cfdata['hdr']['ESO OBS TARG NAME'].replace(' ','_') + '_' +\
    # cfdata['hdr']['ESO DPR CATG'][0:3]                  + '_' +\
    # band                                                + '_' +\
    # cfdata['hdr']['ESO INS DIL ID']                     + '_' +\
    # cfdata['hdr']['ESO INS BCD1 ID']                          +\
    # cfdata['hdr']['ESO INS BCD2 ID']
        
    # #########################################################
    
    # if ifile == starfiles[0]:
    #     cfdata = op_compute_uv(cfdata,plotCoverage)
    # else: 
    #     cfdata = op_compute_uv(cfdata,False)
    
    # uCoord.append(cfdata['OI_BASELINES']['UCOORD'])
    # vCoord.append(cfdata['OI_BASELINES']['VCOORD'])
    
    # if plotCoverage:
    #     op_uv_coverage(uCoord, vCoord,cfdata)
    
    # # if ifile == planetfiles[0]:
    # #     f = [basedir + fi for fi in planetfiles]
    # #     op_uv_coverage(f,cfdata,frame, plotCoverage)
        
    # #########################################################
    # if 1:
    #     cfdata, vis2 = op_extract_simplevis2(cfdata, verbose=verbose, plot=plot)
            
    #     if plotDsp:
    #         #print(mask)
    #         #print(~mask)
    #         notvis2 = np.ma.masked_array(np.ma.getdata(vis2), ~mask)
    #         allvis2 = np.ma.getdata(vis2)
            
    #         #print('Shape of vis2:', vis2.shape)
    #         #print('Shape of notvis2:', notvis2.shape)

    #         fig0, ax0 = plt.subplots(7, 1, figsize=(8, 8), sharex=1, sharey=1)
    #         #print('Shape of ax1:', ax0.shape)
    #         for i in np.arange(7):
    #             #print('i:', i)
    #             ax0[i].plot(wlen, allvis2[i,:], color='lightgray')
    #             ax0[i].plot(wlen, vis2[i,:], color=colors[i])
    #             ax0[i].set_ylabel(f'vis2 {i}')

    #         #print('Basename of starfile:', basen)
    #         plt.suptitle(r'Visibility as a function of $\lambda$, {basen}')
    #         plt.xlim(np.min(wlen), np.max(wlen))
    #         plt.ylim(-0.1, 1.1)
    #         #print(os.path.expanduser(bbasedir+f'{basen}_vis2.png'))
    #         plt.savefig(os.path.expanduser(outdir+f'{basen}_vis2.png'))
    #         #plt.show()

    # #########################################################

    # #cfdem = op_demodulate(cfdata, wlen, verbose=True, plot=False)
    # cfdem = cfdata
    
    # #########################################################
    # outfilename = os.path.expanduser(outdir+f'{basen}_corrflux_oi.fits')
    # hdr = cfdem['hdr']
    # oiwavelength = op_gen_oiwavelength(cfdem, verbose=verbose, wlen_fin='EFF_WAVE')
    # oitarget     = op_gen_oitarget(cfdem, verbose=True, plot=False)
    # oirray       = op_gen_oiarray(cfdem, verbose=True, plot=False)
    # oivis        = op_gen_oivis(cfdem, cfin='CF_piston_corr', verbose=verbose, plot=False)
    # oivis2        = op_gen_oivis2(cfdem, v2in='simplevis2', verbose=verbose, plot=False)
    # op_write_oifits(outfilename, hdr, oiwavelength, oirray, oitarget, oivis, oivis2, oit3=None)
    
    # #########################################################
    
    # if plotCorr:
        
    #     #print('Shape of cfdata:', cfdem['CF']['CF_demod'].shape)
    #     cf = cfdem['CF']['CF_piston_corr']
    #     #cf   = cfdem['CF']['CF_Binned']
    #     nbframes = cf.shape[1]
    #     wlen = cfdata['OI_WAVELENGTH']['EFF_WAVE']
    #     #cf = cfdem['CF']['CF_demod']
    #     #cf = cfdem['CF']['CF_reord']
    #     sumcf = np.sum(cf, axis=1)
    #     avgcf = np.mean(cf, axis=1)
    #     #print('Shape of sumcf:', sumcf.shape)
    #     shp = sumcf.shape
    #     nbs = shp[0]
    
    #     fig1, ax1 = plt.subplots(nbs, 2, figsize=(8, 8), sharex=1, sharey=0)
    #     #print('Shape of ax1:', ax1.shape)
    #     for i in np.arange(nbs):
    #         #print('i:', i)
    #         ax1[i,0].plot(wlen,   np.abs(avgcf[i,:]), color=colors[i])
    #         for j in np.arange(nbframes):
    #             ax1[i,0].plot(wlen, np.abs(cf[i,j,:]), color=colors[i], alpha=0.1)
    #         if i == 0 and nbs == 7:
    #             ax1[i,0].set_ylabel(f'flux {i+1}')
    #         else:
    #             ax1[i,0].set_ylabel(f'corr. flux {i+1}')
    #         ax1[i,1].plot(wlen, np.angle(avgcf[i,:]), color=colors[i])
    #         for j in np.arange(nbframes):
    #             ax1[i,1].plot(wlen, np.angle(cf[i,j,:]), color=colors[i], alpha=0.1)
    #         ax1[i,1].set_ylabel(f'phase {i+1}')
    #     plt.suptitle('Sum CF data (1 exposure)')
    #     plt.tight_layout()
    #     plt.savefig(os.path.expanduser(outdir+f'{basen}_corrflux.png'))
    #     #plt.show()

    # if plotPist:
            
    #     data, OPD_list = op_get_piston_fft(cfdem, verbose=True, plot=plotPist, cfin='CF_piston_corr')

    #     data = op_corr_piston(data, verbose=False, plot=plotPist)
    #     fig, ax = plt.subplots(7, 2, figsize=(8, 8))
    #     fig.suptitle('Piston corrected phase')
    #     for i_base in range(7):
    #         for i_frame in range(6):
    #             ax[i_base,0].plot(wlen, np.angle(data['CF']['CF_Binned'][i_base, i_frame]), color=colors[i_base])
    #             ax[i_base,1].plot(wlen, np.angle(data['CF']['CF_piston_corr'][i_base, i_frame]), color=colors[i_base])
    #     plt.show()
    

    # break