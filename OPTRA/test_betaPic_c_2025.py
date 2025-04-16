#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 16:42:16 2025

@author: nsaucourt
"""

##################################### IMPORT ####################################
from op_corrflux   import *
from op_rawdata    import *
from op_flux       import *
from op_vis        import *
from op_oifits     import *
import numpy as np
import matplotlib.pyplot as plt
from astropy.io    import fits
import os




##################################### Plot et verbose ####################################
plot=False
plotCorr=True
verbose=False

##################################### FILE OPENING ####################################
bbasedir = os.path.expanduser('~/Documents/Planet/beta_pic_c/')
basedir  = bbasedir+'2023-11-29/'

obsfiles = os.listdir(basedir)
fitsfiles = [f for f in obsfiles if ".fits" in f and not "M." in f]
starfiles=[]
planetfiles=[]
skyfilesL     = []
skyfilesL_MJD = []

for fi in fitsfiles:
    #print(fi)
    fh = fits.open(basedir+fi)
    #op_print_fits_header(fh)
    hdr = fh[0].header
    fh.close()
    instr = hdr['INSTRUME']
    catg = hdr['ESO DPR CATG']
    type = hdr['ESO DPR TYPE']
    try:
        chip = hdr['ESO DET CHIP NAME']
    except:
        print('No CHIP in header')
    try:
        dit  = hdr['ESO DET SEQ1 DIT']
    except:
        print('No DIT in header')
    try:
        ndit = hdr['ESO DET NDIT']
    except:
        print('No NDIT in header')
    
    
    if catg == 'SCIENCE' and type == 'OBJECT' and chip == 'HAWAII-2RG' :
        #print("science file!")
        planetfiles.append(fi)
        
    if catg == 'CALIB' and type == 'STD' and chip == 'HAWAII-2RG':
        #print("calibrator file!")
        starfiles.append(fi)# C'est une autre obs
        
    if catg == 'CALIB' and type == 'SKY' and chip == 'HAWAII-2RG' :
        #print("sky file!")
        skyfilesL.append(fi)
        skyfilesL_MJD.append(hdr['MJD-OBS'])
        
skyfilesL_MJD = np.array(skyfilesL_MJD)

planetfiles=sorted(planetfiles)


##################################### PROCESS STAR DATA ####################################

for ifile in planetfiles:
    starfile = basedir + ifile
    fh = fits.open(starfile)
    hdr = fh[0].header
    fh.close()
    mjd_obs = hdr['MJD-OBS']
    
    
    ############ Find Matching sky file ############
    mjdiff = np.abs(skyfilesL_MJD - mjd_obs)
    # Sort skyfiles by ascending distance to the starfile
    skyfilesL_sorted = [x for _,x in sorted(zip(mjdiff,skyfilesL))]
    
    for isky in skyfilesL_sorted:
        skyfile = basedir + isky
        fh = fits.open(skyfile)
        hdrsky = fh[0].header
        fh.close()
        
        keys_to_match = ['INSTRUME','ESO DET CHIP NAME','ESO DET SEQ1 DIT']
        imatch = 0
        for key in keys_to_match:
            if hdr[key] == hdrsky[key]:
                print('Matching key:', key)
                imatch += 1
        if imatch == len(keys_to_match):
            print('Matching sky file:', skyfile)
            break
        
    ############ calibration file ############
    caldir    = '~/Documents/CALIB2024/'
    kappafile = caldir+'KAPPA_MATRIX_L_MED.fits'
    shiftfile = caldir+'SHIFT_L_MED.fits'
    flatfile  = caldir+'FLATFIELD_L_SLOW.fits'
    badfile   = caldir+'BADPIX_L_SLOW.fits'
    
    ##########################################################

    bdata = op_loadAndCal_rawdata(starfile, skyfile, badfile, flatfile, verbose=verbose, plot=plot)

    ##########################################################

    cfdata = op_get_corrflux(bdata, shiftfile, plot=plot, verbose=verbose)

    ##########################################################
    
    basename = os.path.basename(starfile)
    basen    = os.path.splitext(basename)[0]
    
    directory = cfdata['hdr']['DATE-OBS'].split('T')[0]+'_OIFITS/'
    if not os.path.exists(bbasedir+directory):
        os.makedirs(bbasedir+directory)
    chip = cfdata['hdr']['ESO DET CHIP NAME']
    if 'HAWAII' in chip:
        band = 'L'
    elif 'AQUARIUS' in chip:
        band = 'N'
    basen = directory+cfdata['hdr']['INSTRUME'][0:3]    + '_' +\
            cfdata['hdr']['DATE-OBS'].replace(':','-')          + '_' +\
            cfdata['hdr']['ESO OBS TARG NAME'].replace(' ','_') + '_' +\
            cfdata['hdr']['ESO DPR CATG'][0:3]                  + '_' +\
            band                                                + '_' +\
            cfdata['hdr']['ESO INS DIL ID']                     + '_' +\
            cfdata['hdr']['ESO INS BCD1 ID']                          +\
            cfdata['hdr']['ESO INS BCD2 ID']
    
    #########################################################


    
    cf = cfdata['CF']['CF_Binned']
    wlen = cfdata['OI_WAVELENGTH']['EFF_WAVE_Binned']
    
    moycf=np.mean(cf,axis=1)
    nframe = cf.shape[1]
    nbase = moycf.shape[0]
    ymax= np.max(moycf[1:])
    ysmax=np.max(moycf[0])
    if plotCorr:
        fig1, ax1 = plt.subplots(nbase, 2, figsize=(8, 8), sharex=1, sharey=0)
        colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#A133FF', '#33FFF5', 'forestgreen']
        
        for i in range(nbase):
            ax1[i,0].plot(wlen,   np.abs(moycf[i,:]), color='black',alpha=0.95)
            if i == 0 and nbase == 7:
                ax1[i,0].set_ylabel(f'flux {i+1}')
                ax1[i,0].set_ylim(0,ysmax*1.10)
            else:
                ax1[i,0].set_ylabel(f'corr. flux {i+1}')
                ax1[i,0].set_ylim(0,ymax*1.15)
            ax1[i,1].plot(wlen, np.angle(moycf[i,:]), color='black',alpha=0.95)
            ax1[i,1].set_ylabel(f'phase {i+1}')
            ax1[i,1].set_ylim(-0.3,0.3)
            for j in range(nframe):
                ax1[i,0].plot(wlen,   np.abs(cf[i,j,:]), color=colors[i],alpha=0.2)
                ax1[i,1].plot(wlen, np.angle(cf[i,j,:]), color=colors[i],alpha=0.2)
        plt.suptitle('CF data and mean ')
        plt.tight_layout()
        plt.savefig(os.path.expanduser(bbasedir+f'{basen}_corrflux.png'))
        plt.show()
    
    
    
    #########################################################
    outfilename = os.path.expanduser(bbasedir+f'{basen}_corrflux_oi.fits')
    hdr = cfdata['hdr']
    oiwavelength = op_gen_oiwavelength(cfdata, verbose=verbose)
    oitarget     = op_gen_oitarget(cfdata, verbose=True, plot=plot)
    oirray       = op_gen_oiarray(cfdata, verbose=True, plot=plot)
    oivis        = op_gen_oivis(cfdata, cfin='CF_Binned', verbose=verbose, plot=plot)
    op_write_oifits(outfilename, hdr, oiwavelength, oirray, oitarget, oivis, oivis2=None, oit3=None)

    