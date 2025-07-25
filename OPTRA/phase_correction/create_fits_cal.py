#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 14:54:44 2025

@author: nsaucourt
"""

##################################### IMPORT ####################################
import os
import sys

# --- Add parent folder temporarily for import ---
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from op_corrflux   import *
from op_rawdata    import *

import numpy as np
import matplotlib.pyplot as plt
from astropy.io    import fits
from astropy.table import Table
from tqdm import tqdm

sys.path.pop(0)

##################################### Plot et verbose ####################################
plot         = False
plotCorr     = False
plotCoverage = False
verbose      = False
plotSNR      = False
bindata      = False

##################################### FILE OPENING ####################################


####################### FILE'S NAME ########################

# calibration file 
caldir    = '~/Documents/CALIB2024/'
kappafile = caldir+'KAPPA_MATRIX_L_MED.fits'
shiftfile = caldir+'SHIFT_L_MED.fits'
flatfile  = caldir+'FLATFIELD_L_SLOW.fits'
badfile   = caldir+'BADPIX_L_SLOW.fits'

fitfilename = os.path.expanduser('~/Documents/exoMATISSE/OPTRA/phase_correction/fits_phase_CAL.fits')    
basedir = os.path.expanduser('~/Documents/Planet/CALIBRATEUR/')
obsfiles = os.listdir(basedir)
fitsfiles = [f for f in obsfiles if ".fits" in f and not "M." in f]
starfiles     = []
planetfiles   = []
skyfilesL     = []
skyfilesL_MJD = []

for fi in tqdm(fitsfiles,desc='Tri des fichiers'):
    with fits.open(basedir+fi) as fh:
        hdr = fh[0].header

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
        starfiles.append(fi)
        
    if catg == 'CALIB' and type == 'SKY' and chip == 'HAWAII-2RG' :
        #print("sky file!")
        skyfilesL.append(fi)
        skyfilesL_MJD.append(hdr['MJD-OBS'])
        
skyfilesL_MJD = np.array(skyfilesL_MJD)
starfiles = sorted(starfiles)
starfiles= starfiles



##################################### PROCESS STAR DATA ####################################


# Initialize data
nfiles = len(starfiles)
nbases = 6
nwlen = 1560
HUM  = np.zeros((nbases,nfiles)) 
PRES = np.zeros((nbases,nfiles)) 
TEMP = np.zeros((nbases,nfiles)) 
PATH = np.zeros((nbases,nfiles)) 
ERR = np.zeros((nbases,nfiles,nwlen)) 
AMP = np.zeros((nbases,nfiles,nwlen)) 
PHASE = np.zeros((nbases,nfiles,nwlen))


for ifile,file in enumerate(tqdm(starfiles,desc='Traitement des fichiers')):
    print(file)
    starfile = basedir + file
    with fits.open(starfile) as fh:
        hdr = fh[0].header

    ############ Find Matching sky file ############
    mjd_obs = hdr['MJD-OBS']
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
                if verbose:
                    print('Matching key:', key)
                imatch += 1
        if imatch == len(keys_to_match):
            print('Matching sky file:', skyfile)
            break
        

    
    ##########################################################
    # Calibrate raw data
    bdata = op_loadAndCal_rawdata(starfile, skyfile, badfile, flatfile, verbose=verbose, plot=plot)

    #########################################################
    # Apodization
    bdata = op_apodize(bdata, verbose=verbose, plot=plot)
    
    #########################################################
    #compute fft
    bdata = op_calc_fft(bdata,verbose = verbose)

    #########################################################
    # Get the wavelength
    wlen = op_get_wlen(shiftfile, bdata, verbose=verbose)

    #########################################################
    # Get the peaks position
    peaks, peakswd = op_get_peaks_position(bdata, verbose=verbose)

    #########################################################
    # Extract the correlated flux
    bdata = op_extract_CF(bdata, peaks, peakswd, verbose=verbose)

    #########################################################
    # Demodulate MATISSE fringes
    bdata = op_demodulate(bdata, verbose=verbose, plot=plot)
    
    #########################################################
    # Reorder baselines
    bdata, cfdata_reordered = op_reorder_baselines(bdata)
    
    #########################################################
    # Get the atmospheric conditions
    Temp, Pres, hum, dPath    = op_get_amb_conditions(bdata,verbose = verbose)
    
    
    #########################################################
    # Compute the mean and store the error.
    cfin = 'CF_reord'
    bdata=op_get_error_vis(bdata,cfin=cfin,plot = plot)
    mean_cf = op_mean_corrflux(bdata,cfin)

    #########################################################
    # Store values
    cvis = bdata['CF'][cfin]
    nframes = len(cvis[0])
    TEMP[:,ifile]    = Temp
    PRES[:,ifile]    = Pres
    HUM[:,ifile]     = hum
    PATH[:,ifile]    = np.mean(dPath,axis = -1)
    AMP[:,ifile,:]   = np.abs(mean_cf[1:])
    PHASE[:,ifile,:] = np.angle(mean_cf[1:]) 
    ERR[:,ifile,:]   = np.mean(bdata['OI_BASELINES']['VISPHIERR'].reshape(nframes,6,len(wlen)),axis=0)


    if np.max(1/np.mean(bdata['OI_BASELINES']['VISPHIERR'].reshape(nframes,6,len(wlen)),axis=0))<10:
        plt.title(file)
        plt.errorbar(wlen,np.angle(mean_cf[1:])[0],yerr = np.mean(bdata['OI_BASELINES']['VISPHIERR'].reshape(nframes,6,len(wlen)),axis=0)[0])
        plt.show()
        

  
##################################### FITS WRITER ####################################

fits_table = Table()
fits_table.meta['EXTNAME']  = 'OI_VIS'
fits_table.meta['EXTVER']  = 1
fits_table.meta['NFILES']  = nfiles

fits_table['PATH'] = PATH
fits_table['PHASE'] = PHASE
fits_table['ERR'] = ERR
fits_table['HUM'] = HUM
fits_table['TEMP'] = TEMP
fits_table['PRES'] = PRES
fits_table['WLEN'] = np.broadcast_to(wlen,(nbases,nfiles,len(wlen)))
fits_table['AMP'] =AMP
oifits = fits.HDUList()
oifits.append(fits.PrimaryHDU())
oifits[0].header = hdr
oifits.append(fits.BinTableHDU(fits_table))
oifits.writeto(fitfilename, overwrite=True)