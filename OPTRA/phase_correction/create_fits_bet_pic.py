#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 10:54:34 2025

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
import os
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
fitfilename = os.path.expanduser('~/Documents/exoMATISSE/OPTRA/phase_correction/fits_phase_bet_pic.fits') 

BBASEDIR = ['~/Documents/Planet/beta_pic_b/','~/Documents/Planet/beta_pic_c/'] 

BASEDIR_c = ['2023-11-29/','2023-12-28/']
BASEDIR_b = ['2022-11-08/']

############ calibration file ############
caldir    = '~/Documents/CALIB2024/'
kappafile = caldir+'KAPPA_MATRIX_L_MED.fits'
shiftfile = caldir+'SHIFT_L_MED.fits'
flatfile  = caldir+'FLATFIELD_L_SLOW.fits'
badfile   = caldir+'BADPIX_L_SLOW.fits'

planetfile = dict()
sky = dict()
skyMJD = dict()
for bbasedir in BBASEDIR:
    BASEDIR = BASEDIR_b
    if bbasedir[-2]=='c':
        BASEDIR = BASEDIR_c
    for basedir in BASEDIR:
        basedir = os.path.expanduser(bbasedir) + basedir
        obsfiles = os.listdir(basedir)
        fitsfiles = [f for f in obsfiles if ".fits" in f and not "M." in f and not ".Z" in f and not ".gz" in f]
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
                
            if catg == 'CALIB' and type == 'STD' and chip == 'HAWAII-2RG' :
                #print("calibrator file!")
                starfiles.append(fi)
                
            if catg == 'CALIB' and type == 'SKY' and chip == 'HAWAII-2RG' :
                #print("sky file!")
                skyfilesL.append(fi)
                skyfilesL_MJD.append(hdr['MJD-OBS'])
                
        skyfilesL_MJD = np.array(skyfilesL_MJD)
        planetfiles = sorted(planetfiles)
        # planetfiles = sorted(starfiles)
        if '2023-11-29/'  in basedir :
            planetfiles = [] #+[planetfiles[-1]]#+[planetfiles[8]] + #+[planetfiles[0]]+planetfiles[7:9]
        elif '2023-12-28/'  in  basedir:
            planetfiles =  planetfiles[-4:] #+ [planetfiles[0]]
        elif '2022-11-08/' in  basedir:
            planetfiles =planetfiles[8:12]+planetfiles[24:26]+[planetfiles[27]]+planetfiles[40:44]+planetfiles[-4:]

        planetfile[basedir] = planetfiles
        sky[basedir] = skyfilesL
        skyMJD[basedir] = skyfilesL_MJD

planetfiles = [item for sublist in planetfile.values() for item in sublist]

##################################### PROCESS STAR DATA ####################################

# Initialisation 
nfiles = len(planetfiles)
nbases = 6
nwlen = 1560
HUM  = np.zeros((nbases,nfiles)) 
PRES = np.zeros((nbases,nfiles)) 
TEMP = np.zeros((nbases,nfiles)) 
PATH = np.zeros((nbases,nfiles)) 
ERR = np.zeros((nbases,nfiles,nwlen)) 
AMP = np.zeros((nbases,nfiles,nwlen)) 
PHASE = np.zeros((nbases,nfiles,nwlen))

for ifile,file in enumerate(tqdm(planetfiles,desc='Traitement des fichiers')):
    
    # Find the associated basedir of the file
    basedir = next((k for k, v in planetfile.items() if file in v), None)
    
    skyfilesL = sky[basedir]
    skyfilesL_MJD = skyMJD[basedir]
    starfile = basedir + file
    fh = fits.open(starfile)
    hdr = fh[0].header
    fh.close()
    mjd_obs = hdr['MJD-OBS']
    u=[];v=[]
    
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
                if verbose:
                    print('Matching key:', key)
                imatch += 1
        if imatch == len(keys_to_match):
            print('Matching sky file:', skyfile)
            break
        

    
    ##########################################################

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
    if verbose:
        print(wlen)


    #########################################################
    # Get the peaks position
    peaks, peakswd = op_get_peaks_position(bdata, verbose=verbose)
    if verbose:
        print(wlen)

    #########################################################
    # Extract the correlated flux
    bdata = op_extract_CF(bdata, peaks, peakswd, verbose=verbose)
    if verbose:
        print(wlen)
        
    #########################################################
    # Demodulate MATISSE fringes
    bdata = op_demodulate(bdata, verbose=verbose, plot=plot)
    
    #########################################################
    # Reorder baselines
    bdata, cfdata_reordered = op_reorder_baselines(bdata)
    
    #########################################################
    # Get the ambient conditions
    Temp, Pres, hum, dPath    = op_get_amb_conditions(bdata,verbose = verbose)
    
    #########################################################
    # Compute the mean and store the error.
    cfin = 'CF_reord'
    bdata=op_get_error_vis(bdata,cfin=cfin,plot = plot)
    mean_cf = op_mean_corrflux(bdata,cfin = cfin)
    
    
    #########################################################
    # Store values
    cvis = bdata['CF'][cfin]
    nframes = len(cvis[0])
    TEMP[:,ifile]  = Temp
    PRES[:,ifile]  = Pres
    HUM[:,ifile]   = hum
    PATH[:,ifile]  = np.mean(dPath,axis = -1)
    AMP[:,ifile]   = np.abs(mean_cf[1:])
    PHASE[:,ifile] = np.angle(mean_cf[1:]) 
    ERR[:,ifile]   = np.mean(bdata['OI_BASELINES']['VISPHIERR'].reshape(nframes,6,len(wlen)),axis=0)

  

##################################### FITS WRITER ####################################  
   
fits_table = Table()
fits_table.meta['EXTNAME']  = 'OI_VIS'
fits_table.meta['EXTVER']  = 1
fits_table.meta['NFILES']  = nfiles


fits_table['PATH']  = PATH
fits_table['PHASE'] = PHASE
fits_table['AMP']   = AMP
fits_table['ERR']   = ERR
fits_table['HUM']   = HUM
fits_table['TEMP']  = TEMP
fits_table['PRES']  = PRES
fits_table['WLEN']  = np.broadcast_to(wlen,(nbases,nfiles,len(wlen)))

oifits = fits.HDUList()
oifits.append(fits.PrimaryHDU())
oifits[0].header = hdr
oifits.append(fits.BinTableHDU(fits_table))
oifits.writeto(fitfilename, overwrite=True)