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
plot = False
verbose = False

# beta Pic b dataset
#bbasedir = os.path.expanduser('~/SynologyDrive/fmi/driveFlorentin/DATA/')
bbasedir = os.path.expanduser('/Users/fmillour/Desktop/')
### MACAO data
#basedir  = bbasedir+'2022-11-08_Raw_MATISSE-LM_betaPic_b/'
#basedir  = bbasedir+'2023-02-03/'
#basedir  = bbasedir+'2023-02-03_Raw_betaPic_b/'
### GPAO data
#basedir  = bbasedir+'2024-11-17_MATISSE_betaPic_b/'
outdir    = bbasedir+'/beta_pic_b/'


# # beta Pic c datasets. 2023 ones
# bbasedir = os.path.expanduser('~/SynologyDrive/driveFlorentin/DATA/beta_Pic_c/')
# basedir  = bbasedir+'2023-11-27_betaPic_c/'
#basedir  = bbasedir+'2023-11-28_betaPic_c/'
#basedir  = bbasedir+'2023-12-27_betaPic_c/'
basedir  = bbasedir+'2023-12-29_betaPic_c/'
# outdir = os.path.expandvars('$HOME/beta_pic_c_MACAO/')



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
