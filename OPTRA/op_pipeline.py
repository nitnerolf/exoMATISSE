'''
 # @ Create Time: 2024-12-13 16:05:00
 # @ Author: fmillour
 '''

########################################################
# Pipeline functions for OPTRA
#       __           __           __           __
#     .'  `.       .'  `.       .'  `.       .'  `.
#    /      \     /      \     /      \     /      \
# _.'        `._.'        `._.'        `._.'        `._
#
# F. Millour, 2024
########################################################

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

########################################################

def op_sort_files(data_dir):
    files = os.listdir(data_dir)
    fitsfiles = [f for f in files if ".fits" in f]
    
    # select  fits files that correspond to observations
    data= {}
    data['obs']     = []
    data['obs_MJD'] = []
    data['sky']     = []
    data['sky_MJD'] = []
    data['dark']    = []
    data['dark_MJD']= []

    for fi in fitsfiles:
        #print(fi)
        fh = fits.open(basedir+fi)
        #op_print_fits_header(fh)
        hdr = fh[0].header
        fh.close()
        catg = hdr['ESO DPR CATG']
        type = hdr['ESO DPR TYPE']
        mjd  = hdr['MJD-OBS']
        #print(fi, inst, catg, type, chip, dit, ndit)
        if catg == 'SCIENCE' and type == 'OBJECT':
            #print("science file!")
            data['obs'].append(fi)
            data['obs_MJD'].append(mjd)
        if catg == 'CALIB' and type == 'STD':
            #print("calibrator file!")
            data['sky'].append(fi)
            data['sky_MJD'].append(hdr['MJD-OBS'])
        if catg == 'CALIB' and type == 'SKY' :
            #print("sky file!")
            data['sky'].append(fi)
            data['sky_MJD'].append(hdr['MJD-OBS'])

    print('Starfiles:', data['obs'])
    print('Skyfiles:', data['sky'])

    return data

########################################################

def op_assign_sky():
    """
    Assign sky to a given target
    """
    pass

########################################################

def op_assign_bias():
    """
    Assign bias to a given target
    """
    pass

########################################################

def op_assign_flat():
    """
    Assign flat to a given target
    """
    pass

########################################################

def op_assign_bpm():
    """
    Assign bad pixel map to a given target
    """
    pass

########################################################

