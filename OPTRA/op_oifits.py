#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################################
#
# OIFITS handling methods
# Author: fmillour
# Date: 22/11/2024
# Project: OPTRA
#
################################################################################

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u

##############################################
# 
def op_gen_oiarray(cfdata, verbose=True, plot=False):
    print('Generating OI_ARRAY...')
    """
    Save the array information in OIFITS format
    """
    # Create the OI_ARRAY table
    oiarray_table = Table()
    '''
    oiarray_table['ARRNAME'] = cfdata['hdr']['TELESCOP']
    oiarray_table['STATION'] = np.array([''], dtype='S16')
    '''
    
    #oiarray_table.meta['NAME']  = 'OI_ARRAY'
    oiarray_table.meta['EXTNAME'] = 'OI_ARRAY'
    oiarray_table.meta['EXTVER']  = 1
    oiarray_table['TEL_NAME']     = cfdata['OI_ARRAY']['TEL_NAME']
    oiarray_table['STA_NAME']     = cfdata['OI_ARRAY']['STA_NAME']
    oiarray_table['STA_INDEX']    = cfdata['OI_ARRAY']['STA_INDEX']
    oiarray_table['DIAMETER']     = cfdata['OI_ARRAY']['DIAMETER']
    oiarray_table['STAXYZ']       = cfdata['OI_ARRAY']['STAXYZ']
    oiarray_table['FOV']          = 1.0
    oiarray_table['FOVTYPE']      = 'RADIUS'
    
    return oiarray_table

##############################################
# 
def op_gen_oitarget(cfdata, verbose=True, plot=False):
    print('Generating OI_TARGET...')
    """
    Save the target information in OIFITS format
    """
    # Create the OI_TARGET table
    oitarget_table = Table()
    #oitarget_table.meta['NAME']  = 'OI_TARGET'
    oitarget_table.meta['EXTNAME']  = 'OI_TARGET'
    oitarget_table.meta['EXTVER']  = 1
    oitarget_table['TARGET_ID'] = [1]
    oitarget_table['TARGET']    = cfdata['hdr']['ESO OBS TARG NAME']
    oitarget_table['RAEP0']     = cfdata['hdr']['RA']
    oitarget_table['DECEP0']    = cfdata['hdr']['DEC']
    oitarget_table['EQUINOX']   = cfdata['hdr']['EQUINOX']
    oitarget_table['RA_ERR']    = 0.0
    oitarget_table['DEC_ERR']   = 0.0
    oitarget_table['SYSVEL']    = cfdata['hdr']['RADECSYS']
    oitarget_table['VELTYP']    = 'HELIOCEN'
    oitarget_table['VELDEF']    = 'OPTICAL'
    oitarget_table['PMRA']      = 0.0
    oitarget_table['PMDEC']     = 0.0
    oitarget_table['PMRA_ERR']  = 0.0
    oitarget_table['PMDEC_ERR'] = 0.0
    oitarget_table['PARALLAX']  = 0.0
    oitarget_table['PARA_ERR']  = 0.0
    oitarget_table['SPECTYP']   = 'UNKNOWN'
    oitarget_table['CATEGORY']  = cfdata['hdr']['ESO DPR CATG']
       
    return oitarget_table

##############################################
# 
def op_gen_oiwavelength(cfdata, verbose=True, plot=False):
    print('Generating OI_WAVELENGTH...')
    """
    Save the wavelength information in OIFITS format
    """
    # Create the OI_WAVELENGTH table
    oiwavelength_table = Table()
    #oiwavelength_table.meta['NAME']  = 'OI_WAVELENGTH'
    oiwavelength_table.meta['EXTNAME']  = 'OI_WAVELENGTH'
    oiwavelength_table.meta['EXTVER']  = 1
    oiwavelength_table['EFF_WAVE'] = cfdata['OI_WAVELENGTH']['EFF_WAVE_Binned']
    print('Shape of EFF_WAVE:', oiwavelength_table['EFF_WAVE'].shape)
    
    oiwavelength_table['EFF_BAND']  = 0.0
    oiwavelength_table['EFF_REF']   = cfdata['OI_WAVELENGTH']['EFF_REF']
    oiwavelength_table['BANDWIDTH'] = 0.0
    oiwavelength_table['FOV']       = 0.0
    oiwavelength_table['FOVTYPE']   = 'RADIUS'
    
    return oiwavelength_table

##############################################
# 
def op_gen_oivis(cfdata, cfin='CF_achr_phase_corr', verbose=True, plot=False):
    print('Generating OI_VIS...')
    """
    Save the complex visibility in OIFITS format
    """
    complexvis = cfdata['CF'][cfin][1:,...]
    print('Shape of complexvis:', complexvis.shape)
    complexvis2 = np.reshape(np.swapaxes(complexvis, 0,1), (complexvis.shape[0]* complexvis.shape[1],complexvis.shape[2]))
    nbases    = complexvis.shape[0]
    nframes   = complexvis.shape[1]
    # Create the OI_VIS table
    oivis_table = Table()
    #oivis_table.meta['NAME']  = 'OI_VIS'
    oivis_table.meta['EXTNAME']  = 'OI_VIS'
    oivis_table.meta['EXTVER']  = 1
    oivis_table['TARGET_ID'] = np.repeat(cfdata['OI_BASELINES']['TARGET_ID'], nbases)
    print('Shape of target_IDxxx:', oivis_table['TARGET_ID'].shape)
    oivis_table['TARGET']    = cfdata['hdr']['HIERARCH ESO OBS TARG NAME']
    oivis_table['TIME']      = np.repeat(cfdata['OI_BASELINES']['TIME'], nbases)
    oivis_table['MJD']       = np.repeat(cfdata['OI_BASELINES']['MJD'],  nbases)
    oivis_table['INT_TIME']  = np.repeat(cfdata['OI_BASELINES']['INT_TIME'], nbases)
    #print('Shape of complexvisxxx:', complexvis2.shape)
    oivis_table['VISAMP']    = np.abs(complexvis2)
    oivis_table['VISAMPERR'] = 0 #cfdata['OI_BASELINES']['VISAMPERR']
    oivis_table['VISPHI']    = np.angle(complexvis2)
    oivis_table['VISPHIERR'] = 0 #cfdata['OI_BASELINES']['VISAMPERR']
    if np.shape(cfdata['OI_BASELINES']['UCOORD'])[0] == nbases*nframes :
        oivis_table['UCOORD']    = cfdata['OI_BASELINES']['UCOORD']
        oivis_table['VCOORD']    = cfdata['OI_BASELINES']['VCOORD']
    else:
        oivis_table['UCOORD']    = np.repeat(cfdata['OI_BASELINES']['UCOORD'],nframes)
        oivis_table['VCOORD']    = np.repeat(cfdata['OI_BASELINES']['VCOORD'],nframes)
    #print('Shape of STA_INDEX:', np.tile(np.array(cfdata['OI_BASELINES']['STA_INDEX']), (nframes,1)).shape)
    oivis_table['STA_INDEX'] = np.tile(cfdata['OI_BASELINES']['STA_INDEX'], (nframes,1))
    oivis_table['FLAG']      = 0
    
    return oivis_table

##############################################
# 
def op_gen_oivis2(cfdata, v2in='simplevis2', verbose=True, plot=False):
    print('Generating OI_VIS2...')
    """
    Save the v squared in OIFITS format
    """
    vis2in = cfdata['VIS2'][v2in][1:,...]
    print('Shape of vis2:', vis2in.shape)
    #vis2in2 = np.reshape(np.swapaxes(vis2in, 0,1), (vis2in.shape[0]* vis2in.shape[1],vis2in.shape[2]))
    nbases    = vis2in.shape[0]
    nframes   = vis2in.shape[1]
    # Create the OI_VIS table
    oivis_table = Table()
    #oivis_table.meta['NAME']  = 'OI_VIS'
    oivis_table.meta['EXTNAME']  = 'OI_VIS2 '
    oivis_table.meta['EXTVER']  = 1
    oivis_table['TARGET_ID'] = np.repeat(cfdata['OI_BASELINES']['TARGET_ID'], nbases)
    print('Shape of target_IDxxx:', oivis_table['TARGET_ID'].shape)
    oivis_table['TARGET']    = cfdata['hdr']['HIERARCH ESO OBS TARG NAME']
    oivis_table['TIME']      = np.repeat(cfdata['OI_BASELINES']['TIME'], nbases)
    oivis_table['MJD']       = np.repeat(cfdata['OI_BASELINES']['MJD'],  nbases)
    oivis_table['INT_TIME']  = np.repeat(cfdata['OI_BASELINES']['INT_TIME'], nbases)
    #print('Shape of complexvisxxx:', complexvis2.shape)
    oivis_table['VIS2DATA']  = cfdata['VIS2'][v2in]
    oivis_table['VIS2ERR']   = 0.0
    oivis_table['UCOORD']    = 0.0
    oivis_table['VCOORD']    = 0.0
    #print('Shape of STA_INDEX:', np.tile(np.array(cfdata['OI_BASELINES']['STA_INDEX']), (nframes,1)).shape)
    oivis_table['STA_INDEX'] = np.tile(cfdata['OI_BASELINES']['STA_INDEX'], (nframes,1))
    oivis_table['FLAG']      = 0
    
    return oivis2_table

##############################################
# 
def op_write_oifits(filename, hdr, oiwavelength, oirray=None, oitarget=None, oivis=None, oivis2=None, oit3=None):
    print(f'Writing OI fits {filename}...')
    """
    Write the OIFITS file
    """
    # Create the OIFITS file
    oifits = fits.HDUList()
    oifits.append(fits.PrimaryHDU())
    oifits[0].header = hdr
    
    # Create the OI_ARRAY table
    if oirray is not None:
        oifits.append(fits.BinTableHDU(oirray))
    
    # Create the OI_TARGET table
    if oitarget is not None:
        oifits.append(fits.BinTableHDU(oitarget))
    
    # Create the OI_wavelength table
    oifits.append(fits.BinTableHDU(oiwavelength))
    
    # A minimum of one of the three tables must be provided
    if oivis is None and oivis2 is None and oit3 is None:
        raise ValueError("At least one of OI_VIS, OI_VIS2, or OI_T3 tables must be provided")
    
    # Create the OI_VIS table
    if oivis is not None:
        oifits.append(fits.BinTableHDU(oivis))
    
    # Create the OI_VIS2 table
    if oivis2 is not None:
        oifits.append(fits.BinTableHDU(oivis2))
    
    # Create the OI_T3 table
    if oit3 is not None:
        oifits.append(fits.BinTableHDU(oit3))
    
    # Write the OIFITS file
    oifits.writeto(filename, overwrite=True)
    
    print(f'Done!')
    
    return

##############################################
# 
def op_read_oifits(filename):
    fh = fits.open(filename)
    return fh

##############################################
# 
def op_read_oifits_sequence(basedir, filelist):
    
    hdus = []
    vis  = []
    vis2  = []
    dit = []
    for ifile, file in enumerate(filelist):
        print('reading file: ', file)
        
        ihdu = op_read_oifits(basedir + file)
        
        if ifile == 0:
            wlen = ihdu['OI_WAVELENGTH'].data['EFF_WAVE']
            band = ihdu['OI_WAVELENGTH'].data['EFF_BAND']
        dit.append(ihdu[0].header['ESO DET SEQ1 DIT'])
        
        ivis    = ihdu['OI_VIS'].data['VISAMP'] * np.exp(1j * ihdu['OI_VIS'].data['VISPHI'])
        ivis2   = ihdu['OI_VIS2'].data['VIS2DATA']
        nbase   = 6;
        nframes = np.shape(ivis)[0]//nbase
        nwlen   = np.shape(ivis)[1]
        ivisr   = ivis.reshape((nframes,nbase,nwlen))
        ivis2r   = ivis2.reshape((nframes,nbase,nwlen))
        print('ivis shape: ', np.shape(ivisr))
        
        imedvis = np.median(np.abs(ivis), axis=-1)
        print('imedvis : ', imedvis)
        
        vis.append(ivisr)
        vis2.append(ivis2r)
        hdus.append(ihdu)    
        ihdu.close()
    #vis = np.array(vis)
    #dit = np.array(dit)
    
    return hdus, vis, vis2, wlen, band, dit