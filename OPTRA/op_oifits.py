'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
OIFITS handling methods
Author: fmillour
Date: 22/11/2024
Project: OPTRA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u

def op_gen_oiarray(cfdata, verbose=True, plot=False):
    """
    Save the array information in OIFITS format
    """
    # Create the OI_ARRAY table
    oiarray_table = Table()
    '''
    oiarray_table['ARRNAME'] = cfdata['hdr']['TELESCOP']
    oiarray_table['STATION'] = np.array([''], dtype='S16')
    '''
    
    oiarray_table['TEL_NAME']  = cfdata['OI_ARRAY']['TEL_NAME']
    oiarray_table['STA_NAME']  = cfdata['OI_ARRAY']['STA_NAME']
    oiarray_table['STA_INDEX'] = cfdata['OI_ARRAY']['STA_INDEX']
    oiarray_table['DIAMETER']  = cfdata['OI_ARRAY']['DIAMETER']
    oiarray_table['STAXYZ']    = cfdata['OI_ARRAY']['STAXYZ']
    oiarray_table['FOV'] = 1.0
    oiarray_table['FOVTYPE'] = 'RADIUS'
    
    '''
    oiarray_table['FILT'] = np.array([''], dtype='S16')
    oiarray_table['CBEAM'] = np.array([''], dtype='S16')
    oiarray_table['NBEAM'] = np.array([''], dtype='S16')
    oiarray_table['IATM'] = np.array([''], dtype='S16')
    oiarray_table['ATMOBS'] = np.array([''], dtype='S16')
    oiarray_table['ATMOBSERR'] = 0.0
    oiarray_table['TUT1'] = 0.0
    oiarray_table['UT1UTC'] = 0.0
    oiarray_table['DUT1'] = 0.0
    oiarray_table['POLARX'] = 0.0
    oiarray_table['POLARY'] = 0.0
    oiarray_table['LONGITUD'] = cfdata['hdr']['HIERARCH ESO ISS GEOLON']
    oiarray_table['LATITUDE'] = cfdata['hdr']['HIERARCH ESO ISS GEOLAT']
    oiarray_table['ALTITUDE'] = cfdata['hdr']['HIERARCH ESO ISS GEOELEV']
    oiarray_table['OBSLOC'] = np.array([''], dtype='S16')
    oiarray_table['TELESCOP'] = np.array([''], dtype='S16')
    oiarray_table['ARRAYX'] = 0.0
    oiarray_table['ARRAYY'] = 0.0
    oiarray_table['ARRAYZ'] = 0.0
    oiarray_table['FRAME'] = np.array([''], dtype='S16')
    oiarray_table['ARRAYST'] =
    '''
    
    return oiarray_table

def op_gen_oitarget(cfdata, verbose=True, plot=False):
    """
    Save the target information in OIFITS format
    """
    # Create the OI_TARGET table
    oitarget_table = Table()
    oitarget_table['TARGET_ID'] = 1
    oitarget_table['TARGET'] = np.array(cfdata['hdr']['HIERARCH ESO TARG NAME'], dtype='S16')
    oitarget_table['RAEP0']  = cfdata['hdr']['RA']
    oitarget_table['DECEP0'] = cfdata['hdr']['DEC']
    oitarget_table['EQUINOX'] = cfdata['hdr']['EQUINOX']
    oitarget_table['RA_ERR'] = 0.0
    oitarget_table['DEC_ERR'] = 0.0
    oitarget_table['SYSVEL'] = cfdata['hdr']['RADECSYS']
    oitarget_table['VELTYP'] = 'HELIOCEN'
    oitarget_table['VELDEF'] = 'OPTICAL'
    oitarget_table['PMRA'] = 0.0
    oitarget_table['PMDEC'] = 0.0
    oitarget_table['PMRA_ERR'] = 0.0
    oitarget_table['PMDEC_ERR'] = 0.0
    oitarget_table['PARALLAX'] = 0.0
    oitarget_table['PARA_ERR'] = 0.0
    oitarget_table['SPECTYP']  = 'UNKNOWN'
    oitarget_table['CATEGORY'] = cfdata['hdr']['HIERARCH ESO TARG TYPE']
    '''
    oitarget_table['FLUX'] = cfdata['hdr']['HIERARCH ESO SEQ TARG FLUX L']
    oitarget_table['FLUXERR'] = 0.0
    oitarget_table['DIAM'] = 0.0
    oitarget_table['DIAMERR'] = 0.0
    oitarget_table['DISTORT'] = 0.0
    oitarget_table['GCAL_ID'] = np.array([''], dtype='S8')
    oitarget_table['CALSTAT'] = np.array([''], dtype='S8')
    '''
       
    return oitarget_table
    
def op_gen_oiwavelength(cfdata, verbose=True, plot=False):
    """
    Save the wavelength information in OIFITS format
    """
    # Create the OI_WAVELENGTH table
    oiwavelength_table = Table()
    oiwavelength_table['EFF_WAVE'] = cfdata['OI_WAVELENGTH']['EFF_WAVE']
    
    oiwavelength_table['EFF_BAND'] = 0.0
    oiwavelength_table['EFF_REF'] = 0.0
    oiwavelength_table['BANDWIDTH'] = 0.0
    oiwavelength_table['FOV'] = 0.0
    oiwavelength_table['FOVTYPE'] = 'RADIUS'
    
    return oiwavelength_table

def op_gen_oivis(cfdata, verbose=True, plot=False):
    """
    Save the complex visibility in OIFITS format
    """
    complexvis = cfdata['CF']['CF_demod'][1:,...]
    complexvis = np.reshape(complexvis, (complexvis.shape[0]* complexvis.shape[1],complexvis.shape[2]))
    nblines   = complexvis.shape[0]
    # Create the OI_VIS table
    oivis_table = Table()
    print('Shape of target_ID:', nblines)
    oivis_table['TARGET_ID'] = np.ones(nblines)
    oivis_table['TARGET']    = np.repeat(cfdata['hdr']['HIERARCH ESO OBS TARG NAME'], nblines)
    oivis_table['TIME']      = np.repeat(Time(cfdata['hdr']['MJD-OBS'], format='jd'), nblines)
    oivis_table['MJD']       = np.repeat(Time(cfdata['hdr']['MJD-OBS'], format='jd'), nblines)
    oivis_table['INT_TIME']  = np.repeat(cfdata['hdr']['HIERARCH ESO DET SEQ1 DIT'], nblines)
    print('Shape of complexvis:', complexvis.shape)
    oivis_table['VISAMP']    = np.abs(complexvis)
    oivis_table['VISAMPERR'] = 0.0
    oivis_table['VISPHI']    = np.angle(complexvis)
    oivis_table['VISPHIERR'] = 0.0
    oivis_table['UCOORD']    = 0.0
    oivis_table['VCOORD']    = 0.0
    oivis_table['STA_INDEX'] = np.array([1, 2], dtype=np.int32)
    oivis_table['FLAG']      = 0
    
    return oivis_table
    
def op_write_oifits(filename, hdr, oiwavelength, oirray=None, oitarget=None, oivis=None, oivis2=None, oit3=None):
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
    
    return