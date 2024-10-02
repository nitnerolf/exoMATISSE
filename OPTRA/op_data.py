##############################################
# Load/save data
##############################################

import astropy.io.fits as fits

def op_load_rawdata(filename):
    fh = fits.open(filename)
    return fh

