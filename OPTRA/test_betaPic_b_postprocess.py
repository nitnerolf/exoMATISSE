"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example script to post-process MATISSE data of beta Pic b
Author: fmillour
Date: 06/12/2024
Project: OPTRA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

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

bbasedir = os.path.expanduser('~/Documents/ExoMATISSE/beta_Pic_b/')
basedir  = bbasedir+'2022-11-09_OIFITS/'
basedir  = bbasedir+'2022-11-09_OIFITS/'
starfiles = os.listdir(basedir)
fitsfiles = [f for f in starfiles if "oi.fits" in f]
ININfiles = [f for f in starfiles if "ININ" in f]
print(ININfiles)

for ifile in ININfiles:
    print('reading file: ', basedir + ifile)
    fh = op_read_oifits(basedir + ifile)