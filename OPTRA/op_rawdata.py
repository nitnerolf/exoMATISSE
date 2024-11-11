##############################################
# Cosmetics for data
##############################################
from   astropy.io import fits
from   scipy.ndimage import median_filter
from   scipy import *
import numpy as np
import fnmatch
import matplotlib.pyplot as plt

##############################################
# Function to interpolate bad pixels
def op_interpolate_bad_pixels(data, bad_pixel_map, verbose=False):
    if verbose:
        print('Interpolating bad pixels...')
    # Apply a median filter to the data
    filtered_data = median_filter(data, size=3)
    #plt.imshow(filtered_data, cmap='gray')
    # Replace bad pixels with the median filtered values
    data[bad_pixel_map] = filtered_data[bad_pixel_map]
    return data, filtered_data

##############################################
# Load bad pixel map
def op_load_bpm(filename, verbose=True):
    if verbose:
        print('Loading bad pixel map...')
    fh = fits.open(filename)
    bpm = fh[0].data.astype(bool)
    fh.close()
    return bpm

##############################################
# Apply bpm
def op_apply_bpm(rawdata, bpmap, verbose=True):
    if verbose:
        print('Applying bad pixel map...')
    # Subtract sky from rawdata
    
    corner = rawdata['INTERF']['corner']
    naxis  = rawdata['INTERF']['naxis']
    intf  = rawdata['INTERF']['data']
    nframe = np.shape(intf)[0]
    if verbose:
        print(f'Processing {nframe} frames')
    wbpm = bpmap[corner[1]-1:corner[1]+naxis[1]-1, corner[0]-1:corner[0]+naxis[0]-1]
    # Interpolate bad pixels in each frame
    fdata = []
    for i in range(nframe):
        intf[i],filtdata = op_interpolate_bad_pixels(intf[i], wbpm)
        fdata.append(filtdata)
    rawdata['INTERF']['data'] = intf
    
    for key in rawdata['PHOT']:
        corner = rawdata['PHOT'][key]['corner']
        naxis  = rawdata['PHOT'][key]['naxis']
        phot  = rawdata['PHOT'][key]['data']
        wbpm = bpmap[corner[1]-1:corner[1]+naxis[1]-1, corner[0]-1:corner[0]+naxis[0]-1]
        # Interpolate bad pixels in each frame
        fdata = []
        for i in range(nframe):
            phot[i],filtdata = op_interpolate_bad_pixels(phot[i], wbpm)
            fdata.append(filtdata)
        rawdata['PHOT'][key]['data'] = phot
        
    for key in rawdata['OTHER']:
        corner = rawdata['OTHER'][key]['corner']
        naxis  = rawdata['OTHER'][key]['naxis']
        other  = rawdata['OTHER'][key]['data']
        wbpm = bpmap[corner[1]-1:corner[1]+naxis[1]-1, corner[0]-1:corner[0]+naxis[0]-1]
        # Interpolate bad pixels in each frame
        fdata = []
        for i in range(nframe):
            other[i],filtdata = op_interpolate_bad_pixels(other[i], wbpm)
            fdata.append(filtdata)
        rawdata['OTHER'][key]['data'] = other
    return rawdata

##############################################
# Load flat field map
def op_load_ffm(filename, verbose=True):
    if verbose:
        print('Loading flat field...')
    fh = fits.open(filename)
    ffm = fh[0].data.astype(float)
    fh.close()
    return ffm

##############################################
# Apply flat field map
def op_apply_ffm(rawdata, ffmap, verbose=True):
    if verbose:
        print('Applying flat field map...')
    # Subtract sky from rawdata
    
    corner = rawdata['INTERF']['corner']
    naxis  = rawdata['INTERF']['naxis']
    intf   = rawdata['INTERF']['data']
    
    nframe = np.shape(intf)[0]
    if verbose:
        print(f'Processing {nframe} frames')
        
    wffm = ffmap[corner[1]-1:corner[1]+naxis[1]-1, corner[0]-1:corner[0]+naxis[0]-1]
    # Interpolate bad pixels in each frame
    
    for i in range(nframe):
        intf[i] /= wffm
    rawdata['INTERF']['data'] = intf
    
    for key in rawdata['PHOT']:
        corner = rawdata['PHOT'][key]['corner']
        naxis  = rawdata['PHOT'][key]['naxis']
        phot   = rawdata['PHOT'][key]['data']
        wffm   = ffmap[corner[1]-1:corner[1]+naxis[1]-1, corner[0]-1:corner[0]+naxis[0]-1]
        # Interpolate bad pixels in each frame
        
        for i in range(nframe):
            phot[i] /= wffm
        rawdata['PHOT'][key]['data'] = phot
        
    for key in rawdata['OTHER']:
        corner = rawdata['OTHER'][key]['corner']
        naxis  = rawdata['OTHER'][key]['naxis']
        other  = rawdata['OTHER'][key]['data']
        wffm   = ffmap[corner[1]-1:corner[1]+naxis[1]-1, corner[0]-1:corner[0]+naxis[0]-1]
        # correct flat field
        for i in range(nframe):
            other[i] /= wffm
        rawdata['OTHER'][key]['data'] = other
    return rawdata

##############################################
# Subtract sky
def op_subtract_sky(rawdata, skydata, verbose=True):
    if verbose:
        print('Subtracting sky...')
    # Compute robust average of sky
    skydata['INTERF']['data'] = stats.trim_mean(skydata['INTERF']['data'], 0.05, axis=0)
    #print(skydata['INTERF']['data'].shape)
    #print(len(skydata['PHOT']))
    for key in skydata['PHOT']:
        skydata['PHOT'][key]['data'] = stats.trim_mean(skydata['PHOT'][key]['data'], 0.05, axis=0)
        
    # Subtract sky from rawdata
    rawdata['INTERF']['data'] -= skydata['INTERF']['data']
    for key in skydata['PHOT']:
        rawdata['PHOT'][key]['data'] -= skydata['PHOT'][key]['data']
    return rawdata
    
##############################################
# Display the structure of a FITS file
def op_print_fits_structure(fits_data):
    for hdu in fits_data:
        print(f'-------\nHDU: {hdu.name}')
        #print(f'Header:\n{hdu.header}')
        if hdu.data is not None:
            if hdu.is_image:
                print('This HDU contains image data.')
            else:
                print('This is a table.')
            if isinstance(hdu.data, np.recarray):
                print(f'Columns: {hdu.data.dtype.names}')
            print(f'Data shape: {hdu.data.shape}')
        #print('\n')

##############################################
# Load raw data
def op_load_rawdata(filename, verbose=True):
    if verbose:
        print('Loading raw data...')
    fh      =  fits.open(filename)
    data    = {'hdr': fh[0].header}
    nframes = len(fh['IMAGING_DATA'].data)
    nreg    = len(fh['IMAGING_DETECTOR'].data)
    
    data['PHOT'] = {}
    data['INTERF'] = {}
    data['OTHER'] = {}
    
    data['ARRAY_DESCRIPTION'] = fh['ARRAY_DESCRIPTION'].data
    data['ARRAY_GEOMETRY']    = fh['ARRAY_GEOMETRY'].data
    data['OPTICAL_TRAIN']     = fh['OPTICAL_TRAIN'].data
    
    # Load the local OPD table that contains the modulation information
    localopd = []
    for i in np.arange(nframes):
        localopd.append(fh['IMAGING_DATA'].data[i]['LOCALOPD'].astype(float))
    localopd = np.array(localopd) 
    print('Localopd:', localopd)
    
    for j in np.arange(nreg):
        corner = fh['IMAGING_DETECTOR'].data[j]['CORNER']
        naxis  = fh['IMAGING_DETECTOR'].data[j]['NAXIS']
        #print(f'Processing region {j}:{fh['IMAGING_DETECTOR'].data['REGNAME'][j]}')
        datarray = []
        for i in np.arange(nframes):
            datarray.append(fh['IMAGING_DATA'].data[i][j+1].astype(float))
        if fnmatch.fnmatch(fh['IMAGING_DETECTOR'].data['REGNAME'][j], 'INTERF*'):
            data['INTERF']['data']     = datarray
            data['INTERF']['corner']   = corner
            data['INTERF']['naxis']    = naxis
            data['INTERF']['localopd'] = localopd
        elif fnmatch.fnmatch(fh['IMAGING_DETECTOR'].data['REGNAME'][j], 'PHOT*'):
            key = fh['IMAGING_DETECTOR'].data['REGNAME'][j]
            data['PHOT'][key]={}
            data['PHOT'][key]['data']   = datarray
            data['PHOT'][key]['corner'] = corner
            data['PHOT'][key]['naxis']  = naxis
        else:
            key = fh['IMAGING_DETECTOR'].data['REGNAME'][j].strip('\x001')
            data['OTHER'][key]={}
            data['OTHER'][key]['data']   = datarray
            data['OTHER'][key]['corner'] = corner
            data['OTHER'][key]['naxis']  = naxis
    fh.close()
    return data
