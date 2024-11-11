##############################################
# Correlated flux computation
##############################################
from os import error
from astropy.io import fits
from scipy import *
import numpy as np
import astropy.constants as const
import matplotlib.pyplot as plt

##############################################
# Apodization function
def op_apodize(data, verbose=True,plot=False):
    if verbose:
        print('Apodizing data...')
    # Apply an apodizing window to the data
    nframes = np.shape(data['INTERF']['data'])[0]
    nwlen   = np.shape(data['INTERF']['data'])[1]
    nreg    = len(data['PHOT'])
    
    if verbose:
        print('computing apodizing window...')
    intf  = stats.trim_mean(data['INTERF']['data'],axis=0, proportiontocut=0.05)
    argmx = np.argmax(stats.trim_mean(intf, axis=0, proportiontocut=0.05))
    if verbose:
        print('argmx', argmx)
    n     = np.shape(intf)[1]
    if verbose:
        print('n', n)
    dx = int(n - 2*np.abs(n/2 - argmx))
    if verbose:
        print('dx', dx)
    wn   = signal.get_window(('kaiser', 14), dx)
    zewin = np.zeros(n)
    zewin[argmx-dx//2:argmx+dx//2] = wn
    centered_win_intf = zewin
    
    if plot:
        # Plot the Hanning window used for apodization
        plt.figure()
        plt.plot(centered_win_intf)
        plt.title('Window for Apodization')
        plt.xlabel('Pixel Index')
        plt.ylabel('Window Value')
        plt.grid(True)
        plt.show()
    
    data['INTERF']['center'] = argmx
    for i in np.arange(nframes):
        data['INTERF']['data'][i] *= centered_win_intf
        
    for key in data['PHOT']:
        pht   = stats.trim_mean(data['PHOT'][key]['data'],axis=0, proportiontocut=0.05)
        argmx = np.argmax(stats.trim_mean(pht, axis=0, proportiontocut=0.05))
        n     = np.shape(pht)[1]
        dx = int(n - 2*np.abs(n/2 - argmx))
        if verbose:
            print('dx', dx)
        wn   = signal.get_window(('kaiser', 14), dx)
        zewin = np.zeros(n)
        zewin[argmx-dx//2:argmx+dx//2] = wn
        centered_win_pht = zewin
        data['PHOT'][key]['center'] = argmx
        for i in np.arange(nframes):
            data['PHOT'][key]['data'][i] *= centered_win_pht
            
    return data

##############################################
# Function to compute the FFT of interferograms
def op_calc_fft(data, verbose=True):
    if verbose:
        print('Computing FFT of interferograms...')
    intf = data['INTERF']['data']
    nframe = np.shape(intf)[0]
    nwlen  = np.shape(intf)[1]
    npix   = np.shape(intf)[2]
    
    # Compute FFT 1D of intf along the pixels axis
    fft_intf = np.fft.fft(intf, axis=2)
    # Compute the phasor corresponding to the shift of the center of the window
    center_shift = data['INTERF']['center']
    phasor = np.exp(2j * np.pi * center_shift * (np.arange(npix)) / npix)
    fft_intf *= phasor[None,None,:]
    
    fft_intf_magnitude = np.abs(fft_intf)
    dsp_intf = fft_intf_magnitude**2
    sum_dsp_intf = np.sum(dsp_intf, axis=0)
    sdi_resh = np.fft.fftshift(sum_dsp_intf, axes=1)
    freqs = np.fft.fftfreq(npix)
    print('Shape of sum_dsp_intf:', sum_dsp_intf.shape)
    
    data['FFT'] = {'data': fft_intf, 'magnitude': fft_intf_magnitude, 'dsp': dsp_intf, 'sum_dsp': sum_dsp_intf, 'sdi': sdi_resh, 'freqs': freqs}
    return data

##############################################
# Function to compute the wavelength 
def op_get_wlen(shift_map, rawdata, verbose=True, plot=False):
    if verbose:
        print('Computing wavelength map...')
    # Compute wavelength map from shift map
    fh        = fits.open(shift_map)
    shift_map = fh['SHIFT_MAP'].data
    disp      = shift_map['DISP']
    if verbose:
        print('shape of disp:', np.shape(disp))
        print('disp:', disp)
        print('disp0:', disp[0])
        print('disp1:', disp[1])
    
    corner = rawdata['INTERF']['corner']
    px = corner[1]+np.arange(np.shape(rawdata['INTERF']['data'])[1])
    wlen = disp[0][0] + disp[0][1]*px + disp[0][2]*px**2
    
    if verbose:
        print(disp)
        print(wlen)
        
    if plot:
        plt.figure()
        plt.plot(wlen)
        plt.show()
    
    return wlen

##############################################
# Function to get the peaks position
def op_get_peaks_position(fftdata, wlen, instrument, verbose=True):
    npix = np.shape(fftdata['FFT']['data'])[2]
    if instrument == 'MATISSE':
        peaks = np.arange(7)
        interfringe = 17.88*2.75*2*0.85 # in D/lambda
        peakswd = 0.7
        pkwds   = np.ones_like(peaks) * peakswd
        peak    = peaks[:,None] * interfringe / wlen[None,:]
        peakwd  = pkwds[:,None] * interfringe / wlen[None,:]
    else:
        error('Instrument not recognized')
    
    if verbose:
        print('Shape of peak:', np.shape(peak))
        print('Peak:', peak)
    return peak, peakwd

##############################################
# Function to extract the correlated flux
def op_extract_CF(fftdata, wlen, peaks, peakswd, verbose=True, plot=False):
    if verbose:
        print('Extracting correlated flux...')
    bck = np.copy(fftdata['FFT']['data'])
    nfreq = np.shape(bck)[2]
    nfreq2 = int(nfreq/2)
    bck = bck[:,:,0:nfreq2]
    if verbose:
        print('Shape of bck:', np.shape(bck))
        print('nfreq:', nfreq)
    ifreq = np.arange(nfreq2)
    npeaks = np.shape(peaks)[0]
    ibase = np.arange(npeaks)
    FT = []
    CF = []
    NIZ = []
    for i in ibase:
        fti = fftdata['FFT']['data'][:,:,0:nfreq2]
        zone = np.logical_and(ifreq[None,:] >= peaks[i,:][:,None]-peakswd[i,:][:,None]/2, ifreq[None,:] <= peaks[i,:][:,None]+peakswd[i,:][:,None]/2)
        NIZ.append(np.sum(zone, axis=0))
        FT.append(fti*zone)
        CF.append(np.sum(fti*zone, axis=2))
        bck *= (1-zone)
    FT = np.array(FT)
    CF = np.array(CF)
    NIZ = np.array(NIZ)
    
    if verbose:
        print('Shape of FT:', np.shape(FT))
        print('NIZ:', NIZ)
        
    if plot:
        fig, axes = plt.subplots(1, 8, figsize=(16, 8))
        fig.tight_layout()
        FTavg = np.mean(FT, axis=1)
        for i, ax in enumerate(axes.flat):
            if i < npeaks:
                epsilon = 0#1e-6
                #ax.imshow(np.log(np.abs(FTavg[i,...])+epsilon), cmap='gray')
                ax.imshow((np.abs(FTavg[i,...])+epsilon), cmap='gray')
                ax.set_title('Peak {}'.format(i))
            else:
                #ax.imshow(np.log(np.abs(bck[0,...]+epsilon)), cmap='gray')
                ax.imshow((np.abs(bck[0,...]+epsilon)), cmap='gray')
                ax.set_title('Background')
        plt.show()
    
    print('Shape of FT:', np.shape(FT))
    fftdata['CF'] = {'data': FT, 'CF': CF, 'CF_nbpx': NIZ, 'bck': bck}
    return fftdata

##############################################
# Function to demodulate MATISSE fringes
def op_demodulate(CFdata, wlen, verbose=True, plot=False):
    if verbose:
        print('Demodulating correlated flux...')
    npeaks  = np.shape(CFdata['CF']['data'])[0]
    nframes = np.shape(CFdata['CF']['data'])[1]
    nwlen   = np.shape(CFdata['CF']['data'])[2]
    npix    = np.shape(CFdata['CF']['data'])[3]
    if verbose:
        print('npeaks:', npeaks)
        print('nframes:', nframes)
        print('nwlen:', nwlen)
        print('npix:', npix)
        
    localopd = CFdata['INTERF']['localopd']
    if verbose:
        print('Shape of localopd:', np.shape(localopd))
    
    ntel = 4
    # Compute baseline OPD from local OPD
    teli = (3,1,2,2,1,1)
    telj = (4,2,3,4,3,4)
    localopdij = []
    ibase=0
    for itel in np.arange(ntel-1):
        for jtel in np.arange(ntel - itel - 1) + itel + 1:
            print('ij:',itel,jtel)
            if verbose:
                loij = localopd[:,teli[ibase]-1] - localopd[:,telj[ibase]-1]
            localopdij.append(loij)
            print('ij:',itel,jtel, 'localopdij:', loij)
            ibase+=1
    localopdij = np.array(localopdij)
    # Compute the phasor from localopd
    phasor = np.exp(2j * np.pi * localopdij[:,:,None] / wlen[None,None,:] )
    
    CFdata['CF']['mod_phasor'] = phasor
    CFdata['CF']['CF_demod']   = np.zeros_like(CFdata['CF']['CF'])
    CFdata['CF']['CF_demod'][1:,...]   = CFdata['CF']['CF'][1:,...] * np.conjugate(phasor)
    CFdata['CF']['data_demod']   = np.zeros_like(CFdata['CF']['data'])
    CFdata['CF']['data_demod'][1:,...] = CFdata['CF']['data'][1:,...] * np.conjugate(phasor[...,None])
    
    print('wlen:', wlen)
    
    if plot:
        iframe = 0
        colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#A133FF', '#33FFF5', '#F5FF33']
        plt.figure(4)
        for i in np.arange(6):
            plt.plot(np.angle(phasor[i,iframe,:]),color=colors[i])
        plt.show()
    
    if verbose:
        print('Shape of phasor:', np.shape(phasor))
        print('Shape of CF:', np.shape(CFdata['CF']['CF']))
        
    return CFdata

##############################################
# Function to sort out beams    
def op_sortout_beams(beamsin, verbose=True):
    here_do_something = 0

##############################################
# Function to sort out peaks
# The combiner entrance MATISSE pupil looks
# like that in L band (BCD out)
#  S1       S2          S3    S4
#      2        3          1
#  _        _           _     _
# / \      / \         / \   / \
# \_/      \_/         \_/   \_/
# and like this in N band (BCD out)
#  S4   S3           S2       S1
#     1        3          2
#  _     _           _        _
# / \   / \         / \      / \
# \_/   \_/         \_/      \_/
#
# The BCD inverts S1 <-> S2 and S3 <-> S4
# 
def op_sortout_peaks(peaksin, verbose=True):
    if verbose:
        print('Sorting out peaks...')
        
    bcd1 = peaksin['hdr']['HIERARCH ESO INS BCD1 ID']
    bcd2 = peaksin['hdr']['HIERARCH ESO INS BCD2 ID']
    det = peaksin['hdr']['HIERARCH ESO DET CHIP NAME']
    
    if verbose:
        print('BCD1:', bcd1)
        print('BCD2:', bcd2)
    
    tel= peaksin['OPTICAL_TRAIN']['INDEX']
    ntel = len(tel)
    nbases = ntel*(ntel-1)//2
    telname = peaksin['OPTICAL_TRAIN']['TEL_NAME']
    if verbose:
        print('Telescope names:', telname)
    DL_number = peaksin['OPTICAL_TRAIN']['VALUE1']
    IP_number = peaksin['OPTICAL_TRAIN']['VALUE2']
    
    #########################################
    # Internal beams scrambling
    # See MATISSE document: MATISSE-TP-TN-003
    if det   == 'HAWAII-2RG': #  MATISSE_L
        beamscr  = (4,3,2,1)
        band="L"
    elif det == 'AQUARIUS': #  MATISSE_N
        beamscr  = (1,2,3,4)
        band="N"
    else:
        error('Instrument not recognized')
        
    
    bcdin_  = np.array((2,1,0,0))
    bcdout_ = np.array((1,2,0,0))
    bcd_in  = np.array((0,0,4,3))
    bcd_out = np.array((0,0,3,4))
    
    bcdscr  = np.array((0,0,0,0))
    if bcd1 == "IN":
        bcdscr += bcdin_
    else:
        bcdscr += bcdout_
    if bcd2 == "IN":
        bcdscr += bcd_in
    else:
        bcdscr += bcd_out
    
    if verbose:
        for i in range(4):
            print("tel",tel[i], telname[i], "DL",DL_number[i], "IP", IP_number[i], "beamscr", beamscr[i], "bcdscr", bcdscr[i], "beam", tel[bcdscr[beamscr[i]-1]-1])
    
    coding = (1,3,6,7)
    
    ibase=0
    peakscr = np.zeros(nbases)
    for itel in np.arange(ntel-1):
        for jtel in np.arange(ntel - itel - 1) + itel + 1:
            telnamei = telname[tel[bcdscr[beamscr[itel]-1]-1]-1]
            telnamej = telname[tel[bcdscr[beamscr[jtel]-1]-1]-1]
            teli = coding[bcdscr[beamscr[itel]-1]-1]
            telj = coding[bcdscr[beamscr[jtel]-1]-1]
            lng = telj-teli
            print("base",ibase+1, "telescopes", itel, "and", jtel, "tel1",telnamei,"tel2",telnamej, "peak",lng)
            peakscr[ibase] = -lng
            ibase+=1
            
    peakunscr_tmp = np.arange(nbases)
    peakunscr = np.zeros(nbases)
    for i in np.arange(nbases):
        for j in np.arange(nbases):
            if int(np.abs(peakscr[j])-1) == i:
                peakunscr[i] = np.sign(peakscr[j]) * (j+1)
    print("peakscr",peakscr)
    print("peakunscr",peakunscr)
    for i in np.arange(nbases):
        if peakunscr[i] > 0:
            peaksin['CF']['CF'][i+1,...] = peaksin['CF']['CF'][int(peakunscr[i]),...]
        else:
            peaksin['CF']['CF'][i+1,...] = -peaksin['CF']['CF'][int(-peakunscr[i]),...]
            
            
    return peaksin

##############################################
# Function to compute the air refractive index
def op_air_index(wl, T=15, P=1013.25, h=0.1, N_CO2=423, bands='all'):
    """ Compute the refractive index as a function of wavelength at a given temperature,
        pressure, relative humidity and CO2 concentration, using Equation (5) of Voronin & Zheltikov (2017).
        
        Reference: Voronin, A. A. and Zheltikov, A. M. The generalized Sellmeier equation for air. 
        Sci. Rep. 7, 46111; doi: 10.1038/srep46111 (2017).
        
        Inputs:
        - wl: array of wavelengths in microns,
        - T: temperature in °C,
        - P: pressure in hPa,
        - h: relative humidity,
        - N_CO2: CO2 concentration in dry air in ppm,
        - bands: list of absorption bands to include according to the numbering in Table 1 of
            Voronin & Zheltikov (2017). Default: 'all' (includes all 15 bands).
        Output:
        - n_air: array of refractive indices at each wl value.
"""
    ## Characteristic wavelengths (microns)
    wl_abs1 = 1e-3 * np.array([15131, 4290.9, 2684.9, 2011.3, 47862, 6719.0, 2775.6, 1835.6,
               1417.6, 1145.3, 947.73, 85, 127, 87, 128])
    wl_abs2 = 1e-3 * np.array([14218, 4223.1, 2769.1, 1964.6, 16603, 5729.9, 2598.5, 1904.8, 
               1364.7, 1123.2, 935.09, 24.546, 29.469, 22.645, 34.924])
    
    ## Absorption coefficients
    A_abs1 = np.array([4.051e-6, 2.897e-5, 8.573e-7, 1.550e-8, 2.945e-5, 3.273e-6, 
              1.862e-6, 2.544e-7, 1.126e-7, 6.856e-9, 1.985e-9, 1.2029482,
              0.26507582, 0.93132145, 0.25787285])
    A_abs2 = np.array([1.010e-6, 2.728e-5, 6.620e-7, 5.532e-9, 6.583e-8, 3.094e-6,
              2.788e-6, 2.181e-7, 2.336e-7, 9.479e-9, 2.882e-9, 5.796725,
              7.734925, 7.217322, 4.742131])
    
    ## Conversion
    T += 273.15 # °C -> K
    P *= 1e2    # hPa -> Pa
    
    ## Water vapor
    T_cr = 647.096 # critical-point temperature of water (K)
    P_cr = 22.064e6 # critical-point pressure of water (Pa)
    tau = T_cr / T
    th = 1 - T / T_cr
    a1, a2, a3, a4, a5, a6 = -7.85951783, 1.84408259, -11.7866497, 22.6807411, -15.9618719, 1.80122502
    # Saturated water vapor pressure (Pa)
    P_sat = P_cr * np.exp(tau*(a1*th + a2*th**1.5 + a3*th**3 + a4*th**3.5 + a5*th**4 + a6*th**7.5))
    # Water vapor density (m-3)
    N_H2O = h * P_sat / (const.k_B.value * T)

    ## Dry air pressure (Pa)
    P_dry = P - h * P_sat

    ## Gas concentrations (m-3) (mixing ratio in ppm of dry air * concentration of dry air)
    N_air = P_dry / (const.k_B.value * T)
    N_N2 = (780840 * 1e-6) * N_air
    N_O2 = (209460 * 1e-6) * N_air
    N_Ar = (9340 * 1e-6) * N_air
    N_CO2 = (N_CO2 * 1e-6) * N_air
    
    N_gas = np.zeros_like(wl_abs1)
    N_gas[0:4] = N_CO2
    N_gas[4:11] = N_H2O
    N_gas[11] = N_N2
    N_gas[12] = N_O2
    N_gas[13] = N_Ar
    N_gas[14] = N_H2O
    
    ## Critical plasma density (m-3)
    N_cr = const.m_e.value * const.eps0.value * (2 * np.pi * const.c.value / (const.e.value * wl*1e-6))**2
   
    ## Selection of absorption bands
    if bands == 'all':
        bands = np.arange(15)
    else:
        bands = np.array(bands) - 1
    
    ## Refractive index
    n_air = 1
    for i_band in bands:
        dn_air1 = (N_gas[i_band]/N_cr) * A_abs1[i_band] * wl_abs1[i_band]**2 / (wl**2 - wl_abs1[i_band]**2)
        dn_air2 = (N_gas[i_band]/N_cr) * A_abs2[i_band] * wl_abs2[i_band]**2 / (wl**2 - wl_abs2[i_band]**2)
        n_air += dn_air1 + dn_air2
    
    return n_air
   
    
   
