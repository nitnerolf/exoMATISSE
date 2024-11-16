##############################################
# Correlated flux computation
##############################################
from astropy.io import fits
import numpy as np
import astropy.constants as const
import matplotlib.pyplot as plt
from scipy import *

##############################################
def op_apodize(data, verbose=True,plot=False):
    if verbose:
        print('Apodizing data...')
    # Apply a Hamming window to the data
    nframes = np.shape(data['INTERF']['data'])[0]
    nreg    = len(data['PHOT'])
    
    intf  = stats.trim_mean(data['INTERF']['data'],axis=0, proportiontocut=0.05)
    n     = np.shape(intf)[1]
    win   = signal.get_window(('kaiser', 14), n)
    argmx = np.argmax(np.mean(intf, axis=0))
    print('argmx', argmx)
    centered_win_intf = np.roll(win, argmx - n//2)
    
    if plot:
        # Plot the Hanning window used for apodization
        plt.figure()
        plt.plot(centered_win_intf)
        plt.title('Window for Apodization')
        plt.xlabel('Pixel Index')
        plt.ylabel('Window Value')
        plt.grid(True)
        plt.show()
    
    for i in np.arange(nframes):
        data['INTERF']['data'][i] *= centered_win_intf
        
    for key in data['PHOT']:
        pht   = stats.trim_mean(data['PHOT'][key]['data'],axis=0, proportiontocut=0.05)
        n     = np.shape(pht)[1]
        win   = signal.get_window(('kaiser', 14), n)
        argmx = np.argmax(np.mean(pht, axis=0))
        centered_win_pht = np.roll(win, argmx - n//2)
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
    nwlen = np.shape(intf)[1]
    npix = np.shape(intf)[2]
    
    # Compute FFT 1D of intf along the pixels axis
    fft_intf = np.fft.fft(intf, axis=2)
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
        print('interfringe:', interfringe)
        peakswd = 4.
        pkwds = np.ones_like(peaks) * peakswd
        peak = peaks[:,None] * interfringe / wlen[None,:]
        peakwd = pkwds[:,None] * interfringe / wlen[None,:]
    
    if verbose:
        print('Shape of peak:', np.shape(peak))
        print('Peak:', peak)
    return peak, peakwd
    
##############################################
# Function to extract the correlated flux
def op_extract_CF(fftdata, wlen, baselines, baselwd=1, verbose=True):
    if verbose:
        print('Extracting correlated flux...')
    bck = fftdata['FFT']['data']
    FT = []
    for i in baselines:
        FT.append(fftdata['FFT']['data'][:,i-baselwd:i+baselwd+1,:])
        bck[:,i-baselwd:i+baselwd+1,:] = 0
    FT = np.array(FT)
    print('Shape of FT:', np.shape(FT))

##############################################
# Function to sort out peaks
def op_sortout_peaks(peaksin, peaksout, verbose=True):
    tel    = (1,2,3,4)
    telname= ("U1","U2","U3","U4")
    telscr = (3,4,2,1)
    
    bcdin_  = (2,1,0,0)
    bcdout_ = (1,2,0,0)
    bcd_in  = (0,0,4,3)
    bcd_out = (0,0,3,4)
    
    bcd = bcdin_ + bcd_in
    
    coding = (1,3,6,7)
    
    ntel = len(coding)
    for i in np.arange(ntel):
        for j in np.arange(ntel-i-1)+i+1:
            print(i,j)
            telnamei = telname[telscr[bcd[tel[i]-1]-1]-1]
            telnamej = telname[telscr[bcd[tel[j]-1]-1]-1]
            teli = coding[i]
            telj = coding[j]
            lng = telj-teli
            print(telnamei, telnamej, lng)

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
