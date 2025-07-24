#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 11:46:37 2025

@author: nsaucourt
"""
################################## IMPORT ######################################


import os
import sys

# --- Add parent folder temporarily for import ---
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator
from op_rawdata import op_compute_uv,op_MATISSE_L
from astropy.time import Time,TimeDelta


sys.path.pop(0)




###############################################################
#Create a fake VLTI instrument with a scrambling associated
def create_instru(ntel):
    if ntel == 4:
        return op_MATISSE_L
    else:
        return  {  'name': f'fake_MATISSE_{ntel}tel',
                    'ntel': ntel,
                    'scrP': np.arange(1,ntel+1),
                    'scrB':[[i,j] for i in range(0,ntel-1) for j in range(i+1,ntel)] }
        
            
     

###############################################################
# GPS to E,N coordinate converter
def gps_to_E_N( lat, lon):
    """
    Convert lthe GPS coordinates (lat, lon) in meters (E,N)
    based on the origine of the VLTI (lat0, lon0).
    """
    R = 6371000  # Mean Radius of Earth in meters
    lat0 = -24.62794830
    lon0 = -70.40479659

    lat0_rad = np.radians(lat0)
    lat_rad  = np.radians(lat)
    dlat = lat_rad - lat0_rad
    dlon = np.radians(lon - lon0)

    # Mean to correct the latitude
    latm = (lat_rad + lat0_rad) / 2

    dx = R * dlon * np.cos(latm)  # towards East
    dy = R * dlat                 # towards North

    return round(dx,2), round(dy,2)



#################################################################################
#Compute the UTC of 10 points from the start_utc until start_utc + interval 
def generate_date(start_utc, interval):
    start = Time(start_utc)
    end = start + TimeDelta(interval * 3600, format='sec')  # 'interval' hours later

    # 10 points equally spaced between start and end
    times = start + np.linspace(0, (end - start).sec, 10) * TimeDelta(1, format='sec')
    
    return [t.isot for t in times]



#################################################################################
# Compute the header for specific RA,DEC,date,stations
def create_header(RA,DEC,date,stations_list,instrument = op_MATISSE_L , new_coords = dict()):
    
    try:
        assert len(stations_list) == instrument['ntel']
        hdr = {'ORIGIN'                        :'Paranal',
               'RA'                            : RA ,
               'DEC'                           : DEC ,
               'DATE-OBS'                      : date,
               'HIERARCH ESO DET NDIT'         : 6,
               'HIERARCH ESO DET SEQ1 DIT'     : 10,
               'HIERARCH ESO ISS CONF NTEL'    : instrument['ntel'],
               'HIERARCH ESO ISS GEOLON'       : -70.40479659 ,
               'HIERARCH ESO ISS GEOLAT'       : -24.62794830 ,
               'HIERARCH ESO ISS GEOELEV'      :  2635,
               'LST'                           : Time(date).sidereal_time('apparent',longitude=-70.40479659).hour*3600,
               'MJD-OBS'                       : Time(date).mjd}
               
        for i,station in enumerate(stations_list): 
            if station[0]=='U':
                hdr[f'HIERARCH ESO ISS CONF STATION{i+1}'] = station
                hdr[f'HIERARCH ESO ISS CONF T{i+1}NAME'] = f"UT{i+1}"
                if station in new_coords.keys():
                    hdr[f'HIERARCH ESO ISS CONF T{i+1}X']  = new_coords[station]['E']
                    hdr[f'HIERARCH ESO ISS CONF T{i+1}Y']  = new_coords[station]['N']
                    
            else:
                hdr[f'HIERARCH ESO ISS CONF STATION{i+1}'] = station
                hdr[f'HIERARCH ESO ISS CONF T{i+1}NAME'] = f"AT{i+1}"
                if station in new_coords.keys():
                    hdr[f'HIERARCH ESO ISS CONF T{i+1}X']  = new_coords[station]['E']
                    hdr[f'HIERARCH ESO ISS CONF T{i+1}Y']  = new_coords[station]['N']
                
                    
    except:
        print('there is not the same number of telescopes and number of stations')
    
    return hdr



#################################################################################
# Convert UV coords to spatial frequency (Mλ)
def uv_to_sf(u, v, wl, wlen_ref= 3.5e-6):
    return (u / wl *  wlen_ref, v / wl * wlen_ref)



#################################################################################
# Compute the mask of a uv-coverage
def mask_uv_coverage(uCoord,vCoord,wlen,wlen_ref = 3.5e-6, plot = True):
    # Base Resolution
    dxy = 1000
    # Compute max SF radius
    R = 1.50 * np.max(np.hypot(uCoord, vCoord))
    xs = np.linspace(-R, R, dxy)
    ys = np.linspace(-R, R, dxy)
    X, Y = np.meshgrid(xs, ys)
    final_mask = np.zeros((dxy, dxy), dtype=bool)

    # Min and max of wlen
    wl0 = min(wlen)
    wlN = max(wlen)

    # Precompute grid in SF space for mask evaluation
    X_sf_w0, Y_sf_w0 = uv_to_sf(X, Y, wlen_ref, wlen_ref )
    X_sf_wN, Y_sf_wN = uv_to_sf(X, Y, wlen_ref, wlen_ref )
    pts_w0 = np.vstack([X_sf_w0.ravel(), Y_sf_w0.ravel()]).T
    pts_wN = np.vstack([X_sf_wN.ravel(), Y_sf_wN.ravel()]).T


    # Build masks for all tracks
    envelope_paths = []
    for i in range(6):
        u, v = uCoord[i], vCoord[i]
        
        # Build the envelope around the uv coverage
        def build_mask(u, v):
            u0, v0 = uv_to_sf(u, v, wl0, wlen_ref)
            uN, vN = uv_to_sf(u, v, wlN, wlen_ref)
            
            # Compute the polynome around the wl = 2.8 and wl = 4.9
            poly = np.vstack([np.column_stack((u0, v0)),        # first curve 
                              np.column_stack((uN, vN))[::-1]]) #second curve: closed loop 

            path = Path(poly)
            mask = path.contains_points(pts_w0) | path.contains_points(pts_wN)
            envelope_paths.append(path)
            return mask.reshape(final_mask.shape)
        
        
        
        final_mask |= build_mask(u, v)
        final_mask |= build_mask(-u, -v)
        
        
        
        
    ########################## PLOTTING ##########################
    if plot:
        fig, ax = plt.subplots(figsize=(10, 10))
    
        # Plot Mask 
        cmap = ListedColormap(['lightgrey', 'white'])
        ax.imshow(final_mask.astype(int), extent=[-R, R, -R, R], origin='lower',
                  cmap=cmap, interpolation='nearest', zorder=0)
    
        # Plot tracks and UV curves
        colors = ['red', 'blue', 'lightgreen', 'orange', 'purple', 'cyan']
        for i in range(6):
            u, v = uCoord[i], vCoord[i]
            for sign in [+1, -1]:
                u_s, v_s = sign * u, sign * v
                ax.plot(u_s, v_s, color=colors[i], lw=2, zorder=3)
                ax.scatter(u_s, v_s, color=colors[i], zorder=4)
    
                # Plot SF lines at start/mid/end points
                for j in [0, len(u) // 2,len(u)-1]:
                    sf_u, sf_v = uv_to_sf(u_s[j], v_s[j], wlen, wlen_ref)
                    ax.plot(sf_u, sf_v, color=colors[i], lw=2)
    
        # Labels and aesthetics
        ax.set_title('uv-coverage mask', fontsize=16)
        ax.set_xlabel('U (Mλ)', fontsize=12)
        ax.set_ylabel('V (Mλ)', fontsize=12)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.grid(True, linestyle='--', linewidth=0.7)
        plt.tight_layout()
        plt.show()
    
    return final_mask



#################################################################################
# Compute the fft of the mask
def fft_mask(mask, plot = True):
    
    mask = mask.astype(int)
    ny, nx = mask.shape

    # Zero Padding factor 
    pad_factor = 10
    ny_pad = ny * pad_factor
    nx_pad = nx * pad_factor
    pady = (ny_pad - ny) // 2
    padx = (nx_pad - nx) // 2
    mask_padded = np.pad(mask, ((pady, pady), (padx, padx)), mode='constant')

    # Shifting the mask 
    shifted_mask = np.fft.fftshift(mask_padded)

    # Put the 0,0 at the sum
    shifted_mask[0,0]= np.sum(shifted_mask)

    # FFT
    fft = np.fft.fft2(shifted_mask,s=(ny_pad, nx_pad))
    shifted_fft = np.abs(np.fft.fftshift(fft))
    mean = np.median(shifted_fft)



    ########################## PLOT OF FFT ##########################
    if plot :
        plt.figure()
        plt.imshow(shifted_fft - mean,cmap = 'seismic',origin='lower', interpolation='nearest', aspect='equal' )
        plt.colorbar()
        plt.xlim(4500,5500)
        plt.ylim(4500,5500)
        plt.title("Magnitude of FFT")
    
    return shifted_fft