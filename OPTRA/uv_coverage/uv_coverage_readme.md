# UV Coverage Toolkit

This repository provides a Python module and a set of example scripts for computing and analyzing UV-plane coverage with VLTI/MATISSE data.

---

## Overview

This toolkit is designed to generate and visualize UV tracks, compute coverage masks, and analyze the resulting spatial frequency support for optical interferometry experiments. The core logic resides in a reusable module, while the provided scripts serve as applied examples.

---

## Directory Structure

```
uv_coverage/
├── module_uv_coverage.py        # Core module with reusable functions
├── uv_ntel.py                   # Example: UV track computation from FITS
├── uv_mask_Aspro.py             # Example: mask & FFT from UV pickle file
├── uv_mask_Date.py              # Example: end-to-end UV mask from metadata
├── Aspro2_...fits               # Example FITS file from Aspro2
└── README.md                    # This file
```


---

## Python Module: `module_uv_coverage.py`

This file contains the main computational utilities. Each function is briefly described below:

### `create_instru(ntel)`

Returns a fake MATISSE-like instrument dictionary based on the number of telescopes.

PARAMETERS:

  * ntel : number of telescopes.

### `gps_to_E_N( lat, lon)`

Returns the E/N position in meter at the VLT of a given latitude and longitude.

PARAMETERS:

  * lat : latitude.
  * lon : longitude.


### `generate_date(date_str, interval)`

Generates a list of dates in the UTC ISO format (YYYY-MM-DDTHH:MM:SS.sss)


PARAMETERS:

  * date_str : String of a date in the UTC ISO format.
  * interval : Number of hours of observations.



### `create_header(RA, DEC, date, stations_list, instrument = op_MATISSE_L, new_coords = dict())`

Create a header based on the header out of the ESO fits file in the archive

PARAMETERS:

  * RA : Right Ascension.
  * DEC : Declination.
  * date: Specific date in the UTC ISO format.
  * stations_list : liste of used stations.

Optional parameters:

  * instrument : used instrument
  * new_coords : coordinate of stations that are not on the current VLT.


### `uv_to_sf(u, v, wlen, wlen_ref = 3.5e-6)`

Converts UV tracks to spatial frequency units given a wavelength range.

PARAMETERS:

  * u : u Coordinates.
  * v : v Coordinates.
  * wlen: array of used wavelength

Optional parameters:
  * wlen_ref : wavelength's reference


### `mask_uv_coverage(uCoord,vCoord,wlen,wlen_ref = 3.5e-6, plot = True)`

Creates a 2D binary UV mask from UV coordinates

PARAMETERS:

  * uCoord : u Coordinates.
  * vCoord : v Coordinates.
  * wlen: array of used wavelength

Optional parameters:

  * wlen_ref : wavelength's reference
  * plot: boolean for plotting


### `fft_mask(mask, plot = True)`

Compute the fft of the UV mask

PARAMETERS:

  * mask : UV mask (obtained with mask_uv_coverage).

Optional parameters:

  * plot: boolean for plotting


---

## Example Scripts

These are stand-alone scripts demonstrating how to use the module for various tasks:

### `uv_ntel.py`

Generates UV coverage for more than 4 telescopes. 


### `uv_mask_Aspro.py`

Takes an Aspro2 fit file with precomputed UV tracks and builds a binary mask + FFT.

```

### `uv_mask_Date.py`

Computes UV tracks and a mask directly from user-specified  RA/DEC, date and stations.




