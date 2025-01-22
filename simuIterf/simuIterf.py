import numpy as np
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass
import math
import os
from astropy.io import fits

##############################################################################################################################

def simiGetLocation(place, interferometer=None):
    """
    Fills in the SimiLOCATION structure based on the input place and interferometer.

    Parameters:
        place (str): The location (e.g., "paranal", "e-elti", "MWI", "calern").
        interferometer (str, optional): The interferometer (e.g., "VLTI", "CHARA").

    Returns:
        SimiLOCATION: A structure with the location's details.
    
    Raises:
        ValueError: If the place is not recognized.
    """
    loc = SimiLOCATION(name="", tz=0, dst=0, lon=0, lat=0, elev=0)
    
    if place.lower() == "paranal":
        if interferometer is None:
            interferometer = "VLTI"
        loc.name = "ESO, Cerro Paranal"
        loc.tz = 4
        loc.dst = -1
        loc.lon = -4.69365
        loc.lat = -24.62794830
        loc.elev = 2635
        _simiSaveStations(interferometer)
    elif place.lower() == "e-elti":
        loc.name = "ESO, Cerro Armazones"
        loc.tz = 4
        loc.dst = -1
        loc.lon = -4.69365
        loc.lat = -24.62794830
        loc.elev = 3003
        _simiSaveStations(interferometer)
    elif place.lower() == "mwi":
        if interferometer is None:
            interferometer = "CHARA"
        loc.name = "Mount Wilson"
        loc.tz = 4
        loc.dst = -1
        loc.lon = -118.061644
        loc.lat = 34.223758
        loc.elev = 1742
        _simiSaveStations(interferometer)
    elif place.lower() == "calern":
        if interferometer is None:
            interferometer = "GI2T"
        loc.name = "Calern Observatory"
        loc.tz = 1
        loc.dst = 0
        loc.lon = 6.92098
        loc.lat = 43.75389
        loc.elev = 1270
        _simiSaveStations(interferometer)
    else:
        raise ValueError("Place not found!")

    # Logging output for testing
    print(f"You are here: {loc.name}")
    return loc

##############################################################################################################################

def _simiSaveStations(interferometer):
    """
    Simulates saving station details for the given interferometer.

    Parameters:
        interferometer (str): The interferometer name.
    """
    # Placeholder for actual station-saving logic.
    print(f"Saving stations for {interferometer}...")

##############################################################################################################################

def simi_save_stations(interferometer, stations_file=None):
    """
    Saves a table containing the station coordinates in observatory
    coordinates (P & Q) and in E-W, N-S coordinates (meters).

    Parameters:
        interferometer (str): Name of the interferometer.
        stations_file (str, optional): File name to save the station information.

    Returns:
        list: List of station names.
    """
    if stations_file is None:
        stations_file = "~/.interfStations.dat"

    positions = ""

    if interferometer == "VLTI":
        positions = (
            " ID P Q E N ALT D C\n"
            "A0 -32.001 -48.013 -14.642 -55.812 0.0 1.8 1\n"
            "A1 -32.001 -64.021 -9.434 -70.949 0.0 1.8 2\n"
            "B0 -23.991 -48.019 -7.065 -53.212 0.0 1.8 3\n"
            "B1 -23.991 -64.011 -1.863 -68.334 0.0 1.8 4\n"
            "B2 -23.991 -72.011 0.739 -75.899 0.0 1.8 5\n"
            "B3 -23.991 -80.029 3.348 -83.481 0.0 1.8 6\n"
            "B4 -23.991 -88.013 5.945 -91.030 0.0 1.8 7\n"
            "B5 -23.991 -96.012 8.547 -98.594 0.0 1.8 8\n"
            "C0 -16.002 -48.013 0.487 -50.607 0.0 1.8 9\n"
            "C1 -16.002 -64.011 5.691 -65.735 0.0 1.8 10\n"
            "C2 -16.002 -72.019 8.296 -73.307 0.0 1.8 11\n"
            "C3 -16.002 -80.010 10.896 -80.864 0.0 1.8 12\n"
            "D0 0.010 -48.012 15.628 -45.397 0.0 1.8 13\n"
            "D1 0.010 -80.015 26.039 -75.660 0.0 1.8 14\n"
            "D2 0.010 -96.012 31.243 -90.787 0.0 1.8 15\n"
            "E0 16.011 -48.016 30.760 -40.196 0.0 1.8 16\n"
            "G0 32.017 -48.0172 45.896 -34.990 0.0 1.8 17\n"
            "G1 32.020 -112.010 66.716 -95.501 0.0 1.8 18\n"
            "G2 31.995 -24.003 38.063 -12.289 0.0 1.8 19\n"
            "H0 64.015 -48.007 76.150 -24.572 0.0 1.8 20\n"
            "I1 72.001 -87.997 96.711 -59.789 0.0 1.8 21\n"
            "J1 88.016 -71.992 106.648 -39.444 0.0 1.8 22\n"
            "J2 88.016 -96.005 114.460 -62.151 0.0 1.8 23\n"
            "J3 88.016 7.996 80.628 36.193 0.0 1.8 24\n"
            "J4 88.016 23.993 75.424 51.320 0.0 1.8 25\n"
            "J5 88.016 47.987 67.618 74.009 0.0 1.8 26\n"
            "J6 88.016 71.990 59.810 96.706 0.0 1.8 27\n"
            "K0 96.002 -48.006 106.397 -14.165 0.0 1.8 28\n"
            "L0 104.021 -47.998 113.977 -11.549 0.0 1.8 29\n"
            "M0 112.013 -48.000 121.535 -8.951 0.0 1.8 30\n"
            "U1 -16.000 -16.000 -9.925 -20.335 8.504 8 31\n"
            "U2 24.000 24.000 14.887 30.502 8.504 8 32\n"
            "U3 64.0013 47.9725 44.915 66.183 8.504 8 33\n"
            "U4 112.000 8.000 103.306 43.999 8.504 8 34\n"
            "LAB 52.000 -40.000 60 -20 0.0"
        )

    elif interferometer == "NACO_SAM_7":
        mask_positions = [
            [3.51064, -1.99373],
            [3.51064, 2.49014],
            [1.56907, 1.36918],
            [1.56907, 3.61111],
            [-0.372507, -4.23566],
            [-2.31408, 3.61111],
            [-4.25565, 0.248215]
        ]
        mask_diameter = 1.50
        positions = " ID P Q E N ALT D C\n"
        for i, (x, y) in enumerate(mask_positions, 1):
            positions += f"H{i} {x:.3f} {y:.3f} {x:.3f} {y:.3f} 0.0 {mask_diameter:.1f} 1\n"
        positions += "LAB 0.0 0.0 0.0 0.0 0.0 0.0 1"

    elif interferometer == "CHARA":
        positions = (
            " ID P Q E N ALT D C\n"
            "S1 0.0 0.0 0.0 0.0 0.0 1.0 1\n"
            "S2 0.0 0.0 -5.748 33.577 0.637 1.0 2\n"
            "E1 0.0 0.0 125.333 305.928 -5.919 1.0 3\n"
            "E2 0.0 0.0 70.389 269.715 -2.803 1.0 4\n"
            "W1 0.0 0.0 -175.068 216.327 -10.798 1.0 5\n"
            "W2 0.0 0.0 -69.085 199.342 0.471 1.0 6\n"
            "LAB 0.0 0.0 -20.0 200 0.0 0.0 7"
        )

    elif interferometer == "calern":
        positions = (
            " ID P Q E N ALT D C\n"
            "M1 0.0 0.0 0.0 0.0 0.0 1.5 1\n"
            "G1 0.0 0.0 -151 -157 0.0 1.5 2\n"
            "G2 0.0 0.0 -151 -141.5 0.0 1.5 3\n"
            "G1b 0.0 0.0 -151 -181.5 0.0 1.5 4\n"
            "G2b 0.0 0.0 -151 -109.5 0.0 1.5 5\n"
            "C1 0.0 0.0 146 -103 0.0 1.0 6\n"
            "C2 0.0 0.0 167 -103 0.0 1.0 7\n"
            "S1 0.0 0.0 -12 -401.5 0.0 1.5 8"
        )

    elif interferometer == "e-elti":
        D = 42  # ELT diameter
        D2 = 11.76  # ELT central obscuration diameter
        d = 1.47 * np.cos(np.pi / 6)  # Telescope spacing
        nm = int(D / d * np.sqrt(2)) + 1

        positions = " ID P Q E N ALT D C\n"
        count = 0
        for k in range(-nm // 2, nm // 2 + 1):
            for l in range(-nm // 2, nm // 2 + 1):
                y = k * d + 0.5 * d * (l % 2) + 0.5 * d
                x = (l - 1) * d * np.sin(np.pi / 3)
                r = np.sqrt(x**2 + y**2)
                if D2 / 2 + d / 2 < r < D / 2 - d / 2:
                    count += 1
                    positions += f"O{count} {x:.3f} {y:.3f} {x:.3f} {y:.3f} 0.0 {d:.3f} 1\n"
    else:
        raise ValueError("Interferometer not found!")

    # Save the station data to the file
    with open(stations_file, "w") as fh:
        fh.write(positions)

    print(f"Station data saved to {stations_file}")

##############################################################################################################################

def simiGenerateUV(stardata, stations, lst, coloring, usedStations, 
                   hacen=None, harange=None, hastep=None, lambda_=None, colors=None, 
                   mx=None, fp=None, frac=None, msize=None, marker=None, 
                   spFreqFact=None, xyTitles=None, place=None, interferometer=None):
    """
    Generate UV coordinates for interferometric simulations.

    Parameters:
        stardata (dict): Star data with properties like RA, Dec, etc.
        stations (ndarray): Station positions.
        lst (list): Local Sidereal Times (to be filled).
        coloring (list): Colors for visualization (to be filled).
        usedStations (list): Used station data (to be filled).
        hacen (float): Hour angle center.
        harange (float): Hour angle range.
        hastep (float): Hour angle step.
        lambda_ (ndarray): Wavelengths.
        colors (str): Coloring mode ("wlen", "bases", etc.).
        mx, fp, frac, msize, marker, spFreqFact, xyTitles (various): Other parameters.
        place (str): Observation site.
        interferometer (str): Interferometer name.

    Returns:
        uvwTable (ndarray): UVW coordinates.
    """
    if place is None:
        place = "paranal"
    if interferometer is None:
        interferometer = "VLTI"
    
    # Placeholder for simiGetLocation - replace with actual implementation
    loc = simiGetLocation(place, interferometer)

    if lambda_ is None:
        n_lambda = 20
        lambda_min = 1.1 * 1e-6  # Conversion for micro to meters
        lambda_max = 2.5 * 1e-6
        lambda_ = np.linspace(lambda_max, lambda_min, n_lambda)
    else:
        n_lambda = len(lambda_)

    if colors is None:
        colors = "wlen"

    if spFreqFact is None:
        spFreqFact = 1.0

    if xyTitles is None:
        xyTitles = [
            "E <------ Spatial frequency (m^-1^)",
            "Spatial frequency (m^-1^) ------> N"
        ]

    if frac is None:
        frac = 1.0

    if msize is None:
        msize = 1.0

    if marker is None:
        marker = '\1'

    if hacen is None:
        hacen = stardata.get("ra", 0)
    if harange is None:
        harange = 8.0

    hamin = hacen - harange / 2
    hamax = hacen + harange / 2

    if hastep is None:
        hastep = 70.0 / 60.0

    n_obs = stations.shape[1] if stations.ndim > 1 else 1

    uvwTable = np.empty((0, 3))
    lst.clear()
    coloring.clear()
    usedStations.clear()

    for k_obs in range(n_obs):
        nb_tel = stations.shape[0]
        nb_bases = nb_tel * (nb_tel - 1) // 2

        baseNames = []
        B = []
        orig = []

        for i_obs in np.arange(hamin, hamax + hastep, hastep):
            stardata["lst"] = float(i_obs) + hastep / 2.0 * frac * random.uniform(0, 1)
            lst.append(stardata["lst"])

            for i_base in range(nb_bases):
                # Compute base vectors (placeholder for simiComputeBaseVect)
                bvect = B[i_base] if i_base < len(B) else np.zeros(3)
                stardata.update(simiComputeUvwCoord(stardata, bvect))

                uv = np.array([stardata["u"], stardata["v"]])
                uvw = np.array([stardata["u"], stardata["v"], stardata["w"]])

                uvwTable = np.vstack((uvwTable, uvw))

    return uvwTable

##############################################################################################################################

def simiComputeUvwCoord(data, bvect, loc):
    """
    Corrects UVW coordinates using the base vector of an observation and updates the data fields.

    Parameters:
        data (dict): Dictionary containing the `simiSTARVIS` structure to be corrected.
        bvect (list): Base vector of observation [x, y, z].
        loc (dict): Location structure containing latitude.

    Modifies:
        Updates `data` with corrected UVW values, baseline, theta, and delay.
    """
    degr = 180 / math.pi
    hour = degr / 15.0
    Bnorm = math.sqrt(sum([x**2 for x in bvect]))
    print('loc:', loc)

    # Baseline vector in alt-az coordinates
    Balt = math.asin(bvect[2] / (Bnorm + (Bnorm == 0))) * degr
    Baz = math.atan2(bvect[1], bvect[0]) * degr
    print('loc lat:', loc.lat)
    
    # Baseline vector in equatorial coordinates
    Bdec = math.asin(
        math.sin(Balt / degr) * math.sin(loc.lat / degr) +
        math.cos(Balt / degr) * math.cos(loc.lat / degr) * math.cos(Baz / degr)
    ) * degr

    yBha = (math.sin(Balt / degr) * math.cos(loc.lat / degr) -
            math.cos(Balt / degr) * math.cos(Baz / degr) * math.sin(loc.lat / degr))
    zBha = -1.0 * math.cos(Balt / degr) * math.sin(Baz / degr)
    Bha = (math.atan2(zBha, yBha) * hour + 24) % 24

    # Baseline vector in equatorial cartesian frame
    Lx = -(-Bnorm * math.cos(Bdec / degr) * math.cos(Bha / hour))
    Ly = -Bnorm * math.cos(Bdec / degr) * math.sin(Bha / hour)
    Lz = Bnorm * math.sin(Bdec / degr)

    # Projection of the baseline vector on the u, v, w frame
    data['ha'] = data['lst'] - data['ra']
    data['u'] = (math.sin(data['ha'] / hour) * Lx + math.cos(data['ha'] / hour) * Ly)
    data['v'] = (-math.sin(data['dec'] / degr) * math.cos(data['ha'] / hour) * Lx +
                 math.sin(data['dec'] / degr) * math.sin(data['ha'] / hour) * Ly +
                 math.cos(data['dec'] / degr) * Lz)
    data['w'] = (math.cos(data['dec'] / degr) * math.cos(data['ha'] / hour) * Lx -
                 math.cos(data['dec'] / degr) * math.sin(data['ha'] / hour) * Ly +
                 math.sin(data['dec'] / degr) * Lz)

    data['theta'] = math.atan2(data['u'], data['v']) * degr
    data['base'] = math.sqrt(data['u']**2 + data['v']**2)
    data['delay'] = -data['w']

    return data
