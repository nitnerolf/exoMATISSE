#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 14:53:32 2025

@author: nsaucourt

"""
import json
import os
################# GLOBAL PARAMETERS ##################

# PLOT'S COLORS
COLORS10D = [
    '#ff40e0',  # flashy pink
    '#808000',  # olive
    '#000080',  # navy
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#008080',  # teal
    '#ff0000',  # red
    '#00bb00',  # green
    '#0000ff',  # blue
    '#800000',  # maroon
    '#006400',  # dark green
    '#00bfff',  # deep sky blue 
    '#9467bd',  # purple
    '#32cd32',  # lime green
    '#dda0dd'  # plum
]
COLORS7D = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#A133FF', '#33FFF5', '#F5FF33']
COLORS6D = ['red','blue', 'lightgreen', 'orange', 'purple', 'cyan']

# OIFITS PARAMETERS
REVN = 2
PIPELINE_NAME = 'OPTRA'
PIPELINE_VERSION = 'v0.1'
AMPTYPE = 'correlated flux'
PHITYPE = 'differential'

# N_CO2 PARAMETERS
BASE_DIR_NCO2 = os.path.dirname(os.path.abspath(__file__))
FICHIER_NCO2 = os.path.join(BASE_DIR_NCO2, "n_co2", "nco2.json")

with open(FICHIER_NCO2,'r') as f:
    CO2_SAVED = json.load(f)

SLOPE_CO2     = CO2_SAVED['slope']
INTERCEPT_CO2 = CO2_SAVED['intercept']  