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
COLORS10D=['#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # yellow-green
        '#17becf'   # cyan
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
print(CO2_SAVED)

SLOPE_CO2     = CO2_SAVED['slope']      #2.461874809975082
INTERCEPT_CO2 = CO2_SAVED['intercept']  #-4561.3674820094