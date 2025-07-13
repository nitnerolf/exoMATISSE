#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 16:52:04 2025

@author: nsaucourt
"""



########################################## IMPORT ########################################

import os
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d



########################################## FUNCTION ########################################
# Update Co2 file
def update_co2_and_analyze_file(base_dir = './', data_txt_file=None, source="noaa", year_mask=2015,year_max = 2024, plot=False):
    """
    Update the CO₂ model from Scripps CSV data and/or NOAA TXT data.
    
    Parameters:
        base_dir        : directory containing nco2.json and data files
        data_txt_file   : filename of the NOAA data file (if source includes 'noaa')
        source          : "noaa", "scripps", or "both"
        year_mask       : start year for the regression (default 2015)
        year_max       : start year for the regression (default 2024)
        plot            : whether to show plots

    """

    ###### NCO2 json file #####
    fichier_nco2 = os.path.join(os.path.dirname(__file__), "nco2.json")
    with open(fichier_nco2, 'r') as f:
        CO2_SAVED = json.load(f)

    print('Old values : slope =', CO2_SAVED['slope'], 'intercept =', CO2_SAVED['intercept'])
    
    ###### SCRIPPS Processing ######
    if source in ["scripps", "both"]:    
        
        # CSV FILES 
        files = [f for f in os.listdir(base_dir) if f.endswith('.csv') and 'lock' not in f]
        
        # LISTS 
        slopes, intercepts, CO2 = [], [], []
        CO2_values = []; P_recent = [];DATES = []
        
        
        for file in files[:]:
            print('Processing SCRIPPS file:', file)
            file_path = os.path.join(base_dir, file)
            with open(file_path, newline='') as csvfile:
        
                # PASS HEADER OF FILES
                lines = [line for line in csvfile if not line.strip().startswith('"')][3:]
            
            dates, co2_values = [], []
            for row in lines:
                parts = row.split(',')
                try:
                    # TAKE THE DECIMAL YEAR AND THE CO2 VALUES
                    dates.append(float(parts[3])) 
                    co2_values.append(float(parts[-2]))
                except:
                    print(f'file {file} do not have datas')
                    continue
            
            
            dates = np.array(dates)
            co2_values = np.array(co2_values)
            if dates.size == 0:
                files.remove(file)
                print(f'file {file} do not have datas')
                continue
            
            # MASK FOR THE LINEAR REGRESSION
            recent_mask = dates >= year_mask
            if not np.any(recent_mask) or not np.any(dates>=year_max ):
                print(f'file {file} do not have data after {dates[-1]} and it is smaller than year_max = {year_max}')
                files.remove(file)
                continue
            
            # LIST FOR PLOTS OF DATA
            DATES.append(dates)
            CO2_values.append(co2_values)

            # MASKED DATA FOR LINEAR REGRESSION
            dates_recent = dates[recent_mask]
            co2_recent = co2_values[recent_mask]
            
            
            # LINEAR REGRESSION
            coef_recent = np.polyfit(dates_recent, co2_recent, 1)
            p_recent = np.poly1d(coef_recent)
            slopes.append(p_recent[1])
            intercepts.append(p_recent[0])

            # GET THE VARIATIONS OF THE CO2
            co2_variation = co2_recent - (p_recent(dates_recent))
            
            # INTERPOLATE THE VARIATION WITH AVERAGE THEM OVER ONE YEAR
            dates_shared = np.linspace(year_mask, year_max, 2000 * (year_max - year_mask))
            spline = make_interp_spline(dates_recent, co2_variation, k=3)
            co2_interp = spline(dates_shared)
            co2_i = sum(co2_interp[i*2000:(i+1)*2000] for i in range((year_max - year_mask))) / (year_max - year_mask)
            CO2.append(co2_i)

        if plot:
            plt.figure()
            plt.axvline(year_max, label = 'year_max',ls = '--')
            for i,file in enumerate(files):
                if not np.any(recent_mask):
                    continue
                plt.plot(DATES[i], CO2_values[i], alpha=0.5, label=f"{file}")
                
                
                plt.title('SCRIPPS')
                plt.legend()
                plt.grid()


        if slopes:
            # SAVE DATAS IN JSON FILE
            CO2_SAVED['variations'] = (- np.mean(CO2,axis = 0)).tolist()
            CO2_SAVED['slope'] = float(np.mean(slopes))
            CO2_SAVED['intercept'] = float(np.mean(intercepts))
            with open(fichier_nco2, 'w') as f:
                json.dump(CO2_SAVED, f)
            print('New values : slope =', CO2_SAVED['slope'], 'intercept =', CO2_SAVED['intercept'])



    if source in ["noaa", "both"]:
        if data_txt_file is None:
            raise ValueError("You must provide data_txt_file if source is 'noaa' or both")
        print('Processing NOAA file:', data_txt_file)
        co2_val, date = [], []
        with open(os.path.join(base_dir, data_txt_file), 'r') as f:
            try:
                # SKIP HEADER
                lines = f.readlines()[160:]
                for line in lines:
                    parts = line.split()
                    try:
                        val = float(parts[10])
                        d = float(parts[8])
                        if val != -999.99:
                            co2_val.append(val)
                            date.append(d)
                    except:
                        continue
            except:
                print('file given is not compatible. Are you sure it is a NOAA file ?')
                
        
        date = np.array(date)
        co2_val = np.array(co2_val)
        
        # MASK FOR THE LINEAR REGRESSION
        mask = (date > year_mask) & (date < year_max)
        if not np.any(mask) or not np.any(date>=year_max ):
            print('reduce year_max as the last decimal year file is {date[-1]}')
            raise ValueError('{year_max}>{date[-1]}')
        
        # MASKED DATA FOR LINEAR REGRESSION
        date = date[mask]
        co2_val = co2_val[mask]
        
        # LINEAR REGRESSION to delete bad data
        coef = np.polyfit(date, co2_val, 1)
        p = np.poly1d(coef)
        
        # DELETE BAD DATA
        variations = np.abs(co2_val - p(date))
        mask_error = variations < np.percentile(variations, 90)
        date = date[mask_error]
        co2_val = co2_val[mask_error]
    
        
        # MEAN OVER DOUBLE DATES 
        unique_date, inverse_indices = np.unique(date, return_inverse=True)
        sum_co2 = np.bincount(inverse_indices, weights=co2_val)
        count_co2 = np.bincount(inverse_indices)
        mean_co2 = sum_co2 / count_co2

        # 2nd LINEAR REGRESSION to get variations
        coef = np.polyfit(date, co2_val, 1)
        p = np.poly1d(coef)
        co2_variations = mean_co2 - p(unique_date)
        
        
        # INTERPOLATE THE VARIATION WITH AVERAGE THEM OVER ONE YEA
        dates_interp = np.linspace(year_mask, year_max, 2000 * (year_max - year_mask))
        spline = make_interp_spline(unique_date, co2_variations, k=1)
        co2_interp = spline(dates_interp)
        co2_i = 0
        for i in range(year_max - year_mask):
            co2_i += co2_interp[i*2000:(i+1)*2000]/ (year_max-year_mask)  
            
        # SMOOTHEN THE DATA
        co2_smooth = gaussian_filter1d(co2_i, sigma=101, mode='wrap')
        
        # SAVE DATAS IN JSON FILE
        if source == 'noaa':
            CO2_SAVED['slope'] = p[1]
            CO2_SAVED['intercept'] = p[0]
            print('New values : slope =', CO2_SAVED['slope'], 'intercept =', CO2_SAVED['intercept'])
        CO2_SAVED['variations'] = co2_smooth.tolist()
        
        with open(fichier_nco2, 'w') as f:
            json.dump(CO2_SAVED, f)

        date_co2 = np.linspace(year_max-1, year_max, 2000)
        if plot:
            plt.figure()
            plt.plot(date_co2, co2_i + CO2_SAVED['slope'] * date_co2 + CO2_SAVED['intercept'], label="NOAA raw regression")
            plt.plot(date_co2, co2_smooth + CO2_SAVED['slope'] * date_co2 + CO2_SAVED['intercept'], label="NOAA + smoothed residuals")
            plt.title("Interpolated CO₂ 2024–2025")
            plt.legend()
            plt.grid()
            # plt.show()



    if plot:
        plt.show()
    return None



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Update CO₂ model from Scripps and/or NOAA data")
    parser.add_argument("--base_dir", type=str, default=".",
                        help="Directory containing Scripps CSV data (default: current directory)")
    parser.add_argument("--file", type=str,
                        help="NOAA data TXT file (optional if only using Scripps)")
    parser.add_argument("--source", type=str, choices=["scripps", "noaa", "both"], default="noaa",
                        help="Force a specific data source ('scripps', 'noaa', or 'both'); otherwise noaa")
    parser.add_argument("--year", type=int, default=2015,
                        help="Start year for linear regression (default: 2015)")
    parser.add_argument("--year_max", type=int, default=2024,
                        help="END year for linear regression (default: 2024)")
    parser.add_argument("--plot", action="store_true", help="Enable plotting")

    args = parser.parse_args()
    print(args.plot)
    
    # Auto-detect sources if not explicitly provided
    detected_sources = []
    scripps_files = [f for f in os.listdir(args.base_dir) if f.endswith('.csv')]
    if scripps_files:
        detected_sources.append("scripps")
    if args.file:
        detected_sources.append("noaa")

    if args.source:
        source = args.source
    else:
        if not detected_sources:
            raise RuntimeError("No valid data source detected. Provide Scripps CSVs in base_dir or a NOAA file with --file.")
        source = "both" if len(detected_sources) == 2 else detected_sources[0]
    
    update_co2_and_analyze_file(
        base_dir=args.base_dir,
        data_txt_file=args.file,
        source=source,
        year_mask=args.year,
        year_max = args.year_max,
        plot=args.plot
    )


# update_co2_and_analyze_file(data_txt_file = 'co2_ush_surface-flask_1_ccgg_event.txt',plot = True)