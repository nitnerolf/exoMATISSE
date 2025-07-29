Exoplanet spectrum extraction for MATISSE !

phase_correction.py correct the phase dispertion and from the mat_raw_estimates files, it generates FITS files for each frame of the star and the planet, containing the corrected interferometric quantities. The achromatic phase dispersion is corrected using the model of *Veronin and Zelthikov 2017*. 
The instrumental phase induced by the optical path difference (OPD) between MATISSE and GRAVITY and/or by a piston is also corrected. To correct it you need to indicate two limits (opd_min, opd_max) in the 'input parameters' section.
You have to specify the path of the raw data and the path where you want the resulting FITS files to be saved.


launch_array.sh is a script to launch the first step of the spectrum extraction. The script is designed to be run from the command line: `./launch_array.sh`. It will run the script prepare_astrometry.py which compute the mean correlated flux of the star. Afterwards, it will launch the astrometry_multiprocessing_array.slurm, which will put each frame on one core of the cluster running astrometry_multiprocessing_realimagpoly.py. 

To correctly run the astrometry_multiprocessing_realimagpoly.py, you need to set the “input parameters” section. 
(x,y) are the coordinate of the planet + an offset to build the chi2 map where you will test the planet position.
Cps_file is the path where you have the star-to-planet contrast file, or in a first approximation you can use a flat contrast (i.e. 'flat 1.5e-4' ).
n_poly is the degree of the polynomial used to fit the stellar contamination at the planet position.
stellar_residuals_poly_coeffs_init are the initial guess for the coefficient of the stellar contamination. Keep in mind that you have 2 polynomials for each frame (real and imaginary part), so you have 2*(n_poly+1) coefficients to guess.
You also need to specify how the errors on the coherent flux ratio are computed. To do this, you must set the weighted_least_squares boolean accordingly. The error can either be derived from the pipeline uncertainties ('errors_from_pipeline') or estimated using the standard deviation over a single exposure ('errors_from_data').

astrometry_complex_model_drawer_realimagfit.py is a script to draw the astrometric figures: the chi2 maps, model of the data (amplitude, phase, real and imaginary part), the polynomial fit of the stellar contamination and the residuals of the fit. All this is done per frame, for all baselines, and for all frames.
To run this script you need to specify the **same** input parameters as in astrometry_multiprocessing_realimagpoly.py. You also need to specify the path to the directory where the results of the astrometry are stored and the path to the directory where you want to save the figures.

spectrometry_complex_nochi2.py extract the spectrum from the later results on the astrometry. It also must be parametrized the **same** way as astrometry_multiprocessing_realimagpoly.py. The path to the directory where the results of the astrometry are stored must also be specified. The script will also ask for the path to the directory where you want to save the spectrum and figures. 


bin_data.py is to spectrally bin the data. The default binning value is 5. As well as the other scripts you need to specify the path of your **FITS (output files of the phase correction script)**  files and the path where you want to save the binned data.
