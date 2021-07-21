#!/usr/bin/env python -O

"""
This script reads in SN 2010jl data from fits files and text files and
bins the data and makes several plots.
Version 1.0
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib.font_manager as fm
import matplotlib.type1font as t1f
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, MaxNLocator)
import numpy as np
import pandas as pd
from scipy import stats
from scipy import optimize
import math
import json
import os.path
import glob
import logging
from astropy.io import fits
from astropy import units as u
from astropy.visualization import quantity_support
import sys

# The inv_quad funtion in this script is recursive and this
# relaxes the number of iterations.
sys.setrecursionlimit(1500)

# Don't print a full array if it's bigger than 20 elements.
# Use a threshold of 'np.inf' to print all elements.
np.set_printoptions(threshold=20)

# Uncommenting the following will restore the plotting parameters to classic
# rather than 2.0.
# See https://matplotlib.org/users/dflt_style_changes.html
#mpl.style.use('classic')

# rc parameters
# These could also be set in the ~/.matplotlib/matplotlibrc file.

# use tex fonts, set the font family, and set the base font size
mpl.rc('text', usetex=True)
mpl.rc('font',**{'family':'serif','serif':['Computer Modern']})
mpl.rc('font',**{'size':14})

# This may be necessary to properly scale a postscript file.
#mpl.rc('ps',usedistiller=xpdf)

# Restore some classic mpl styles that were changed with mpl 2.0.
# See https://matplotlib.org/users/dflt_style_changes.html 
#mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.major.width'] = 0.5
mpl.rcParams['ytick.major.width'] = 0.5
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12

"""
The following block sets up logging that can be used for  
debugging purposes. The basicConfig is the basic setup 
but this script will use a more sophisticated method with 
multiple handlers.
"""

#logging.basicConfig(filename='sn2010jl.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# create a logger called 'main_logger'
logger = logging.getLogger(__name__)

# set the logging level to:
# NOTSET (0), DEBUG (10), INFO (20), WARNING (30), # ERROR (40), CRITICAL (50)
# The number in parentheses is the numberical level 
logger.setLevel(logging.DEBUG)

# create a file handler which logs DEBUG and higher messages
fh = logging.FileHandler('sn2010jl.log')
fh.setLevel(logging.DEBUG)

# create a console handler (i.e. stdout) that prints INFO and higher messages
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create a formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

# First log the runtime of the script
logger.debug("Script runtime.")

# Read constant parameters from a json configuration file
config = json.load(open("qu_config_sn2010jl.json"))

"""
In the following block use the number of directories that correspond
to different SPOL epochs to read in all the needed fits files.

First define the variable lists.  Note the square brackets make these of type
list rather than tuple. One cannont append to a tuple.
"""

q_header = []
q = []

u_header = []
u = []

qsig_header = []
qsig = []

usig_header = []
usig = []

qsum_header = []
qsum = []

usum_header = []
usum = []

flux_header = []
flux = []

for e_dir in config["epoch_dirs"]:

    q_header.append(fits.open(os.path.join(config["base_path"],e_dir,config["q_file"]))[0].header)
    q.append(fits.open(os.path.join(config["base_path"],e_dir,config["q_file"]))[0].data)

    u_header.append(fits.open(os.path.join(config["base_path"],e_dir,config["u_file"]))[0].header)
    u.append(fits.open(os.path.join(config["base_path"],e_dir,config["u_file"]))[0].data)

    qsig_header.append(fits.open(os.path.join(config["base_path"],e_dir,config["qsig_file"]))[0].header)
    qsig.append(fits.open(os.path.join(config["base_path"],e_dir,config["qsig_file"]))[0].data)

    usig_header.append(fits.open(os.path.join(config["base_path"],e_dir,config["usig_file"]))[0].header)
    usig.append(fits.open(os.path.join(config["base_path"],e_dir,config["usig_file"]))[0].data)

    qsum_header.append(fits.open(os.path.join(config["base_path"],e_dir,config["qsum_file"]))[0].header)
    qsum.append(fits.open(os.path.join(config["base_path"],e_dir,config["qsum_file"]))[0].data)

    usum_header.append(fits.open(os.path.join(config["base_path"],e_dir,config["usum_file"]))[0].header)
    usum.append(fits.open(os.path.join(config["base_path"],e_dir,config["usum_file"]))[0].data)

    flux_header.append(fits.open(os.path.join(config["base_path"],e_dir,config["flux_file"]))[0].header)
    flux.append(fits.open(os.path.join(config["base_path"],e_dir,config["flux_file"]))[0].data)

# This was a test to see whether it would be better to use a panda dataframe.
# It looks good but I haven't explored the implications of running with this.
df = pd.DataFrame(q)
ddf = df.T

print("Type of q_header")
print(type(q_header))
print("Type of q")
print(type(q))
print("Data from q")
print(q)
print(len(q[0]), len(q[1]), len(q[2]), len(q[3]), len(q[4]))
print(len(q[5]), len(q[6]), len(q[7]), len(q[8]), len(q[9]), len(q[10]))
print("Pandas DataFrame")
print(ddf)

# Redefine the list of vectors (or arrays) into numpy arrays.
# There's not an easy way to define the numpy array first and then read
# data into it but appending to a list and then converting is easy
# Note, it's required to use dtype=object when the individual rows in the
# array vary in length (are ragged).
q = np.asarray(q,dtype=object)
u = np.asarray(u,dtype=object)
qsig = np.asarray(qsig,dtype=object)
usig = np.asarray(usig,dtype=object)
qsum = np.asarray(qsum,dtype=object)
usum = np.asarray(usum,dtype=object)
flux = np.asarray(flux,dtype=object)

print("Type of numpy array q")
print(type(q))
print("Data from numpy array q")
print(q)
print(q.shape)


"""
In this section we calculate the wavelength values for each epoch based on
the header information.
Here we assume q, qsig, and qsum (and separately u, usig and usum) all have
the same wavelength bins. This is a good assumption but I've also confirmed
it in some individual spectra.
"""

q_lambdas = []
u_lambdas = []
flux_lambdas = []


for q_h, u_h, f_h, q_d, u_d, f_d in zip(q_header, u_header, flux_header, q, u, flux):
    q_lambdas.append((((np.arange(len(q_d)) + 1.0 - q_h['CRPIX1']) * q_h['CD1_1']) + q_h['CRVAL1'])/(1.0 + config["redshift"]))
    u_lambdas.append((((np.arange(len(u_d)) + 1.0 - u_h['CRPIX1']) * u_h['CD1_1']) + u_h['CRVAL1'])/(1.0 + config["redshift"]))
    flux_lambdas.append((((np.arange(len(f_d)) + 1.0 - f_h['CRPIX1']) * f_h['CD1_1']) + f_h['CRVAL1'])/(1.0 + config["redshift"]))

# Redefine the list of vectors (or arrays) into a numpy arrays.
# Again, it's required to set the datatype to object when the individual
# vectors in the array are of different length.
q_lambdas = np.asarray(q_lambdas,dtype=object)
u_lambdas = np.asarray(u_lambdas,dtype=object)
flux_lambdas = np.asarray(flux_lambdas,dtype=object)


"""
Read in the Lick Observatory text files.
"""

lick_lambdas_epoch = []
lick_flux_big_epoch = []

lick_q_epoch = []
lick_qsig_epoch = []

lick_u_epoch = []
lick_usig_epoch = []

lick_p_epoch = []
lick_psig_epoch = []

lick_theta_epoch = []
lick_thetasig_epoch = []

lick_lambdas = []
lick_flux_big = []
lick_flux = []

lick_q = []
lick_qsig = []

lick_u = []
lick_usig = []

lick_Qflx = []

lick_Uflx = []

lick_p = []
lick_psig = []

lick_theta = []
lick_thetasig = []

# MAKE SURE TO USE 'sorted' here since 'glob' is unordered and therefore the
# data will be out of order)
lick_data_files = sorted(glob.glob(os.path.join(config["base_path"],'lick','sn2010jl_kast_*_specpol.txt')))

for lickfile in lick_data_files:
    lick_lambdas_epoch,lick_flux_big_epoch,lick_q_epoch,lick_qsig_epoch,lick_u_epoch,lick_usig_epoch,lick_p_epoch,lick_psig_epoch,lick_theta_epoch,lick_thetasig_epoch = np.genfromtxt(lickfile,delimiter=None,dtype=float,comments='#',unpack=True)
    lick_lambdas.append(lick_lambdas_epoch / (1.0 + config["redshift"]))
    lick_flux_big.append(lick_flux_big_epoch)
    lick_flux.append(np.multiply(lick_flux_big_epoch,1.0e-15))
    lick_q.append(lick_q_epoch)
    lick_qsig.append(lick_qsig_epoch)
    lick_u.append(lick_u_epoch)
    lick_usig.append(lick_usig_epoch)
    lick_p.append(lick_p_epoch)
    lick_psig.append(lick_psig_epoch)
    lick_theta.append(lick_theta_epoch)
    lick_thetasig.append(lick_thetasig_epoch)

print("Type of lick_lambdas")
print(type(lick_lambdas))
print("Data from lick_lambdas")
print(lick_lambdas)
print(len(lick_q[0]), len(lick_q[1]), len(lick_q[2]), len(lick_q[3]), len(lick_q[4]), len(lick_q[5]))

# Redefine the list of vectors (or arrays) into a numpy arrays.
# I believe all these vectors are of the same length so there it isn't
# necessary to add dtype=object
lick_lambdas = np.asarray(lick_lambdas)
lick_flux_big = np.asarray(lick_flux_big)
lick_flux = np.asarray(lick_flux)
lick_q = np.asarray(lick_q)
lick_qsig = np.asarray(lick_qsig)
lick_u = np.asarray(lick_u)
lick_usig = np.asarray(lick_usig)
lick_p = np.asarray(lick_p)
lick_psig = np.asarray(lick_psig)
lick_theta = np.asarray(lick_theta)
lick_thetasig = np.asarray(lick_thetasig)

print("Type of lick_q after np.asarray")
print(type(lick_q))
print("Data from lick_q after np.asarray")
print(lick_q)
print(lick_q.shape)

"""
Here we'll read in the photometry from Fransson et al.
"""

photfile = os.path.join(config["base_path"],config["phot_file"])

phot_jd, phot_epoch, phot_u, phot_u_err, phot_B, phot_B_err, phot_V, phot_V_err, phot_r, phot_r_err, phot_i, phot_i_err = np.genfromtxt(photfile,delimiter=None,dtype=float,skip_header=1,comments='#',usecols=(range(12)),unpack=True)

"""
Here we'll create the binned data
"""

bin_by = config["bin_by"]
num_bins = math.floor(len(q_lambdas[0])/bin_by)

print('number of bins')
print(num_bins)

lick_bin_by = config["lick_bin_by"]
lick_num_bins = math.floor(len(lick_lambdas[0])/lick_bin_by)

print('Number of Lick bins')
print(lick_num_bins)

"""
This function is defined as the adding errors in quadrature as
should be done for binning the variance spectra.
The result is sqrt( (sig_0)^2 + (sig_1)^2 + (sig_2)^2 + ...)
"""

def quad(numList):
    if len(numList) == 1:
        return numList[0]
    else:
        theResult = math.sqrt((numList[0])**2 + (quad(numList[1:])**2))
        return theResult

"""
7/9/2021 - THIS FUNCTION WON'T BE USED, IT WASN'T THE CORRECT CALCULATION.
This function is defined as the adding variances in inverse quadrature as
should be done for binning the variance spectra.
The result is 1.0/sqrt( (1/sig_0)^2 + (1/sig_1)^2 + (1/sig_2)^2 + ...)
"""

def inv_quad(numList):
    if len(numList) == 1:
        return numList[0]
    else:
        theResult = 1.0/(math.sqrt((1.0/numList[0])**2 + (1.0/inv_quad(numList[1:])**2)))
        return theResult

Qflx = []
Uflx = []
Qflx_bin = []
Uflx_bin = []

Qflx_bin_band1 = []
Uflx_bin_band1 = []
Qflx_bin_band2 = []
Uflx_bin_band2 = []
Qflx_bin_band3 = []
Uflx_bin_band3 = []

Qflx_bin_edges = []
Uflx_bin_edges = []

Qflx_bin_num = []
Uflx_bin_num = []

qsum_bin = []
usum_bin = []

qsum_bin_band1 = []
usum_bin_band1 = []
qsum_bin_band2 = []
usum_bin_band2 = []
qsum_bin_band3 = []
usum_bin_band3 = []

qsig_bin = []
usig_bin = []

qsig_bin_band1 = []
usig_bin_band1 = []
qsig_bin_band2 = []
usig_bin_band2 = []
qsig_bin_band3 = []
usig_bin_band3 = []

for q_d, u_d, qsum_d, usum_d, qsig_d, usig_d, q_lam, u_lam in zip(q, u, qsum, usum, qsig, usig, q_lambdas, u_lambdas):
    Qflx.append(q_d * qsum_d)
    Uflx.append(u_d * usum_d)
    Qflx_bin.append(stats.binned_statistic(q_lam, (q_d * qsum_d), statistic='sum', bins=num_bins)[0])
    Uflx_bin.append(stats.binned_statistic(u_lam, (u_d * usum_d), statistic='sum', bins=num_bins)[0])
    Qflx_bin_edges.append(stats.binned_statistic(q_lam, (q_d * qsum_d), statistic='sum', bins=num_bins)[1])
    Uflx_bin_edges.append(stats.binned_statistic(u_lam, (u_d * usum_d), statistic='sum', bins=num_bins)[1])
    Qflx_bin_num.append(stats.binned_statistic(q_lam, (q_d * qsum_d), statistic='sum', bins=num_bins)[2])
    Uflx_bin_num.append(stats.binned_statistic(u_lam, (u_d * usum_d), statistic='sum', bins=num_bins)[2])
    qsum_bin.append(stats.binned_statistic(q_lam, qsum_d, statistic='sum', bins=num_bins)[0])
    usum_bin.append(stats.binned_statistic(u_lam, usum_d, statistic='sum', bins=num_bins)[0])
    qsum_bin_band1.append(stats.binned_statistic(q_lam, qsum_d, statistic='sum', bins=1, range=config["band1_range"])[0])
    usum_bin_band1.append(stats.binned_statistic(u_lam, usum_d, statistic='sum', bins=1, range=config["band1_range"])[0])
    qsum_bin_band2.append(stats.binned_statistic(q_lam, qsum_d, statistic='sum', bins=1, range=config["band2_range"])[0])
    usum_bin_band2.append(stats.binned_statistic(u_lam, usum_d, statistic='sum', bins=1, range=config["band2_range"])[0])
    qsum_bin_band3.append(stats.binned_statistic(q_lam, qsum_d, statistic='sum', bins=1, range=config["band3_range"])[0])
    usum_bin_band3.append(stats.binned_statistic(u_lam, usum_d, statistic='sum', bins=1, range=config["band3_range"])[0])
    qsig_bin.append(stats.binned_statistic(q_lam, qsig_d, statistic=inv_quad, bins=num_bins)[0])
    usig_bin.append(stats.binned_statistic(u_lam, usig_d, statistic=inv_quad, bins=num_bins)[0])
    qsig_bin_band1.append(stats.binned_statistic(q_lam, qsig_d, statistic=inv_quad, bins=1, range=config["band1_range"])[0])
    usig_bin_band1.append(stats.binned_statistic(u_lam, usig_d, statistic=inv_quad, bins=1, range=config["band1_range"])[0])
    qsig_bin_band2.append(stats.binned_statistic(q_lam, qsig_d, statistic=inv_quad, bins=1, range=config["band2_range"])[0])
    usig_bin_band2.append(stats.binned_statistic(u_lam, usig_d, statistic=inv_quad, bins=1, range=config["band2_range"])[0])
    qsig_bin_band3.append(stats.binned_statistic(q_lam, qsig_d, statistic=inv_quad, bins=1, range=config["band3_range"])[0])
    usig_bin_band3.append(stats.binned_statistic(u_lam, usig_d, statistic=inv_quad, bins=1, range=config["band3_range"])[0])
    Qflx_bin_band1.append(stats.binned_statistic(q_lam, (q_d * qsum_d), statistic='sum', bins=1, range=config["band1_range"])[0])
    Uflx_bin_band1.append(stats.binned_statistic(u_lam, (u_d * usum_d), statistic='sum', bins=1, range=config["band1_range"])[0])
    Qflx_bin_band2.append(stats.binned_statistic(q_lam, (q_d * qsum_d), statistic='sum', bins=1, range=config["band2_range"])[0])
    Uflx_bin_band2.append(stats.binned_statistic(u_lam, (u_d * usum_d), statistic='sum', bins=1, range=config["band2_range"])[0])
    Qflx_bin_band3.append(stats.binned_statistic(q_lam, (q_d * qsum_d), statistic='sum', bins=1, range=config["band3_range"])[0])
    Uflx_bin_band3.append(stats.binned_statistic(u_lam, (u_d * usum_d), statistic='sum', bins=1, range=config["band3_range"])[0])

# Redefine the list of vectors (or arrays) into a numpy arrays.
Qflx = np.asarray(Qflx,dtype=object)
Uflx = np.asarray(Uflx,dtype=object)
Qflx_bin = np.asarray(Qflx_bin)
Uflx_bin = np.asarray(Uflx_bin)
Qflx_bin_edges = np.asarray(Qflx_bin_edges,dtype=object)
Uflx_bin_edges = np.asarray(Uflx_bin_edges,dtype=object)
Qflx_bin_num = np.asarray(Qflx_bin_num,dtype=object)
Uflx_bin_num = np.asarray(Uflx_bin_num,dtype=object)
qsum_bin = np.asarray(qsum_bin)
usum_bin = np.asarray(usum_bin)
qsum_bin_band1 = np.asarray(qsum_bin_band1)
usum_bin_band1 = np.asarray(usum_bin_band1)
qsum_bin_band2 = np.asarray(qsum_bin_band2)
usum_bin_band2 = np.asarray(usum_bin_band2)
qsum_bin_band3 = np.asarray(qsum_bin_band3)
usum_bin_band3 = np.asarray(usum_bin_band3)
qsig_bin = np.asarray(qsig_bin)
usig_bin = np.asarray(usig_bin)
qsig_bin_band1 = np.asarray(qsig_bin_band1)
usig_bin_band1 = np.asarray(usig_bin_band1)
qsig_bin_band2 = np.asarray(qsig_bin_band2)
usig_bin_band2 = np.asarray(usig_bin_band2)
qsig_bin_band3 = np.asarray(qsig_bin_band3)
usig_bin_band3 = np.asarray(usig_bin_band3)
Qflx_bin_band1 = np.asarray(Qflx_bin_band1)
Uflx_bin_band1 = np.asarray(Uflx_bin_band1)
Qflx_bin_band2 = np.asarray(Qflx_bin_band2)
Uflx_bin_band2 = np.asarray(Uflx_bin_band2)
Qflx_bin_band3 = np.asarray(Qflx_bin_band3)
Uflx_bin_band3 = np.asarray(Uflx_bin_band3)

lick_Qflx = []
lick_Uflx = []
lick_Qflx_bin = []
lick_Uflx_bin = []
lick_Qflx_bin_band1 = []
lick_Uflx_bin_band1 = []
lick_Qflx_bin_band2 = []
lick_Uflx_bin_band2 = []
lick_Qflx_bin_band3 = []
lick_Uflx_bin_band3 = []
lick_Qflx_bin_edges = []
lick_Uflx_bin_edges = []
lick_Qflx_bin_num = []
lick_Uflx_bin_num = []
lick_flux_big_bin = []
lick_flux_big_bin_band1 = []
lick_flux_big_bin_band2 = []
lick_flux_big_bin_band3 = []
lick_qsig_bin = []
lick_usig_bin = []
lick_qsig_bin_band1 = []
lick_usig_bin_band1 = []
lick_qsig_bin_band2 = []
lick_usig_bin_band2 = []
lick_qsig_bin_band3 = []
lick_usig_bin_band3 = []

for l_q_d, l_u_d, l_qsig_d, l_usig_d, l_flux_big_d, l_lam in zip(lick_q, lick_u, lick_qsig, lick_usig, lick_flux_big, lick_lambdas):
    lick_Qflx.append(l_q_d * l_flux_big_d)
    lick_Uflx.append(l_u_d * l_flux_big_d)
    lick_Qflx_bin.append(stats.binned_statistic(l_lam, (l_q_d * l_flux_big_d), statistic='sum', bins=lick_num_bins)[0])
    lick_Uflx_bin.append(stats.binned_statistic(l_lam, (l_u_d * l_flux_big_d), statistic='sum', bins=lick_num_bins)[0])
    lick_Qflx_bin_edges.append(stats.binned_statistic(l_lam, (l_q_d * l_flux_big_d), statistic='sum', bins=lick_num_bins)[1])
    lick_Uflx_bin_edges.append(stats.binned_statistic(l_lam, (l_u_d * l_flux_big_d), statistic='sum', bins=lick_num_bins)[1])
    lick_Qflx_bin_num.append(stats.binned_statistic(l_lam, (l_q_d * l_flux_big_d), statistic='sum', bins=lick_num_bins)[2])
    lick_Uflx_bin_num.append(stats.binned_statistic(l_lam, (l_u_d * l_flux_big_d), statistic='sum', bins=lick_num_bins)[2])
    lick_flux_big_bin.append(stats.binned_statistic(l_lam, l_flux_big_d, statistic='sum', bins=lick_num_bins)[0])
    lick_flux_big_bin_band1.append(stats.binned_statistic(l_lam, l_flux_big_d, statistic='sum', bins=1, range=config["band1_range"])[0])
    lick_flux_big_bin_band2.append(stats.binned_statistic(l_lam, l_flux_big_d, statistic='sum', bins=1, range=config["band2_range"])[0])
    lick_flux_big_bin_band3.append(stats.binned_statistic(l_lam, l_flux_big_d, statistic='sum', bins=1, range=config["band3_range"])[0])
    lick_qsig_bin.append(stats.binned_statistic(l_lam, l_qsig_d, statistic=inv_quad, bins=lick_num_bins)[0])
    lick_usig_bin.append(stats.binned_statistic(l_lam, l_usig_d, statistic=inv_quad, bins=lick_num_bins)[0])
    lick_qsig_bin_band1.append(stats.binned_statistic(l_lam, l_qsig_d, statistic=inv_quad, bins=1, range=config["band1_range"])[0])
    #lick_qsig_bin_band1.append(stats.binned_statistic(l_lam, l_qsig_d, statistic='sum', bins=1, range=config["band1_range"])[0])
    lick_usig_bin_band1.append(stats.binned_statistic(l_lam, l_usig_d, statistic=inv_quad, bins=1, range=config["band1_range"])[0])
    lick_qsig_bin_band2.append(stats.binned_statistic(l_lam, l_qsig_d, statistic=inv_quad, bins=1, range=config["band2_range"])[0])
    lick_usig_bin_band2.append(stats.binned_statistic(l_lam, l_usig_d, statistic=inv_quad, bins=1, range=config["band2_range"])[0])
    lick_qsig_bin_band3.append(stats.binned_statistic(l_lam, l_qsig_d, statistic=inv_quad, bins=1, range=config["band3_range"])[0])
    lick_usig_bin_band3.append(stats.binned_statistic(l_lam, l_usig_d, statistic=inv_quad, bins=1, range=config["band3_range"])[0])
    lick_Qflx_bin_band1.append(stats.binned_statistic(l_lam, (l_q_d * l_flux_big_d), statistic='sum', bins=1, range=config["band1_range"])[0])
    lick_Uflx_bin_band1.append(stats.binned_statistic(l_lam, (l_u_d * l_flux_big_d), statistic='sum', bins=1, range=config["band1_range"])[0])
    lick_Qflx_bin_band2.append(stats.binned_statistic(l_lam, (l_q_d * l_flux_big_d), statistic='sum', bins=1, range=config["band2_range"])[0])
    lick_Uflx_bin_band2.append(stats.binned_statistic(l_lam, (l_u_d * l_flux_big_d), statistic='sum', bins=1, range=config["band2_range"])[0])
    lick_Qflx_bin_band3.append(stats.binned_statistic(l_lam, (l_q_d * l_flux_big_d), statistic='sum', bins=1, range=config["band3_range"])[0])
    lick_Uflx_bin_band3.append(stats.binned_statistic(l_lam, (l_u_d * l_flux_big_d), statistic='sum', bins=1, range=config["band3_range"])[0])

# Redefine the list of vectors (or arrays) into a numpy arrays.
lick_Qflx = np.asarray(lick_Qflx)
lick_Uflx = np.asarray(lick_Uflx)
lick_Qflx_bin = np.asarray(lick_Qflx_bin)
lick_Uflx_bin = np.asarray(lick_Uflx_bin)
lick_Qflx_bin_edges = np.asarray(lick_Qflx_bin_edges)
lick_Uflx_bin_edges = np.asarray(lick_Uflx_bin_edges)
lick_Qflx_bin_num = np.asarray(lick_Qflx_bin_num)
lick_Uflx_bin_num = np.asarray(lick_Uflx_bin_num)
lick_flux_big_bin = np.asarray(lick_flux_big_bin)
lick_qsig_bin = np.asarray(lick_qsig_bin)
lick_usig_bin = np.asarray(lick_usig_bin)
lick_qsig_bin_band1 = np.asarray(lick_qsig_bin_band1)
lick_usig_bin_band1 = np.asarray(lick_usig_bin_band1)
lick_qsig_bin_band2 = np.asarray(lick_qsig_bin_band2)
lick_usig_bin_band2 = np.asarray(lick_usig_bin_band2)
lick_qsig_bin_band3 = np.asarray(lick_qsig_bin_band3)
lick_usig_bin_band3 = np.asarray(lick_usig_bin_band3)
lick_Qflx_bin_band1 = np.asarray(lick_Qflx_bin_band1)
lick_Uflx_bin_band1 = np.asarray(lick_Uflx_bin_band1)
lick_Qflx_bin_band2 = np.asarray(lick_Qflx_bin_band2)
lick_Uflx_bin_band2 = np.asarray(lick_Uflx_bin_band2)
lick_Qflx_bin_band3 = np.asarray(lick_Qflx_bin_band3)
lick_Uflx_bin_band3 = np.asarray(lick_Uflx_bin_band3)

"""
We can't divide multiple lists but we can use list comprehension to separate
and divide.  The multiplication by 1.0 is to make the list a float.
"""
q_bin = np.divide(Qflx_bin,qsum_bin)
print('q_bin')
print(type(q_bin))
print(q_bin.dtype)
print(q_bin)
u_bin = np.divide(Uflx_bin,usum_bin)
q_bin_band1 = np.divide(Qflx_bin_band1,qsum_bin_band1)
u_bin_band1 = np.divide(Uflx_bin_band1,usum_bin_band1)
q_bin_band2 = np.divide(Qflx_bin_band2,qsum_bin_band2)
u_bin_band2 = np.divide(Uflx_bin_band2,usum_bin_band2)
q_bin_band3 = np.divide(Qflx_bin_band3,qsum_bin_band3)
u_bin_band3 = np.divide(Uflx_bin_band3,usum_bin_band3)
lick_q_bin = np.divide(lick_Qflx_bin,lick_flux_big_bin)
lick_u_bin = np.divide(lick_Uflx_bin,lick_flux_big_bin)
lick_q_bin_band1 = np.divide(lick_Qflx_bin_band1,lick_flux_big_bin_band1)
lick_u_bin_band1 = np.divide(lick_Uflx_bin_band1,lick_flux_big_bin_band1)
lick_q_bin_band2 = np.divide(lick_Qflx_bin_band2,lick_flux_big_bin_band2)
lick_u_bin_band2 = np.divide(lick_Uflx_bin_band2,lick_flux_big_bin_band2)
lick_q_bin_band3 = np.divide(lick_Qflx_bin_band3,lick_flux_big_bin_band3)
lick_u_bin_band3 = np.divide(lick_Uflx_bin_band3,lick_flux_big_bin_band3)
#q_bin = [(ai * 1.0)/bi for ai,bi in zip(Qflx_bin,qsum_bin)]
#u_bin = [(ai * 1.0)/bi for ai,bi in zip(Uflx_bin,usum_bin)]
#q_bin_band1 = [(ai * 1.0)/bi for ai,bi in zip(Qflx_bin_band1,qsum_bin_band1)]
#test_q_bin_band1 = np.divide(Qflx_bin_band1,qsum_bin_band1)
#u_bin_band1 = [(ai * 1.0)/bi for ai,bi in zip(Uflx_bin_band1,usum_bin_band1)]
#lick_q_bin = [(ai * 1.0)/bi for ai,bi in zip(lick_Qflx_bin,lick_flux_big_bin)]
#lick_u_bin = [(ai * 1.0)/bi for ai,bi in zip(lick_Uflx_bin,lick_flux_big_bin)]

q_isp = np.subtract(q,config["q_isp"]/100.0)
u_isp = np.subtract(u,config["u_isp"]/100.0)
q_bin_isp = np.subtract(q_bin,config["q_isp"]/100.0)
u_bin_isp = np.subtract(u_bin,config["u_isp"]/100.0)
q_bin_band1_isp = np.subtract(q_bin_band1,config["q_isp"]/100.0)
u_bin_band1_isp = np.subtract(u_bin_band1,config["u_isp"]/100.0)
q_bin_band2_isp = np.subtract(q_bin_band2,config["q_isp"]/100.0)
u_bin_band2_isp = np.subtract(u_bin_band2,config["u_isp"]/100.0)
q_bin_band3_isp = np.subtract(q_bin_band3,config["q_isp"]/100.0)
u_bin_band3_isp = np.subtract(u_bin_band3,config["u_isp"]/100.0)

lick_q_isp = np.subtract(lick_q,config["q_isp"]/100.0)
lick_u_isp = np.subtract(lick_u,config["u_isp"]/100.0)
lick_q_bin_isp = np.subtract(lick_q_bin,config["q_isp"]/100.0)
lick_u_bin_isp = np.subtract(lick_u_bin,config["u_isp"]/100.0)
lick_q_bin_band1_isp = np.subtract(lick_q_bin_band1,config["q_isp"]/100.0)
lick_u_bin_band1_isp = np.subtract(lick_u_bin_band1,config["u_isp"]/100.0)
lick_q_bin_band2_isp = np.subtract(lick_q_bin_band2,config["q_isp"]/100.0)
lick_u_bin_band2_isp = np.subtract(lick_u_bin_band2,config["u_isp"]/100.0)
lick_q_bin_band3_isp = np.subtract(lick_q_bin_band3,config["q_isp"]/100.0)
lick_u_bin_band3_isp = np.subtract(lick_u_bin_band3,config["u_isp"]/100.0)

"""
Now calculate P and Theta using the Wardle and Kronberg 1974 debiasing.
Use the ISP subtracted q's and u's.
"""

psig_squared = ((q_isp**2 * qsig**2) + (u_isp**2 * usig**2))/(q_isp**2 + u_isp**2)
#p_squared = q_isp**2 + u_isp**2 - (((q_isp**2 * qsig**2) + (u_isp**2 * usig**2))/(q_isp**2 + u_isp**2))
p_squared = q_isp**2 + u_isp**2 - psig_squared
print('psig_squared')
print(type(psig_squared))
print(psig_squared)
#psig = np.sqrt(psig_squared)
#psig = psig_squared**0.5
#p = np.sign(p_squared) * np.sqrt(np.absolute(p_squared))
#theta = 0.5 * (np.rad2deg(np.arctan2(u_isp,q_isp)))
#theta_sig = np.rad2deg((0.5 / (q_isp**2 + u_isp**2)) * np.sqrt(q_isp**2 * usig**2 + u_isp**2 * qsig**2))

print('p_squared')
print(type(p_squared))
print(p_squared)

#print('theta_sig')
#print(type(theta_sig))
#print(theta_sig)

psig_bin_squared = ((q_bin_isp**2 * qsig_bin**2) + (u_bin_isp**2 * usig_bin**2))/(q_bin_isp**2 + u_bin_isp**2)
#p_bin_squared = q_bin_isp**2 + u_bin_isp**2 - (((q_bin_isp**2 * qsig_bin**2) + (u_bin_isp**2 * usig_bin**2))/(q_bin_isp**2 + u_bin_isp**2))
p_bin_squared = q_bin_isp**2 + u_bin_isp**2 - psig_bin_squared
psig_bin = np.sqrt(psig_bin_squared)
p_bin = np.sign(p_bin_squared) * np.sqrt(np.absolute(p_bin_squared))
theta_bin = 0.5 * (np.rad2deg(np.arctan2(u_bin_isp,q_bin_isp)))
theta_sig_bin = np.rad2deg((0.5 / (q_bin_isp**2 + u_bin_isp**2)) * np.sqrt(q_bin_isp**2 * usig_bin**2 + u_bin_isp**2 * qsig_bin**2))

print('p_bin_squared')
print(type(p_bin_squared))
print(p_bin_squared)


p_bin_band1_squared = q_bin_band1_isp**2 + u_bin_band1_isp**2 - (((q_bin_band1_isp**2 * qsig_bin_band1**2) + (u_bin_band1_isp**2 * usig_bin_band1**2))/(q_bin_band1_isp**2 + u_bin_band1_isp**2))
p_bin_band2_squared = q_bin_band2_isp**2 + u_bin_band2_isp**2 - (((q_bin_band2_isp**2 * qsig_bin_band2**2) + (u_bin_band2_isp**2 * usig_bin_band2**2))/(q_bin_band2_isp**2 + u_bin_band2_isp**2))
p_bin_band3_squared = q_bin_band3_isp**2 + u_bin_band3_isp**2 - (((q_bin_band3_isp**2 * qsig_bin_band3**2) + (u_bin_band3_isp**2 * usig_bin_band3**2))/(q_bin_band3_isp**2 + u_bin_band3_isp**2))

p_bin_band1 = np.sign(p_bin_band1_squared) * np.sqrt(np.absolute(p_bin_band1_squared))
p_bin_band2 = np.sign(p_bin_band2_squared) * np.sqrt(np.absolute(p_bin_band2_squared))
p_bin_band3 = np.sign(p_bin_band3_squared) * np.sqrt(np.absolute(p_bin_band3_squared))

psig_bin_band1 = (1.0/np.sqrt(q_bin_band1_isp**2 + u_bin_band1_isp**2)) * np.sqrt((q_bin_band1_isp**2 * qsig_bin_band1**2) + (u_bin_band1_isp**2 * usig_bin_band1**2))
psig_bin_band2 = (1.0/np.sqrt(q_bin_band2_isp**2 + u_bin_band2_isp**2)) * np.sqrt((q_bin_band2_isp**2 * qsig_bin_band2**2) + (u_bin_band2_isp**2 * usig_bin_band2**2))
psig_bin_band3 = (1.0/np.sqrt(q_bin_band3_isp**2 + u_bin_band3_isp**2)) * np.sqrt((q_bin_band3_isp**2 * qsig_bin_band3**2) + (u_bin_band3_isp**2 * usig_bin_band3**2))

theta_bin_band1 = 0.5 * (np.rad2deg(np.arctan2(u_bin_band1_isp,q_bin_band1_isp)))
theta_bin_band2 = 0.5 * (np.rad2deg(np.arctan2(u_bin_band2_isp,q_bin_band2_isp)))
theta_bin_band3 = 0.5 * (np.rad2deg(np.arctan2(u_bin_band3_isp,q_bin_band3_isp)))

thetasig_bin_band1 = 0.5 * (1.0/(q_bin_band1_isp**2 + u_bin_band1_isp**2)) * np.sqrt((q_bin_band1_isp**2 * usig_bin_band1**2) + (u_bin_band1_isp**2 * qsig_bin_band1**2))
thetasig_bin_band2 = 0.5 * (1.0/(q_bin_band2_isp**2 + u_bin_band2_isp**2)) * np.sqrt((q_bin_band2_isp**2 * usig_bin_band2**2) + (u_bin_band2_isp**2 * qsig_bin_band2**2))
thetasig_bin_band3 = 0.5 * (1.0/(q_bin_band3_isp**2 + u_bin_band3_isp**2)) * np.sqrt((q_bin_band3_isp**2 * usig_bin_band3**2) + (u_bin_band3_isp**2 * qsig_bin_band3**2))

#lick_p_bin_squared = [ai**2 + bi**2 - (((ai**2 * ci**2) + (bi**2 * di**2))/(ai**2 + bi**2)) for ai, bi, ci, di in zip(lick_q_bin_isp,lick_u_bin_isp,lick_qsig_bin,lick_usig_bin)]
lick_p_bin_squared = lick_q_bin_isp**2 + lick_u_bin_isp**2 - (((lick_q_bin_isp**2 * lick_qsig_bin**2) + (lick_u_bin_isp**2 * lick_usig_bin**2))/(lick_q_bin_isp**2 + lick_u_bin_isp**2))
lick_p_bin = np.sign(lick_p_bin_squared) * np.sqrt(np.absolute(lick_p_bin_squared))
lick_theta_bin = 0.5 * (np.rad2deg(np.arctan2(lick_u_bin_isp,lick_q_bin_isp)))

lick_p_bin_band1_squared = lick_q_bin_band1_isp**2 + lick_u_bin_band1_isp**2 - (((lick_q_bin_band1_isp**2 * lick_qsig_bin_band1**2) + (lick_u_bin_band1_isp**2 * lick_usig_bin_band1**2))/(lick_q_bin_band1_isp**2 + lick_u_bin_band1_isp**2))
lick_p_bin_band2_squared = lick_q_bin_band2_isp**2 + lick_u_bin_band2_isp**2 - (((lick_q_bin_band2_isp**2 * lick_qsig_bin_band2**2) + (lick_u_bin_band2_isp**2 * lick_usig_bin_band2**2))/(lick_q_bin_band2_isp**2 + lick_u_bin_band2_isp**2))
lick_p_bin_band3_squared = lick_q_bin_band3_isp**2 + lick_u_bin_band3_isp**2 - (((lick_q_bin_band3_isp**2 * lick_qsig_bin_band3**2) + (lick_u_bin_band3_isp**2 * lick_usig_bin_band3**2))/(lick_q_bin_band3_isp**2 + lick_u_bin_band3_isp**2))

lick_p_bin_band1 = np.sign(lick_p_bin_band1_squared) * np.sqrt(np.absolute(lick_p_bin_band1_squared))
lick_p_bin_band2 = np.sign(lick_p_bin_band2_squared) * np.sqrt(np.absolute(lick_p_bin_band2_squared))
lick_p_bin_band3 = np.sign(lick_p_bin_band3_squared) * np.sqrt(np.absolute(lick_p_bin_band3_squared))

lick_psig_bin_band1 = (1.0/np.sqrt(lick_q_bin_band1_isp**2 + lick_u_bin_band1_isp**2)) * np.sqrt((lick_q_bin_band1_isp**2 * lick_qsig_bin_band1**2) + (lick_u_bin_band1_isp**2 * lick_usig_bin_band1**2))
lick_psig_bin_band2 = (1.0/np.sqrt(lick_q_bin_band2_isp**2 + lick_u_bin_band2_isp**2)) * np.sqrt((lick_q_bin_band2_isp**2 * lick_qsig_bin_band2**2) + (lick_u_bin_band2_isp**2 * lick_usig_bin_band2**2))
lick_psig_bin_band3 = (1.0/np.sqrt(lick_q_bin_band3_isp**2 + lick_u_bin_band3_isp**2)) * np.sqrt((lick_q_bin_band3_isp**2 * lick_qsig_bin_band3**2) + (lick_u_bin_band3_isp**2 * lick_usig_bin_band3**2))

lick_theta_bin_band1 = 0.5 * (np.rad2deg(np.arctan2(lick_u_bin_band1_isp,lick_q_bin_band1_isp)))
lick_theta_bin_band2 = 0.5 * (np.rad2deg(np.arctan2(lick_u_bin_band2_isp,lick_q_bin_band2_isp)))
lick_theta_bin_band3 = 0.5 * (np.rad2deg(np.arctan2(lick_u_bin_band3_isp,lick_q_bin_band3_isp)))

lick_thetasig_bin_band1 = 0.5 * (1.0/(lick_q_bin_band1_isp**2 + lick_u_bin_band1_isp**2)) * np.sqrt((lick_q_bin_band1_isp**2 * lick_usig_bin_band1**2) + (lick_u_bin_band1_isp**2 * lick_qsig_bin_band1**2))
lick_thetasig_bin_band2 = 0.5 * (1.0/(lick_q_bin_band2_isp**2 + lick_u_bin_band2_isp**2)) * np.sqrt((lick_q_bin_band2_isp**2 * lick_usig_bin_band2**2) + (lick_u_bin_band2_isp**2 * lick_qsig_bin_band2**2))
lick_thetasig_bin_band3 = 0.5 * (1.0/(lick_q_bin_band3_isp**2 + lick_u_bin_band3_isp**2)) * np.sqrt((lick_q_bin_band3_isp**2 * lick_usig_bin_band3**2) + (lick_u_bin_band3_isp**2 * lick_qsig_bin_band3**2))

print('Band1 q')
print(q_bin_band1)
print(type(q_bin_band1))
print('Band1 u')
print(u_bin_band1)
print('Band1 qsig')
print(qsig_bin_band1)
print(type(qsig_bin_band1))
print('Band1 usig')
print(usig_bin_band1)
print('Band1 p')
print(p_bin_band1)
print('Lick Band1 p')
print(lick_p_bin_band1)
print('Band1 theta')
print(theta_bin_band1)
print('Lick Band1 theta')
print(lick_theta_bin_band1)


"""
Now combine the SNSPOL data and the Lick data. Combining several lists. 
"""

all_flux = [flux[0], lick_flux[0], lick_flux[1], flux[1], flux[2], lick_flux[2], lick_flux[3], lick_flux[4], lick_flux[5], flux[3], flux[4], flux[5], flux[6], flux[7], flux[8], flux[9], flux[10]]
all_lambdas = [flux_lambdas[0], lick_lambdas[0], lick_lambdas[1], flux_lambdas[1], flux_lambdas[2], lick_lambdas[2], lick_lambdas[3], lick_lambdas[4], lick_lambdas[5], flux_lambdas[3], flux_lambdas[4], flux_lambdas[5], flux_lambdas[6], flux_lambdas[7], flux_lambdas[8], flux_lambdas[9], flux_lambdas[10]]

all_q_isp = [q_isp[0], lick_q_isp[0], lick_q_isp[1], q_isp[1], q_isp[2], lick_q_isp[2], lick_q_isp[3], lick_q_isp[4], lick_q_isp[5], q_isp[3], q_isp[4], q_isp[5], q_isp[6], q_isp[7], q_isp[8], q_isp[9], q_isp[10]]
all_qsig = [qsig[0], lick_qsig[0], lick_qsig[1], qsig[1], qsig[2], lick_qsig[2], lick_qsig[3], lick_qsig[4], lick_qsig[5], qsig[3], qsig[4], qsig[5], qsig[6], qsig[7], qsig[8], qsig[9], qsig[10]]

all_q_bin_isp = [q_bin_isp[0], lick_q_bin_isp[0], lick_q_bin_isp[1], q_bin_isp[1], q_bin_isp[2], lick_q_bin_isp[2], lick_q_bin_isp[3], lick_q_bin_isp[4], lick_q_bin_isp[5], q_bin_isp[3], q_bin_isp[4], q_bin_isp[5], q_bin_isp[6], q_bin_isp[7], q_bin_isp[8], q_bin_isp[9], q_bin_isp[10]]
all_qsig_bin = [qsig_bin[0], lick_qsig_bin[0], lick_qsig_bin[1], qsig_bin[1], qsig_bin[2], lick_qsig_bin[2], lick_qsig_bin[3], lick_qsig_bin[4], lick_qsig_bin[5], qsig_bin[3], qsig_bin[4], qsig_bin[5], qsig_bin[6], qsig_bin[7], qsig_bin[8], qsig_bin[9], qsig_bin[10]]

#all_q_bin_band1_isp = np.asarray([q_bin_band1_isp[0], lick_q_bin_band1_isp[0], lick_q_bin_band1_isp[1], q_bin_band1_isp[1], q_bin_band1_isp[2], lick_q_bin_band1_isp[2], lick_q_bin_band1_isp[3], lick_q_bin_band1_isp[4], lick_q_bin_band1_isp[5], q_bin_band1_isp[3], q_bin_band1_isp[4], q_bin_band1_isp[5], q_bin_band1_isp[6], q_bin_band1_isp[7], q_bin_band1_isp[8], q_bin_band1_isp[9], q_bin_band1_isp[10]])
all_q_bin_band1_isp = np.concatenate([q_bin_band1_isp[0], lick_q_bin_band1_isp[0], lick_q_bin_band1_isp[1], q_bin_band1_isp[1], q_bin_band1_isp[2], lick_q_bin_band1_isp[2], lick_q_bin_band1_isp[3], lick_q_bin_band1_isp[4], lick_q_bin_band1_isp[5], q_bin_band1_isp[3], q_bin_band1_isp[4], q_bin_band1_isp[5], q_bin_band1_isp[6], q_bin_band1_isp[7], q_bin_band1_isp[8], q_bin_band1_isp[9], q_bin_band1_isp[10]])
all_q_bin_band2_isp = np.concatenate([q_bin_band2_isp[0], lick_q_bin_band2_isp[0], lick_q_bin_band2_isp[1], q_bin_band2_isp[1], q_bin_band2_isp[2], lick_q_bin_band2_isp[2], lick_q_bin_band2_isp[3], lick_q_bin_band2_isp[4], lick_q_bin_band2_isp[5], q_bin_band2_isp[3], q_bin_band2_isp[4], q_bin_band2_isp[5], q_bin_band2_isp[6], q_bin_band2_isp[7], q_bin_band2_isp[8], q_bin_band2_isp[9], q_bin_band2_isp[10]])
all_q_bin_band3_isp = np.concatenate([q_bin_band3_isp[0], lick_q_bin_band3_isp[0], lick_q_bin_band3_isp[1], q_bin_band3_isp[1], q_bin_band3_isp[2], lick_q_bin_band3_isp[2], lick_q_bin_band3_isp[3], lick_q_bin_band3_isp[4], lick_q_bin_band3_isp[5], q_bin_band3_isp[3], q_bin_band3_isp[4], q_bin_band3_isp[5], q_bin_band3_isp[6], q_bin_band3_isp[7], q_bin_band3_isp[8], q_bin_band3_isp[9], q_bin_band3_isp[10]])
all_qsig_bin_band1 = np.concatenate([qsig_bin_band1[0], lick_qsig_bin_band1[0], lick_qsig_bin_band1[1], qsig_bin_band1[1], qsig_bin_band1[2], lick_qsig_bin_band1[2], lick_qsig_bin_band1[3], lick_qsig_bin_band1[4], lick_qsig_bin_band1[5], qsig_bin_band1[3], qsig_bin_band1[4], qsig_bin_band1[5], qsig_bin_band1[6], qsig_bin_band1[7], qsig_bin_band1[8], qsig_bin_band1[9], qsig_bin_band1[10]])
all_qsig_bin_band2 = np.concatenate([qsig_bin_band2[0], lick_qsig_bin_band2[0], lick_qsig_bin_band2[1], qsig_bin_band2[1], qsig_bin_band2[2], lick_qsig_bin_band2[2], lick_qsig_bin_band2[3], lick_qsig_bin_band2[4], lick_qsig_bin_band2[5], qsig_bin_band2[3], qsig_bin_band2[4], qsig_bin_band2[5], qsig_bin_band2[6], qsig_bin_band2[7], qsig_bin_band2[8], qsig_bin_band2[9], qsig_bin_band2[10]])
all_qsig_bin_band3 = np.concatenate([qsig_bin_band3[0], lick_qsig_bin_band3[0], lick_qsig_bin_band3[1], qsig_bin_band3[1], qsig_bin_band3[2], lick_qsig_bin_band3[2], lick_qsig_bin_band3[3], lick_qsig_bin_band3[4], lick_qsig_bin_band3[5], qsig_bin_band3[3], qsig_bin_band3[4], qsig_bin_band3[5], qsig_bin_band3[6], qsig_bin_band3[7], qsig_bin_band3[8], qsig_bin_band3[9], qsig_bin_band3[10]])
print("What type is this?")
print(type(all_q_bin_band1_isp))
print(all_q_bin_band1_isp)

all_u_isp = [u_isp[0], lick_u_isp[0], lick_u_isp[1], u_isp[1], u_isp[2], lick_u_isp[2], lick_u_isp[3], lick_u_isp[4], lick_u_isp[5], u_isp[3], u_isp[4], u_isp[5], u_isp[6], u_isp[7], u_isp[8], u_isp[9], u_isp[10]]
all_usig = [usig[0], lick_usig[0], lick_usig[1], usig[1], usig[2], lick_usig[2], lick_usig[3], lick_usig[4], lick_usig[5], usig[3], usig[4], usig[5], usig[6], usig[7], usig[8], usig[9], usig[10]]

all_u_bin_isp = [u_bin_isp[0], lick_u_bin_isp[0], lick_u_bin_isp[1], u_bin_isp[1], u_bin_isp[2], lick_u_bin_isp[2], lick_u_bin_isp[3], lick_u_bin_isp[4], lick_u_bin_isp[5], u_bin_isp[3], u_bin_isp[4], u_bin_isp[5], u_bin_isp[6], u_bin_isp[7], u_bin_isp[8], u_bin_isp[9], u_bin_isp[10]]
all_usig_bin = [usig_bin[0], lick_usig_bin[0], lick_usig_bin[1], usig_bin[1], usig_bin[2], lick_usig_bin[2], lick_usig_bin[3], lick_usig_bin[4], lick_usig_bin[5], usig_bin[3], usig_bin[4], usig_bin[5], usig_bin[6], usig_bin[7], usig_bin[8], usig_bin[9], usig_bin[10]]

all_u_bin_band1_isp = np.concatenate([u_bin_band1_isp[0], lick_u_bin_band1_isp[0], lick_u_bin_band1_isp[1], u_bin_band1_isp[1], u_bin_band1_isp[2], lick_u_bin_band1_isp[2], lick_u_bin_band1_isp[3], lick_u_bin_band1_isp[4], lick_u_bin_band1_isp[5], u_bin_band1_isp[3], u_bin_band1_isp[4], u_bin_band1_isp[5], u_bin_band1_isp[6], u_bin_band1_isp[7], u_bin_band1_isp[8], u_bin_band1_isp[9], u_bin_band1_isp[10]])
all_u_bin_band2_isp = np.concatenate([u_bin_band2_isp[0], lick_u_bin_band2_isp[0], lick_u_bin_band2_isp[1], u_bin_band2_isp[1], u_bin_band2_isp[2], lick_u_bin_band2_isp[2], lick_u_bin_band2_isp[3], lick_u_bin_band2_isp[4], lick_u_bin_band2_isp[5], u_bin_band2_isp[3], u_bin_band2_isp[4], u_bin_band2_isp[5], u_bin_band2_isp[6], u_bin_band2_isp[7], u_bin_band2_isp[8], u_bin_band2_isp[9], u_bin_band2_isp[10]])
all_u_bin_band3_isp = np.concatenate([u_bin_band3_isp[0], lick_u_bin_band3_isp[0], lick_u_bin_band3_isp[1], u_bin_band3_isp[1], u_bin_band3_isp[2], lick_u_bin_band3_isp[2], lick_u_bin_band3_isp[3], lick_u_bin_band3_isp[4], lick_u_bin_band3_isp[5], u_bin_band3_isp[3], u_bin_band3_isp[4], u_bin_band3_isp[5], u_bin_band3_isp[6], u_bin_band3_isp[7], u_bin_band3_isp[8], u_bin_band3_isp[9], u_bin_band3_isp[10]])
all_usig_bin_band1 = np.concatenate([usig_bin_band1[0], lick_usig_bin_band1[0], lick_usig_bin_band1[1], usig_bin_band1[1], usig_bin_band1[2], lick_usig_bin_band1[2], lick_usig_bin_band1[3], lick_usig_bin_band1[4], lick_usig_bin_band1[5], usig_bin_band1[3], usig_bin_band1[4], usig_bin_band1[5], usig_bin_band1[6], usig_bin_band1[7], usig_bin_band1[8], usig_bin_band1[9], usig_bin_band1[10]])
all_usig_bin_band2 = np.concatenate([usig_bin_band2[0], lick_usig_bin_band2[0], lick_usig_bin_band2[1], usig_bin_band2[1], usig_bin_band2[2], lick_usig_bin_band2[2], lick_usig_bin_band2[3], lick_usig_bin_band2[4], lick_usig_bin_band2[5], usig_bin_band2[3], usig_bin_band2[4], usig_bin_band2[5], usig_bin_band2[6], usig_bin_band2[7], usig_bin_band2[8], usig_bin_band2[9], usig_bin_band2[10]])
all_usig_bin_band3 = np.concatenate([usig_bin_band3[0], lick_usig_bin_band3[0], lick_usig_bin_band3[1], usig_bin_band3[1], usig_bin_band3[2], lick_usig_bin_band3[2], lick_usig_bin_band3[3], lick_usig_bin_band3[4], lick_usig_bin_band3[5], usig_bin_band3[3], usig_bin_band3[4], usig_bin_band3[5], usig_bin_band3[6], usig_bin_band3[7], usig_bin_band3[8], usig_bin_band3[9], usig_bin_band3[10]])

#all_Qflx_bin_edges = np.concatenate([Qflx_bin_edges[0], lick_Qflx_bin_edges[0], lick_Qflx_bin_edges[1], Qflx_bin_edges[1], Qflx_bin_edges[2], lick_Qflx_bin_edges[2], lick_Qflx_bin_edges[3], lick_Qflx_bin_edges[4], lick_Qflx_bin_edges[5], Qflx_bin_edges[3], Qflx_bin_edges[4], Qflx_bin_edges[5], Qflx_bin_edges[6], Qflx_bin_edges[7], Qflx_bin_edges[8], Qflx_bin_edges[9], Qflx_bin_edges[10]])
all_Qflx_bin_edges = [Qflx_bin_edges[0], lick_Qflx_bin_edges[0], lick_Qflx_bin_edges[1], Qflx_bin_edges[1], Qflx_bin_edges[2], lick_Qflx_bin_edges[2], lick_Qflx_bin_edges[3], lick_Qflx_bin_edges[4], lick_Qflx_bin_edges[5], Qflx_bin_edges[3], Qflx_bin_edges[4], Qflx_bin_edges[5], Qflx_bin_edges[6], Qflx_bin_edges[7], Qflx_bin_edges[8], Qflx_bin_edges[9], Qflx_bin_edges[10]]
all_Uflx_bin_edges = [Uflx_bin_edges[0], lick_Uflx_bin_edges[0], lick_Uflx_bin_edges[1], Uflx_bin_edges[1], Uflx_bin_edges[2], lick_Uflx_bin_edges[2], lick_Uflx_bin_edges[3], lick_Uflx_bin_edges[4], lick_Uflx_bin_edges[5], Uflx_bin_edges[3], Uflx_bin_edges[4], Uflx_bin_edges[5], Uflx_bin_edges[6], Uflx_bin_edges[7], Uflx_bin_edges[8], Uflx_bin_edges[9], Uflx_bin_edges[10]]

#all_p = [p[0], lick_p[0], lick_p[1], p[1], p[2], lick_p[2], lick_p[3], lick_p[4], lick_p[5], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10]]
#all_psig = [psig[0], lick_psig[0], lick_psig[1], psig[1], psig[2], lick_psig[2], lick_psig[3], lick_psig[4], lick_psig[5], psig[3], psig[4], psig[5], psig[6], psig[7], psig[8], psig[9], psig[10]]

all_p_bin = [p_bin[0], lick_p_bin[0], lick_p_bin[1], p_bin[1], p_bin[2], lick_p_bin[2], lick_p_bin[3], lick_p_bin[4], lick_p_bin[5], p_bin[3], p_bin[4], p_bin[5], p_bin[6], p_bin[7], p_bin[8], p_bin[9], p_bin[10]]
#all_psig_bin = [psig_bin[0], lick_psig_bin[0], lick_psig_bin[1], psig_bin[1], psig_bin[2], lick_psig_bin[2], lick_psig_bin[3], lick_psig_bin[4], lick_psig_bin[5], psig_bin[3], psig_bin[4], psig_bin[5], psig_bin[6], psig_bin[7], psig_bin[8], psig_bin[9], psig_bin[10]]

all_p_bin_band1 = np.concatenate([p_bin_band1[0], lick_p_bin_band1[0], lick_p_bin_band1[1], p_bin_band1[1], p_bin_band1[2], lick_p_bin_band1[2], lick_p_bin_band1[3], lick_p_bin_band1[4], lick_p_bin_band1[5], p_bin_band1[3], p_bin_band1[4], p_bin_band1[5], p_bin_band1[6], p_bin_band1[7], p_bin_band1[8], p_bin_band1[9], p_bin_band1[10]])
all_p_bin_band2 = np.concatenate([p_bin_band2[0], lick_p_bin_band2[0], lick_p_bin_band2[1], p_bin_band2[1], p_bin_band2[2], lick_p_bin_band2[2], lick_p_bin_band2[3], lick_p_bin_band2[4], lick_p_bin_band2[5], p_bin_band2[3], p_bin_band2[4], p_bin_band2[5], p_bin_band2[6], p_bin_band2[7], p_bin_band2[8], p_bin_band2[9], p_bin_band2[10]])
all_p_bin_band3 = np.concatenate([p_bin_band3[0], lick_p_bin_band3[0], lick_p_bin_band3[1], p_bin_band3[1], p_bin_band3[2], lick_p_bin_band3[2], lick_p_bin_band3[3], lick_p_bin_band3[4], lick_p_bin_band3[5], p_bin_band3[3], p_bin_band3[4], p_bin_band3[5], p_bin_band3[6], p_bin_band3[7], p_bin_band3[8], p_bin_band3[9], p_bin_band3[10]])

print('all_p_bin_band1')
print(all_p_bin_band1)
print('p_bin_band1 and lick_p_bin_band1')
print(p_bin_band1)
print(lick_p_bin_band1)

#all_theta = [theta[0], lick_theta[0], lick_theta[1], theta[1], theta[2], lick_theta[2], lick_theta[3], lick_theta[4], lick_theta[5], theta[3], theta[4], theta[5], theta[6], theta[7], theta[8], theta[9], theta[10]]
#all_thetasig = [thetasig[0], lick_thetasig[0], lick_thetasig[1], thetasig[1], thetasig[2], lick_thetasig[2], lick_thetasig[3], lick_thetasig[4], lick_thetasig[5], thetasig[3], thetasig[4], thetasig[5], thetasig[6], thetasig[7], thetasig[8], thetasig[9], thetasig[10]]

all_theta_bin = [theta_bin[0], lick_theta_bin[0], lick_theta_bin[1], theta_bin[1], theta_bin[2], lick_theta_bin[2], lick_theta_bin[3], lick_theta_bin[4], lick_theta_bin[5], theta_bin[3], theta_bin[4], theta_bin[5], theta_bin[6], theta_bin[7], theta_bin[8], theta_bin[9], theta_bin[10]]
#all_thetasig_bin = [thetasig_bin[0], lick_thetasig_bin[0], lick_thetasig_bin[1], thetasig_bin[1], thetasig_bin[2], lick_thetasig_bin[2], lick_thetasig_bin[3], lick_thetasig_bin[4], lick_thetasig_bin[5], thetasig_bin[3], thetasig_bin[4], thetasig_bin[5], thetasig_bin[6], thetasig_bin[7], thetasig_bin[8], thetasig_bin[9], thetasig_bin[10]]

all_theta_bin_band1 = np.concatenate([theta_bin_band1[0], lick_theta_bin_band1[0], lick_theta_bin_band1[1], theta_bin_band1[1], theta_bin_band1[2], lick_theta_bin_band1[2], lick_theta_bin_band1[3], lick_theta_bin_band1[4], lick_theta_bin_band1[5], theta_bin_band1[3], theta_bin_band1[4], theta_bin_band1[5], theta_bin_band1[6], theta_bin_band1[7], theta_bin_band1[8], theta_bin_band1[9], theta_bin_band1[10]])
all_theta_bin_band2 = np.concatenate([theta_bin_band2[0], lick_theta_bin_band2[0], lick_theta_bin_band2[1], theta_bin_band2[1], theta_bin_band2[2], lick_theta_bin_band2[2], lick_theta_bin_band2[3], lick_theta_bin_band2[4], lick_theta_bin_band2[5], theta_bin_band2[3], theta_bin_band2[4], theta_bin_band2[5], theta_bin_band2[6], theta_bin_band2[7], theta_bin_band2[8], theta_bin_band2[9], theta_bin_band2[10]])
all_theta_bin_band3 = np.concatenate([theta_bin_band3[0], lick_theta_bin_band3[0], lick_theta_bin_band3[1], theta_bin_band3[1], theta_bin_band3[2], lick_theta_bin_band3[2], lick_theta_bin_band3[3], lick_theta_bin_band3[4], lick_theta_bin_band3[5], theta_bin_band3[3], theta_bin_band3[4], theta_bin_band3[5], theta_bin_band3[6], theta_bin_band3[7], theta_bin_band3[8], theta_bin_band3[9], theta_bin_band3[10]])

#print('Combined data')
#print(flux)
#print(all_flux)

"""
Save the results to a text file using numpy to put the results in columns
"""

#np.savetxt('test_save.txt',[q_bin_band1,qsig_bin_band1,u_bin_band1,usig_bin_band1,p_bin_band1,theta_bin_band1])
#np.savetxt('test_save.txt',[q_bin_band1[1],qsig_bin_band1,u_bin_band1,usig_bin_band1,p_bin_band1,theta_bin_band1])
#np.savetxt('test_save.txt',(p_bin_band1,theta_bin_band1))
#np.savetxt('test_save.txt', all_q_bin_band1_isp.flatten(), all_qsig_bin_band1.flatten(), all_u_bin_band1_isp.flatten(), all_usig_bin_band1.flatten(), all_p_bin_band1.flatten(), all_theta_bin_band1.flatten())
#np.savetxt('test_save.txt', np.column_stack((all_q_bin_band1_isp, all_qsig_bin_band1)),fmt='%1.4e',delimiter=', ')
#np.savetxt('test_save.txt', np.column_stack((config["all_epoch_days"],all_q_bin_band1_isp, all_qsig_bin_band1, all_u_bin_band1_isp, all_usig_bin_band1, all_p_bin_band1 * 100.0, all_theta_bin_band1)), fmt='%i, %1.4e, %1.4e, %1.4e, %1.4e, %1.3f, %3.1f', delimiter=', ',header="day, q, qsig, u, usig, p, theta")
#np.savetxt(os.path.join(config["output_path"],config["continuum_file"]), np.column_stack((config["all_epoch_days"], all_q_bin_band1_isp * 100.0, all_qsig_bin_band1 * 100.0, all_u_bin_band1_isp * 100.0, all_usig_bin_band1 * 100.0, all_q_bin_band2_isp * 100.0, all_qsig_bin_band2 * 100.0, all_u_bin_band2_isp * 100.0, all_usig_bin_band2 * 100.0, all_q_bin_band3_isp * 100.0, all_qsig_bin_band3 * 100.0, all_u_bin_band3_isp * 100.0, all_usig_bin_band3 * 100.0, all_p_bin_band1 * 100.0, all_theta_bin_band1, all_p_bin_band2 * 100.0, all_theta_bin_band2, all_p_bin_band3 * 100.0, all_theta_bin_band3)), fmt='%i, %1.3f, %1.3f, %1.3f, %1.3f, %1.3f, %3.1f', delimiter=', ',header="day, q (%), qsig (%), u (%), usig (%), p (%), theta (deg)")
np.savetxt(os.path.join(config["output_path"],config["continuum_file"]), np.column_stack((config["all_epoch_days"], all_q_bin_band1_isp * 100.0, all_qsig_bin_band1 * 100.0, all_u_bin_band1_isp * 100.0, all_usig_bin_band1 * 100.0, all_q_bin_band2_isp * 100.0, all_qsig_bin_band2 * 100.0, all_u_bin_band2_isp * 100.0, all_usig_bin_band2 * 100.0, all_q_bin_band3_isp * 100.0, all_qsig_bin_band3 * 100.0, all_u_bin_band3_isp * 100.0, all_usig_bin_band3 * 100.0, all_p_bin_band1 * 100.0, all_theta_bin_band1, all_p_bin_band2 * 100.0, all_theta_bin_band2, all_p_bin_band3 * 100.0, all_theta_bin_band3)), fmt='%i' + ', %3.3f'*12 + ', %3.3f, %3.1f'*3, delimiter=', ',header="day, q1 (%), qsig1 (%), u1 (%), usig1 (%), q2 (%), qsig2 (%), u2 (%), usig2 (%), q3 (%), qsig3 (%), u3 (%), usig3 (%), p1 (%), theta1 (deg), p2 (%), theta2 (deg), p3 (%), theta3 (deg)")

"""
To turn debugging on remove the "-O" in the python call in the first line.
To turn debugging off add a "-O" in the python call in the first line.
"""
if __debug__:
    print(q[0])
    print(config["q_file"])
    print(config["base_path"])
    print(config["epoch_dirs"])
    print(q_lambdas[0])
    print(Qflx_bin, Qflx_bin_edges, Qflx_bin_num)
    print(p_bin_squared)
    print(p_bin)
    print(theta_bin)
    print(qsum[0])
    print(usum[0])
else:
    print('Debugging is OFF')

"""
Set up a new figure for plotting q and u as a function of wavelength
"""

fig = plt.figure()
fig.subplots_adjust(hspace=0.001)
fig.subplots_adjust(bottom=0.15)

# Plot flux on the first row of three
ax1 = fig.add_subplot(311)
ax1.plot(flux_lambdas[config["plot_epoch"]-1],flux[config["plot_epoch"]-1],color='b',alpha=0.9,linewidth=1.0)
# The following could be used to add error bars.
#ax1.plot(Qflx_bin_edges[1][1:],q_bin[1]*100.0,color='b',alpha=0.9,drawstyle='steps-mid')

# Add minor 4 or 5 minor ticks (putting a number in the () will force a value).
# Search for 'matplotlib ticker' for detailed documentation.
ax1.yaxis.set_minor_locator(AutoMinorLocator())

# Plot q, unbinned and binned, on the second row of three and share the x axis
ax2 = fig.add_subplot(312, sharex=ax1)
ax2.plot(q_lambdas[config["plot_epoch"]-1],q_isp[config["plot_epoch"]-1]*100.0,color='r',alpha=0.2,linewidth=1.0)
ax2.plot(Qflx_bin_edges[config["plot_epoch"]-1][1:],q_bin_isp[config["plot_epoch"]-1]*100.0,color='b',alpha=0.9,linewidth=0.8,drawstyle='steps-pre')

# Add minor 4 or 5 minor ticks (putting a number in the () will force a value).
# Search for 'matplotlib ticker' for detailed documentation.
ax2.yaxis.set_minor_locator(AutoMinorLocator())
#ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
#ax2.yaxis.set_major_locator(MultipleLocator(0.5))
#ax2.yaxis.set_minor_locator(MaxNLocator(10))

# Plot u, unbinned and binned, on the third row of three and share the x axis
ax3 = fig.add_subplot(313, sharex=ax1)
ax3.plot(u_lambdas[config["plot_epoch"]-1],u_isp[config["plot_epoch"]-1]*100.0,color='r',alpha=0.2,linewidth=1.0)
ax3.plot(Uflx_bin_edges[config["plot_epoch"]-1][1:],u_bin_isp[config["plot_epoch"]-1]*100.0,color='b',alpha=0.9,linewidth=0.8,drawstyle='steps-pre')

# Add minor 4 or 5 minor ticks (putting a number in the () will force a value).
# Search for 'matplotlib ticker' for detailed documentation.
ax3.yaxis.set_minor_locator(AutoMinorLocator())

# Set the x and y boundaries for the plots above.
ax1.set_xbound(config["lambda_min"],config["lambda_max"])
#ax2.set_ybound(-3.5,2.5)
#ax3.set_ybound(-3.0,3.0)
#ax1.set_ybound(0.0,4.5e-14)
ax2.set_ybound(-1.0,1.5)
ax3.set_ybound(-2.5,0.0)

# Don't show the x tick labels on the first two plots.
xticklabels1 = ax1.get_xticklabels()
xticklabels2 = ax2.get_xticklabels()
plt.setp(xticklabels1, visible=False)
plt.setp(xticklabels2, visible=False)

# Don't show the first (and sometimes last) tick labels.
yticks_vals1 = ax1.yaxis.get_major_ticks()
yticks_vals1[0].label1.set_visible(False)

yticks_vals2 = ax2.yaxis.get_major_ticks()
yticks_vals2[0].label1.set_visible(False)
yticks_vals2[-1].label1.set_visible(False)

yticks_vals3 = ax3.yaxis.get_major_ticks()
yticks_vals3[0].label1.set_visible(False)
yticks_vals3[-1].label1.set_visible(False)

# Add a title. Comment out for publications.
#ax1.set_title(config["plot_title"], fontsize=20)

# Label the plot with the name of the SN and the epoch being plotted.
plt.figtext(0.75,0.79,
            config["plot_title"]
	    + '\n' + 'Epoch ' + str(config["plot_epoch"]),
	    size=14)

# Label the x and y axes.
ax3.set_xlabel(r'$\mathrm{\lambda (\AA)}$', fontsize=14)
ax1.set_ylabel(r'$\mathrm{F_\lambda}$', fontsize=14)
ax2.set_ylabel(r'$\mathrm{q (\%)}$', fontsize=14)
ax3.set_ylabel(r'$\mathrm{u (\%)}$', fontsize=14)

plt.savefig(os.path.join(config["output_path"],'qu_plot1.pdf'),format='pdf')
#plt.savefig('qu_plot1.eps',format='eps',bbox_inches='tight')


"""
Set up a new figure for plotting p and theta as a function of wavelength
"""

fig = plt.figure()
fig.subplots_adjust(hspace=0.001)
fig.subplots_adjust(bottom=0.15)

# Plot flux on the first row of three
ax1 = fig.add_subplot(311)
ax1.plot(flux_lambdas[config["plot_epoch"]-1],
         flux[config["plot_epoch"]-1],
	 color='g',alpha=0.9,linewidth=1.0)

# Add minor 4 or 5 minor ticks (putting a number in the () will force a value).
# Search for 'matplotlib ticker' for detailed documentation.
ax1.yaxis.set_minor_locator(AutoMinorLocator())

# Plot binned P on the second row of three
ax2 = fig.add_subplot(312, sharex=ax1)
ax2.plot(Qflx_bin_edges[config["plot_epoch"]-1][1:],p_bin[config["plot_epoch"]-1]*100.0,color='g',alpha=0.9, linewidth=0.8, drawstyle='steps-pre')
ax2.plot(lick_Qflx_bin_edges[config["plot_epoch"]-1][1:],lick_p_bin[config["plot_epoch"]-1]*100.0,color='b',alpha=0.9, linewidth=0.8, drawstyle='steps-pre')

# Add minor 4 or 5 minor ticks (putting a number in the () will force a value).
# Search for 'matplotlib ticker' for detailed documentation.
ax2.yaxis.set_minor_locator(AutoMinorLocator())

# Plot binned theta on the third row of three
ax3 = fig.add_subplot(313, sharex=ax1)
ax3.plot(Qflx_bin_edges[config["plot_epoch"]-1][1:],theta_bin[config["plot_epoch"]-1],color='g',alpha=0.9, linewidth=0.8, drawstyle='steps-pre')
ax3.plot(lick_Qflx_bin_edges[config["plot_epoch"]-1][1:],lick_theta_bin[config["plot_epoch"]-1],color='b',alpha=0.9, linewidth=0.8, drawstyle='steps-pre')

# Add minor 4 or 5 minor ticks (putting a number in the () will force a value).
# Search for 'matplotlib ticker' for detailed documentation.
ax3.yaxis.set_minor_locator(AutoMinorLocator())
ax3.yaxis.set_major_locator(MaxNLocator(3))

# Set the boundaries of the axes
ax1.set_xbound(config["lambda_min"],config["lambda_max"])
#ax2.set_ybound(-1.0,3.0)
#ax3.set_ybound(-90.0,90.0)
ax2.set_ybound(0.0,2.5)
ax3.set_ybound(-60.0,-20.0)

# Don't show the x tick labels on the first two plots.
xticklabels1 = ax1.get_xticklabels()
xticklabels2 = ax2.get_xticklabels()
plt.setp(xticklabels1, visible=False)
plt.setp(xticklabels2, visible=False)

# Suppresses the first (and sometimes last) tick labels
yticks_vals1 = ax1.yaxis.get_major_ticks()
yticks_vals1[0].label1.set_visible(False)

yticks_vals2 = ax2.yaxis.get_major_ticks()
yticks_vals2[0].label1.set_visible(False)
yticks_vals2[-1].label1.set_visible(False)

yticks_vals3 = ax3.yaxis.get_major_ticks()
yticks_vals3[0].label1.set_visible(False)
yticks_vals3[-1].label1.set_visible(False)

# Add a title. Comment out for publications.
#ax1.set_title(config["plot_title"], fontsize=20)

# Label the plot with the name of the SN and the epoch being plotted.
plt.figtext(0.75,0.79,
            config["plot_title"]
	    + '\n' + 'Epoch ' + str(config["plot_epoch"]),
	    size=14)

# Label the x and y axes.
ax3.set_xlabel(r'$\mathrm{\lambda (\AA)}$', fontsize=18)
ax1.set_ylabel(r'$\mathrm{F_\lambda}$', fontsize=18)
ax2.set_ylabel(r'$\mathrm{P (\%)}$', fontsize=18)
ax3.set_ylabel(r'$\theta (^\circ)$', fontsize=18)

plt.savefig(os.path.join(config["output_path"],'qu_plot2.pdf'),format='pdf')
#plt.savefig('qu_plot2.eps',format='eps',bbox_inches='tight')

# If we want to mouse over the plot show it to the screen.
#plt.show()


"""
Set up a new figure for the q/u plots 
"""

# Create a square figure, 6.0" x 6.0".
# A square makes it easier for setting coordinates with aspect ratio equal.
fig = plt.figure(figsize=(6.0,6.0))

# Set a square plotting region.
# Leave more room on the right for a color bar.
# The hspace and wspace are overridden by the aspect and adjustable.
fig.subplots_adjust(bottom=0.3, top=0.9, left=0.2, right=0.8, hspace=0.001, wspace=0.001)

# Establish the color scale.
cmin = config["lambda_min"]
cmax = config["lambda_max"]

# define a colormap, jet was default and now viridis is.
# See https://matplotlib.org/users/dflt_style_changes.html
# I like jet slightly better than rainbow.
#use_cmap="rainbow"
use_cmap="jet"

# Get the range of indices for data > 4200 and < 7600 since S/N is low outside that range.
indice_range0 = (np.logical_and(Qflx_bin_edges[0][:-1] > config["lambda_min"],Qflx_bin_edges[0][:-1] < config["lambda_max"])).nonzero()
epoch1_colors = Qflx_bin_edges[0][indice_range0]

# Generate the binned q/u plots for epoch one. 
# Keep the aspect ratio equal as it should be.
# The line below threw an error becaus 'box-forced' isn't allowed.
# I'm changing it to 'box'.
#ax1 = fig.add_subplot(221, aspect='equal', adjustable='box-forced')
ax1 = fig.add_subplot(221, aspect='equal', adjustable='box')
e1_plot=ax1.scatter((q_bin_isp[0][indice_range0]*100.0),(u_bin_isp[0][indice_range0]*100.0),c=epoch1_colors,edgecolors='k',cmap=use_cmap,vmin=cmin,vmax=cmax,alpha=0.8,s=25)

# Make crosshairs at 0,0 and use zorder to put them below the data points.
ax1.axhline(0, color='grey', linewidth=0.8, zorder=0)
ax1.axvline(0, color='grey', linewidth=0.8, zorder=0)

# Make lighter crosshairs of the origin before ISP subtraction.
ax1.axhline(-1.0*config["q_isp"], color='grey',alpha=0.2,zorder=0)
ax1.axvline(-1.0*config["u_isp"], color='grey',alpha=0.2,zorder=0)

# Draw five concentric circles denoting %P intervals of 0.5%
ax1.add_artist(plt.Circle((0, 0), 0.5, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax1.add_artist(plt.Circle((0, 0), 1.0, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax1.add_artist(plt.Circle((0, 0), 1.5, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax1.add_artist(plt.Circle((0, 0), 2.0, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax1.add_artist(plt.Circle((0, 0), 2.5, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax1.add_artist(plt.Circle((0, 0), 3.0, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))

# Get the range of indices for data > 4200 and < 7600 since S/N is low outside that range.
indice_range1 = (np.logical_and(Qflx_bin_edges[1][:-1] > config["lambda_min"],Qflx_bin_edges[1][:-1] < config["lambda_max"])).nonzero()
epoch2_colors = Qflx_bin_edges[1][indice_range1]

# Generate the binned q/u plots for epoch two. 
# The line below threw an error becaus 'box-forced' isn't allowed.
# I'm changing it to 'box'.
#ax2 = fig.add_subplot(222, aspect='equal', adjustable='box-forced', sharex=ax1, sharey=ax1)
ax2 = fig.add_subplot(222, aspect='equal', adjustable='box', sharex=ax1, sharey=ax1)
ax2.scatter((q_bin_isp[1][indice_range1]*100.0),(u_bin_isp[1][indice_range1]*100.0),c=epoch2_colors,edgecolors='k',cmap=use_cmap,vmin=cmin,vmax=cmax,alpha=0.8,s=25)

# Make crosshairs at 0,0 and use zorder to put them below the data points.
ax2.axhline(0, color='grey', linewidth=0.8, zorder=0)
ax2.axvline(0, color='grey', linewidth=0.8, zorder=0)

# Make lighter crosshairs of the origin before ISP subtraction.
ax2.axhline(-1.0*config["q_isp"], color='grey',alpha=0.2,zorder=0)
ax2.axvline(-1.0*config["u_isp"], color='grey',alpha=0.2,zorder=0)

# Draw five concentric circles denoting %P intervals of 0.5%
ax2.add_artist(plt.Circle((0, 0), 0.5, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax2.add_artist(plt.Circle((0, 0), 1.0, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax2.add_artist(plt.Circle((0, 0), 1.5, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax2.add_artist(plt.Circle((0, 0), 2.0, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax2.add_artist(plt.Circle((0, 0), 2.5, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax2.add_artist(plt.Circle((0, 0), 3.0, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))

# Get the range of indices for data > 4200 and < 7600 since S/N is low outside that range.
indice_range2 = (np.logical_and(Qflx_bin_edges[2][:-1] > config["lambda_min"],Qflx_bin_edges[2][:-1] < config["lambda_max"])).nonzero()
epoch3_colors = Qflx_bin_edges[2][indice_range2]

# Generate the binned q/u plots for epoch three. 
#ax3 = fig.add_subplot(223, aspect='equal', adjustable='box-forced', sharex=ax1, sharey=ax1)
ax3 = fig.add_subplot(223, aspect='equal', adjustable='box', sharex=ax1, sharey=ax1)
ax3.scatter((q_bin_isp[2][indice_range2]*100.0),(u_bin_isp[2][indice_range2]*100.0),c=epoch3_colors,edgecolors='k',cmap=use_cmap,vmin=cmin,vmax=cmax,alpha=0.8,s=25)

# Make crosshairs at 0,0 and use zorder to put them below the data points.
ax3.axhline(0, color='grey', linewidth=0.8, zorder=0)
ax3.axvline(0, color='grey', linewidth=0.8, zorder=0)

# Make lighter crosshairs of the origin before ISP subtraction.
ax3.axhline(-1.0*config["q_isp"], color='grey',alpha=0.2,zorder=0)
ax3.axvline(-1.0*config["u_isp"], color='grey',alpha=0.2,zorder=0)

# Draw five concentric circles denoting %P intervals of 0.5%
ax3.add_artist(plt.Circle((0, 0), 0.5, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax3.add_artist(plt.Circle((0, 0), 1.0, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax3.add_artist(plt.Circle((0, 0), 1.5, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax3.add_artist(plt.Circle((0, 0), 2.0, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax3.add_artist(plt.Circle((0, 0), 2.5, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax3.add_artist(plt.Circle((0, 0), 3.0, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))

# Get the range of indices for data > 4200 and < 7600 since S/N is low outside that range.
indice_range3 = (np.logical_and(Qflx_bin_edges[3][:-1] > config["lambda_min"],Qflx_bin_edges[3][:-1] < config["lambda_max"])).nonzero()
epoch4_colors = Qflx_bin_edges[3][indice_range3]

# Generate the binned q/u plots for epoch four. 
#ax4 = fig.add_subplot(224, aspect='equal', adjustable='box-forced', sharex=ax1, sharey=ax1)
ax4 = fig.add_subplot(224, aspect='equal', adjustable='box', sharex=ax1, sharey=ax1)
ax4.scatter((q_bin_isp[3][indice_range3]*100.0),(u_bin_isp[3][indice_range3]*100.0),c=epoch4_colors,edgecolors='k',cmap=use_cmap,vmin=cmin,vmax=cmax,alpha=0.8,s=25)

# Make crosshairs at 0,0 and use zorder to put them below the data points.
ax4.axhline(0, color='grey', linewidth=0.8, zorder=0)
ax4.axvline(0, color='grey', linewidth=0.8, zorder=0)

# Make lighter crosshairs of the origin before ISP subtraction.
ax4.axhline(-1.0*config["q_isp"], color='grey',alpha=0.2,zorder=0)
ax4.axvline(-1.0*config["u_isp"], color='grey',alpha=0.2,zorder=0)

# Draw five concentric circles denoting %P intervals of 0.5%
ax4.add_artist(plt.Circle((0, 0), 0.5, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax4.add_artist(plt.Circle((0, 0), 1.0, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax4.add_artist(plt.Circle((0, 0), 1.5, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax4.add_artist(plt.Circle((0, 0), 2.0, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax4.add_artist(plt.Circle((0, 0), 2.5, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax4.add_artist(plt.Circle((0, 0), 3.0, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))

# Set the boundaries.  This sets them for all the plots.
ax1.set_xbound(-2.0,2.0)
ax1.set_ybound(-3.0,1.0)

# Don't display tick labels for the shared axes 
xticklabels1 = ax1.get_xticklabels()
xticklabels2 = ax2.get_xticklabels()
yticklabels2 = ax2.get_yticklabels()
yticklabels4 = ax4.get_yticklabels()
plt.setp(xticklabels1, visible=False)
plt.setp(xticklabels2, visible=False)
plt.setp(yticklabels2, visible=False)
plt.setp(yticklabels4, visible=False)

# Set the major and minor tick marks.
# For some reason you only need to do one of the two y or x axes.
ax1.yaxis.set_ticks(np.arange(-2.5, 1.5, 1.0))
ax1.yaxis.set_ticks(np.arange(-3.0, 1.0, 0.5),minor=True)

ax3.xaxis.set_ticks(np.arange(-1.5, 2.5, 1.0))
ax3.xaxis.set_ticks(np.arange(-2.0, 2.0, 0.5),minor=True)

# Add a title to the plot. Comment out for publications
fig.suptitle(config["plot_title"], fontsize=20)

# Label the plot with the name of the SN and the epochs being plotted.
#plt.figtext(0.22, 0.86, config["plot_title"], size=16)

plt.figtext(0.21, 0.86, 'Epoch 1', size=16)
plt.figtext(0.51, 0.86, 'Epoch 2', size=16)
plt.figtext(0.21, 0.56, 'Epoch 3', size=16)
plt.figtext(0.51, 0.56, 'Epoch 4', size=16)

# Label the axes.
ax1.set_ylabel(r'$\mathrm{u (\%)}$', fontsize=18)
ax3.set_ylabel(r'$\mathrm{u (\%)}$', fontsize=18)
ax3.set_xlabel(r'$\mathrm{q (\%)}$', fontsize=18)
ax4.set_xlabel(r'$\mathrm{q (\%)}$', fontsize=18)

# Adjust the right side of the subplots to fit in a colorbar.
#fig.subplots_adjust(right=0.8)

# Add a color bar.
cbar_ax = fig.add_axes([0.85, 0.3, 0.05, 0.6])
cb = fig.colorbar(e1_plot, cax=cbar_ax)

plt.savefig(os.path.join(config["output_path"],'qu_plot3.pdf'),format='pdf')
#plt.savefig('qu_plot3.eps',format='eps',bbox_inches='tight')

"""
Set up a new figure for plotting the LICK q and u as a function of wavelength
"""

fig = plt.figure()
fig.subplots_adjust(hspace=0.001)
fig.subplots_adjust(bottom=0.15)

# Plot flux on the first row of three
ax1 = fig.add_subplot(311)
ax1.plot(lick_lambdas[config["plot_epoch"]-1],lick_flux[config["plot_epoch"]-1]*1.3,color='b',alpha=0.9,linewidth=1.0)
ax1.plot(flux_lambdas[config["plot_epoch"]-1],flux[config["plot_epoch"]-1],color='g',alpha=0.7,linewidth=1.0)
# The following could be used to add error bars.
#ax1.plot(Qflx_bin_edges[1][1:],q_bin[1]*100.0,color='b',alpha=0.9,drawstyle='steps-mid')

# Add minor 4 or 5 minor ticks (putting a number in the () will force a value).
# Search for 'matplotlib ticker' for detailed documentation.
ax1.yaxis.set_minor_locator(AutoMinorLocator())

# Plot q, unbinned and binned, on the second row of three and share the x axis
ax2 = fig.add_subplot(312, sharex=ax1)
ax2.plot(lick_lambdas[config["plot_epoch"]-1],lick_q_isp[config["plot_epoch"]-1]*100.0,color='r',alpha=0.2,linewidth=1.0,drawstyle='steps-pre')
ax2.plot(Qflx_bin_edges[config["plot_epoch"]-1][1:],q_bin_isp[config["plot_epoch"]-1]*100.0,color='g',alpha=0.6,linewidth=0.8,drawstyle='steps-pre')
ax2.plot(lick_Qflx_bin_edges[config["plot_epoch"]-1][1:],lick_q_bin_isp[config["plot_epoch"]-1]*100.0,color='b',alpha=0.9,linewidth=0.8,drawstyle='steps-pre')

# Add minor 4 or 5 minor ticks (putting a number in the () will force a value).
# Search for 'matplotlib ticker' for detailed documentation.
ax2.yaxis.set_minor_locator(AutoMinorLocator())
#ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
#ax2.yaxis.set_major_locator(MultipleLocator(0.5))
#ax2.yaxis.set_minor_locator(MaxNLocator(10))

# Plot u, unbinned and binned, on the third row of three and share the x axis
ax3 = fig.add_subplot(313, sharex=ax1)
ax3.plot(lick_lambdas[config["plot_epoch"]-1],lick_u_isp[config["plot_epoch"]-1]*100.0,color='r',alpha=0.2,linewidth=1.0,drawstyle='steps-pre')
ax3.plot(Uflx_bin_edges[config["plot_epoch"]-1][1:],u_bin_isp[config["plot_epoch"]-1]*100.0,color='g',alpha=0.6,linewidth=0.8,drawstyle='steps-pre')
ax3.plot(lick_Uflx_bin_edges[config["plot_epoch"]-1][1:],lick_u_bin_isp[config["plot_epoch"]-1]*100.0,color='b',alpha=0.9,linewidth=0.8,drawstyle='steps-pre')

# Add minor 4 or 5 minor ticks (putting a number in the () will force a value).
# Search for 'matplotlib ticker' for detailed documentation.
ax3.yaxis.set_minor_locator(AutoMinorLocator())

# Set the x and y boundaries for the plots above.
ax1.set_xbound(config["lambda_min"],config["lambda_max"])
#ax2.set_ybound(-3.5,2.5)
#ax3.set_ybound(-3.0,3.0)
#ax1.set_ybound(0.0,4.5e-14)
ax2.set_ybound(-1.0,1.5)
ax3.set_ybound(-2.5,0.0)

# Don't show the x tick labels on the first two plots.
xticklabels1 = ax1.get_xticklabels()
xticklabels2 = ax2.get_xticklabels()
plt.setp(xticklabels1, visible=False)
plt.setp(xticklabels2, visible=False)

# Don't show the first (and sometimes last) tick labels.
yticks_vals1 = ax1.yaxis.get_major_ticks()
yticks_vals1[0].label1.set_visible(False)

yticks_vals2 = ax2.yaxis.get_major_ticks()
yticks_vals2[0].label1.set_visible(False)
yticks_vals2[-1].label1.set_visible(False)

yticks_vals3 = ax3.yaxis.get_major_ticks()
yticks_vals3[0].label1.set_visible(False)
yticks_vals3[-1].label1.set_visible(False)

# Add a title. Comment out for publications.
#ax1.set_title(config["plot_title"], fontsize=20)

# Label the plot with the name of the SN and the epoch being plotted.
plt.figtext(0.75,0.79,
            config["plot_title"]
	    + '\n' + 'Epoch ' + str(config["plot_epoch"]),
	    size=14)

# Label the x and y axes.
ax3.set_xlabel(r'$\mathrm{\lambda (\AA)}$', fontsize=14)
ax1.set_ylabel(r'$\mathrm{F_\lambda}$', fontsize=14)
ax2.set_ylabel(r'$\mathrm{q (\%)}$', fontsize=14)
ax3.set_ylabel(r'$\mathrm{u (\%)}$', fontsize=14)

plt.savefig(os.path.join(config["output_path"],'qu_plot4.pdf'),format='pdf')

"""
Set up a new figure for plotting all the q data with flux over plotted.
"""

fig = plt.figure()
fig.subplots_adjust(hspace=0.001)
fig.subplots_adjust(bottom=0.15)

# Plot q, unbinned and binned, and overplot flux and share the x axis
ax1 = fig.add_subplot(311)

ax1.plot(all_lambdas[config["first_plot_epoch"]-1],all_q_isp[config["first_plot_epoch"]-1]*100.0,color='k',alpha=0.2,linewidth=1.0)
ax1.plot(all_Qflx_bin_edges[config["first_plot_epoch"]-1][1:],all_q_bin_isp[config["first_plot_epoch"]-1]*100.0,color='k',alpha=0.9,linewidth=0.8,drawstyle='steps-pre')

# Add minor 4 or 5 minor ticks (putting a number in the () will force a value).
# Search for 'matplotlib ticker' for detailed documentation.
ax1.yaxis.set_minor_locator(AutoMinorLocator())

# The following could be used to add error bars.
#ax1.plot(Qflx_bin_edges[1][1:],q_bin[1]*100.0,color='b',alpha=0.9,drawstyle='steps-mid')

# Plot flux times 10^14 on the right y axis sharing the x axis 
ax2 = ax1.twinx()
ax2.plot(all_lambdas[config["first_plot_epoch"]-1],all_flux[config["first_plot_epoch"]-1] * 10**14,color='r',alpha=0.7,linewidth=1.0)

# Plot q, unbinned and binned, and overplot flux and share the x axis
ax3 = fig.add_subplot(312, sharex=ax1)
ax3.plot(all_lambdas[config["first_plot_epoch"]+0],all_q_isp[config["first_plot_epoch"]+0]*100.0,color='k',alpha=0.2,linewidth=1.0,drawstyle='steps-pre')
ax3.plot(all_Qflx_bin_edges[config["first_plot_epoch"]+0][1:],all_q_bin_isp[config["first_plot_epoch"]+0]*100.0,color='k',alpha=0.9,linewidth=0.8,drawstyle='steps-pre')

# Add minor 4 or 5 minor ticks (putting a number in the () will force a value).
# Search for 'matplotlib ticker' for detailed documentation.
ax3.yaxis.set_minor_locator(AutoMinorLocator())

# The following could be used to add error bars.
#ax1.plot(Qflx_bin_edges[1][1:],q_bin[1]*100.0,color='b',alpha=0.9,drawstyle='steps-mid')

# Plot flux times 10^14 on the right y axis sharing the x axis 
ax4 = ax3.twinx()
ax4.plot(all_lambdas[config["first_plot_epoch"]+0],all_flux[config["first_plot_epoch"]+0] * 10**14,color='r',alpha=0.7,linewidth=1.0)

# Plot q, unbinned and binned, and overplot flux and share the x axis
ax5 = fig.add_subplot(313, sharex=ax1)
ax5.plot(all_lambdas[config["first_plot_epoch"]+1],all_q_isp[config["first_plot_epoch"]+1]*100.0,color='k',alpha=0.2,linewidth=1.0,drawstyle='steps-pre')
ax5.plot(all_Qflx_bin_edges[config["first_plot_epoch"]+1][1:],all_q_bin_isp[config["first_plot_epoch"]+1]*100.0,color='k',alpha=0.9,linewidth=0.8,drawstyle='steps-pre')

# Add minor 4 or 5 minor ticks (putting a number in the () will force a value).
# Search for 'matplotlib ticker' for detailed documentation.
ax5.yaxis.set_minor_locator(AutoMinorLocator())

# The following could be used to add error bars.
#ax1.plot(Qflx_bin_edges[1][1:],q_bin[1]*100.0,color='b',alpha=0.9,drawstyle='steps-mid')

# Plot flux on the right y axis sharing the x axis 
ax6 = ax5.twinx()
ax6.plot(all_lambdas[config["first_plot_epoch"]+1],all_flux[config["first_plot_epoch"]+1] * 10**14,color='r',alpha=0.7,linewidth=1.0)

# Shade the continuum regions.
ax1.axvspan(config["band1_range"][0], config["band1_range"][1], alpha=0.2, color='#2874A6')
ax3.axvspan(config["band1_range"][0], config["band1_range"][1], alpha=0.2, color='#2874A6')
ax5.axvspan(config["band1_range"][0], config["band1_range"][1], alpha=0.2, color='#2874A6')

ax1.axvspan(config["band2_range"][0], config["band2_range"][1], alpha=0.2, color='green')
ax3.axvspan(config["band2_range"][0], config["band2_range"][1], alpha=0.2, color='green')
ax5.axvspan(config["band2_range"][0], config["band2_range"][1], alpha=0.2, color='green')

ax1.axvspan(config["band3_range"][0], config["band3_range"][1], alpha=0.2, color='#C0392B')
ax3.axvspan(config["band3_range"][0], config["band3_range"][1], alpha=0.2, color='#C0392B')
ax5.axvspan(config["band3_range"][0], config["band3_range"][1], alpha=0.2, color='#C0392B')

# add lines and labels for emission lines
ax1.axvline(config["H_alpha"], alpha=0.3, color='black')
ax3.axvline(config["H_alpha"], alpha=0.3, color='black')
ax5.axvline(config["H_alpha"], alpha=0.3, color='black')

ax1.axvline(config["H_beta"], alpha=0.3, color='black')
ax3.axvline(config["H_beta"], alpha=0.3, color='black')
ax5.axvline(config["H_beta"], alpha=0.3, color='black')

ax1.axvline(config["HeI"], alpha=0.3, color='black')
ax3.axvline(config["HeI"], alpha=0.3, color='black')
ax5.axvline(config["HeI"], alpha=0.3, color='black')

# Label the lines on the left side aligned right.
ax5.text(config["H_alpha"] - 25.0, -0.95, r'H$\alpha$', size=8, horizontalalignment='right', transform = ax5.transData)
ax5.text(config["HeI"] - 25.0, -0.95, r'HeI', size=8, horizontalalignment='right', transform = ax5.transData)
ax5.text(config["H_beta"] - 25.0, -0.95, r'H$\beta$', size=8, horizontalalignment='right', transform = ax5.transData)

# Set the x and y boundaries for the plots above.
ax1.set_xbound(config["lambda_min"],config["lambda_max"])
ax1.set_ybound(-1.0,1.5)
ax3.set_ybound(-1.0,1.5)
ax5.set_ybound(-1.0,1.5)

# Don't show the x tick labels on the first two plots.
xticklabels1 = ax1.get_xticklabels()
xticklabels3 = ax3.get_xticklabels()
plt.setp(xticklabels1, visible=False)
plt.setp(xticklabels3, visible=False)

# Don't show the first (and sometimes last) tick labels.
yticks_vals1 = ax1.yaxis.get_major_ticks()
yticks_vals1[0].label1.set_visible(False)

#yticks_vals2 = ax2.yaxis.get_major_ticks()
#yticks_vals2[0].label1.set_visible(False)
#yticks_vals2[-1].label1.set_visible(False)

yticks_vals3 = ax3.yaxis.get_major_ticks()
yticks_vals3[0].label1.set_visible(False)
#yticks_vals3[-1].label1.set_visible(False)

# Label the plot with the day being plotted.
ax1.text(4500,1.0,
	    'Day ' + str(config["all_epoch_days"][config["first_plot_epoch"] - 1]),
	    size=14, transform = ax1.transData)
ax3.text(4500,1.0,
	    'Day ' + str(config["all_epoch_days"][config["first_plot_epoch"]]),
	    size=14, transform = ax3.transData)
ax5.text(4500,1.0,
	    'Day ' + str(config["all_epoch_days"][config["first_plot_epoch"] + 1]),
	    size=14, transform = ax5.transData)

# Label the plot with the name of the SN or add a title. Comment out for publications??
ax1.text(6700, 1.0, config["plot_title"], size=14, transform = ax1.transData)
#ax1.set_title(config["plot_title"], fontsize=20)

# Label the x and y axes.
ax5.set_xlabel(r'$\mathrm{\lambda (\AA)}$', fontsize=14)
ax3.set_ylabel(r'$\mathrm{q (\%)}$', fontsize=14)
# the \: below produce a medium space in latex math mode
ax4.set_ylabel(r'$\mathrm{F_\lambda (10^{-14}\: erg\: cm^{-2}\: s^{-1}\: \AA^{-1})}$', fontsize=14)

plt.savefig(os.path.join(config["output_path"],'qu_plot5q.pdf'),format='pdf')

"""
Set up a new figure for plotting all the u data with flux over plotted.
"""

fig = plt.figure()
fig.subplots_adjust(hspace=0.001)
fig.subplots_adjust(bottom=0.15)

# Plot u, unbinned and binned, and overplot flux and share the x axis
ax1 = fig.add_subplot(311)

ax1.plot(all_lambdas[config["first_plot_epoch"]-1],all_u_isp[config["first_plot_epoch"]-1]*100.0,color='k',alpha=0.2,linewidth=1.0)
ax1.plot(all_Uflx_bin_edges[config["first_plot_epoch"]-1][1:],all_u_bin_isp[config["first_plot_epoch"]-1]*100.0,color='k',alpha=0.9,linewidth=0.8,drawstyle='steps-pre')

# Add minor 4 or 5 minor ticks (putting a number in the () will force a value).
# Search for 'matplotlib ticker' for detailed documentation.
ax1.yaxis.set_minor_locator(AutoMinorLocator())

# The following could be used to add error bars.
#ax1.plot(Uflx_bin_edges[1][1:],u_bin[1]*100.0,color='b',alpha=0.9,drawstyle='steps-mid')

# Plot flux times 10^14 on the right y axis sharing the x axis 
ax2 = ax1.twinx()
ax2.plot(all_lambdas[config["first_plot_epoch"]-1],all_flux[config["first_plot_epoch"]-1] * 10**14,color='r',alpha=0.7,linewidth=1.0)

# Plot u, unbinned and binned, and overplot flux and share the x axis
ax3 = fig.add_subplot(312, sharex=ax1)
ax3.plot(all_lambdas[config["first_plot_epoch"]+0],all_u_isp[config["first_plot_epoch"]+0]*100.0,color='k',alpha=0.2,linewidth=1.0,drawstyle='steps-pre')
ax3.plot(all_Uflx_bin_edges[config["first_plot_epoch"]+0][1:],all_u_bin_isp[config["first_plot_epoch"]+0]*100.0,color='k',alpha=0.9,linewidth=0.8,drawstyle='steps-pre')

# Add minor 4 or 5 minor ticks (putting a number in the () will force a value).
# Search for 'matplotlib ticker' for detailed documentation.
ax3.yaxis.set_minor_locator(AutoMinorLocator())

# The following could be used to add error bars.
#ax1.plot(Uflx_bin_edges[1][1:],u_bin[1]*100.0,color='b',alpha=0.9,drawstyle='steps-mid')

# Plot flux times 10^14 on the right y axis sharing the x axis 
ax4 = ax3.twinx()
ax4.plot(all_lambdas[config["first_plot_epoch"]+0],all_flux[config["first_plot_epoch"]+0] * 10**14,color='r',alpha=0.7,linewidth=1.0)

# Plot u, unbinned and binned, and overplot flux and share the x axis
ax5 = fig.add_subplot(313, sharex=ax1)
ax5.plot(all_lambdas[config["first_plot_epoch"]+1],all_u_isp[config["first_plot_epoch"]+1]*100.0,color='k',alpha=0.2,linewidth=1.0,drawstyle='steps-pre')
ax5.plot(all_Uflx_bin_edges[config["first_plot_epoch"]+1][1:],all_u_bin_isp[config["first_plot_epoch"]+1]*100.0,color='k',alpha=0.9,linewidth=0.8,drawstyle='steps-pre')

# Add minor 4 or 5 minor ticks (putting a number in the () will force a value).
# Search for 'matplotlib ticker' for detailed documentation.
ax5.yaxis.set_minor_locator(AutoMinorLocator())

# The following could be used to add error bars.
#ax1.plot(Uflx_bin_edges[1][1:],u_bin[1]*100.0,color='b',alpha=0.9,drawstyle='steps-mid')

# Plot flux on the right y axis sharing the x axis 
ax6 = ax5.twinx()
ax6.plot(all_lambdas[config["first_plot_epoch"]+1],all_flux[config["first_plot_epoch"]+1] * 10**14,color='r',alpha=0.7,linewidth=1.0)

# Shade the continuum regions.
ax1.axvspan(config["band1_range"][0], config["band1_range"][1], alpha=0.2, color='#2874A6')
ax3.axvspan(config["band1_range"][0], config["band1_range"][1], alpha=0.2, color='#2874A6')
ax5.axvspan(config["band1_range"][0], config["band1_range"][1], alpha=0.2, color='#2874A6')

ax1.axvspan(config["band2_range"][0], config["band2_range"][1], alpha=0.2, color='green')
ax3.axvspan(config["band2_range"][0], config["band2_range"][1], alpha=0.2, color='green')
ax5.axvspan(config["band2_range"][0], config["band2_range"][1], alpha=0.2, color='green')

ax1.axvspan(config["band3_range"][0], config["band3_range"][1], alpha=0.2, color='#C0392B')
ax3.axvspan(config["band3_range"][0], config["band3_range"][1], alpha=0.2, color='#C0392B')
ax5.axvspan(config["band3_range"][0], config["band3_range"][1], alpha=0.2, color='#C0392B')

# add lines and labels for emission lines
ax1.axvline(config["H_alpha"], alpha=0.3, color='black')
ax3.axvline(config["H_alpha"], alpha=0.3, color='black')
ax5.axvline(config["H_alpha"], alpha=0.3, color='black')

ax1.axvline(config["H_beta"], alpha=0.3, color='black')
ax3.axvline(config["H_beta"], alpha=0.3, color='black')
ax5.axvline(config["H_beta"], alpha=0.3, color='black')

ax1.axvline(config["HeI"], alpha=0.3, color='black')
ax3.axvline(config["HeI"], alpha=0.3, color='black')
ax5.axvline(config["HeI"], alpha=0.3, color='black')

# Label the lines on the left side aligned right.
ax5.text(config["H_alpha"] - 25.0, -2.45, r'H$\alpha$', size=8, horizontalalignment='right', transform = ax5.transData)
ax5.text(config["HeI"] - 25.0, -2.45, r'HeI', size=8, horizontalalignment='right', transform = ax5.transData)
ax5.text(config["H_beta"] - 25.0, -2.45, r'H$\beta$', size=8, horizontalalignment='right', transform = ax5.transData)

# Set the x and y boundaries for the plots above.
ax1.set_xbound(config["lambda_min"],config["lambda_max"])
ax1.set_ybound(-2.5,0.0)
ax3.set_ybound(-2.5,0.0)
ax5.set_ybound(-2.5,0.0)

# Don't show the x tick labels on the first two plots.
xticklabels1 = ax1.get_xticklabels()
xticklabels3 = ax3.get_xticklabels()
plt.setp(xticklabels1, visible=False)
plt.setp(xticklabels3, visible=False)

# Don't show the first (and sometimes last) tick labels.
yticks_vals1 = ax1.yaxis.get_major_ticks()
yticks_vals1[0].label1.set_visible(False)

#yticks_vals2 = ax2.yaxis.get_major_ticks()
#yticks_vals2[0].label1.set_visible(False)
#yticks_vals2[-1].label1.set_visible(False)

yticks_vals3 = ax3.yaxis.get_major_ticks()
yticks_vals3[0].label1.set_visible(False)
#yticks_vals3[-1].label1.set_visible(False)

# Label the plot with the day being plotted.
ax1.text(4500,-0.5,
	    'Day ' + str(config["all_epoch_days"][config["first_plot_epoch"] - 1]),
	    size=14, transform = ax1.transData)
ax3.text(4500,-0.5,
	    'Day ' + str(config["all_epoch_days"][config["first_plot_epoch"]]),
	    size=14, transform = ax3.transData)
ax5.text(4500,-0.5,
	    'Day ' + str(config["all_epoch_days"][config["first_plot_epoch"] + 1]),
	    size=14, transform = ax5.transData)

# Label the plot with the name of the SN or add a title. Comment out for publications??
ax1.text(6700, -0.5, config["plot_title"], size=14, transform = ax1.transData)
#ax1.set_title(config["plot_title"], fontsize=20)

# Label the x and y axes.
ax5.set_xlabel(r'$\mathrm{\lambda (\AA)}$', fontsize=14)
ax3.set_ylabel(r'$\mathrm{u (\%)}$', fontsize=14)
# the \: below produce a medium space in latex math mode
ax4.set_ylabel(r'$\mathrm{F_\lambda (10^{-14}\: erg\: cm^{-2}\: s^{-1}\: \AA^{-1})}$', fontsize=14)

plt.savefig(os.path.join(config["output_path"],'qu_plot5u.pdf'),format='pdf')

"""
Set up a new figure for plotting all the p data with flux over plotted.
"""

fig = plt.figure()
fig.subplots_adjust(hspace=0.001)
fig.subplots_adjust(bottom=0.15)

# Plot q, unbinned and binned, and overplot flux and share the x axis
ax1 = fig.add_subplot(311)

#ax1.plot(all_lambdas[config["first_plot_epoch"]-1],all_p[config["first_plot_epoch"]-1]*100.0,color='k',alpha=0.2,linewidth=1.0)
ax1.plot(all_Qflx_bin_edges[config["first_plot_epoch"]-1][1:],all_p_bin[config["first_plot_epoch"]-1]*100.0,color='k',alpha=0.9,linewidth=0.8,drawstyle='steps-pre')
#ax1.plot(q_lambdas[config["plot_epoch"]-1],q_isp[config["plot_epoch"]-1]*100.0,color='k',alpha=0.2,linewidth=1.0)
#ax1.plot(Qflx_bin_edges[config["plot_epoch"]-1][1:],q_bin_isp[config["plot_epoch"]-1]*100.0,color='k',alpha=0.9,linewidth=0.8,drawstyle='steps-pre')

# Add minor 4 or 5 minor ticks (putting a number in the () will force a value).
# Search for 'matplotlib ticker' for detailed documentation.
ax1.yaxis.set_minor_locator(AutoMinorLocator())

# The following could be used to add error bars.
#ax1.plot(Qflx_bin_edges[1][1:],q_bin[1]*100.0,color='b',alpha=0.9,drawstyle='steps-mid')

# Plot flux on the right y axis sharing the x axis 
ax2 = ax1.twinx()
ax2.plot(all_lambdas[config["first_plot_epoch"]-1],all_flux[config["first_plot_epoch"]-1],color='r',alpha=0.7,linewidth=1.0)

# Plot q, unbinned and binned, and overplot flux and share the x axis
ax3 = fig.add_subplot(312, sharex=ax1)
#ax3.plot(all_lambdas[config["first_plot_epoch"]+0],all_p[config["first_plot_epoch"]+0]*100.0,color='k',alpha=0.2,linewidth=1.0,drawstyle='steps-pre')
ax3.plot(all_Qflx_bin_edges[config["first_plot_epoch"]+0][1:],all_p_bin[config["first_plot_epoch"]+0]*100.0,color='k',alpha=0.9,linewidth=0.8,drawstyle='steps-pre')

# Add minor 4 or 5 minor ticks (putting a number in the () will force a value).
# Search for 'matplotlib ticker' for detailed documentation.
ax3.yaxis.set_minor_locator(AutoMinorLocator())

# The following could be used to add error bars.
#ax1.plot(Qflx_bin_edges[1][1:],q_bin[1]*100.0,color='b',alpha=0.9,drawstyle='steps-mid')

# Plot flux on the right y axis sharing the x axis 
ax4 = ax3.twinx()
ax4.plot(all_lambdas[config["first_plot_epoch"]+0],all_flux[config["first_plot_epoch"]+0],color='r',alpha=0.7,linewidth=1.0)

# Plot q, unbinned and binned, and overplot flux and share the x axis
ax5 = fig.add_subplot(313, sharex=ax1)
#ax5.plot(all_lambdas[config["first_plot_epoch"]+1],all_p[config["first_plot_epoch"]+1]*100.0,color='k',alpha=0.2,linewidth=1.0,drawstyle='steps-pre')
ax5.plot(all_Qflx_bin_edges[config["first_plot_epoch"]+1][1:],all_p_bin[config["first_plot_epoch"]+1]*100.0,color='k',alpha=0.9,linewidth=0.8,drawstyle='steps-pre')

# Add minor 4 or 5 minor ticks (putting a number in the () will force a value).
# Search for 'matplotlib ticker' for detailed documentation.
ax5.yaxis.set_minor_locator(AutoMinorLocator())

# The following could be used to add error bars.
#ax1.plot(Qflx_bin_edges[1][1:],q_bin[1]*100.0,color='b',alpha=0.9,drawstyle='steps-mid')

# Plot flux on the right y axis sharing the x axis 
ax6 = ax5.twinx()
ax6.plot(all_lambdas[config["first_plot_epoch"]+1],all_flux[config["first_plot_epoch"]+1],color='r',alpha=0.7,linewidth=1.0)

# Set the x and y boundaries for the plots above.
ax1.set_xbound(config["lambda_min"],config["lambda_max"])
#ax2.set_ybound(-3.5,2.5)
#ax3.set_ybound(-3.0,3.0)
#ax1.set_ybound(0.0,4.5e-14)
ax1.set_ybound(0.0,2.5)
ax3.set_ybound(0.0,2.5)
ax5.set_ybound(0.0,2.5)
#ax3.set_ybound(-2.5,0.0)

# Don't show the x tick labels on the first two plots.
xticklabels1 = ax1.get_xticklabels()
xticklabels3 = ax3.get_xticklabels()
plt.setp(xticklabels1, visible=False)
plt.setp(xticklabels3, visible=False)

# Don't show the first (and sometimes last) tick labels.
yticks_vals1 = ax1.yaxis.get_major_ticks()
yticks_vals1[0].label1.set_visible(False)

#yticks_vals2 = ax2.yaxis.get_major_ticks()
#yticks_vals2[0].label1.set_visible(False)
#yticks_vals2[-1].label1.set_visible(False)

yticks_vals3 = ax3.yaxis.get_major_ticks()
yticks_vals3[0].label1.set_visible(False)
#yticks_vals3[-1].label1.set_visible(False)

# Add a title. Comment out for publications.
#ax1.set_title(config["plot_title"], fontsize=20)

# Label the plot with the name of the SN and the epoch being plotted.
plt.figtext(0.75,0.79,
            config["plot_title"]
	    + '\n' + 'Epoch ' + str(config["first_plot_epoch"]),
	    size=14)

# Label the x and y axes.
ax5.set_xlabel(r'$\mathrm{\lambda (\AA)}$', fontsize=14)
#ax1.set_ylabel(r'$\mathrm{q (\%)}$', fontsize=14)
#ax2.set_ylabel(r'$\mathrm{F_\lambda}$', fontsize=14)
ax3.set_ylabel(r'$\mathrm{P (\%)}$', fontsize=14)
ax4.set_ylabel(r'$\mathrm{F_\lambda}$', fontsize=14)
#ax3.set_ylabel(r'$\mathrm{u (\%)}$', fontsize=14)

plt.savefig(os.path.join(config["output_path"],'qu_plot6.pdf'),format='pdf')

"""
Set up a new figure for plotting all the p data with flux over plotted.
"""

fig = plt.figure()
fig.subplots_adjust(hspace=0.001)
fig.subplots_adjust(bottom=0.15)

# Plot q, unbinned and binned, and overplot flux and share the x axis
ax1 = fig.add_subplot(311)

#ax1.plot(all_lambdas[config["first_plot_epoch"]-1],all_p[config["first_plot_epoch"]-1]*100.0,color='k',alpha=0.2,linewidth=1.0)
ax1.plot(all_Qflx_bin_edges[config["first_plot_epoch"]-1][1:],all_theta_bin[config["first_plot_epoch"]-1],color='k',alpha=0.9,linewidth=0.8,drawstyle='steps-pre')
#ax1.plot(q_lambdas[config["plot_epoch"]-1],q_isp[config["plot_epoch"]-1]*100.0,color='k',alpha=0.2,linewidth=1.0)
#ax1.plot(Qflx_bin_edges[config["plot_epoch"]-1][1:],q_bin_isp[config["plot_epoch"]-1]*100.0,color='k',alpha=0.9,linewidth=0.8,drawstyle='steps-pre')

# Add minor 4 or 5 minor ticks (putting a number in the () will force a value).
# Search for 'matplotlib ticker' for detailed documentation.
ax1.yaxis.set_minor_locator(AutoMinorLocator())

# The following could be used to add error bars.
#ax1.plot(Qflx_bin_edges[1][1:],q_bin[1]*100.0,color='b',alpha=0.9,drawstyle='steps-mid')

# Plot flux on the right y axis sharing the x axis 
ax2 = ax1.twinx()
ax2.plot(all_lambdas[config["first_plot_epoch"]-1],all_flux[config["first_plot_epoch"]-1],color='r',alpha=0.7,linewidth=1.0)

# Plot q, unbinned and binned, and overplot flux and share the x axis
ax3 = fig.add_subplot(312, sharex=ax1)
#ax3.plot(all_lambdas[config["first_plot_epoch"]+0],all_p[config["first_plot_epoch"]+0]*100.0,color='k',alpha=0.2,linewidth=1.0,drawstyle='steps-pre')
ax3.plot(all_Qflx_bin_edges[config["first_plot_epoch"]+0][1:],all_theta_bin[config["first_plot_epoch"]+0],color='k',alpha=0.9,linewidth=0.8,drawstyle='steps-pre')

# Add minor 4 or 5 minor ticks (putting a number in the () will force a value).
# Search for 'matplotlib ticker' for detailed documentation.
ax3.yaxis.set_minor_locator(AutoMinorLocator())

# The following could be used to add error bars.
#ax1.plot(Qflx_bin_edges[1][1:],q_bin[1]*100.0,color='b',alpha=0.9,drawstyle='steps-mid')

# Plot flux on the right y axis sharing the x axis 
ax4 = ax3.twinx()
ax4.plot(all_lambdas[config["first_plot_epoch"]+0],all_flux[config["first_plot_epoch"]+0],color='r',alpha=0.7,linewidth=1.0)

# Plot q, unbinned and binned, and overplot flux and share the x axis
ax5 = fig.add_subplot(313, sharex=ax1)
#ax5.plot(all_lambdas[config["first_plot_epoch"]+1],all_p[config["first_plot_epoch"]+1]*100.0,color='k',alpha=0.2,linewidth=1.0,drawstyle='steps-pre')
ax5.plot(all_Qflx_bin_edges[config["first_plot_epoch"]+1][1:],all_theta_bin[config["first_plot_epoch"]+1],color='k',alpha=0.9,linewidth=0.8,drawstyle='steps-pre')

# Add minor 4 or 5 minor ticks (putting a number in the () will force a value).
# Search for 'matplotlib ticker' for detailed documentation.
ax5.yaxis.set_minor_locator(AutoMinorLocator())

# The following could be used to add error bars.
#ax1.plot(Qflx_bin_edges[1][1:],q_bin[1]*100.0,color='b',alpha=0.9,drawstyle='steps-mid')

# Plot flux on the right y axis sharing the x axis 
ax6 = ax5.twinx()
ax6.plot(all_lambdas[config["first_plot_epoch"]+1],all_flux[config["first_plot_epoch"]+1],color='r',alpha=0.7,linewidth=1.0)

# Set the x and y boundaries for the plots above.
ax1.set_xbound(config["lambda_min"],config["lambda_max"])
#ax2.set_ybound(-3.5,2.5)
#ax3.set_ybound(-3.0,3.0)
#ax1.set_ybound(0.0,4.5e-14)
ax1.set_ybound(-60.0,-20.0)
ax3.set_ybound(-60.0,-20.0)
ax5.set_ybound(-60.0,-20.0)
#ax3.set_ybound(-2.5,0.0)

# Don't show the x tick labels on the first two plots.
xticklabels1 = ax1.get_xticklabels()
xticklabels3 = ax3.get_xticklabels()
plt.setp(xticklabels1, visible=False)
plt.setp(xticklabels3, visible=False)

# Don't show the first (and sometimes last) tick labels.
yticks_vals1 = ax1.yaxis.get_major_ticks()
yticks_vals1[0].label1.set_visible(False)

#yticks_vals2 = ax2.yaxis.get_major_ticks()
#yticks_vals2[0].label1.set_visible(False)
#yticks_vals2[-1].label1.set_visible(False)

yticks_vals3 = ax3.yaxis.get_major_ticks()
yticks_vals3[0].label1.set_visible(False)
#yticks_vals3[-1].label1.set_visible(False)

# Add a title. Comment out for publications.
#ax1.set_title(config["plot_title"], fontsize=20)

# Label the plot with the name of the SN and the epoch being plotted.
plt.figtext(0.75,0.79,
            config["plot_title"]
	    + '\n' + 'Epoch ' + str(config["first_plot_epoch"]),
	    size=14)

# Label the x and y axes.
ax5.set_xlabel(r'$\mathrm{\lambda (\AA)}$', fontsize=14)
#ax1.set_ylabel(r'$\mathrm{q (\%)}$', fontsize=14)
#ax2.set_ylabel(r'$\mathrm{F_\lambda}$', fontsize=14)
ax3.set_ylabel(r'$\mathrm{\theta (^\circ)}$', fontsize=14)
ax4.set_ylabel(r'$\mathrm{F_\lambda}$', fontsize=14)
#ax3.set_ylabel(r'$\mathrm{u (\%)}$', fontsize=14)

plt.savefig(os.path.join(config["output_path"],'qu_plot7.pdf'),format='pdf')

"""
Set up a new figure for plotting broadband polarization as a function of time.
"""

# Make this plot square at 6" x 6"
fig = plt.figure(figsize=(6.0,6.0))

# Generate an axes for plotting, define the axis labels, and set the x limit
ax1 = fig.add_subplot(111)

ax1.set_ylabel(r'$P (\%)$')
ax1.set_xlabel('Days Post Maximum')

ax1.set_xlim([-10.0,600.0])
ax1.set_ylim([-0.5, 2.7])

# Define 5 minor tick marks on the x and y axes.
ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
ax1.yaxis.set_minor_locator(AutoMinorLocator(5))


#  ***** BELOW TESTING THAT SHOULD BE REMOVED *****

jelly_bean = p_bin_band1[:,0]
jelly_belly = np.squeeze(p_bin_band1)

print('Just checking what the array dimensions are: ')
print(p_bin_band1.shape)
print(qsig_bin_band1.shape)
print(jelly_bean.shape)
print(jelly_bean)
print(jelly_belly.shape)
print(jelly_belly)

#  ***** ABOVE TESTING THAT SHOULD BE REMOVED *****

# The order of the markers matters, the last ones plotted will be on top.

# I use np.squeeze to reduce the dimensions of the array to 1x
# Use blue squares for band1, the color 'blue' is too intense.
marker1 = ax1.errorbar(config["epoch_days"], p_bin_band1*100.0 - 0.0, yerr=qsig_bin_band1*100, markerfacecolor='#2874A6', markeredgecolor='k', markeredgewidth=0.6, markersize=7, alpha=0.9, elinewidth=1, ecolor='k', capsize=3, fmt='s')
marker2 = ax1.errorbar(config["lick_epoch_days"], lick_p_bin_band1*100.0 - 0.0, yerr=lick_qsig_bin_band1*100, markerfacecolor='#2874A6', markeredgecolor='k', markeredgewidth=0.6, markersize=7, alpha=0.4, elinewidth=1, ecolor='k', capsize=3, fmt='s')

# Use red diamonds for band3, the color 'red' is too intense.
marker5 = ax1.errorbar(config["epoch_days"], p_bin_band3*100.0 - 0.5, yerr=qsig_bin_band3*100, markerfacecolor='#C0392B', markeredgecolor='k', markeredgewidth=0.6, markersize=7, alpha=0.9, elinewidth=1, ecolor='k', capsize=3, fmt='D')
marker6 = ax1.errorbar(config["lick_epoch_days"], lick_p_bin_band3*100.0 - 0.5, yerr=lick_qsig_bin_band3*100, markerfacecolor='#C0392B', markeredgecolor='k', markeredgewidth=0.6, markersize=7, alpha=0.4, elinewidth=1, ecolor='k', capsize=3, fmt='D')

# Use green circles for band2, green could be #1E8449, yellow would be #F1C40F
# Having these come last also puts them on top of the other two marker types
marker3 = ax1.errorbar(config["epoch_days"], p_bin_band2*100.0 - 0.25, yerr=qsig_bin_band2*100, markerfacecolor='green', markeredgecolor='k', markeredgewidth=0.6, markersize=7, alpha=0.9, elinewidth=1, ecolor='k', capsize=3, fmt='o')
marker4 = ax1.errorbar(config["lick_epoch_days"], lick_p_bin_band2*100.0 - 0.25, yerr=lick_qsig_bin_band2*100, markerfacecolor='green', markeredgecolor='k', markeredgewidth=0.6, markersize=7, alpha=0.4, elinewidth=1, ecolor='k', capsize=3, fmt='o')

# Make a second y axis on the right that shares the x axis of ax1
ax2 = ax1.twinx()

# The following two lines put the markers for ax2 behind the markers for ax1
ax1.set_zorder(10)
ax1.patch.set_visible(False)

# The AutoMinorLocator sets 5 minor tick marks
ax2.set_ylabel('V Magnitude')
ax2.set_ylim([19.5, 13.5])
ax2.yaxis.set_minor_locator(AutoMinorLocator(5))

# The comma after marker7 is necessary because "plot" returns a list of Line2D
# and "legend" can't take a list.
# the comma could have been omitted if legend included 'marker7[0]', i.e. the
# first element of the list.
marker7, = ax2.plot(phot_jd - config["mjd_at_max"], phot_V, 'd', color='0.8', alpha=1.0)

# Put a legend in the upper right
ax1.legend([marker1, marker3, marker5, marker7], [r'$P(4450-4750 \mathrm{\AA})$', r'$P(5250-5550 \mathrm{\AA}) - 0.25$', r'$P(6050-6350 \mathrm{\AA}) - 0.50$', 'V Magnitude'], loc = 'upper right', fontsize = 'small')

# This is an example of exluding non-valid data point in the plot.
#ax1.plot(MJD[np.where( V != 99.999 )], V[np.where( V != 99.999 )], 'ro', MJD[np.where( R != 99.999 )], R[np.where( R != 99.999 )],'bs')


# Define an exponential decay function.
def exp_decay(x, p_0, x_0):
    return p_0 * np.exp(-1.0 * x / x_0)

# Define an inverse time decay function.
def inv_t_decay(x, n_0, x_0):
    return (x - x_0) ** (-1.0 * n_0)

# Fit all three bands to an exponential decay.
# Use the weighted (inverse sigmas) optimize curve fit function
params_band1, params_band1_covariance = optimize.curve_fit(exp_decay, config["all_epoch_days"], all_p_bin_band1, p0 = [0.025, 250.0], sigma = all_qsig_bin_band1)
params_band1_inv, params_band1_inv_covariance = optimize.curve_fit(inv_t_decay, config["all_epoch_days"], all_p_bin_band1, p0 = [1.0, 0.0], sigma = all_qsig_bin_band1)
params_band2, params_band2_covariance = optimize.curve_fit(exp_decay, config["all_epoch_days"], all_p_bin_band2, p0 = [0.025, 250.0], sigma = all_qsig_bin_band2)
#params_band2_inv, params_band2_inv_covariance = optimize.curve_fit(inv_t_decay, config["all_epoch_days"], all_p_bin_band2, p0 = [1.0, 0.0], sigma = all_qsig_bin_band2)
params_band3, params_band3_covariance = optimize.curve_fit(exp_decay, config["all_epoch_days"], all_p_bin_band3, p0 = [0.025, 250.0], sigma = all_qsig_bin_band3)

# Fit just the first n data points of band1 to an exponential decay.
params_band1_part, params_band1_part_covariance = optimize.curve_fit(exp_decay, config["all_epoch_days"][:13], all_p_bin_band1[:13], p0 = [0.025, 250.0], sigma = all_qsig_bin_band1[:13])

print('The best fit parameters to band1 are p_0, t_0: ')
print(params_band1)
print('The best fit parameters to ONLY the first 13 point of band1 are p_0, t_0: ')
print(params_band1_part)
print('The best fit parameters to band1 inv_t are n_0, t_0: ')
print(params_band1_inv)
print('The best fit parameters to band2 are p_0, t_0: ')
print(params_band2)
print('The best fit parameters to band3 are p_0, t_0: ')
print(params_band3)

# Create a filled array from 0 to 600 in increments of 20
all_days = np.arange(0,600,20)
ax1.plot(all_days, 100.0 * exp_decay(all_days, params_band1[0], params_band1[1]) - 0.0, label='Best Fit to band1', linewidth = 1.0, color = 'k', alpha = 1.0)
ax1.plot(all_days, 100.0 * exp_decay(all_days, params_band2[0], params_band2[1]) - 0.25, label='Best Fit to band2', linewidth = 1.0, color = 'k', alpha = 0.3)
ax1.plot(all_days, 100.0 * exp_decay(all_days, params_band3[0], params_band3[1]) - 0.5, label='Best Fit to band3', linewidth = 1.0, color = 'k', alpha = 0.3)

ax1.plot(all_days, 100.0 * exp_decay(all_days, params_band1_part[0], params_band1_part[1]) - 0.0, label='Best Fit to part of band1', linewidth = 0.7, linestyle = 'dashed', color = 'k', alpha = 1.0)

ax1.plot(all_days, 100.0 * inv_t_decay(all_days, params_band1_inv[0], params_band1_inv[1]) - 0.0, label='Best Fit to inverse time', linewidth = 1.0, linestyle = 'dotted', color = 'k', alpha = 1.0)

ax1.text(320, 0.92, r'$(t - t_0)^{-n}$', size=12, transform = ax1.transData)
ax1.text(360, 0.62, r'$P_0 e^{-t/\tau}$', size=12, transform = ax1.transData)

plt.savefig(os.path.join(config["output_path"],'qu_plot8.pdf'),format='pdf')

"""
Set up a new figure for plotting a q/u plot for the broadband data.
"""

# Make this plot square at 6" x 6"
fig = plt.figure(figsize=(6.0,6.0))
fig.subplots_adjust(wspace=0.001)
###fig.subplots_adjust(bottom=0.15)

# Generate an axes for plotting, define the axis labels, and set the x limit
ax1 = fig.add_subplot(131)

# Only put a y label on the first plot, the x label will go on the second
ax1.set_ylabel(r'$u (\%)$')

ax1.set_xlim([-0.50, 0.50])
ax1.set_ylim([-2.5, 0.5])

# The order of the markers matters, the last ones plotted will be on top.

# I use np.squeeze to reduce the dimensions of the array to 1x
# Use blue squares for band1, the color 'blue' is too intense.
marker1 = ax1.errorbar(q_bin_band1_isp*100.0, u_bin_band1_isp*100.0, xerr=qsig_bin_band1*100, yerr=usig_bin_band1*100, markerfacecolor='#2874A6', markeredgecolor='k', markeredgewidth=0.6, markersize=7, alpha=0.9, elinewidth=1, ecolor='k', capsize=3, fmt='s')
marker2 = ax1.errorbar(lick_q_bin_band1_isp*100.0, lick_u_bin_band1_isp*100.0, xerr=lick_qsig_bin_band1*100, yerr=lick_usig_bin_band1*100, markerfacecolor='#2874A6', markeredgecolor='k', markeredgewidth=0.6, markersize=7, alpha=0.4, elinewidth=1, ecolor='k', capsize=3, fmt='s')
line1 = ax1.plot(all_q_bin_band1_isp*100.0, all_u_bin_band1_isp*100.0, '-', color='#2874A6', alpha=1.0)

# Put a marker for the measure ISP location:
isp_marker1 = ax1.plot([0.031], [0.024], marker="x", color='#2874A6')
isp_marker2 = ax1.plot([-0.041], [0.014], marker="+", color='#2874A6')

# Create the third plot in column 3 and share the y-axis
ax2 = fig.add_subplot(132, sharey=ax1)
ax2.set_xlim([-0.50, 0.50])

# Put an x label on the second plot only.
ax2.set_xlabel(r'$q (\%)$')

# Don't show the y tick labels on the second plot.
yticklabels2 = ax2.get_yticklabels()
plt.setp(yticklabels2, visible=False)

# Use green circles for band2, green could be #1E8449, yellow would be #F1C40F
# Having these come last also puts them on top of the other two marker types
marker3 = ax2.errorbar(q_bin_band2_isp*100.0, u_bin_band2_isp*100.0, xerr=qsig_bin_band2*100, yerr=usig_bin_band2*100, markerfacecolor='green', markeredgecolor='k', markeredgewidth=0.6, markersize=7, alpha=0.9, elinewidth=1, ecolor='k', capsize=3, fmt='o')
marker4 = ax2.errorbar(lick_q_bin_band2_isp*100.0, lick_u_bin_band2_isp*100.0, xerr=lick_qsig_bin_band2*100, yerr=lick_usig_bin_band2*100, markerfacecolor='green', markeredgecolor='k', markeredgewidth=0.6, markersize=7, alpha=0.4, elinewidth=1, ecolor='k', capsize=3, fmt='o')
line2 = ax2.plot(all_q_bin_band2_isp*100.0, all_u_bin_band2_isp*100.0, '-', color='green', alpha=1.0)

# Put a marker for the measure ISP location:
isp_marker3 = ax2.plot([0.024], [0.023], marker="x", color='green')
isp_marker4 = ax2.plot([-0.105], [-0.022], marker="+", color='green')

# Create the second plot in column 2 and share the y-axis
ax3 = fig.add_subplot(133, sharey=ax2)
ax3.set_xlim([-0.50, 0.50])

# Define 4 or 5 minor tick marks on the x and y axes.
ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
ax2.xaxis.set_minor_locator(AutoMinorLocator(4))
ax3.xaxis.set_minor_locator(AutoMinorLocator(4))

# Define which tick labels to show (so they don't overlape).
ax1.set_xticks([-0.4,0,0.4])
ax2.set_xticks([-0.4,0,0.4])
ax3.set_xticks([-0.4,0,0.4])

# Don't show the y tick labels on the second plot.
yticklabels3 = ax3.get_yticklabels()
plt.setp(yticklabels3, visible=False)

# Use red diamonds for band3, the color 'red' is too intense.
marker5 = ax3.errorbar(q_bin_band3_isp*100.0, u_bin_band3_isp*100.0, xerr=qsig_bin_band3*100, yerr=usig_bin_band3*100, markerfacecolor='#C0392B', markeredgecolor='k', markeredgewidth=0.6, markersize=7, alpha=0.9, elinewidth=1, ecolor='k', capsize=3, fmt='D')
marker6 = ax3.errorbar(lick_q_bin_band3_isp*100.0, lick_u_bin_band3_isp*100.0, xerr=lick_qsig_bin_band3*100, yerr=lick_usig_bin_band3*100, markerfacecolor='#C0392B', markeredgecolor='k', markeredgewidth=0.6, markersize=7, alpha=0.4, elinewidth=1, ecolor='k', capsize=3, fmt='D')
line3 = ax3.plot(all_q_bin_band3_isp*100.0, all_u_bin_band3_isp*100.0, '-', color='#C0392B', alpha=1.0)

# Put a marker for the measure ISP location:
isp_marker3 = ax3.plot([-0.006], [-0.001], marker="x", color='#C0392B')
isp_marker4 = ax3.plot([-0.084], [0.000], marker="+", color='#C0392B')

# Put a legend in the upper right
ax3.legend([marker1, marker3, marker5], [r'$4450-4750 \mathrm{\AA}$', r'$5250-5550 \mathrm{\AA}$', r'$6050-6350 \mathrm{\AA}$'], loc = 'upper right', fontsize = 'x-small')

# Make crosshairs at 0,0 and use zorder to put them below the data points.
ax1.axhline(0, color='grey', linewidth=0.8, zorder=0)
ax1.axvline(0, color='grey', linewidth=0.8, zorder=0)
ax2.axhline(0, color='grey', linewidth=0.8, zorder=0)
ax2.axvline(0, color='grey', linewidth=0.8, zorder=0)
ax3.axhline(0, color='grey', linewidth=0.8, zorder=0)
ax3.axvline(0, color='grey', linewidth=0.8, zorder=0)

# Draw five concentric circles denoting %P intervals of 0.5%
ax1.add_artist(plt.Circle((0, 0), 0.5, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax1.add_artist(plt.Circle((0, 0), 1.0, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax1.add_artist(plt.Circle((0, 0), 1.5, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax1.add_artist(plt.Circle((0, 0), 2.0, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax1.add_artist(plt.Circle((0, 0), 2.5, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax2.add_artist(plt.Circle((0, 0), 0.5, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax2.add_artist(plt.Circle((0, 0), 1.0, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax2.add_artist(plt.Circle((0, 0), 1.5, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax2.add_artist(plt.Circle((0, 0), 2.0, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax2.add_artist(plt.Circle((0, 0), 2.5, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax3.add_artist(plt.Circle((0, 0), 0.5, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax3.add_artist(plt.Circle((0, 0), 1.0, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax3.add_artist(plt.Circle((0, 0), 1.5, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax3.add_artist(plt.Circle((0, 0), 2.0, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))
ax3.add_artist(plt.Circle((0, 0), 2.5, color='grey', fill=False, linewidth=0.5, linestyle='dotted', zorder=0))

plt.savefig(os.path.join(config["output_path"],'qu_plot9.pdf'),format='pdf')
