#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 12:56:19 2021

@author: mars

Script for the exploration of the peak resolution for the different measurements
taken at CNA on 21/7. Gate, gain, Th were variated.

The aloha source has 3 peaks close between each other:
    
    Pu239   5,156.59 keV
    Am241   5,485.56 keV
    Cm244   5,80482 keV

"""

#%% ######## 0) Module importing ##########################


import sys                   #to import functions from other folders!!
sys.path.insert(0, '/home/mars/Python/Python_functions_home_made-main')  
         #path where I have the functions
         
import Read_root_COMPASS   #module to load .root data
import Fits   #module that contains many fit functions

import time      #to measure the running time
import numpy as np      #numpy array object, very useful
import pandas as pd    #dataframe object, extremely useful
import matplotlib.pyplot as plt   #for plotting




#%% ########## 1) Data loading ############

#Note that a series of data with a record length of 8k has not been loaded because
#looking at its spectrum, we can see 2 3 alpha structures, whihc would mean that
#we changed parameters online.

dat_448 = Read_root_COMPASS.ReadRootSingleCOMPASS('SDataF_record_length_8000_Th_330LSB_gate_448_gain_40.root',
                                               Waveform_saving = False )
plt.savefig('Spectrum_gate_448.png', format='png')      #plot saving

dat_1k = Read_root_COMPASS.ReadRootSingleCOMPASS('SDataF_medida_6_rec_len_262112_pre_trigger_992_Th_430_g.root', 
                                                 Waveform_saving= False )
plt.savefig('Spectrum_gate_1k.png', format='png')      #plot saving

dat_8k = Read_root_COMPASS.ReadRootSingleCOMPASS('SDataF_rec_len_16k_Gate_8000_Th_330_gain_640.root',
                                                  Waveform_saving=False )
plt.savefig('Spectrum_gate_8k.png', format='png')      #plot saving


dat_25k = Read_root_COMPASS.ReadRootSingleCOMPASS('SDataF_rec_len_32k_gate_25k_Th_330_gain_640_pregate_200.root',
                                                  Waveform_saving=False )
plt.savefig('Spectrum_gate_25k.png', format='png')      #plot saving

dat_40k = Read_root_COMPASS.ReadRootSingleCOMPASS('SDataF_rec_len_26212_pre_trigger_992_Th_400LSB_gate.root',
                                                  Waveform_saving=False)
plt.savefig('Spectrum_gate_40k.png', format='png')      #plot saving



#%% ############ 2) R calc #################################

#Lets do the gaussian fit to the gamma () of Cs137, to see the FWHM as a function
#of the scintillation crystal

def gaussian(x, Heigh, Mean, Std_dev):
    return Heigh * np.exp(- (x-Mean)**2 / (2 * Std_dev**2)) 
    #

        #this is a gaussian function (more general than normal distribution)
        
        
############ GATE = 448 ##############


###1st peak (to appear)

index_min = 171          #index that starts the peak (by looking the graph)
index_max = 179          #index that ends the peak (by looking the graph)

counts_peak = np.array( dat_448['Hist']['Counts'][index_min-1:index_max-1] )
ch_peak = np.array( dat_448['Hist']['E[ch]'][index_min-1:index_max-1] )


fit_448_peak_1 = Fits.Gaussian_fit(ch_peak,counts_peak)                #fit


###2nd peak (to appear)

index_min = 182          #index that starts the peak (by looking the graph)
index_max = 190          #index that ends the peak (by looking the graph)

counts_peak = np.array( dat_448['Hist']['Counts'][index_min-1:index_max-1] )
ch_peak = np.array( dat_448['Hist']['E[ch]'][index_min-1:index_max-1] )


fit_448_peak_2 = Fits.Gaussian_fit(ch_peak,counts_peak)                #fit


###3rd peak (to appear)

index_min = 191          #index that starts the peak (by looking the graph)
index_max = 199          #index that ends the peak (by looking the graph)

counts_peak = np.array( dat_448['Hist']['Counts'][index_min-1:index_max-1] )
ch_peak = np.array( dat_448['Hist']['E[ch]'][index_min-1:index_max-1] )


fit_448_peak_3 = Fits.Gaussian_fit(ch_peak,counts_peak)                #fit



#Resolution plot
#will plot the resolution with a bar plot (error also included, of course)

plt.figure(figsize=(8,5))  #width, heigh 6.4*4.8 inches by default
plt.bar(['Pu239','Am241', 'Cm244'], 
        np.array( [ fit_448_peak_1['R[%]'], 
                   fit_448_peak_1['R[%]'], 
                   fit_448_peak_1['R[%]'] ] ), 
        yerr = np.array( [ fit_448_peak_1['\Delta(R[%])'], 
                          fit_448_peak_1['\Delta(R[%])'], 
                          fit_448_peak_1['\Delta(R[%])'] ] ), 
        edgecolor="black")
plt.title("Resolution of the alpha peaks for gate = 448ns", fontsize=22, wrap=True)           #title
plt.ylabel("R (%)", fontsize=14)              #ylabel
plt.tick_params(axis='both', labelsize=14)              #size of axis
plt.grid(True) 
plt.savefig('Resolution_gate_448ns.png', format='png')



############ GATE = 1k ##############


###1st peak (to appear)

index_min = 169          #index that starts the peak (by looking the graph)
index_max = 178          #index that ends the peak (by looking the graph)

counts_peak = np.array( dat_1k['Hist']['Counts'][index_min-1:index_max-1] )
ch_peak = np.array( dat_1k['Hist']['E[ch]'][index_min-1:index_max-1] )


fit_1k_peak_1 = Fits.Gaussian_fit(ch_peak,counts_peak)                #fit


###2nd peak (to appear)

index_min = 179          #index that starts the peak (by looking the graph)
index_max = 190          #index that ends the peak (by looking the graph)

counts_peak = np.array( dat_1k['Hist']['Counts'][index_min-1:index_max-1] )
ch_peak = np.array( dat_1k['Hist']['E[ch]'][index_min-1:index_max-1] )


fit_1k_peak_2 = Fits.Gaussian_fit(ch_peak,counts_peak)                #fit


###3rd peak (to appear)

index_min = 190          #index that starts the peak (by looking the graph)
index_max = 199          #index that ends the peak (by looking the graph)

counts_peak = np.array( dat_1k['Hist']['Counts'][index_min-1:index_max-1] )
ch_peak = np.array( dat_1k['Hist']['E[ch]'][index_min-1:index_max-1] )


fit_1k_peak_3 = Fits.Gaussian_fit(ch_peak,counts_peak)                #fit


#Plot
plt.figure(figsize=(8,5))  #width, heigh 6.4*4.8 inches by default
plt.bar(['Pu239','Am241', 'Cm244'], 
        np.array( [ fit_1k_peak_1['R[%]'], 
                   fit_1k_peak_1['R[%]'], 
                   fit_1k_peak_1['R[%]'] ] ), 
        yerr = np.array( [ fit_1k_peak_1['\Delta(R[%])'], 
                          fit_1k_peak_1['\Delta(R[%])'], 
                          fit_1k_peak_1['\Delta(R[%])'] ] ), 
        edgecolor="black")
plt.title("Resolution of the alpha peaks for gate = 1kns", fontsize=22, wrap=True)           #title
plt.ylabel("R (%)", fontsize=14)              #ylabel
plt.tick_params(axis='both', labelsize=14)              #size of axis
plt.grid(True) 
plt.savefig('Resolution_gate_1kns.png', format='png')



############ GATE = 8k ##############


###1st peak (to appear)

index_min = 169          #index that starts the peak (by looking the graph)
index_max = 179          #index that ends the peak (by looking the graph)

counts_peak = np.array( dat_8k['Hist']['Counts'][index_min-1:index_max-1] )
ch_peak = np.array( dat_8k['Hist']['E[ch]'][index_min-1:index_max-1] )


fit_8k_peak_1 = Fits.Gaussian_fit(ch_peak,counts_peak)                #fit


###2nd peak (to appear)

index_min = 180          #index that starts the peak (by looking the graph)
index_max = 189          #index that ends the peak (by looking the graph)

counts_peak = np.array( dat_8k['Hist']['Counts'][index_min-1:index_max-1] )
ch_peak = np.array( dat_8k['Hist']['E[ch]'][index_min-1:index_max-1] )


fit_8k_peak_2 = Fits.Gaussian_fit(ch_peak,counts_peak)                #fit


###3rd peak (to appear)

index_min = 191          #index that starts the peak (by looking the graph)
index_max = 199          #index that ends the peak (by looking the graph)

counts_peak = np.array( dat_8k['Hist']['Counts'][index_min-1:index_max-1] )
ch_peak = np.array( dat_8k['Hist']['E[ch]'][index_min-1:index_max-1] )


fit_8k_peak_3 = Fits.Gaussian_fit(ch_peak,counts_peak)                #fit


#Plot
plt.figure(figsize=(8,5))  #width, heigh 6.4*4.8 inches by default
plt.bar(['Pu239','Am241', 'Cm244'], 
        np.array( [ fit_8k_peak_1['R[%]'], 
                   fit_8k_peak_1['R[%]'], 
                   fit_8k_peak_1['R[%]'] ] ), 
        yerr = np.array( [ fit_8k_peak_1['\Delta(R[%])'], 
                          fit_8k_peak_1['\Delta(R[%])'], 
                          fit_8k_peak_1['\Delta(R[%])'] ] ), 
        edgecolor="black")
plt.title("Resolution of the alpha peaks for gate = 8kns", fontsize=22, wrap=True)           #title
plt.ylabel("R (%)", fontsize=14)              #ylabel
plt.tick_params(axis='both', labelsize=14)              #size of axis
plt.grid(True) 
plt.savefig('Resolution_gate_8kns.png', format='png')




##################3

#For the remaining values, 25 and 40k, those calcs do not apply since the 3 peaks are no longer distinguisable