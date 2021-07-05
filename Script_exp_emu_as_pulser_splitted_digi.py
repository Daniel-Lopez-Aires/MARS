#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 10:00:44 2021

@author: dla

This script contains the calcs of the exp of using the Emu as pulser, splits
its signal, and to the digi.
"""

#######General packages useful##

import matplotlib.pyplot as plt  #for simplicity, to not write matplotlib.pyplot
        #everytime we want to plot something

#from scipy.stats import norm               ##norm.fit() fit to gaussian
import numpy as np
    #np contain linspaces as np.linspace(a,b,N)
import pandas as pd
        
from plotly.graph_objs import Bar, Layout
from plotly import offline
from scipy import stats as stats     #to find the most commom element in a list

import sys                   #to import functions from other folders!!
sys.path.insert(0, '/home/dla/Python/Functions_homemade')   #path where I have the functions

import Read_root_COMPASS
######3

#plt.close("all")




#%% #########################################################
#########################1), Data loading #####################
#############################################################


########Filtered data in .csv########
#The format of the header is very weird, so I have creaed a function to process
#this data.

data_ch14 = Read_root_COMPASS.ReadRootSingleCOMPASS('DataF_CH14@V1725S_646_run.root')
data_ch15 = Read_root_COMPASS.ReadRootSingleCOMPASS('DataF_CH15@V1725S_646_run.root')

#%% #########################################################
#########################2), Wave plotting #####################
#############################################################


plt.figure(figsize=(10,8))  #width, heigh 6.4*4.8 inches by default
#plt.plot(np.linspace(1, len(data_csv['Sample']), len(data_csv['Sample']) ),data_csv['Sample'], 'b.-')        
plt.plot(data_ch14['Time_wave[ns]'] ,data_ch14['Voltage_wave[ch]'], 'b.-')        

plt.title("Waveform ch14", fontsize=22)           #title
plt.xlabel("time (ns)", fontsize=14)                        #xlabel
plt.ylabel("ADC channels", fontsize=14)              #ylabel
# Set size of tick labels.
plt.tick_params(axis='both', labelsize=14)              #size of axis
plt.grid(True) 
#plt.xlim(min(ADC_channel_8ampl),3000)                       #limits of x axis
plt.savefig('wave_ch14.png', format='png')



plt.figure(figsize=(10,8))  #width, heigh 6.4*4.8 inches by default
#plt.plot(np.linspace(1, len(data_csv['Sample']), len(data_csv['Sample']) ),data_csv['Sample'], 'b.-')        
plt.plot(data_ch15['Time_wave[ns]'] ,data_ch15['Voltage_wave[ch]'], 'b.-')        

plt.title("Waveform ch15", fontsize=22)           #title
plt.xlabel("time (ns)", fontsize=14)                        #xlabel
plt.ylabel("ADC channels", fontsize=14)              #ylabel
# Set size of tick labels.
plt.tick_params(axis='both', labelsize=14)              #size of axis
plt.grid(True) 
#plt.xlim(min(ADC_channel_8ampl),3000)                       #limits of x axis
plt.savefig('wave_ch15.png', format='png')


#%% #########################################################
#########################3), single E plotting #####################
#############################################################

#To plot the Energy (and similar variables) as a bar plot (the hist is weird)
#, you could do:
    
u_14, inv_14 = np.unique(data_ch14['E[ch]'], return_inverse=True)
counts_14 = np.bincount(inv_14)

plt.figure(figsize=(10,8))  #width, heigh 6.4*4.8 inches by default
plt.bar(u_14, counts_14, width = u_14[1]-u_14[0], edgecolor="black")   
plt.title("Spectra in channels ch14", fontsize=22)           #title
plt.xlabel("ADC Channels", fontsize=14)                        #xlabel
plt.ylabel("Counts", fontsize=14)              #ylabel
# Set size of tick labels.
plt.tick_params(axis='both', labelsize=14)              #size of axis
plt.grid(True) 
#plt.xlim(min(ADC_channel_8ampl),3000)                       #limits of x axis
plt.savefig('E_ch14.png', format='png')

u_15, inv_15 = np.unique(data_ch15['E[ch]'], return_inverse=True)
counts_15 = np.bincount(inv_15)

plt.figure(figsize=(10,8))  #width, heigh 6.4*4.8 inches by default
plt.bar(u_15, counts_15, width = u_15[1]-u_15[0], edgecolor="black")   
plt.title("Spectra in channels ch15", fontsize=22)           #title
plt.xlabel("ADC Channels", fontsize=14)                        #xlabel
plt.ylabel("Counts", fontsize=14)              #ylabel
# Set size of tick labels.
plt.tick_params(axis='both', labelsize=14)              #size of axis
plt.grid(True) 
#plt.xlim(min(ADC_channel_8ampl),3000)                       #limits of x axis
plt.savefig('E_ch14.png', format='png')



#%% #########################################################
#########################4) 2D E plotting #####################
#############################################################


###########Coincidence plot################

#Coindence: for each entry, which have to be the same for all the channels,
#events are measured, store them. Finally, plot those variable, the energies 
#from the detectors for those entries. This can be done with a loop:

E_14_c = np.array( [] )         #store of the coincidence Ener of ch14
E_15_c = np.array( [] )         #store of the coincidence Ener of ch15


for i in range(0, data_ch14['n_events']):         #loop through all the events
    if data_ch14['E[ch]'][i]>= 0 and data_ch15['E[ch]'][i]>= 0:
             #if both channels have a non zero energy value for the event, 
             #store it
    
            E_14_c = np.append(E_14_c,data_ch14['E[ch]'][i])
            E_15_c = np.append(E_15_c,data_ch15['E[ch]'][i])
            #counts_coinc = np.append(counts_coinc,counts_14[i]+ counts_15[j])
            
            
plt.figure(figsize=(10,8))  #width, heigh 6.4*4.8 inches by default
plt.plot(E_14_c,E_15_c,'b.')
plt.xlim(0,4096)
plt.ylim(0,4096)
plt.title("2D Energy spectra (ch14 and 15 coincidence)", fontsize=22)           #title
plt.xlabel("E(ch) [ch14]", fontsize=14)                        #xlabel
plt.ylabel("E(ch) [ch 15]", fontsize=14)              #ylabel
plt.tick_params(axis='both', labelsize=14)              #size of axis
plt.grid(True) 
#plt.xlim(min(ADC_channel_8ampl),3000)                       #limits of x axis
plt.savefig('E_2D_ch14_15.png', format='png')




















#%% #########################################################
######################### GARBAJE #####################
#############################################################

# ###########Plot without coincidence############

# #For the plot I need to create a meshgrid for the energy values of both spectras:
# u_14_m, u_15_m = np.meshgrid(u_14,u_15)         #meshgrid for the 3D plot axes

# #and I also need to mesh the counts of each spectra:
# counts_14_m,counts_15_m = np.meshgrid(counts_14,counts_15)  #meshgrid for the z axis

# #The z axis should be the frequency of spectra1+spectra2, so:
    
# plt.figure(figsize=(10,8))  #width, heigh 6.4*4.8 inches by default
# plt.contourf(u_14_m, u_15_m, counts_14_m+ counts_15_m, cmap="terrain")
# cbar=plt.colorbar()
# cbar.set_label("Counts")
# plt.title("2D Energy spectra (ch14 and 15)", fontsize=22)           #title
# plt.xlabel("ADC channels [ch14]", fontsize=14)                        #xlabel
# plt.ylabel("ADC channels [ch 15]", fontsize=14)              #ylabel
# # Set size of tick labels.
# plt.tick_params(axis='both', labelsize=14)              #size of axis
# plt.grid(True) 
# #plt.xlim(min(ADC_channel_8ampl),3000)                       #limits of x axis
# plt.savefig('E_2D_ch14_15.png', format='png')

