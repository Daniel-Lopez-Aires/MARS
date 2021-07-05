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



#%% #########################################################
####################1), Data loading and plotting #####################
#############################################################
#The .root file contains folders, the folder have the instograms, whose
#type is TH1D, which are histograms with 1 double (type of variable)
# for each bin


#######################Root file with only hist!!!!#################

data_root = Read_root_COMPASS.ReadRootFullCOMPASS("HcompassF_run_20210702_110336.root")



######3 root file with samples (contain both #############

data_comb = Read_root_COMPASS.ReadRootSingleCOMPASS('SDataF_run.root')


#%% #########################################################
####################2)Coincidence calc #####################
#############################################################
#This version if valid if storing the single .root with all the channels with
#the option "Time sort" enabled.
#
#To do the coincidence, event by event, have to:
    #1) Choose an event
    #2) Check if the next one is from the other Channel or not
    #3) If yes, coincidences are possible. Check if the time interval
    #       betweeen those events are small enough.
    #4) Store the single energy values if the time interval is small enough
    


n_events = len(data_comb['Hist'])   #number of events; rows goes from 0 to 
            #n_events - 1
epsilon = .5            #(t_2-t_1)/t_1, t_2>=t_1 (time sorted) 
            #to do the coincidence!


#Storing
E_14 = np.array( [] )
E_15 = np.array( [] )
E_14_c = np.array( [] )
E_15_c = np.array( [] )

for i in range(0,n_events-2):
        #last values are
    
    if data_comb['Hist']['Ch digitizer'][i] == 14:  #If the event is ch 14
        
        E_14 = np.append(E_14, data_comb['Hist']['E[ch]'][i] )  #storing of the
                    #single energy
        
        #For the coincidence
        
        if data_comb['Hist']['Ch digitizer'][i + 1]==15: #the following row
                    #is the other channel, the 15 ==> coincidence possible

            comp = (data_comb['Hist']['Timestamp[ps]'][i + 1] - 
                    data_comb['Hist']['Timestamp[ps]'][i] ) / data_comb['Hist']['Timestamp[ps]'][i] 
            #variable to find the coincidences, delta_t (t_2-t_1)/t_1 >= 0
            
            if comp <= epsilon : 
                E_14_c = np.append(E_14_c, data_comb['Hist']['E[ch]'][i] )
                E_15_c = np.append(E_15_c, data_comb['Hist']['E[ch]'][i+1] )       
    
    if data_comb['Hist']['Ch digitizer'][i] == 15:  #If the event is ch 15
        
        E_15 = np.append(E_15, data_comb['Hist']['E[ch]'][i] )  #storing of the
                    #single energy    

        
        #For the coincidence
        
        if data_comb['Hist']['Ch digitizer'][i + 1]==14: #the following row
                    #is the other channel, 14 ==> coincidence possible
           
            comp = (data_comb['Hist']['Timestamp[ps]'][i + 1] - 
                    data_comb['Hist']['Timestamp[ps]'][i] ) / data_comb['Hist']['Timestamp[ps]'][i] 
            #variable to find the coincidences, delta_t (t_2-t_1)/t_1 >= 0
            
            if comp <= epsilon : 
                E_14_c = np.append(E_14_c, data_comb['Hist']['E[ch]'][i+1] )
                E_15_c = np.append(E_15_c, data_comb['Hist']['E[ch]'][i] )           





##################### Plot ##############3

####Single plots

#To plot the Energy (and similar variables) as a bar plot (the hist is weird)
#, you could do:
    
u_14, inv_14 = np.unique(E_14, return_inverse=True)
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

u_15, inv_15 = np.unique(E_15, return_inverse=True)
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
plt.savefig('E_ch15.png', format='png')

####Coincidence plot

plt.figure(figsize=(10,8))  #width, heigh 6.4*4.8 inches by default
plt.plot(E_14_c,E_15_c,'b.')
#plt.xlim(0,data_root['n_Channels'])
#plt.ylim(0,data_root['n_Channels'])
plt.title("2D Energy spectra (ch14 and 15 coincidence)", fontsize=22)           #title
plt.xlabel("E(ch) [ch14]", fontsize=14)                        #xlabel
plt.ylabel("E(ch) [ch 15]", fontsize=14)              #ylabel
plt.tick_params(axis='both', labelsize=14)              #size of axis
plt.grid(True) 
#plt.xlim(min(ADC_channel_8ampl),3000)                       #limits of x axis
plt.savefig('E_2D_ch14_15.png', format='png')


















#%% RESIDUOS


# ######## Try: frequency of appearance of the timestamp ##############


# counts = data_comb['Ch_digi|Timestamp']['Timestamp[ps]'].value_counts().to_dict()

# import operator

# max_key = max(counts.items(), key=operator.itemgetter(1))[0] #the key whose
#         #value is the max
# print(counts[max_key])  #the max value =2.
# #there are 2 keys whose appearance is 2.





