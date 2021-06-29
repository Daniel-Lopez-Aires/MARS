#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 10:00:44 2021

@author: dla

This script contains the calcs of the PSA using the emulator and the Digi from MARS. 
Data saved from COMPASS in .csv
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

import Read_csv_COMPASS
import Peak_analysis_oscilloscope
import Read_root_COMPASS
######3

#plt.close("all")




#%% #########################################################
#########################1), Data loading #####################
#############################################################


########Filtered data in .csv########
#The format of the header is very weird, so I have creaed a function to process
#this data.

data_csv = Read_csv_COMPASS.ReadCsvCOMPASS('DataF_CH15@V1725S_646_run.csv')
#iWav = 0                        #index of the waveform

######Signal from the emulator######

data_emu = pd.read_csv('TEK0000.CSV', sep=',', 
                       names = ['a', 'b', 'Time[s]', 'Voltage[V]', 'c'])
            #I only name the important rows to us xD
#iPul = 1                                                 #pulser, index

#%% #########################################################
#########################2), Data plotting #####################
#############################################################


###################### .csv ##################

#the x axis, time, would be like: t0, t0+4,t0+8, t0+12 etc, for an initial time t0
#(the sampling time is 4ns). Since I do 1000samples, I can get the time:


plt.figure(figsize=(10,8))  #width, heigh 6.4*4.8 inches by default
#plt.plot(np.linspace(1, len(data_csv['Sample']), len(data_csv['Sample']) ),data_csv['Sample'], 'b.-')        
plt.plot(data_csv['Time_wave[ns]'] ,data_csv['Voltage_wave[ch]'], 'b.-')        

plt.title("Waveform .csv", fontsize=22)           #title
plt.xlabel("time (ns)", fontsize=14)                        #xlabel
plt.ylabel("ADC channels", fontsize=14)              #ylabel
# Set size of tick labels.
plt.tick_params(axis='both', labelsize=14)              #size of axis
plt.grid(True) 
#plt.xlim(min(ADC_channel_8ampl),3000)                       #limits of x axis
plt.savefig('wave_positive_csv.png', format='png')

###### emu #########

plt.figure(figsize=(10,8))  #width, heigh 6.4*4.8 inches by default
#plt.plot(np.linspace(1, len(data_csv['Sample']), len(data_csv['Sample']) ),data_csv['Sample'], 'b.-')        
plt.plot(data_emu['Time[s]'] ,data_emu['Voltage[V]'], 'b.-')        

plt.title("Emulator Waveform", fontsize=22)           #title
plt.xlabel("time (ns)", fontsize=14)                        #xlabel
plt.ylabel("Voltage (V)", fontsize=14)              #ylabel
# Set size of tick labels.
plt.tick_params(axis='both', labelsize=14)              #size of axis
plt.grid(True) 
#plt.xlim(min(ADC_channel_8ampl),3000)                       #limits of x axis
plt.savefig('wave_emu.png', format='png')


#%% #########################################################
#########################3), PSA analysis #####################
#############################################################


####################################waveform ##########################3

peak_csv = Peak_analysis_oscilloscope.Peak_analysis_oscillo(data_csv['Voltage_wave[ch]'], 
                            data_csv['Time_wave[ns]']*1e-9,'raw', 120, 290, 
                            1/5 *0, 1/5 * 0 )

#Since this is ch and not V, will redefine the ylabel in the plot:
plt.ylabel("ADC channel", fontsize=14)              #ylabel 

####################################emu ##########################3

peak_emu = Peak_analysis_oscilloscope.Peak_analysis_oscillo(data_emu['Voltage[V]'], 
                            data_emu['Time[s]'],'raw', 250, 780, 
                            1/5 *.5, 1/5 * 500e-9 )


##########################
#The units are different, V for the emu, and Channels for the waveform from COMPASS,
#so only the time can be compared


##############################################################################
##################PLOT OF TIMES####################################################
###################################################################################

plt.figure(figsize=(13,6))  #width, heigh 6.4*4.8 inches by default
plt.subplot(1, 2, 1)
plt.suptitle("Rise and decay time of the signals", fontsize=22, wrap=True)           #title
plt.bar(['emu', 'digi'], np.array([peak_emu['t_rise[s]'],peak_csv['t_rise[s]']])*1e6, 
        yerr = np.array([peak_emu['\Delta(t_rise[s])'],peak_csv['\Delta(t_rise[s])']])*1e6, 
        edgecolor="black")
#plt.xlabel("ADC channels", fontsize=10)                        #xlabel
plt.ylabel("Rise time (us)", fontsize=14)              #ylabel
# Set size of tick labels.
plt.tick_params(axis='both', labelsize=14)              #size of axis
plt.grid(True) 

plt.subplot(1, 2, 2)
plt.bar(['emu', 'digi'], np.array([peak_emu['t_decay[s]'],peak_csv['t_decay[s]']])*1e6, 
        yerr = np.array([peak_emu['\Delta(t_decay[s])'],peak_csv['\Delta(t_decay[s])']])*1e6, 
        edgecolor="black")
#plt.xlabel("ADC channels", fontsize=10)                        #xlabel
plt.ylabel("Decay time (us)", fontsize=14)              #ylabel
# Set size of tick labels.
plt.tick_params(axis='both', labelsize=14)              #size of axis
plt.grid(True) 
plt.savefig('Rise_decay_time.png', format='png')


#Print:
print('Decay time of emu: (' + str(peak_emu['t_decay[s]']*1e6) + ' +/- ' + str(peak_emu['\Delta(t_decay[s])']*1e6) + ')us')
print('Decay time of digi: (' + str(peak_csv['t_decay[s]']*1e6) + ' +/- ' + str(peak_csv['\Delta(t_decay[s])']*1e6) + ')us' + '\n')

#################################################



####################################################################
##################DEcay time ratios#############################
##################################################################
#ratio between the decay time from the pre and from the detector

ratio_decay = peak_emu['t_decay[s]']/peak_csv['t_decay[s]']
                  #The order is: LYSO/CsI,  LYSO/BGO,  BGO/CsI; first raw and then pre

    
#Error calculation of the ratio
auxiliar2 = (peak_emu['\Delta(t_decay[s])']/peak_emu['t_decay[s]'])**2 + (peak_csv['\Delta(t_decay[s])']/peak_csv['t_decay[s]'])**2       
            #this are (delta_t1/t1)^2 +  (delta_t2/t2)^2

delta_ratios = ratio_decay * np.sqrt(auxiliar2)
                                              #this will be though

print('Decay time ratio emu/digi: (' + str(ratio_decay) + ' +/- ' + str(delta_ratios) + '\n')


#####################################################


############################################################
##################Amplitudes and integrals #############################
##################################################################
#ratio between the amplitudes from the pre and from the detector


#Amplitudes (voltage integration)
print('Amplitude (Vmax) emu (V) = ' + str(peak_emu['amplitude[V]']) + ' +- ' + str(peak_emu['\Delta(amplitude[V])']))
print('Amplitude (Vmax) digi (ch) = ' + str(peak_csv['amplitude[V]']) + ' +- ' + str(peak_csv['\Delta(amplitude[V])']) + '\n')


print('Integral emu (V*s) = ' + str(peak_emu['integral[V*s]']) + ' +- ' + str(peak_emu['\Delta(integral[V*s])']))
print('Integral digi (ch*s) = ' + str(peak_csv['integral[V*s]']) + ' +- ' + str(peak_csv['\Delta(integral[V*s])']) + '\n')

