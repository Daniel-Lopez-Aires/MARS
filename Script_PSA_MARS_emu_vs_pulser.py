#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 10:00:44 2021

@author: dla

This script contains the calcs of the B.1.5. experiment using the signals from the oscilloscope. The signals are from the pre (divided with the disivor, to see the histogram) and directly from the detector.
The wrong way to compute the light yield ratio is also included, commented.

The sample is elevated, 1 cylinder only (plus the little ring placed just above the scintillator (see pictures in logbook))
"""

#######General packages useful##

import matplotlib.pyplot as plt  #for simplicity, to not write matplotlib.pyplot
        #everytime we want to plot something

#from scipy.stats import norm               ##norm.fit() fit to gaussian
import numpy as np
    #np contain linspaces as np.linspace(a,b,N)
import pandas as pd
        
#from plotly.graph_objs import Bar, Layout
#from plotly import offline
import sys                   #to import functions from other folders!!
sys.path.insert(0, '/home/dla/Python/Functions_homemade')   #path where I have the functions

import Peak_analysis_oscilloscope
######3

#plt.close("all")




#%% #########################################################
#########################1), Data loading #####################
#############################################################
#Load from the .csv of the oscilloscope!


#Variables that will store the results
voltage_stored = np.array(np.array([]))
time_stored = np.array([])


########Ortec Pulser, raw########
Pul_r = pd.read_csv('TEK0000.CSV', sep=',', header = None, 
                   names = ['a', 'b', 'c','t[s]' , 'V[V]' , 'e'])
                #, as sepparator, no header. I rename the columns, with letters
                #the unimportant ones
                
iPulser_raw = 0                                                  #pulser_raw, index

#Storing of the values
voltage_stored = np.append(voltage_stored,Pul_r['V[V]']) #one to last line
time_stored = np.append(time_stored,Pul_r['t[s]'])  #two to last line

############EMU RAW#################
Emu_r = pd.read_csv('TEK0002.CSV', sep=',', header = None, 
                   names = ['a', 'b', 'c','t[s]' , 'V[V]' , 'e']) 
iEmu_raw = 1                                                  #emu_raw, index

#Storing of the values
voltage_stored = np.column_stack((voltage_stored,Emu_r['V[V]'])) #one to last line
time_stored = np.column_stack((time_stored, Emu_r['t[s]']) )  #two to last line 
  
#########Ortec pulser, pre##########
Pul_p = pd.read_csv('TEK0001.CSV', sep=',', header = None, 
                   names = ['a', 'b', 'c','t[s]' , 'V[V]' , 'e'])
iPulser_pre = 2                     #pulser_pre, index

#Storing of the values
voltage_stored = np.column_stack((voltage_stored,Pul_p['V[V]'])) #one to last line
time_stored = np.column_stack((time_stored,Pul_p['t[s]']))  #two to last line 
        #have to write column stack so it creates columns!

#############EMU PRE####################
Emu_p = pd.read_csv('TEK0003.CSV', sep=',', header = None, 
                   names = ['a', 'b', 'c','t[s]' , 'V[V]' , 'e'])  
iEmu_pre = 3                                                  #emu_pre, index

#Storing of the values
voltage_stored = np.column_stack((voltage_stored,Emu_p['V[V]'])) #one to last line
time_stored = np.column_stack((time_stored,Emu_p['t[s]']))  #two to last line    
   

        
# # #Plot

# plt.figure(figsize=(10,6))  #width, heigh 6.4*4.8 inches by default
# plt.plot(1e6 *time_stored,1e3 * voltage_stored, 'bo-')    #-1 chooses last element, which is the
#         #one that have been added to the lsit the lastest ;)    
#         #widht so that each bar touches each other!
# plt.title("Raw Waveform of Cs137 with pulser", fontsize=22)           #title
# plt.xlabel("time (us)", fontsize=14)                        #xlabel
# plt.ylabel("voltage (mV)", fontsize=14)              #ylabel
# # Set size of tick labels.
# plt.tick_params(axis='both', labelsize=14)              #size of axis
# plt.grid(True) 
# #plt.xlim(0,max(ADC_channel))                       #limits of x axis     
# plt.savefig('Raw_signal_pulser.png', format='png')



###############Plot of all the waves##################################

#####Raw####

#since not all the waveforms have the same baseline, for a good plot, they should all
#have the same baseline, so the idea will be to move (substract) ti akk the waveforms to 
#decrease them to the waveform with the minimum baseline

baseliniEmu_rawaw = [
            min(voltage_stored[:,iPulser_raw]), min(voltage_stored[:,iEmu_raw]) ]
                #Pulser, EMU

baseliniEmu_rawaw_min = min(baseliniEmu_rawaw)                           #min value of the baseline (raw)
baseliniEmu_rawaw_min_index = np.where(baseliniEmu_rawaw == baseliniEmu_rawaw_min)        #index of the min value

#now that we have the index (note the right order of the basline indexes, 0, 1, 2) we can substract.

#plot
plt.figure(figsize=(10,6))  #width, heigh 6.4*4.8 inches by default
plt.title("Raw signals", fontsize=22)           #title
plt.plot(1e6 *time_stored[:,iPulser_raw], voltage_stored[:,iPulser_raw] - (baseliniEmu_rawaw[iPulser_raw]- baseliniEmu_rawaw_min), 'k-')
plt.plot(1e6 *time_stored[:,iEmu_raw], voltage_stored[:,iEmu_raw] - (baseliniEmu_rawaw[iEmu_raw]- baseliniEmu_rawaw_min), 'b-')
plt.xlabel("time (us)", fontsize=14)                        #xlabel
plt.ylabel("voltage (V)", fontsize=14)              #ylabel
plt.legend(['Pulser', 'Emulator'], fontsize=16) 
# Set size of tick labels.
plt.tick_params(axis='both', labelsize=14)              #size of axis
plt.grid(True) 
#plt.xlim(0,max(ADC_channel))                       #limits of x axis     
plt.savefig('Raw_signals.png', format='png')



######Pre####

plt.figure(figsize=(10,6))  #width, heigh 6.4*4.8 inches by default
plt.title("Signals", fontsize=22)           #title

plt.plot(1e6 *time_stored[:,iPulser_pre], voltage_stored[:,iPulser_pre], 'k-')
plt.plot(1e6 *time_stored[:,iEmu_pre], voltage_stored[:,iEmu_pre], 'b-')

plt.xlabel("time (us)", fontsize=14)                        #xlabel
plt.ylabel("voltage (V)", fontsize=14)              #ylabel
plt.legend(['Pulser', 'Emulator'], fontsize=16) 
# Set size of tick labels.
plt.tick_params(axis='both', labelsize=14)              #size of axis
plt.grid(True) 
#plt.xlim(0,max(ADC_channel))                       #limits of x axis     
plt.savefig('Signals.png', format='png')



###############Plot of both waves for each crystal######################

####Pulser####
plt.figure(figsize=(10,6))  #width, heigh 6.4*4.8 inches by default
plt.title("Pulser waveforms", fontsize=22)           #title

plt.plot(1e6 *time_stored[:,iPulser_raw], voltage_stored[:,iPulser_raw] )
plt.plot(1e6 *time_stored[:,iPulser_pre], voltage_stored[:,iPulser_pre] +.4)
plt.xlabel("time (us)", fontsize=14)                        #xlabel
plt.ylabel("voltage (V)", fontsize=14)              #ylabel
plt.legend(['from detector', 'from A1422'], fontsize=16, loc = 'lower right') 
# Set size of tick labels.
plt.tick_params(axis='both', labelsize=16)              #size of axis
plt.grid(True) 
#plt.xlim(0,max(ADC_channel))                       #limits of x axis    
plt.savefig('Waves_pulser.png', format='png')


#EMU####
plt.figure(figsize=(10,6))  #width, heigh 6.4*4.8 inches by default
plt.title("Emulator waveforms", fontsize=22)           #title

plt.plot(1e6 *time_stored[:,iEmu_raw], voltage_stored[:,iEmu_raw]  )
plt.plot(1e6 *time_stored[:,iEmu_pre], voltage_stored[:,iEmu_pre] )
plt.xlabel("time (us)", fontsize=14)                        #xlabel
plt.ylabel("voltage (V)", fontsize=14)              #ylabel
plt.legend(['from detector', 'from A1422'], fontsize=16, loc = 'lower right') 
# Set size of tick labels.
plt.tick_params(axis='both', labelsize=14)              #size of axis
plt.grid(True) 
#plt.xlim(0,max(ADC_channel))                       #limits of x axis   
plt.savefig('Waves_emu.png', format='png')




 #%% ########################################################################
 ###################3) Pulse analysis ############################################
###############################################################################

#Since the signals from the Pre are way better, will use them. Remember they are
#the last stored (6 total stored), so from 3 to 5 (0 the first). pulser, emu, BGO the order
#Note we are not sure about the light yield concept, but computing the amplitude will be good.

#I have also implemented the integral of the curve. I will use trapz,which is the easiest way

     # numpy.trapz(y, x, dx=1.0, axis=-1)[source]
     #    Integrate along the given axis using the composite trapezoidal rule.
     #    Integrate y (x) along given axis.

voltagiEmu_preeak_st = np.array(np.array([]))                #storage of the total voltage of the peak,
                    #removing the baseline!!!!!!

baseline_st = np.array([])               #storage of the baseline
peak_st = np.array([])                  #storage of the peak value, for the max amplitude
n_elements_peak_st = np.array([])                        #this will store the number of voltages I sum,
                                            #for each peak, for the error calc
delta_single_V_measurement = np.array([])         #storage of the error of the voltage 
                                                #measurements, for the error calc
amplitude_st = np.array( [] )               #storage of the amplitudes                                           
delta_amplitude_st = np.array([])           #storage of \Delta(amplitude)
integral_st = np.array([])                      #peak integral
delta_integral_st = np.array([])                #error of the peak integral (overstimation)

t_rise_st = np.array([])
t_decay_st = np.array([])
delta_t_rise_st = np.array([])
delta_t_decay_st = np.array([])



####################################pulser RAW  ##########################3
pulser_raw = Peak_analysis_oscilloscope.Peak_analysis_oscillo(Pul_r['V[V]'],Pul_r['t[s]'],
                                                'raw', 490, 1100, 
                                                 1/5 *50e-3, 1/5 * 1e-3 )

#Storing

baseline_st = np.append(baseline_st, pulser_raw['baseline[V]'] )  
peak_st = np.append(peak_st, pulser_raw['|peak|[V]'] )
n_elements_peak_st = np.append(n_elements_peak_st, pulser_raw['N_peak'] )
delta_single_V_measurement = np.append(delta_single_V_measurement, pulser_raw['\Delta(V[V])'] )
integral_st = np.append(integral_st, pulser_raw['integral[V*s]'] )
delta_integral_st = np.append(delta_integral_st, pulser_raw['\Delta(integral[V*s])'] )
amplitude_st = np.append(amplitude_st, pulser_raw['amplitude[V]'])
delta_amplitude_st = np.append(delta_amplitude_st, pulser_raw['\Delta(amplitude[V])'] )

t_rise_st = np.append(t_rise_st,pulser_raw['t_rise[s]'])
t_decay_st = np.append(t_decay_st,pulser_raw['t_decay[s]'])
delta_t_rise_st = np.append(delta_t_rise_st,pulser_raw['\Delta(t_rise[s])'])
delta_t_decay_st = np.append(delta_t_decay_st,pulser_raw['\Delta(t_decay[s])'])



####################################emu RAW ##########################3

emu_raw = Peak_analysis_oscilloscope.Peak_analysis_oscillo(Emu_r['V[V]'],Emu_r['t[s]'],
                                                'raw', 480, 800, 
                                                 1/5 *2, 1/5 * 500e-9 )

#Storing

baseline_st = np.append(baseline_st, emu_raw['baseline[V]'] )  
peak_st = np.append(peak_st, emu_raw['|peak|[V]'] )
n_elements_peak_st = np.append(n_elements_peak_st, emu_raw['N_peak'] )
delta_single_V_measurement = np.append(delta_single_V_measurement, emu_raw['\Delta(V[V])'] )
integral_st = np.append(integral_st, emu_raw['integral[V*s]'] )
delta_integral_st = np.append(delta_integral_st, emu_raw['\Delta(integral[V*s])'] )
amplitude_st = np.append(amplitude_st, emu_raw['amplitude[V]'])
delta_amplitude_st = np.append(delta_amplitude_st, emu_raw['\Delta(amplitude[V])'] )

t_rise_st = np.append(t_rise_st,emu_raw['t_rise[s]'])
t_decay_st = np.append(t_decay_st,emu_raw['t_decay[s]'])
delta_t_rise_st = np.append(delta_t_rise_st,emu_raw['\Delta(t_rise[s])'])
delta_t_decay_st = np.append(delta_t_decay_st,emu_raw['\Delta(t_decay[s])'])


################################PULSER PRE ###############################

pulser_pre = Peak_analysis_oscilloscope.Peak_analysis_oscillo(Pul_p['V[V]'],Pul_p['t[s]'],
                                                'pre', 495, 800, 
                                                 1/5 *50e-3, 1/5 * 500e-6 )


#Storing

baseline_st = np.append(baseline_st, pulser_pre['baseline[V]'] )  
peak_st = np.append(peak_st, pulser_pre['|peak|[V]'] )
n_elements_peak_st = np.append(n_elements_peak_st, pulser_pre['N_peak'] )
delta_single_V_measurement = np.append(delta_single_V_measurement, pulser_pre['\Delta(V[V])'] )
integral_st = np.append(integral_st, pulser_pre['integral[V*s]'] )
delta_integral_st = np.append(delta_integral_st, pulser_pre['\Delta(integral[V*s])'] )
amplitude_st = np.append(amplitude_st, pulser_pre['amplitude[V]'])
delta_amplitude_st = np.append(delta_amplitude_st, pulser_pre['\Delta(amplitude[V])'] )

t_rise_st = np.append(t_rise_st,pulser_pre['t_rise[s]'])
t_decay_st = np.append(t_decay_st,pulser_pre['t_decay[s]'])
delta_t_rise_st = np.append(delta_t_rise_st,pulser_pre['\Delta(t_rise[s])'])
delta_t_decay_st = np.append(delta_t_decay_st,pulser_pre['\Delta(t_decay[s])'])


#############################EMULATOR PRE   ###############################3

emu_pre = Peak_analysis_oscilloscope.Peak_analysis_oscillo(Emu_p['V[V]'],Emu_p['t[s]'],
                                                'pre', 490, 800, 
                                                 1/5 *5, 1/5 * 500e-9 )

#Storing

baseline_st = np.append(baseline_st, emu_pre['baseline[V]'] )  
peak_st = np.append(peak_st, emu_pre['|peak|[V]'] )
n_elements_peak_st = np.append(n_elements_peak_st, emu_pre['N_peak'] )
delta_single_V_measurement = np.append(delta_single_V_measurement, emu_pre['\Delta(V[V])'] )
integral_st = np.append(integral_st, emu_pre['integral[V*s]'] )
delta_integral_st = np.append(delta_integral_st, emu_pre['\Delta(integral[V*s])'] )
amplitude_st = np.append(amplitude_st, emu_pre['amplitude[V]'])
delta_amplitude_st = np.append(delta_amplitude_st, emu_pre['\Delta(amplitude[V])'] )

t_rise_st = np.append(t_rise_st,emu_pre['t_rise[s]'])
t_decay_st = np.append(t_decay_st,emu_pre['t_decay[s]'])
delta_t_rise_st = np.append(delta_t_rise_st,emu_pre['\Delta(t_rise[s])'])
delta_t_decay_st = np.append(delta_t_decay_st,emu_pre['\Delta(t_decay[s])'])





##############################################################################
##################PLOT OF TIMES####################################################
###################################################################################


#have to reorder the data so that they are arranged from lowest to highers: pulser-->BGO-->emu
#the current order (storing order) is pulser-->emu-->BGO. I must not change the storing order since
#it would affect everyhing, so I will alter the plot order!!

#RAW

plt.figure(figsize=(13,6))  #width, heigh 6.4*4.8 inches by default
plt.subplot(1, 2, 1)
plt.suptitle("Rise and decay time of the raw signals coming from the crystals", fontsize=22, wrap=True)           #title

plt.bar(['pulser', 'emu'], t_rise_st[iPulser_raw:iEmu_raw+1] *1e6, 
        yerr = delta_t_rise_st[iPulser_raw:iEmu_raw+1] *1e6, edgecolor="black")
#plt.xlabel("ADC channels", fontsize=10)                        #xlabel
plt.ylabel("Rise time (us)", fontsize=14)              #ylabel
# Set size of tick labels.
plt.tick_params(axis='both', labelsize=14)              #size of axis
plt.grid(True) 

plt.subplot(1, 2, 2)
plt.bar(['pulser', 'emu'], t_decay_st[iPulser_raw:iEmu_raw+1] *1e6, 
        yerr = delta_t_decay_st[iPulser_raw:iEmu_raw+1] *1e6, edgecolor="black")
#plt.xlabel("ADC channels", fontsize=10)                        #xlabel
plt.ylabel("Decay time (us)", fontsize=14)              #ylabel
# Set size of tick labels.
plt.tick_params(axis='both', labelsize=14)              #size of axis
plt.grid(True) 
plt.savefig('Rise_decay_timiEmu_rawaw.png', format='png')


#Print:
print('Decay time of raw pulser: (' + str(t_decay_st[iPulser_raw]*1e6) + ' +/- ' + str(delta_t_decay_st[iPulser_raw]*1e6) + ')us')
print('Decay time of raw emu: (' + str(t_decay_st[iEmu_raw]*1e6) + ' +/- ' + str(delta_t_decay_st[iEmu_raw]*1e6) + ')us' +"\n")

#################################################

#Pre

plt.figure(figsize=(13,6))  #width, heigh 6.4*4.8 inches by default
plt.subplot(1, 2, 1)
plt.suptitle("Rise and decay time of the signals coming from the crystals", fontsize=22, wrap=True)    #title
plt.bar(['pulser', 'emu'], t_rise_st[iPulser_pre:iEmu_pre+1] *1e6, 
        yerr = delta_t_rise_st[iPulser_pre:iEmu_pre+1] *1e6, edgecolor="black")
#plt.xlabel("ADC channels", fontsize=10)                        #xlabel
plt.ylabel("Rise time (us)", fontsize=14)              #ylabel
# Set size of tick labels.
plt.tick_params(axis='both', labelsize=14)              #size of axis
plt.grid(True) 

plt.subplot(1, 2, 2)
plt.bar(['pulser', 'emu'], t_decay_st[iPulser_pre:iEmu_pre+1] *1e6, 
        yerr = delta_t_decay_st[iPulser_pre:iEmu_pre+1] *1e6, edgecolor="black")
#plt.xlabel("ADC channels", fontsize=10)                        #xlabel
plt.ylabel("Decay time (us)", fontsize=14)              #ylabel
# Set size of tick labels.
plt.tick_params(axis='both', labelsize=14)              #size of axis
plt.grid(True) 
plt.savefig('Rise_decay_timiEmu_prere.png', format='png')


#Print:
print('Decay time of pre pulser: (' + str(t_decay_st[iPulser_pre]*1e6) + ' +/- ' + str(delta_t_decay_st[iPulser_pre]*1e6) + ')us')
print('Decay time of pre emu: (' + str(t_decay_st[iEmu_pre]*1e6) + ' +/- ' + str(delta_t_decay_st[iEmu_pre]*1e6) + ')us' +"\n")
##############################################



####################################################################
##################DEcay time ratios#############################
##################################################################
#ratio between the decay time from the pre and from the detector

ratio_t_decay_priEmu_rawaw = t_decay_st[iPulser_pre:iEmu_pre+1]/t_decay_st[iPulser_raw:iEmu_raw+1] #correct indexes, yes

delta_ratio_t_decay_priEmu_rawaw = ratio_t_decay_priEmu_rawaw * np.sqrt( 
    np.array( [ (delta_t_decay_st[iPulser_pre]/t_decay_st[iPulser_pre])**2 + (delta_t_decay_st[iPulser_raw]/t_decay_st[iPulser_raw])**2, 
    (delta_t_decay_st[iEmu_pre]/t_decay_st[iEmu_pre])**2 + (delta_t_decay_st[iEmu_raw]/t_decay_st[iEmu_raw])**2, 
    ] ) )                               #delta of the amplitudes, following the order defined in the variable

#Print:
print('Decay time ratio pre/raw pulser: (' + str(ratio_t_decay_priEmu_rawaw[iPulser_raw] ) + ' +/- ' + str(delta_ratio_t_decay_priEmu_rawaw[iPulser_raw]) )
print('Decay time ratio pre/raw emu: (' + str(ratio_t_decay_priEmu_rawaw[iEmu_raw] ) + ' +/- ' + str(delta_ratio_t_decay_priEmu_rawaw[iEmu_raw]) + "\n")



############################################################
##################Amplitudes ratios#############################
##################################################################
#ratio between the amplitudes from the pre and from the detector

print('Amplitude (Vmax) pulser (V) = ' + str(amplitude_st[iPulser_pre]) + ' +- ' + str(delta_amplitude_st[iPulser_pre]))
print('Amplitude (Vmax) emu (V) =' + str(amplitude_st[iEmu_pre]) + ' +- ' + str(delta_amplitude_st[iEmu_pre]) +'\n')

print('Amplitude (Vmax) raw pulser (V) = ' + str(amplitude_st[iPulser_raw]) + ' +- ' + str(delta_amplitude_st[iPulser_raw]))
print('Amplitude (Vmax) raw emu (V) =' + str(amplitude_st[iEmu_raw]) + ' +- ' + str(delta_amplitude_st[iEmu_raw]) +'\n')


ratio_ampl_priEmu_rawaw = amplitude_st[iEmu_raw+1:iEmu_pre+1]/amplitude_st[iPulser_raw:iEmu_raw+1]  #yup, correct indexes

delta_ratio_ampl_priEmu_rawaw = ratio_ampl_priEmu_rawaw * np.sqrt( 
    np.array( [ (delta_amplitude_st[iPulser_raw]/amplitude_st[iPulser_raw])**2 + (delta_amplitude_st[iPulser_pre]/amplitude_st[iPulser_pre])**2, 
    (delta_amplitude_st[iEmu_raw]/amplitude_st[iEmu_raw])**2 + (delta_amplitude_st[iPulser_pre]/amplitude_st[iPulser_pre])**2
    ] ) )          #delta of the amplitudes, following the order defined in the variable

#print('Ratio amplitude pre/raw: (pulser, emu, BGO) ' + str(ratio_ampl_priEmu_rawaw) + '\n')

print('Amplitude ratio pulser pre/raw = ' + str(ratio_ampl_priEmu_rawaw[iPulser_raw]) + ' +- ' + str(delta_ratio_ampl_priEmu_rawaw[iPulser_raw]) )
print('Amplitude ratio emu pre/raw = ' + str(ratio_ampl_priEmu_rawaw[iEmu_raw]) + ' +- ' + str(delta_ratio_ampl_priEmu_rawaw[iEmu_raw]) + '\n')



print('Integral pulser (baseline removed) (pre) (V*s) = ' + str(integral_st[iPulser_pre]) + ' +- ' + str(delta_integral_st[iPulser_pre]) )
print('Integral emu (baseline removed) (pre) (V*s) =' + str(integral_st[iEmu_pre]) + ' +- ' + str(delta_integral_st[iEmu_pre]) + '\n')