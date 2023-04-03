import scipy.signal as signal
from scipy import interpolate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import Slider, Button
from pathlib import Path
from objects import Table, Actuator, Sensor
from setups import Setup2, Setup3, Setup3_2, Setup3_4, Setup6, Setup9
from constants import SAMPLE_RATE, CHANNEL_NAMES, CHIRP_CHANNEL_NAMES
from data_processing import cross_correlation_position as ccp
from csv_to_df import csv_to_df, csv_to_df_thesis
from data_viz_files.visualise_data import compare_signals, plot_vphs, plot_fft, plot_plate_speed_sliders_book, plot_estimated_reflections_with_sliders, compare_signals_v2, plot_compare_signals_v2
from data_processing.preprocessing import get_first_index_above_threshold, interpolate_waveform, crop_data, filter_general, compress_chirp, get_phase_and_vph_of_compressed_signal,cut_out_signal, manual_cut_signal, compress_df_touch, cut_out_pulse_wave, shift_signal
from data_processing.detect_echoes import find_first_peak, get_hilbert_envelope, get_travel_times
from data_processing.find_propagation_speed import find_propagation_speed_with_delay
from data_viz_files.drawing import plot_legend_without_duplicates
import timeit
import data_processing.wave_properties as wp
import data_processing.sensor_testing as st
from data_viz_files.visualise_data import inspect_touch, figure_size_setup, to_dB
import data_processing.wave_properties as wp
import data_processing.sensor_testing as st
from matplotlib import style
import data_viz_files.visualise_data as vd
import os

def results_setup1():
    ## Results for phase velocity test in the beginning of the thesis.
    custom_chirp = csv_to_df(file_folder='Measurements\\div_files', file_name='chirp_custom_fs_150000_tmax_2_100-40000_method_linear', channel_names=CHIRP_CHANNEL_NAMES)
    # df1 = csv_to_df('Measurements\\setup3_0\\', 'chirp_100_40000_2s_v1')
    # phase10, freq10  = wp.phase_plotting(df1, chirp=custom_chirp, use_recorded_chirp=True,start_stops=[(241000,508570),(241000,508570)], BANDWIDTH=[100,40000], save_fig=False, file_name='phase_plot_10cm0_45.svg', file_format='svg',figsize=0.45, n_pi=1)
    # wp.plot_velocities(phase10, freq10, 0.10, savefig=False, filename='phase_velocity_10cm.svg', file_format='svg')
    df_PE = csv_to_df_thesis('plate20mm\\setup1\\chirp', 'chirp_100_40000_2s_v1')
    df_teflon = csv_to_df_thesis('plate10mm\\setup1\\chirp', 'chirp_100_40000_2s_v1')
    #filter signal
    df_PE_filt = filter_general(df_PE, filtertype='bandpass', cutoff_highpass=10000, cutoff_lowpass=15000, order=4)
    #df_teflon_filt = filter_general(df_teflon, filtertype='bandpass', cutoff_highpass=100, cutoff_lowpass=40000, order=4)
    phase_PE, freq_PE  = wp.phase_plotting_chirp(df_PE, BANDWIDTH=[100,40000], save_fig=False, file_name='phase_plot_PE_45.svg', file_format='svg',figsize=0.45, n_pi=1)
    #phase_teflon, freq_teflon  = wp.phase_plotting_chirp(df_teflon, BANDWIDTH=[5000,40000], save_fig=False, file_name='phase_plot_teflon_45.svg', file_format='svg',figsize=0.45, n_pi=1)
    #wp.plot_velocities(phase_PE, freq_PE, 0.10, savefig=False, filename='phase_velocity_PE.svg', file_format='svg')
    #wp.plot_velocities(phase_teflon, freq_teflon, 0.10, savefig=False, filename='phase_velocity_teflon.svg', file_format='svg')
    #wp.plot_velocities(phase_PE, freq_PE, 0.10, material='HDPE', savefig=False, filename='phase_velocity_teflon.svg', file_format='svg')
    #wp.plot_velocities(phase_PE, freq_PE, 0.10, material='LDPE', savefig=False, filename='phase_velocity_teflon.svg', file_format='svg')
    print(wp.max_peak_velocity(df_PE, material='HDPE'))
    #print(wp.max_peak_velocity(df_teflon))
    #print(wp.max_peak_velocity(df_PE))

def data_viz(viz_type, folder, filename, semester='thesis', channel='wave_gen'):
    if semester == 'thesis':
        df = csv_to_df_thesis(folder, filename)
    else:
        df = csv_to_df(folder, filename)
    if viz_type == 'scaleogram':
        
        vd.plot_scaleogram(df, channels=['wave_gen'])
    elif viz_type == 'wvd':
        vd.wigner_ville_dist(df, channel)
    elif viz_type == 'custom_wvd':
        vd.custom_wigner_ville_batch(df, channel)
    elif viz_type == 'ssq':
        vd.ssqueeze_spectrum(df, channel)


def load_simulated_data1():
    signals = ['sg8.txt','sg28.txt', 'sg108.txt']
    freq_signals = ['wt8.txt','wt28.txt', 'wt108.txt']
    channels = []
    for sig in signals:
        path = os.path.join(r'C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_7mm', sig)
        data = np.loadtxt(path)
        channels.append(data)
    return channels

def simulated_data_vel():
    distances = [0.173, 0.41, 1.359]
    channels = load_simulated_data1()
    for ch in channels:
        plt.plot(ch)
    plt.show()
    fft1 = np.fft.fft(channels[0])
    fft2 = np.fft.fft(channels[1])
    freq = np.fft.fftfreq(len(channels[1]), d=1000)
    plt.plot(freq, fft1)
    plt.plot(freq, fft2)
    plt.show()
    phase = wp.phase_difference_sub(channels[0], channels[1])

    freq = np.linspace(1000,50000,1000)
    plt.plot(phase)
    plt.show()
    
def wave_type_plots():
    df = csv_to_df_thesis('plate10mm\\setup2\\chirp', 'chirp3_ch3top_ch2bot_ch1_sidemid_v1')
    df_no_wave = df.drop(['wave_gen'], axis=1)
    time_axis = np.linspace(0, len(df) // 150e3, num=len(df))
    plt.plot(time_axis, df_no_wave)
    plt.show()

    #plotting just above and under the plate
    plt.plot(time_axis, df['channel 2'], label='channel 2')
    plt.plot(time_axis, df['channel 3'], label='channel 3')
    plt.legend()
    plt.show()

    #filter the signals
    df_filt = filter_general(df_no_wave, filtertype='highpass', cutoff_highpass=100, cutoff_lowpass=15000, order=4)
    plt.plot(time_axis, df_filt)
    plt.show()

    #plotting just above and under the plate
    plt.plot(time_axis, df_filt['channel 2'], label='channel 2')
    plt.plot(time_axis, df_filt['channel 3'], label='channel 3')
    plt.legend()
    plt.show()

    #normalize channel 2 and 3 so that they are on the same scale
    df_filt['channel 2'] = df_filt['channel 2'] / np.max(df_filt['channel 2'])
    df_filt['channel 3'] = df_filt['channel 3'] / np.max(df_filt['channel 3'])
    #plot the difference between the two signals
    plt.plot(time_axis, df_filt['channel 2'] + df_filt['channel 3'], label='difference')
    #plt.plot(time_axis, df_filt['channel 2'], label='channel 2')
    #plt.plot(time_axis, df_filt['channel 3'], label='channel 3')
    plt.legend()
    plt.show()

def velocities():
    df = csv_to_df_thesis('plate20mm\\setup1\\chirp', 'chirp_100_40000_2s_v1')
    #frequency range from 100 to 40000 Hz samplerate of 150000 Hz
    freqs = np.linspace(100, 40000, 150000)
    
    #wp.plot_theoretical_velocities(freqs)
    phase10, freq10  = wp.phase_plotting_chirp(df,
                channels=['channel 2', 'channel 3'], 
                detrend=True,
                method='div', 
                n_pi=0, 
                SAMPLE_RATE=150000, 
                BANDWIDTH=[100,40000], 
                save_fig=False,
                file_name='phase_difference.png',
                file_format='png',
                figsize=0.75,
                use_recorded_chirp=True)
    wp.plot_velocities(phase10, freq10, 0.10, savefig=False, filename='phase_velocity_10cm.svg', file_format='svg', material='HDPE')

if __name__ == '__main__':
    CROSS_CORR_PATH1 = '\\Measurements\\setup2_korrelasjon\\'
    CROSS_CORR_PATH2 = '\\first_test_touch_passive_setup2\\'
    custom_chirp = csv_to_df(file_folder='Measurements\\div_files', file_name='chirp_custom_fs_150000_tmax_2_100-40000_method_linear', channel_names=CHIRP_CHANNEL_NAMES)
    #set base new base after new measurements:
    #execution_time = timeit.timeit(ccp.SetBase, number=1)
    #print(f'Execution time: {execution_time}')
    #ccp.SetBase(CROSS_CORR_PATH2)
    #ccp.run_test(tolatex=True)
    #ccp.run_test(tolatex=True,data_folder='\\OneDrive - NTNU\\NTNU\\ProsjektOppgave\\first_test_touch_passive_setup2\\',filename='results_correlation_old_samples.csv')
    #find position
    #ccp.FindTouchPosition(f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave\\Measurements\\setup2_korrelasjon\\A2_v3.csv')
    df1 = csv_to_df('Measurements\\setup3_0\\', 'prop_speed_chirp3_setup3_0_v1')
    df = csv_to_df('\\Measurements\\setup10_propagation_speed_15cm\\chirp\\', 'chirp_v1')
    sensor_test_df = csv_to_df('\\Measurements\\sensortest\\rot_clock_123', 'chirp_v1')
    df_touch_10cm = csv_to_df('Measurements\\setup9_propagation_speed_short\\touch\\', 'touch_v1')
    time = np.linspace(0, len(df1) / SAMPLE_RATE, num=len(df1))
    