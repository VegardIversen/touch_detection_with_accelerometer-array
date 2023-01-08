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
from csv_to_df import csv_to_df
from data_viz_files.visualise_data import compare_signals, plot_vphs, plot_fft, plot_plate_speed_sliders_book, plot_estimated_reflections_with_sliders, compare_signals_v2, plot_compare_signals_v2
from data_processing.preprocessing import crop_data, filter_general, compress_chirp, get_phase_and_vph_of_compressed_signal,cut_out_signal, manual_cut_signal, compress_df_touch, cut_out_pulse_wave
from data_processing.detect_echoes import find_first_peak, get_hilbert_envelope, get_travel_times
from data_processing.find_propagation_speed import find_propagation_speed_with_delay
from data_viz_files.drawing import plot_legend_without_duplicates
import timeit
import data_processing.wave_properties as wp
import data_processing.sensor_testing as st
from data_viz_files.visualise_data import inspect_touch

def results_setup5_touch():
    FILTER = False
    SETUP = Setup9()
    BANDWIDTH = np.array([100, 40000])
    df_touch_10cm = csv_to_df('\\setup9_propagation_speed_short\\touch\\', 'touch_v1')
    df_chirp_10cm = csv_to_df('vegard_og_niklas\\setup3\\', 'prop_speed_chirp3_setup3_0_v1')
    #threshold of "csv_to_df('\\setup9_propagation_speed_short\\touch\\', 'touch_v1')" is 0.0006 ch1. 0.0013 ch3 
    custom_chirp = csv_to_df(file_folder='div_files', file_name='chirp_custom_fs_150000_tmax_2_100-40000_method_linear', channel_names=CHIRP_CHANNEL_NAMES)
    phase10, freq10  = wp.phase_plotting(df_chirp_10cm, chirp=custom_chirp, use_recorded_chirp=True, start_stops=[(95000,390500),(94000,390400)], BANDWIDTH=[100,40000], save_fig=False, file_name='phase_plot_10cm.svg', file_format='svg')
    phase_vel = wp.phase_velocity(phase10, freq10, distance=0.1, plot=True)
    max_vel = np.max(phase_vel)
    max_samples_direct = int(0.1/max_vel * SAMPLE_RATE)
    prop_speed = max_vel
    #get the compressed touch signal. check where i got the images from
    if FILTER:
        measurements_filt = filter_general(df_touch_10cm,
                                           filtertype='highpass',
                                           cutoff_highpass=100,
                                           # cutoff_lowpass=BANDWIDTH[1],
                                           order=4)
    else:
        measurements_filt = df_touch_10cm
    measurements_comp, start_indexes = compress_df_touch(measurements_filt, set_threshold_man=False,thresholds=[0.0006,0.0007,0.0013], n_sampl=max_samples_direct)
    measurements_hilb = get_hilbert_envelope(measurements_filt)

    measurements_comp_hilb = get_hilbert_envelope(measurements_comp)
    SETUP.draw()
    actuator, sensors = SETUP.get_objects()
    arrival_times = np.array([])
    for idx, sensor in enumerate(sensors):
        time, _ = get_travel_times(actuator[0],
                                   sensor,
                                   prop_speed,
                                   ms=False,
                                   print_info=True,
                                   relative_first_reflection=False,
                                   sig_start=start_indexes[idx])
        #time = time + 2.5
        print(time)
        #time = int(time*SAMPLE_RATE)
        
        arrival_times = np.append(arrival_times, time)
    arrival_times = np.reshape(arrival_times, (len(sensors), len(arrival_times) // len(sensors)))

    dynamic_range_db = 60
    vmin = 10 * np.log10(np.max(measurements_comp['channel 1'])) - dynamic_range_db
    for i, sensor in enumerate(sensors):
        plt.subplot(311 + i, sharex=plt.gca())
        plt.title('Correlation between chirp and channel ' + str(i + 1))
        plt.specgram(measurements_comp['channel ' + str(i + 1)], Fs=SAMPLE_RATE, NFFT=16, noverlap=(16 // 2), vmin=vmin)
        plt.axis(ymax=BANDWIDTH[1])
        plt.title('Spectrogram')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.colorbar()
        plt.axvline(arrival_times[i][0], linestyle='--', color='r', label='Direct wave')
        x = [plt.axvline(line, linestyle='--', color='g', label='1st reflections') for line in (arrival_times[i][1:5])]
        y = [plt.axvline(line, linestyle='--', color='purple', label='2nd reflections') for line in (arrival_times[i][5:])]
        plt.xlabel('Time [ms]')
        plt.ylabel('Amplitude [V]')
        plot_legend_without_duplicates()
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    time_axis_corr = np.linspace(-1000 * len(measurements_filt) / SAMPLE_RATE,
                            1000 * len(measurements_filt) / SAMPLE_RATE,
                            len(measurements_filt))
    
    #arrival_times *= 1000   # Convert to ms
    for i, sensor in enumerate(sensors):
        plt.subplot(311 + i, sharex=plt.gca())
        plt.title('Correlation between chirp and channel ' + str(i + 1))
        plt.plot(time_axis_corr, measurements_comp['channel ' + str(i + 1)], label='Correlation')
        plt.plot(time_axis_corr, measurements_comp_hilb['channel ' + str(i + 1)], label='Hilbert envelope')
        plt.axvline(arrival_times[i][0], linestyle='--', color='r', label='Direct wave')
        #plt.axvline(start_indexes[i], linestyle='--', color='r', label='Direct wave')
        [plt.axvline(line, linestyle='--', color='g', label='1st reflections') for line in (arrival_times[i][1:5])]
        [plt.axvline(line, linestyle='--', color='purple', label='2nd reflections') for line in (arrival_times[i][5:])]
        print([plt.axvline(line, linestyle='--', color='g', label='1st reflections') for line in (arrival_times[i][1:5])])
        plt.xlabel('Time [ms]')
        plt.ylabel('Amplitude [V]')
        plot_legend_without_duplicates()
        plt.grid()
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    arrival_times /= 1000   # Convert back to s


def results_setup5_chirp():
    SAMPLE_RATE = 150000
    use_recorded_chirp=True
    channels=['channel 1', 'channel 3']
    custom_chirp = csv_to_df(file_folder='div_files', file_name='chirp_custom_fs_150000_tmax_2_100-40000_method_linear', channel_names=CHIRP_CHANNEL_NAMES)
    df1 = csv_to_df('vegard_og_niklas\\setup3\\', 'prop_speed_chirp3_setup3_0_v1')
    df_touch_15cm = csv_to_df('\\vegard_og_niklas\\setup10_propagation_speed_15cm\\chirp\\', 'chirp_v1')
    df_touch_10cm = csv_to_df('\\setup9_propagation_speed_short\\touch\\', 'touch_v1')
    time = np.linspace(0, len(df_touch_10cm) / SAMPLE_RATE, num=len(df_touch_10cm))
    df_touch_10cm_detrend = wp.preprocess_df(df_touch_10cm)
    df_touch_15cm_detrend = wp.preprocess_df(df_touch_15cm)
    chirp_10cm = df_touch_10cm_detrend['wave_gen'].to_numpy()
    chirp_15cm = df_touch_15cm_detrend['wave_gen'].to_numpy()
    temp_arr_15cm = np.zeros((len(df_touch_15cm_detrend[channels[0]]),len(channels)))
    temp_arr_10cm = np.zeros((len(df_touch_10cm_detrend[channels[0]]),len(channels)))
    #filling the array
    start_stops_15cm=[(146000,441200),(146000,442000)]
    start_stops_10cm=[(95000,390500),(94000,390400)]
    start_stops=[(95000,390500),(94000,390400)]
    start1_15cm, end1_15cm = start_stops_15cm[0]
    start2_15cm, end2_15cm = start_stops_15cm[1]
    start1_10cm, end1_10cm = start_stops_10cm[0]
    start2_10cm, end2_10cm = start_stops_10cm[1]

    temp_arr_15cm[start1_15cm:end1_15cm,0] = df_touch_15cm_detrend[channels[0]].iloc[start1_15cm:end1_15cm]
    temp_arr_15cm[start2_15cm:end2_15cm,1] = df_touch_15cm_detrend[channels[1]].iloc[start2_15cm:end2_15cm]
    temp_arr_10cm[start1_10cm:end1_10cm,0] = df_touch_10cm_detrend[channels[0]].iloc[start1_10cm:end1_10cm]
    temp_arr_10cm[start2_10cm:end2_10cm,1] = df_touch_10cm_detrend[channels[1]].iloc[start2_10cm:end2_10cm]
    df_sig_only_15cm = pd.DataFrame(temp_arr_15cm, columns=channels)
    df_sig_only_10cm = pd.DataFrame(temp_arr_10cm, columns=channels)

    compressed_df15 = wp.compress_chirp(df_sig_only_15, chirp_15cm, use_recorded_chirp=use_recorded_chirp)
    compressed_df10 = wp.compress_chirp(df_sig_only_10, chirp_10cm, use_recorded_chirp=use_recorded_chirp)


if __name__ == '__main__':
    CROSS_CORR_PATH1 = '\\vegard_og_niklas\\setup2_korrelasjon\\'
    CROSS_CORR_PATH2 = '\\first_test_touch_passive_setup2\\'
    custom_chirp = csv_to_df(file_folder='div_files', file_name='chirp_custom_fs_150000_tmax_2_100-40000_method_linear', channel_names=CHIRP_CHANNEL_NAMES)
    #set base new base after new measurements:
    #execution_time = timeit.timeit(ccp.SetBase, number=1)
    #print(f'Execution time: {execution_time}')
    #ccp.SetBase(CROSS_CORR_PATH2)
    #ccp.run_test(tolatex=True)
    #ccp.run_test(tolatex=True,data_folder='\\OneDrive - NTNU\\NTNU\\ProsjektOppgave\\first_test_touch_passive_setup2\\',filename='results_correlation_old_samples.csv')
    #find position
    #ccp.FindTouchPosition(f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave\\vegard_og_niklas\\setup2_korrelasjon\\A2_v3.csv')
    df1 = csv_to_df('vegard_og_niklas\\setup3\\', 'prop_speed_chirp3_setup3_0_v1')
    df = csv_to_df('\\vegard_og_niklas\\setup10_propagation_speed_15cm\\chirp\\', 'chirp_v1')
    sensor_test_df = csv_to_df('\\vegard_og_niklas\\sensortest\\rot_clock_123', 'chirp_v1')
    df_touch_10cm = csv_to_df('\\setup9_propagation_speed_short\\touch\\', 'touch_v1')
    time = np.linspace(0, len(df1) / SAMPLE_RATE, num=len(df1))
    