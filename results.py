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

def compression_figs():
    # Generate the chirp signal
    # sns.set_theme(style="darkgrid")
    # sns.set(font_scale=12/10)
    t = np.linspace(0, 1, 1000)
    f0 = 3
    f1 = 30
    k = (f1 - f0) / 1
    signal = np.sin(2 * np.pi * (f0 * t + k * t * t / 2))

    # Compress the signal using matched filtering
    signal_compressed = np.convolve(signal, signal[::-1], mode='same')
    #time array for the compressed signal from negative to positive
    t_compressed = np.linspace(-0.5, 0.5, 1000)
    # Plot the original and compressed signals in separate subplots
    fig, axs = figure_size_setup(0.33)
    
    axs.plot(t, signal)
    axs.set_xlabel('Time')
    axs.set_ylabel('Amplitude')
    fig.savefig('compression_signal.png', dpi=300, format='png')
    fig.savefig('compression_signal.svg', dpi=300, format='svg')
    #clear figure
    plt.show()
    plt.clf()
    fig, axs = figure_size_setup(0.33)
    #ax1.set_title('Original Signal')
    axs.plot(t_compressed, signal_compressed)
    axs.set_xlabel('Time')
    axs.set_ylabel('Amplitude')
    #ax2.set_title('Compressed Signal')
    fig.savefig('compression_result.png', dpi=300, format='png')
    fig.savefig('compression_result.svg', dpi=300, format='svg')
    #plt.tight_layout()
    plt.show()
    plt.clf()
    # plot the fft of the respective signals with only positive frequencies
    fig, axs = figure_size_setup(0.33)
    axs.plot(np.fft.rfftfreq(len(signal), 1 / 1000), np.abs(np.fft.rfft(signal)))
    axs.set_xlabel('Frequency')
    axs.set_ylabel('Amplitude')
    fig.savefig('compression_fft_signal.png', dpi=300, format='png')
    #save as svg
    fig.savefig('compression_fft_signal.svg', dpi=300, format='svg')
    plt.show()
    plt.clf()
    fig, axs = figure_size_setup(0.33)
    axs.plot(np.fft.rfftfreq(len(signal_compressed), 1 / 1000), np.abs(np.fft.rfft(signal_compressed)))
    axs.set_xlabel('Frequency')
    axs.set_ylabel('Amplitude')
    fig.savefig('compression_fft_result.png', dpi=300, format='png')
    fig.savefig('compression_fft_result.svg', dpi=300, format='svg')
    plt.show()
    plt.clf()
    #plot the ffts in the same plot with labels "signal" and "compressed signal" and limit the maximum frequency to 100 Hz and normalize the amplitude so they are around the same amplitude
    fig, axs = figure_size_setup(0.33)
    #plot normalized ffts
    axs.plot(np.fft.rfftfreq(len(signal), 1 / 1000), np.abs(np.fft.rfft(signal))/np.max(np.abs(np.fft.rfft(signal))), label='signal')
    axs.plot(np.fft.rfftfreq(len(signal_compressed), 1 / 1000), np.abs(np.fft.rfft(signal_compressed))/np.max(np.abs(np.fft.rfft(signal_compressed))), label='compressed')
    axs.set_xlabel('Frequency')
    axs.set_ylabel('Amplitude')
    axs.set_xlim(0, 100)
    axs.legend(loc='upper right')
    fig.savefig('compression_fft_both_norm.png', dpi=300, format='png')
    fig.savefig('compression_fft_both_norm.svg', dpi=300, format='svg')
    plt.show()

def table_figures():
    SETUP = Setup2()
    SETUP.draw(actuator_show=False, save_fig=True, file_format='svg', fig_name='table_setup2_correlation', show_tab=True)
    SETUP.draw(actuator_show=False, save_fig=True, file_format='png', fig_name='table_setup2_correlation', show_tab=True)
def results_setup1():
    st.transfer_function_plate('\\vegard_og_niklas\\sensortest\\rot_clock_123', n_files=1,plot_response=True, channels=['channel 1','channel 2', 'channel 3'])
    st.transfer_function_plate('\\vegard_og_niklas\\sensortest\\rot_clock_123', n_files=1,plot_response=True, channels=['channel 1','channel 2', 'channel 3'], file_format='svg')
    st.transfer_function_plate('\\vegard_og_niklas\\sensortest\\rot_clock_123', n_files=1,plot_response=True, channels=['channel 1'])
    st.transfer_function_plate('\\vegard_og_niklas\\sensortest\\rot_clock_123', n_files=1,plot_response=True, channels=['channel 1'], file_format='svg')
    st.transfer_function_plate('\\vegard_og_niklas\\sensortest\\rot_clock_123', n_files=1,plot_response=False, channels=['channel 1','channel 2', 'channel 3'])
    st.transfer_function_plate('\\vegard_og_niklas\\sensortest\\rot_clock_123', n_files=1,plot_response=False, channels=['channel 1','channel 2', 'channel 3'], file_format='svg')
    st.transfer_function_plate('\\vegard_og_niklas\\sensortest\\rot_clock_123', n_files=1,plot_response=False, channels=['channel 1'])
    st.transfer_function_plate('\\vegard_og_niklas\\sensortest\\rot_clock_123', n_files=1,plot_response=False, channels=['channel 1'], file_format='svg')
    st.fft_analysis_one_sens_all_pos(folder='\\vegard_og_niklas\\sensortest\\', n_runs=1, plot_chirp_fft=True,savefig=True, file_format='svg')
    st.fft_analysis_one_sens_all_pos(folder='\\vegard_og_niklas\\sensortest\\', n_runs=1, plot_chirp_fft=True,savefig=True, file_format='png')
    st.fft_analysis_all_sensors('\\vegard_og_niklas\\sensortest\\', position=-3, n_runs=1, plot_chirp_fft=True,savefig=True, file_format='svg')
    st.fft_analysis_all_sensors('\\vegard_og_niklas\\sensortest\\', position=-2, n_runs=1, plot_chirp_fft=True,savefig=True, file_format='svg')
    st.fft_analysis_all_sensors('\\vegard_og_niklas\\sensortest\\', position=-1, n_runs=1, plot_chirp_fft=True,savefig=True, file_format='svg')
    st.fft_analysis_all_sensors('\\vegard_og_niklas\\sensortest\\', position=-3, n_runs=1, plot_chirp_fft=True,savefig=True, file_format='png')
    st.fft_analysis_all_sensors('\\vegard_og_niklas\\sensortest\\', position=-2, n_runs=1, plot_chirp_fft=True,savefig=True, file_format='png')
    st.fft_analysis_all_sensors('\\vegard_og_niklas\\sensortest\\', position=-1, n_runs=1, plot_chirp_fft=True,savefig=True, file_format='png')
    st.fft_analysis_all_sensors_compare(folder='\\vegard_og_niklas\\sensortest\\', n_runs=1, plot_chirp_fft=True,savefig=True, file_format='svg')
    st.fft_analysis_all_sensors_compare(folder='\\vegard_og_niklas\\sensortest\\', n_runs=1, plot_chirp_fft=True,savefig=True, file_format='png')
    st.fft_analysis_all_positions_compare(folder='\\vegard_og_niklas\\sensortest\\rot_clock_123\\', savefig=True, file_format='png')

def results_setup2():
    #bug in code here. Done on different branch to save time
    pass
    # """Inspect touch data by plotting the raw from channel 1 and the spectogram of the raw data on this channel.
    # share axis between the two plots"""
    # df = csv_to_df('\\setup9_propagation_speed_short\\touch\\', 'touch_v1')
    # dynamic_range_db = 60
    # #filter the dataframe with a highpassfilter with cutoff at 100 Hz
    # #df = filter_general(df, 'highpass', cutoff_highpass=100)
    # measurements = interpolate_waveform(df)

    # measurements_full_touch = crop_data(measurements,
    #                                     time_start=2.05,
    #                                     time_end=2.5075)

    # """Filter signals to remove 50 Hz and other mud"""
    # measurements_full_touch = filter_general(measurements_full_touch,
    #                                          filtertype='highpass',
    #                                          cutoff_highpass=100)
    # fig, axs = figure_size_setup(0.45)
    # time_axis = np.linspace(0, len(df) / SAMPLE_RATE, len(df))
    # axs.plot(time_axis[315900:370000],df['channel 1'].iloc[315900:370000])
    # #axs.set_title('Raw data from channel 1')
    # axs.set_ylabel('Amplitude [V]')
    # axs.set_xlabel('Time [s]')
    # fig.savefig(f'inspect_touch_fullsignalch1.svg', dpi=300, format='svg')
    # plt.show()
    # plt.clf()

    # fig, axs = figure_size_setup(0.45)
    # style.use('default')
    # spec = axs.specgram(measurements_full_touch['channel 1'], Fs=SAMPLE_RATE, NFFT=256, noverlap=128)
    # spec[3].set_clim(to_dB(np.max(spec[0])) - dynamic_range_db,
    #                          to_dB(np.max(spec[0])))
    # #set xlim for spectrogram
    # fig.colorbar(spec[3],ax=axs)
    # #axs.set_xlim(315900, 370000)
    # #axs.set_title('Spectrogram of raw data from channel 1')
    # axs.set_xlabel('Time [s]')
    # axs.set_ylabel('Frequency [Hz]')
    # #plt.tight_layout()
    # fig.savefig(f'inspect_touch_spetrogramch1.svg', dpi=300, format='svg')
    # fig.savefig(f'inspect_touch_spetrogramch1.png', dpi=300, format='png')
    # plt.show()
    # plt.clf()
    # fig, axs = figure_size_setup(0.45)
    # time_axis = np.linspace(0, len(df) / SAMPLE_RATE, len(df))
    # axs.plot(time_axis[315900:370000],df['channel 3'].iloc[315900:370000])
    # #axs.set_title('Raw data from channel 3')
    # axs.set_ylabel('Amplitude [V]')
    # axs.set_xlabel('Time [s]')
    # fig.savefig(f'inspect_touch_fullsignalch3.svg', dpi=300, format='svg')
    
    # plt.show()
    # plt.clf()

    # fig, axs = figure_size_setup(0.45)
    # style.use('default')
    # spec = axs.specgram(df['channel 3'], Fs=SAMPLE_RATE, NFFT=256, noverlap=128)
    # #spec[3].set_clim(to_dB(np.max(spec[0])) - 60,
    # #                         to_dB(np.max(spec[0])))
    # #set xlim for spectrogram
    # fig.colorbar(spec[3],ax=axs)
    # #axs.set_xlim(315900, 370000)
    # #axs.set_title('Spectrogram of raw data from channel 3')
    # axs.set_xlabel('Time [s]')
    # axs.set_ylabel('Frequency [Hz]')
    # #plt.tight_layout()
    # fig.savefig(f'inspect_touch_spetrogramch3.svg', dpi=300, format='svg')
    # fig.savefig(f'inspect_touch_spetrogramch3.png', dpi=300, format='png')
    # plt.show()
    # plt.clf()
def results_setup4():
    custom_chirp = csv_to_df(file_folder='div_files', file_name='chirp_custom_fs_150000_tmax_2_100-40000_method_linear', channel_names=CHIRP_CHANNEL_NAMES)
    df1 = csv_to_df('vegard_og_niklas\\setup3\\', 'prop_speed_chirp3_setup3_0_v1')
    df = csv_to_df('\\vegard_og_niklas\\setup10_propagation_speed_15cm\\chirp\\', 'chirp_v1')
    sensor_test_df = csv_to_df('\\vegard_og_niklas\\sensortest\\rot_clock_123', 'chirp_v1')
    df_touch_10cm = csv_to_df('\\setup9_propagation_speed_short\\touch\\', 'touch_v1')
    phase15, freq15  = wp.phase_plotting(df, chirp=custom_chirp, use_recorded_chirp=True, BANDWIDTH=[100,40000], start_stops=[(146000,441200),(146000,442000)], save_fig=True, file_name='phase_plot_15cm0_45.svg', file_format='svg',figsize=0.45)
    phase10, freq10  = wp.phase_plotting(df1, chirp=custom_chirp, use_recorded_chirp=True, start_stops=[(95000,390500),(94000,390400)], BANDWIDTH=[100,40000], save_fig=True, file_name='phase_plot_10cm0_45.svg', file_format='svg',figsize=0.45)
    #phase_vel = wp.phase_velocity(phase, freq, distance=0.15, plot=True)
    wp.plot_velocities_2distance(phase15, freq15, 0.15, phase10, freq10,  0.1,  savefig=True, filename='phase_velocity_10_cm_15_cm.png', file_format='png')
    wp.plot_velocities_2distance(phase15, freq15, 0.15, phase10, freq10,  0.1,  savefig=True, filename='phase_velocity_10_cm_15_cm.svg', file_format='svg')
    wp.plot_velocities(phase10, freq10, 0.10, savefig=True, filename='phase_velocity_10cm.svg', file_format='svg')

def results_setup5_touch():
    FILTER = False
    SETUP = Setup9()
    BANDWIDTH = np.array([100, 40000])
    df_touch_10cm = csv_to_df('\\setup9_propagation_speed_short\\touch\\', 'touch_v1')
    #df_touch_10cm = csv_to_df('\\setup9_propagation_speed_short\\touch\\', 'touch_v2')
    df_chirp_10cm = csv_to_df('vegard_og_niklas\\setup3\\', 'prop_speed_chirp3_setup3_0_v1')
    #threshold of "csv_to_df('\\setup9_propagation_speed_short\\touch\\', 'touch_v1')" is 0.0006 ch1. 0.0013 ch3 
    custom_chirp = csv_to_df(file_folder='div_files', file_name='chirp_custom_fs_150000_tmax_2_100-40000_method_linear', channel_names=CHIRP_CHANNEL_NAMES)
    phase10, freq10  = wp.phase_plotting(df_chirp_10cm, chirp=custom_chirp, use_recorded_chirp=True, start_stops=[(95000,390500),(94000,390400)], BANDWIDTH=[100,40000], save_fig=False, file_name='phase_plot_10cm.svg', file_format='svg')
    phase_vel = wp.phase_velocity(phase10, freq10, distance=0.1, plot=False)
    max_vel = np.max(phase_vel)
    max_samples_direct = int(0.1/max_vel * SAMPLE_RATE)
    prop_speed = max_vel
    print(prop_speed)
    prop_speed = 300
    #get the compressed touch signal. check where i got the images from
    if FILTER:
        measurements_filt = filter_general(df_touch_10cm,
                                           filtertype='highpass',
                                           cutoff_highpass=100,
                                           # cutoff_lowpass=BANDWIDTH[1],
                                           order=4)
    else:
        measurements_filt = df_touch_10cm
    print(f'length of measurements_filt: {len(measurements_filt)}')
    measurements_comp, start_indexes = compress_df_touch(measurements_filt, set_threshold_man=False,thresholds=[0.0010,0.0007,0.0013], n_sampl=max_samples_direct)
    
    
    print(f'length of measurements_comp: {len(measurements_comp)}')
    measurements_hilb = get_hilbert_envelope(measurements_filt)
    print(f'length of measurements_hilb: {len(measurements_hilb)}')
    measurements_comp_hilb = get_hilbert_envelope(measurements_comp)
    print(f'length of measurements_comp_hilb: {len(measurements_comp_hilb)}')
    #SETUP.draw()
    
    actuator, sensors = SETUP.get_objects()
    arrival_times = np.array([])
    for idx, sensor in enumerate(sensors):
        time, _ = get_travel_times(actuator[0],
                                   sensor,
                                   prop_speed,
                                   ms=False,
                                   print_info=False,
                                   relative_first_reflection=True,
                                   sig_start=start_indexes[idx])
        #time *= -1
        #time = time + 2.5
        print(time)
        #time = int(time*SAMPLE_RATE)
        
        arrival_times = np.append(arrival_times, time)
    arrival_times = np.reshape(arrival_times, (len(sensors), len(arrival_times) // len(sensors)))

    # dynamic_range_db = 60
    # vmin = 10 * np.log10(np.max(measurements_comp['channel 1'])) - dynamic_range_db
    # for i, sensor in enumerate(sensors):
    #     plt.subplot(311 + i, sharex=plt.gca())
    #     plt.title('Correlation between chirp and channel ' + str(i + 1))
    #     plt.specgram(measurements_comp['channel ' + str(i + 1)], Fs=SAMPLE_RATE, NFFT=16, noverlap=(16 // 2), vmin=vmin)
    #     plt.axis(ymax=BANDWIDTH[1])
    #     plt.title('Spectrogram')
    #     plt.xlabel('Time [s]')
    #     plt.ylabel('Frequency [Hz]')
    #     plt.colorbar()
    #     plt.axvline(arrival_times[i][0], linestyle='--', color='r', label='Direct wave')
    #     x = [plt.axvline(line, linestyle='--', color='g', label='1st reflections') for line in (arrival_times[i][1:5])]
    #     y = [plt.axvline(line, linestyle='--', color='purple', label='2nd reflections') for line in (arrival_times[i][5:])]
    #     plt.xlabel('Time [ms]')
    #     plt.ylabel('Amplitude [V]')
    #     plot_legend_without_duplicates()
    #only plot from -0.1 to 1
    
    plt.subplots_adjust(hspace=0.5)
    plt.show()

    # Sampling rate of the recorded signal (in Hz)
    sampling_rate = 150_000

    # Length of the cross-correlation function in samples
    n_samples = 750_000

    # Duration of the recorded signal (in seconds)
    duration = n_samples / sampling_rate

    # Time axis for the cross-correlation function (in seconds)
    time_axis = np.linspace(-duration/2, duration/2, n_samples)
    fig, axs = figure_size_setup()
    # Time axis in milliseconds
    #time_axis_ms = time_axis * 1000
    axs.plot(time_axis, measurements_comp['channel 1'], label='Correlation')
    axs.plot(time_axis, measurements_comp_hilb['channel 1'], label='Hilbert envelope')
    #plt.axvline(0.0001333, linestyle='--', color='r', label='Direct wave')
    #plt.axvline(0.00081093, linestyle='--', color='r', label='1st reflections')
    axs.axvline(arrival_times[0][0], linestyle='--', color='r', label='Direct wave')
    [axs.axvline(line, linestyle='--', color='g', label='1st reflections') for line in (arrival_times[0][1:5])]
    [axs.axvline(line, linestyle='--', color='purple', label='2nd reflections') for line in (arrival_times[0][5:])]
    axs.set_xlabel('Time [s]')
    axs.set_ylabel('Amplitude [V]')
    plot_legend_without_duplicates()
    axs.set_xlim(-0.01, 0.04)
    fig.savefig(f'reflections_speed{np.round(prop_speed, 2)}.png',dpi=300, format='png')
    fig.savefig(f'reflections_speed{np.round(prop_speed, 2)}.svg',dpi=300, format='svg')
    # for i, sensor in enumerate(sensors):
    #     print(f'Length of channel {i+1}: {len(measurements_comp["channel " + str(i + 1)])}')
    #     plt.subplot(311 + i, sharex=plt.gca())
    #     plt.title('Correlation between chirp and channel ' + str(i + 1))
    #     plt.plot(time_axis, measurements_comp['channel ' + str(i + 1)], label='Correlation')
    #     plt.plot(time_axis, measurements_comp_hilb['channel ' + str(i + 1)], label='Hilbert envelope')
    #     plt.axvline(arrival_times[i][0], linestyle='--', color='r', label='Direct wave')
    #     #plt.axvline(start_indexes[i], linestyle='--', color='r', label='Direct wave')
    #     [plt.axvline(line, linestyle='--', color='g', label='1st reflections') for line in (arrival_times[i][1:5])]
    #     [plt.axvline(line, linestyle='--', color='purple', label='2nd reflections') for line in (arrival_times[i][5:])]
    #     #print([plt.axvline(line, linestyle='--', color='g', label='1st reflections') for line in (arrival_times[i][1:5])])
    #     plt.xlabel('Time [ms]')
    #     plt.ylabel('Amplitude [V]')
    #     plot_legend_without_duplicates()
    #     plt.grid()
    #plt.subplots_adjust(hspace=0.5)
    plt.show()
    arrival_times /= 1000   # Convert back to s

def results_setup5_chirp():
    SETUP = Setup9()
    SETUP.draw()
    df_touch_10cm = csv_to_df('\\setup9_propagation_speed_short\\touch\\', 'touch_v1')
    df_chirp_ref_10cm = csv_to_df('\\setup9_propagation_speed_short\\chirp\\100Hz_to_40kHz_single_chirp\\', 'chirp_v1')
    custom_chirp = csv_to_df(file_folder='div_files', file_name='chirp_custom_fs_150000_tmax_2_100-40000_method_linear', channel_names=CHIRP_CHANNEL_NAMES)
    df_chirp_10cm = csv_to_df('vegard_og_niklas\\setup3\\', 'prop_speed_chirp3_setup3_0_v1')
    custom_chirp = csv_to_df(file_folder='div_files', file_name='chirp_custom_fs_150000_tmax_2_100-40000_method_linear', channel_names=CHIRP_CHANNEL_NAMES)
    phase10, freq10  = wp.phase_plotting(df_chirp_10cm, chirp=custom_chirp, use_recorded_chirp=True, start_stops=[(95000,390500),(94000,390400)], BANDWIDTH=[100,40000], save_fig=False, file_name='phase_plot_10cm.svg', file_format='svg')
    phase_vel = wp.phase_velocity(phase10, freq10, distance=0.1, plot=False)
    max_vel = np.max(phase_vel)
    max_samples_direct = int(0.1/max_vel * SAMPLE_RATE)
    #prop_speed = max_vel
    prop_speed = 150
    #prop_speed = max_vel
    #compress chirp
    measurements_compressed = compress_chirp(df_chirp_ref_10cm, custom_chirp)
    measurements_comp_touch, start_indexes = compress_df_touch(df_touch_10cm, set_threshold_man=False,thresholds=[0.0010,0.0007,0.0013], n_sampl=max_samples_direct)
    measurements_compressed_hilbert = get_hilbert_envelope(measurements_compressed)
    wrong_start = get_first_index_above_threshold(measurements_comp_touch['channel 1'], 0.0001)
    print(f'index of threshold above 0.0001: {wrong_start}')
    print(f'start index: {start_indexes}')
    print(f'difference: {start_indexes[0] - wrong_start}')
    corrected_start = start_indexes[0] - wrong_start
    touch_comp_shifted = shift_signal(measurements_comp_touch['channel 1'], -59100)
    measurements_comp_touch_hilb = get_hilbert_envelope(touch_comp_shifted)
    actuator, sensors = SETUP.get_objects()
    arrival_times = np.array([])
    for idx, sensor in enumerate(sensors):
        time, _ = get_travel_times(actuator[0],
                                   sensor,
                                   prop_speed,
                                   ms=False,
                                   print_info=False,
                                   relative_first_reflection=False,
                                   )
        #time *= -1
        #time = time + 2.5
        print(time)
        #time = int(time*SAMPLE_RATE)
        
        arrival_times = np.append(arrival_times, time)
    arrival_times = np.reshape(arrival_times, (len(sensors), len(arrival_times) // len(sensors)))
    sampling_rate = 150_000

    # Length of the cross-correlation function in samples
    n_samples = 750_000

    # Duration of the recorded signal (in seconds)
    duration = n_samples / sampling_rate

    # Time axis for the cross-correlation function (in seconds)
    time_axis = np.linspace(-duration/2, duration/2, n_samples)
    fig, axs = figure_size_setup()
    # Time axis in milliseconds
    #time_axis_ms = time_axis * 1000
    axs.plot(time_axis, measurements_compressed['channel 1'], label='Correlation')
    axs.plot(time_axis, measurements_compressed_hilbert['channel 1'], label='Hilbert envelope')
    
    axs.axvline(arrival_times[0][0], linestyle='--', color='r', label='Direct wave')
    [axs.axvline(line, linestyle='--', color='g', label='1st reflections') for line in (arrival_times[0][1:5])]
    [axs.axvline(line, linestyle='--', color='purple', label='2nd reflections') for line in (arrival_times[0][5:])]
    axs.set_xlabel('Time [s]')
    axs.set_ylabel('Amplitude [V]')
    plot_legend_without_duplicates()
    axs.set_xlim(0, 0.0025)
    fig.savefig(f'chirp_reflections_speed{np.round(prop_speed, 2)}.png',dpi=300, format='png')
    fig.savefig(f'chirp_reflections_speed{np.round(prop_speed, 2)}.svg',dpi=300, format='svg')
    plt.show()
    plt.clf()
    fig, axs = figure_size_setup()
    axs.plot(time_axis, touch_comp_shifted, label='Correlation')
    axs.plot(time_axis, measurements_comp_touch_hilb, label='Hilbert envelope')
    #axs.plot( touch_comp_shifted, label='Correlation')
    #axs.plot( measurements_comp_touch_hilb, label='Hilbert envelope')
    axs.axvline(arrival_times[0][0], linestyle='--', color='r', label='Direct wave')
    [axs.axvline(line, linestyle='--', color='g', label='1st reflections') for line in (arrival_times[0][1:5])]
    [axs.axvline(line, linestyle='--', color='purple', label='2nd reflections') for line in (arrival_times[0][5:])]
    axs.set_xlabel('Time [s]')
    axs.set_ylabel('Amplitude [V]')
    plot_legend_without_duplicates()
    axs.set_xlim(-0.002, 0.02)
    fig.savefig(f'touch_reflections_speed{np.round(prop_speed, 2)}.png',dpi=300, format='png')
    fig.savefig(f'touch_reflections_speed{np.round(prop_speed, 2)}.svg',dpi=300, format='svg')
    plt.show()

def results_setup5_chirp1():
    SAMPLE_RATE = 150000
    use_recorded_chirp=True
    #channels=['channel 1', 'channel 3']
    custom_chirp = csv_to_df(file_folder='div_files', file_name='chirp_custom_fs_150000_tmax_2_100-40000_method_linear', channel_names=CHIRP_CHANNEL_NAMES)
    #df1 = csv_to_df('vegard_og_niklas\\setup3\\', 'prop_speed_chirp3_setup3_0_v1')
    #df_touch_15cm = csv_to_df('\\vegard_og_niklas\\setup10_propagation_speed_15cm\\chirp\\', 'chirp_v1')
    df_touch_10cm = csv_to_df('\\setup9_propagation_speed_short\\touch\\', 'touch_v1')
    df_chirp_10cm = csv_to_df('\\setup9_propagation_speed_short\\chirp\\100Hz_to_40kHz_single_chirp\\', 'chirp_v1')
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
    