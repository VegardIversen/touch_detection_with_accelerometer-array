import scipy.signal as signal
from scipy import interpolate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import Slider, Button
from pathlib import Path
from objects import Table, Actuator, Sensor
from setups import Setup2, Setup3, Setup3_2, Setup3_4, Setup6
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
import seaborn as sns
if __name__ == '__main__':
    # Set the font scale to match the text size of the document (12pt)
    sns.set_context("paper", font_scale=1.2)
    fig, ax = plt.subplots()

    # Generate some random data and plot it
    x = np.random.rand(10)
    y = np.random.rand(10)
    ax.plot(x, y, label='random')

    # Set the x and y labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Add a legend
    ax.legend()

    # Save the figure
    plt.savefig('random_plot_3sub.png',format='png')
    plt.savefig('random_plot_3sub.svg',format='svg')

    # Show the plot
    plt.show()
    exit()



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
    #inspect_touch(df_touch_10cm)
    
    #st.time_domain_analysis(sensor_test_df, 'channel 1')
    #st.fft_analysis_plot(sensor_test_df, 'channel 1')
    # st.transfer_function_plate('\\vegard_og_niklas\\sensortest\\rot_clock_123', n_files=1,plot_response=True, channels=['channel 1','channel 2', 'channel 3'])
    # st.transfer_function_plate('\\vegard_og_niklas\\sensortest\\rot_clock_123', n_files=1,plot_response=True, channels=['channel 1','channel 2', 'channel 3'], file_format='svg')
    # st.transfer_function_plate('\\vegard_og_niklas\\sensortest\\rot_clock_123', n_files=1,plot_response=True, channels=['channel 1'])
    # st.transfer_function_plate('\\vegard_og_niklas\\sensortest\\rot_clock_123', n_files=1,plot_response=True, channels=['channel 1'], file_format='svg')
    # st.transfer_function_plate('\\vegard_og_niklas\\sensortest\\rot_clock_123', n_files=1,plot_response=False, channels=['channel 1','channel 2', 'channel 3'])
    # st.transfer_function_plate('\\vegard_og_niklas\\sensortest\\rot_clock_123', n_files=1,plot_response=False, channels=['channel 1','channel 2', 'channel 3'], file_format='svg')
    # st.transfer_function_plate('\\vegard_og_niklas\\sensortest\\rot_clock_123', n_files=1,plot_response=False, channels=['channel 1'])
    # st.transfer_function_plate('\\vegard_og_niklas\\sensortest\\rot_clock_123', n_files=1,plot_response=False, channels=['channel 1'], file_format='svg')
    #st.fft_analysis_one_sens_all_pos(folder='\\vegard_og_niklas\\sensortest\\', n_runs=1, plot_chirp_fft=True,savefig=True, file_format='svg')
    #st.fft_analysis_one_sens_all_pos(folder='\\vegard_og_niklas\\sensortest\\', n_runs=1, plot_chirp_fft=True,savefig=True, file_format='png')
    # st.fft_analysis_all_sensors('\\vegard_og_niklas\\sensortest\\', position=-3, n_runs=1, plot_chirp_fft=True,savefig=True, file_format='svg')
    # st.fft_analysis_all_sensors('\\vegard_og_niklas\\sensortest\\', position=-2, n_runs=1, plot_chirp_fft=True,savefig=True, file_format='svg')
    # st.fft_analysis_all_sensors('\\vegard_og_niklas\\sensortest\\', position=-1, n_runs=1, plot_chirp_fft=True,savefig=True, file_format='svg')
    # st.fft_analysis_all_sensors('\\vegard_og_niklas\\sensortest\\', position=-3, n_runs=1, plot_chirp_fft=True,savefig=True, file_format='png')
    # st.fft_analysis_all_sensors('\\vegard_og_niklas\\sensortest\\', position=-2, n_runs=1, plot_chirp_fft=True,savefig=True, file_format='png')
    # st.fft_analysis_all_sensors('\\vegard_og_niklas\\sensortest\\', position=-1, n_runs=1, plot_chirp_fft=True,savefig=True, file_format='png')
    #st.fft_analysis_all_sensors_compare(folder='\\vegard_og_niklas\\sensortest\\', n_runs=1, plot_chirp_fft=True,savefig=True, file_format='svg')
    #st.fft_analysis_all_sensors_compare(folder='\\vegard_og_niklas\\sensortest\\', n_runs=1, plot_chirp_fft=True,savefig=True, file_format='png')
    #st.fft_analysis_all_positions_compare(folder='\\vegard_og_niklas\\sensortest\\rot_clock_123\\', savefig=True, file_format='png')
    #wp.phase_plotting(df, chirp=custom_chirp, BANDWIDTH=[100,40000])
    phase15, freq15  = wp.phase_plotting(df, chirp=custom_chirp, use_recorded_chirp=True, BANDWIDTH=[100,40000], start_stops=[(146000,441200),(146000,442000)], save_fig=True, file_name='phase_plot_15cm.svg', file_format='svg')
    #phase10, freq10  = wp.phase_plotting(df1, chirp=custom_chirp, use_recorded_chirp=True, start_stops=[(95000,390500),(94000,390400)], BANDWIDTH=[100,40000], save_fig=True, file_name='phase_plot_10cm.svg', file_format='svg')
    #phase_vel = wp.phase_velocity(phase, freq, distance=0.15, plot=True)
    #wp.plot_velocities_2distance(phase15, freq15, 0.15, phase10, freq10,  0.1,  savefig=True, filename='phase_velocity_10_cm_15_cm.png', file_format='png')
    #wp.plot_velocities(phase1, freq1, 0.10, savefig=True, filename='phase_velocity_10cm.svg', file_format='svg')
    #wp.plot_results(df, chirp=custom_chirp, BANDWIDTH=[100,40000])
    #get_phase_and_vph_of_compressed_signal(df, custom_chirp, BANDWIDTH=[100,40000])
    #compressed_df = compress_chirp(df, custom_chirp)
    #compressed_df_no_pressurewave = cut_out_pulse_wave(compressed_df, start_stops=[(146000,441200),(146000,442000)])
    # phase, v_ph, freq = get_phase_and_vph_of_compressed_signal(
    #                                                         compressed_signal=compressed_df_no_pressurewave,
    #                                                         distance=0.15, 
    #                                                         threshold1=400, 
    #                                                         threshold2=800, 
    #                                                         bandwidth=[100,40000],
    #                                                         plot=True)