import scipy.signal as signal
from scipy import interpolate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_viz_files.visualise_data import compare_signals, plot_data_vs_noiseavg, plot_data, plot_fft, plot_2fft
from data_processing.noise import adaptive_filter_RLS, adaptive_filter_NLMS, noise_reduce_signal
from data_processing.find_propagation_speed import find_propagation_speed, find_propagation_speed_plot, find_propagation_speed_with_delay
from data_processing.detect_echoes import find_first_peak, find_indices_of_peaks, get_hilbert_envelope, get_expected_reflections_pos
from data_processing.preprocessing import crop_data, crop_data_threshold, filter_general, filter_notches
from data_processing.transfer_function import transfer_function
import matplotlib.pyplot as plt
from data_processing.signal_separation import signal_sep
from csv_to_df import csv_to_df
print('\n' + __file__ + '\n')

# Sample rate in Hz
SAMPLE_RATE = 150000

# Crop limits in seconds
TIME_START = 0
TIME_END = 0.006

FREQ_START = 1000
FREQ_END = 2000

CHIRP_CHANNEL_NAMES = ['channel 1', 'channel 2', 'channel 3', 'chirp']
SIGNAL_CHANNEL_NAMES = ['channel 1', 'channel 2', 'channel 3', 'WAVE_GEN']

def main():
    signal_touch_hold_df = csv_to_df(file_folder='div_files/setup3',
                         file_name='sign_integ_test_touchhold_fulldurr_150k_5s_setup3_1_v1', channel_names=SIGNAL_CHANNEL_NAMES)
    signal_touch_df = csv_to_df(file_folder='div_files/setup3',
                         file_name='sign_integ_test_touch_150k_5s_setup3_1_v1', channel_names=SIGNAL_CHANNEL_NAMES)
    signal_touch_short_df = csv_to_df(file_folder='div_files',
                         file_name='prop_speed_fingernail_hard_setup3_v3', channel_names=SIGNAL_CHANNEL_NAMES)
    signal_touch_hold_df_filtered = filter_general(signal_touch_hold_df.drop(labels=['WAVE_GEN'], axis=1), 'bandpass', cutoff_low=FREQ_START, cutoff_high=FREQ_END, fs=150000, order=5)
    signal_touch_df_filt = filter_general(signal_touch_df.drop(labels=['WAVE_GEN'], axis=1), 'bandpass', cutoff_low=FREQ_START, cutoff_high=FREQ_END, fs=150000, order=5)
    signal_touch_short_df_filt = filter_general(signal_touch_short_df.drop(labels=['WAVE_GEN'], axis=1), 'bandpass', cutoff_low=FREQ_START, cutoff_high=FREQ_END, fs=150000, order=5)
    compare_signals( signal_touch_df['channel 2'], signal_touch_df['channel 3'], df1_name='touch  ch2', df2_name='touch filt ch3')
    #plot_data(np.abs(signal_df))
    #corr = signal.correlate(signal_df['channel 1'], signal_df['channel 3'])
    #plot_data(signal_touch_short_df_filt)
    speed1 = find_propagation_speed_with_delay(signal_touch_df_filt, 'channel 1', 'channel 2', distance=0.267)
    speed2 = find_propagation_speed_with_delay(signal_touch_df_filt, 'channel 1', 'channel 3', distance=0.267*2)
    speed3 = find_propagation_speed_with_delay(signal_touch_df_filt, 'channel 2', 'channel 3', distance=0.267)
    speed1_CORR = find_propagation_speed(signal_touch_df_filt, 'channel 1', 'channel 2', distance=0.267)
    speed2_CORR = find_propagation_speed(signal_touch_df_filt, 'channel 1', 'channel 3', distance=0.267*2)
    speed3_CORR  = find_propagation_speed(signal_touch_df_filt, 'channel 2', 'channel 3', distance=0.267)
    speed_short = find_propagation_speed_with_delay(signal_touch_short_df_filt, 'channel 1', 'channel 3', distance=0.1)
    speed_short_corr = find_propagation_speed(signal_touch_short_df_filt, 'channel 1', 'channel 3', distance=0.1)
    print(f'filters with low cutoff {FREQ_START} Hz and high cutoff {FREQ_END} Hz')
    print(f'speed1: {speed1}, speed2: {speed2}, speed3: {speed3}')
    print(f'speed1_CORR: {speed1_CORR}, speed2_CORR: {speed2_CORR}, speed3_CORR: {speed3_CORR}')
    print(f'speed_short: {speed_short}, speed_short_corr: {speed_short_corr}')
    avg_speed = (speed1 + speed2 + speed3)/3
    avg_speed_CORR = (speed1_CORR + speed2_CORR + speed3_CORR)/3
    print(f'avg_speed: {avg_speed}, avg_speed_CORR: {avg_speed_CORR}')
    
    peak = find_first_peak(signal_touch_df_filt['channel 2'], 0.0005)
    print(f'peak: {peak}')
    s = [0.4695, 1.1385]
    lines = get_expected_reflections_pos(avg_speed, peak, s=s)
    lines.append(peak)
    vlines = ['s1','s2','peak']
    plt.plot(signal_touch_df_filt['channel 2'])
    for idx, s in enumerate(vlines):
        print(lines[idx])
        plt.vlines(lines[idx],0,0.02, label=s)
        plt.text(lines[idx], -0.05, s, color='black')
    plt.show()
    #print(len(signal_df['channel 1']))
    #print(0.1/((len(signal_df['channel 1']) - np.argmax(corr))/SAMPLE_RATE))
    #t = np.linspace(-len(signal_df['channel 1']), len(signal_df['channel 1']) , len(signal_df['channel 1'])*2 -1)
    #plt.plot(t, corr)
    #plt.show()
    #speed = find_propagation_speed(signal_df, 'channel 1', 'channel 3', SAMPLE_RATE)
    #speed1 = find_propagation_speed_with_delay(signal_df, 'channel 1', 'channel 3')
    #speed2 = find_propagation_speed_with_delay(signal_df, 'channel 1', 'channel 3', hilbert=False)
    #print(f'speed calculated with correlation: {speed}')
    #print(f'speed calculated with delay using hilbert: {speed1}')
    #print(f'speed calculated with delay: {speed2}')
    #signal_df = csv_to_df(file_folder='first_test_touch_passive_setup2', file_name='touch_test_passive_setup2_place_A3_center_v2')
    #corr, chirp = transfer_function()
    # filt_signal_df = filter_general(signal_df, 'bandpass', cutoff_low=1000, cutoff_high=2000, fs=150000, order=5)
    # ax = plt.subplot(211)
    # plt.title('signal un-filtered')
    # plt.plot(signal_df)
    # plt.subplot(212, sharex=ax, sharey=ax)
    # plt.title('signal filtered bandpass 1000-2000' )
    # plt.plot(filt_signal_df)
    # plt.show()
    # #peaks_ch1 = find_indices_of_peaks(signal_df['channel 1'])
    #peaks_ch3 = find_indices_of_peaks(signal_df['channel 3'])
    #freq, speeds, _ = find_propagation_speed_plot(signal_df, 10, 20000)
    #plt.stem(freq, speeds)
    #plt.ylabel(ylabel='Speed (m/s)')
    #plt.xlabel(xlabel='Frequency (Hz)')
    #plt.show()
    #corr_df = pd.Series(corr, name='Corr sig')
    # chirp, sens = signal_sep()
    # peak = find_first_peak(sens['channel 2'], 0.0005)
    # speed = find_propagation_speed(sens, 'channel 1', 'channel 2', 150000)
    # lines = get_expected_reflections_pos(speed, peak)
    # lines.append(peak)
    

    # print(type(lines))
    # vlines = ['s1','s2','s3','s4','peak']
    # #sens = pd.DataFrame(sens)
    # #print(sens.loc[(sens>0.006)])
    # #print(isinstance(sens, pd.DataFrame))
    # #crop = crop_data_threshold(sens)
    # #fix, axs = plt.subplot(111)
    # for idx, s in enumerate(vlines):
    #     print(lines[idx])
    #     plt.vlines(lines[idx],0,1, label=s)
    #     plt.text(lines[idx], -0.05, s, color='black')
        
    # #plt.vlines(n,0,1, label=vlines)
    # plt.plot(sens)
    # plt.legend()
    # plt.show()
    #plot_fft(sens)
    #plt.plot(corr_df.iloc[333100:333200])
    #plt.show()
    #plot_2fft(corr_df, chirp)
    #plot_fft(corr_df.iloc[333100:333200])
    #chirp_df = csv_to_df(file_folder='div_files',
    #                      file_name='chirp_test_fs_96000_t_max_2s_2000-20000hz_1vpp_1cyc_setup3_v2', channel_names=CHIRP_CHANNEL_NAMES)
    #signal_df = csv_to_df(file_folder='first_test_touch_passive_setup2', file_name='touch_test_passive_setup2_place_A3_center_v2')
    #signal_df = crop_data(signal_df, time_start=TIME_START, time_end=TIME_END)
    # signal_df = crop_data(signal_df, time_start=TIME_START, time_end=TIME_END)


if __name__ == '__main__':
    main()
