import scipy.signal as signal
from scipy import interpolate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_viz_files.visualise_data import compare_signals, plot_data_vs_noiseavg, plot_data, plot_fft, plot_2fft
from data_processing.noise import adaptive_filter_RLS, adaptive_filter_NLMS, noise_reduce_signal
<<<<<<< Updated upstream
from data_processing.find_propagation_speed import find_propagation_speed, find_propagation_speed_plot
=======
from data_processing.find_propagation_speed import find_propagation_speed, find_propagation_speed_plot, find_propagation_speed_with_delay
>>>>>>> Stashed changes
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

FREQ_START = 20000
FREQ_END = 40000

CHIRP_CHANNEL_NAMES = ['channel 1', 'channel 2', 'channel 3', 'chirp']
SIGNAL_CHANNEL_NAMES = ['channel 1', 'channel 2', 'channel 3', 'WAVE_GEN']

def main():
<<<<<<< Updated upstream
    signal_df = csv_to_df(file_folder='first_test_touch_passive_setup2', file_name='touch_test_passive_setup2_place_C3_center_v2')
    #plot_data(signal_df, crop=False)
    #speed = find_propagation_speed(signal_df, ch1='channel 3', ch2='channel 2', sr=SAMPLE_RATE, distance_between_sensors=np.sqrt(0.8))
    #print(speed)
=======
    signal_df = csv_to_df(file_folder='div_files',
                         file_name='prop_speed_fingertouch_hard_setup3_v1')
    #plot_data(np.abs(signal_df))
    corr = signal.correlate(signal_df['channel 1'], signal_df['channel 3'])
    print(len(signal_df['channel 1']))
    print(0.1/((len(signal_df['channel 1']) - np.argmax(corr))/SAMPLE_RATE))
    #t = np.linspace(-len(signal_df['channel 1']), len(signal_df['channel 1']) , len(signal_df['channel 1'])*2 -1)
    #plt.plot(t, corr)
    #plt.show()
    speed = find_propagation_speed(signal_df, 'channel 1', 'channel 3', SAMPLE_RATE)
    speed1 = find_propagation_speed_with_delay(signal_df, 'channel 1', 'channel 3')
    speed2 = find_propagation_speed_with_delay(signal_df, 'channel 1', 'channel 3', hilbert=False)
    print(f'speed calculated with correlation: {speed}')
    print(f'speed calculated with delay using hilbert: {speed1}')
    print(f'speed calculated with delay: {speed2}')
    #signal_df = csv_to_df(file_folder='first_test_touch_passive_setup2', file_name='touch_test_passive_setup2_place_A3_center_v2')
>>>>>>> Stashed changes
    #corr, chirp = transfer_function()
    peaks_ch1 = find_indices_of_peaks(signal_df['channel 1'])
    peaks_ch3 = find_indices_of_peaks(signal_df['channel 3'])
    freq, speeds, _ = find_propagation_speed_plot(signal_df, 10, 20000)
    plt.stem(freq, speeds)
    plt.ylabel(ylabel='Speed (m/s)')
    plt.xlabel(xlabel='Frequency (Hz)')
    plt.show()
    #corr_df = pd.Series(corr, name='Corr sig')
<<<<<<< Updated upstream
    #plot_2fft(corr_df, chirp)
    #corr_df = pd.Series(corr, name='Corr sig')
=======
>>>>>>> Stashed changes
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
