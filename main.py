import scipy.signal as signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_viz_files.visualise_data import compare_signals, plot_data_vs_noiseavg, plot_data, plot_fft, plot_2fft
from data_processing.noise import adaptive_filter_RLS, adaptive_filter_NLMS, noise_reduce_signal
from data_processing.find_propagation_speed import find_propagation_speed, find_propagation_speed_corr
from data_processing.detect_echoes import find_first_peak, find_indices_of_peaks, get_hilbert_envelope, get_expected_reflections_pos
from data_processing.preprocessing import crop_data, crop_data_threshold, hp_or_lp_filter
from data_processing.transfer_function import transfer_function
import matplotlib.pyplot as plt
from data_processing.signal_separation import signal_sep
print('\n' + __file__ + '\n')

# Sample rate in Hz
SAMPLE_RATE = 150000

# Crop limits in seconds
TIME_START = 0
TIME_END = 5

FREQ_START = 20000
FREQ_END = 60000

CHIRP_CHANNEL_NAMES = ['channel 1', 'channel 2', 'channel 3', 'chirp']


def main():
    #corr, chirp = transfer_function()
    #corr_df = pd.Series(corr, name='Corr sig')
    chirp, sens = signal_sep()
    peak = find_first_peak(sens['channel 2'], 0.0005)
    speed = find_propagation_speed(sens, 'channel 1', 'channel 2', 150000)
    n = get_expected_reflections_pos(speed, peak)
    n = n.append(peak)
    print(type(peak))
    vlines = ['s1','s2','s3','s4','peak']
    #sens = pd.DataFrame(sens)
    #print(sens.loc[(sens>0.006)])
    #print(isinstance(sens, pd.DataFrame))
    #crop = crop_data_threshold(sens)
    for idx, s in enumerate(vlines):
        print(lines[idx])
        plt.vlines(lines[idx],0,1, label=s)
    #plt.vlines(n,0,1, label=vlines)
    plt.plot(sens)
    plt.legend()
    plt.show()
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
