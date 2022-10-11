import scipy.signal as signal
import pandas as pd
from data_viz_files.visualise_data import compare_signals, plot_data_vs_noiseavg, plot_data, plot_fft, plot_2fft
from data_processing.noise import adaptive_filter_RLS, adaptive_filter_NLMS, noise_reduce_signal
from csv_to_df import csv_to_df
from data_processing.find_propagation_speed import find_propagation_speed
from data_processing.detect_echoes import find_indices_of_peaks, get_hilbert_envelope
from data_processing.preprocessing import crop_data, crop_data_threshold, hp_or_lp_filter
from data_processing.transfer_function import transfer_function
import matplotlib.pyplot as plt
print('\n' + __file__ + '\n')

# Sample rate in Hz
SAMPLE_RATE = 150000

# Crop limits in seconds
TIME_START = 1.3
TIME_END = 2

CHIRP_CHANNEL_NAMES = ['channel 1', 'channel 2', 'channel 3', 'chirp']
def main():
    corr, chirp = transfer_function()
    corr_df = pd.Series(corr, name='Corr sig')
    plt.plot(corr_df.iloc[333100:333200])
    plt.show()
    #plot_2fft(corr_df, chirp)
    plot_fft(corr_df.iloc[333100:333200])
    #chirp_df = csv_to_df(file_folder='div_files',
    #                      file_name='chirp_test_fs_96000_t_max_2s_2000-20000hz_1vpp_1cyc_setup3_v2', channel_names=CHIRP_CHANNEL_NAMES)
    #signal_df = csv_to_df(file_folder='first_test_touch_passive_setup2', file_name='touch_test_passive_setup2_place_A3_center_v2')
    #signal_df = crop_data(signal_df, time_start=TIME_START, time_end=TIME_END)
    # signal_df = crop_data(signal_df, time_start=TIME_START, time_end=TIME_END)

    # signal_df_filtered = hp_or_lp_filter(signal_df, filtertype='highpass', cutoff=1000, order=8)
    # # signal_df_filtered = hp_or_lp_filter(signal_df_filtered, filtertype='lowpass', cutoff=5000, order=8)
    # # signal_df_filtered = df(get_hilbert_envelope(signal_df_filtered['channel 1'].values), columns=['channel 1'])
    # signal_df_filtered = get_hilbert_envelope(signal_df_filtered['channel 1'].values)

    # compare_signals(signal_df,
    #                 signal_df_filtered,
    #                 sample_rate=SAMPLE_RATE,
    #                 time_start=TIME_START,
    #                 time_end=TIME_END)

    # find_indices_of_peaks(signal_df_filtered, plot=True)

    #compare_signals(signal_df['channel 1'], signal_df['channel 1'], sample_rate=SAMPLE_RATE, time_start=TIME_START, time_end=TIME_END)
    #find_propagation_speed(chirp_df, sr=SAMPLE_RATE)

if __name__ == '__main__':
    main()
