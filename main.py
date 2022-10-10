import scipy.signal as signal
from pandas import DataFrame as df
from data_viz_files.visualise_data import compare_signals, plot_data_vs_noiseavg
from data_processing.noise import adaptive_filter_RLS, adaptive_filter_NLMS, noise_reduce_signal
from csv_to_df import csv_to_df
from data_processing.find_propagation_speed import find_propagation_speed
from data_processing.detect_echoes import find_indices_of_peaks, get_hilbert_envelope
from data_processing.preprocessing import crop_data, crop_data_threshold, hp_or_lp_filter


print('\n' + __file__ + '\n')

# Sample rate in Hz
SAMPLE_RATE = 150000

# Crop limits in seconds
TIME_START = 1.3
TIME_END = 2


def main():
    signal_df = csv_to_df(file_folder='first_test_touch_passive_setup2',
                          file_name='touch_test_passive_setup2_place_C3_center_v2')

    signal_df = crop_data(signal_df, time_start=TIME_START, time_end=TIME_END)

    signal_df_filtered = hp_or_lp_filter(signal_df, filtertype='highpass', cutoff=1000, order=8)
    # signal_df_filtered = hp_or_lp_filter(signal_df_filtered, filtertype='lowpass', cutoff=5000, order=8)
    # signal_df_filtered = df(get_hilbert_envelope(signal_df_filtered['channel 1'].values), columns=['channel 1'])
    signal_df_filtered = get_hilbert_envelope(signal_df_filtered['channel 1'].values)

    compare_signals(signal_df,
                    signal_df_filtered,
                    sample_rate=SAMPLE_RATE,
                    time_start=TIME_START,
                    time_end=TIME_END)

    # find_indices_of_peaks(signal_df_filtered, plot=True)


if __name__ == '__main__':
    main()
