from data_viz_files.visualise_data import compare_signals, plot_data_vs_noiseavg
from data_processing.noise import adaptive_filter_RLS, adaptive_filter_NLMS, noise_reduce_signal
from csv_to_df import csv_to_df
from data_processing.find_propagation_speed import find_propagation_speed
from data_processing.detect_echoes import find_indices_of_peaks
from data_processing.preprocessing import crop_data


print('\n' + __file__ + '\n')

# Sample rate in Hz
SAMPLE_RATE = 150000

# Crop limits in seconds
TIME_START = 1.46
TIME_END = 2


def main():
    signal_df = csv_to_df(file_folder='first_test_touch_passive_setup2',
                          file_name='touch_test_passive_setup2_place_C3_center_v2')

    signal_df = crop_data(signal_df, time_start=TIME_START, time_end=TIME_END)

    find_indices_of_peaks(signal_df['channel 1'], plot=True)


if __name__ == '__main__':
    main()
