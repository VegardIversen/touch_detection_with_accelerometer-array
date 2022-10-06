from data_viz_files.visualise_data import compare_signals
from data_processing.noise import adaptive_filter_RLS, adaptive_filter_NLMS, noise_reduce_signal
import pandas as pd
from csv_to_df import csv_to_df


print(__file__)

# Sample rate in Hz
SAMPLE_RATE = 150000

# Crop limits in seconds
TIME_START = 0
TIME_END = 5


if __name__ == '__main__':

    # plot_data_subtracted_noise(data_file)
    noise_df = csv_to_df(file_folder='base_data',
                         file_name='df_average_noise')

    df = csv_to_df(file_folder='first_test_touch_passive_setup2',
                   file_name='touch_test_passive_setup2_place_A1_center_v1')

    reduced_noise = noise_reduce_signal(df['channel 1'], noise_df['channel 1'])
    # y, e, w = adaptive_filter_NLMS(df['channel 1'])
    # print(reduced_noise.shape)
    compare_signals(df['channel 1'], pd.Series(reduced_noise, copy=False), save=True, filename='signal_and_reduced_nois_signal.png')
    # cell = FindTouchPosition(DATA_FOLDER + '\\first_test_touch_passive_setup2\\touch_test_passive_setup2_place_B3_center_v2.csv')
    # draw(cell)
