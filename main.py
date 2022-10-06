from data_processing.cross_correlation_position import FindTouchPosition
from data_viz_files.display_table_grid import draw
from data_viz_files.visualise_data import compare_signals
from data_processing.noise import adaptive_filter_RLS, adaptive_filter_NLMS, noise_reduce_signal
from pathlib import Path
import pandas as pd


print(__file__)

DATA_FOLDER = f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave'
DATA_DELIMITER = ","
CHANNEL_NAMES = ['channel 1', 'channel 2', 'channel 3']
SAMPLE_RATE = 150000     # Hz

# Crop limits in seconds
TIME_START = 0
TIME_END = 5


if __name__ == '__main__':
    SAMPLE_RATE = 150000     # Hz

    # Crop limits in seconds
    TIME_START = 0
    TIME_END = 5

    DATA_DELIMITER = ","
    noise_path = f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave\\base_data\\df_average_noise.csv'
    data_folder = f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave'
    test_file1 = data_folder + ('\\holdfinger_test_active_setup2_5\\'
                                'hold_test_B1_setup2_5_sinus_2khz_10vpp_cyclcount_1_burstp_1s_v1.csv')
    test_file2 = data_folder + ('\\holdfinger_test_active_setup2_5'
                                '\\hold_test_B1_setup2_5_sinus_2khz_10vpp_cyclcount_1_burstp_1s_v2.csv')
    data_file = data_folder + '\\first_test_touch_passive_setup2\\touch_test_passive_setup2_place_A1_center_v1.csv'
    # plot_data_subtracted_noise(data_file)
    noise = pd.read_csv(noise_path, delimiter=DATA_DELIMITER)
    df = pd.read_csv(data_file, delimiter=DATA_DELIMITER, names=CHANNEL_NAMES)
    reduced_noise = noise_reduce_signal(df['channel 1'], noise['channel 1'])
    # y, e, w = adaptive_filter_NLMS(df['channel 1'])
    # print(reduced_noise.shape)
    compare_signals(df['channel 1'], pd.Series(reduced_noise, copy=False), save=True, filename='signal_and_reduced_nois_signal.png')
    # cell = FindTouchPosition(DATA_FOLDER + '\\first_test_touch_passive_setup2\\touch_test_passive_setup2_place_B3_center_v2.csv')
    # draw(cell)
