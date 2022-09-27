import numpy as np
import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
SAMPLE_RATE = 80000     # Hz

CROP_MODE = "Auto"      # Auto or Manual
CROP_BEFORE = 80000     # samples
CROP_AFTER = 120000     # samples

DATA_DELIMITER = ","

def crop_data(data):
    data_cropped = data.loc[(data>0.0006).any(1)]
    return data_cropped
#data_folder = Path(f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave'+'\\first_test_touch_passive_setup2').rglob('*.csv')
data_folder = f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave'
#for x in os.listdir(data_folder):
#    print(x)

test_file = data_folder + '\\first_test_touch_passive_setup2\\touch_test_passive_setup2_place_C2_center_v2.csv'
A1_file_path = data_folder + '\\first_test_touch_passive_setup2\\touch_test_passive_setup2_place_A1_center_v1.csv'
A2_file_path = data_folder + '\\first_test_touch_passive_setup2\\touch_test_passive_setup2_place_A2_center_v1.csv'
A3_file_path = data_folder + '\\first_test_touch_passive_setup2\\touch_test_passive_setup2_place_A3_center_v1.csv'
B1_file_path = data_folder + '\\first_test_touch_passive_setup2\\touch_test_passive_setup2_place_B1_center_v1.csv'
B2_file_path = data_folder + '\\first_test_touch_passive_setup2\\touch_test_passive_setup2_place_B2_center_v1.csv'
B3_file_path = data_folder + '\\first_test_touch_passive_setup2\\touch_test_passive_setup2_place_B3_center_v1.csv'
C1_file_path = data_folder + '\\first_test_touch_passive_setup2\\touch_test_passive_setup2_place_C1_center_v1.csv'
C2_file_path = data_folder + '\\first_test_touch_passive_setup2\\touch_test_passive_setup2_place_C2_center_v1.csv'
C3_file_path = data_folder + '\\first_test_touch_passive_setup2\\touch_test_passive_setup2_place_C3_center_v1.csv'
#C:\Users\vegar\OneDrive - NTNU\NTNU\ProsjektOppgave\first_test_touch_passive_setup2
print(A1_file_path)
test_signal = pd.read_csv(test_file, delimiter=DATA_DELIMITER, names=['channel 1', 'channel 2', 'channel 3']) 
A1 = pd.read_csv(A1_file_path, delimiter=DATA_DELIMITER, names=['channel 1', 'channel 2', 'channel 3'])
A2 = pd.read_csv(A2_file_path, delimiter=DATA_DELIMITER, names=['channel 1', 'channel 2', 'channel 3'])
A3 = pd.read_csv(A3_file_path, delimiter=DATA_DELIMITER, names=['channel 1', 'channel 2', 'channel 3'])
B1 = pd.read_csv(B1_file_path, delimiter=DATA_DELIMITER, names=['channel 1', 'channel 2', 'channel 3'])
B2 = pd.read_csv(B2_file_path, delimiter=DATA_DELIMITER, names=['channel 1', 'channel 2', 'channel 3'])
B3 = pd.read_csv(B3_file_path, delimiter=DATA_DELIMITER, names=['channel 1', 'channel 2', 'channel 3'])
C1 = pd.read_csv(C1_file_path, delimiter=DATA_DELIMITER, names=['channel 1', 'channel 2', 'channel 3'])
C2 = pd.read_csv(C2_file_path, delimiter=DATA_DELIMITER, names=['channel 1', 'channel 2', 'channel 3'])
C3 = pd.read_csv(C3_file_path, delimiter=DATA_DELIMITER, names=['channel 1', 'channel 2', 'channel 3'])

data_frames = [A1, A2, A3, B1, B2, B3, C1, C2, C3]
A1_cropped = crop_data(A1)
A2_cropped = crop_data(A2)
A3_cropped = crop_data(A3)
B1_cropped = crop_data(B1)
B2_cropped = crop_data(B2)
B3_cropped = crop_data(B3)
C1_cropped = crop_data(C1)
C2_cropped = crop_data(C2)
C3_cropped = crop_data(C3)
test_cropped = crop_data(test_signal)
#test_cropped.plot()
#plt.show()
#plt.clf()
#A1_cropped.plot()
#plt.show()
#plt.clf()
x1 = np.correlate(C3_cropped['channel 1'], test_cropped['channel 1'], mode='valid')
print(np.max(x1))
# x2 = np.correlate(A2, test_file)
# x3 = np.correlate(A3, test_file)
# x4 = np.correlate(B1, test_file)
# x5 = np.correlate(B2, test_file)
# x6 = np.correlate(B3, test_file)
# x7 = np.correlate(C1, test_file)
# x8 = np.correlate(C2, test_file)
# x9 = np.correlate(C3, test_file)
x_ax = np.linspace(-len(x1)//2, len(x1)//2, num=len(x1))
#plt.plot(x_ax, x1)
#A1.plot()
#plt.show()

