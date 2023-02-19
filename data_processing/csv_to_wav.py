import pandas as pd
import numpy 
from scipy.io import wavfile
from pathlib import Path


data_folder = f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave'
output = data_folder + '\\noise_test_veg_gudsong_5s_samplerate_150khz.wav'
file_path = data_folder + '\\Measurements\\div_files\\noise_test_veg_gudsong_5s_samplerate_150khz.csv'
A1 = pd.read_csv(file_path, delimiter=',', names=['channel 1', 'channel 2', 'channel 3'])
data = A1.to_numpy()
SAMPLE_RATE = 150000

print(f'saved at: {output}')
wavfile.write(filename=output, data=data, rate=SAMPLE_RATE)
