import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.signal import correlate
from pathlib import Path
import scipy
sb.set_theme(style="darkgrid")

def manual_cut_signal(signal):
    print('Find start of cut')
    plt.plot(signal)
    plt.show()
    start = int(input('Start: '))
    print('Find end of cut')
    plt.plot(signal)
    plt.show()
    end = int(input('End: '))
    print(f'Start: {start}, End: {end}')
    return start, end

SAMPLERATE = 150000
path = f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave\\setup9_propagation_speed_short\\touch\\touch_v1.csv'
path_chirp = f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave\\setup9_propagation_speed_short\\chirp\\100Hz_to_40kHz_single_chirp\\chirp_v1.csv'
data = pd.read_csv(path, names=['channel 1', 'channel 2', 'channel 3', 'wave_gen'])
data_chirp = pd.read_csv(path_chirp, names=['channel 1', 'channel 2', 'channel 3', 'wave_gen'])
time_ax = np.linspace(0, len(data)/SAMPLERATE, len(data))
data['time'] = time_ax
data_chirp['time'] = time_ax
# plt.plot(data['time'], data['channel 1'])
# plt.ylabel(ylabel='Amplitude')
# plt.xlabel(xlabel='Time [s]')
# plt.show()
#start1, end1 = manual_cut_signal(data['channel 1']) #touchv1 = 316004, 316856
#start2, end2 = manual_cut_signal(data['channel 3']) #touchv1 = 3160030, 316883
plt.plot(data_chirp['time'], data_chirp['channel 1'])
plt.ylabel(ylabel='Amplitude')
plt.xlabel(xlabel='Time [s]')
plt.show()
plt.plot(data_chirp['time'], data_chirp['wave_gen'])
plt.ylabel(ylabel='Amplitude')
plt.xlabel(xlabel='Time [s]')
plt.show()
fig, ax = plt.subplots(1, 2)
compressed_signal1 = correlate(data_chirp['channel 1'].to_numpy(), data_chirp['wave_gen'].to_numpy(), mode='same')
compressed_signal2 = correlate(data_chirp['channel 3'].to_numpy(), data_chirp['wave_gen'].to_numpy(), mode='same')


ax[0].plot(data_chirp['time'], compressed_signal1)
ax[0].set_ylabel('Amplitude')
ax[0].set_xlabel('Time [s]')
ax[0].set_title('Compressed signal channel 1')
ax[1].plot(data_chirp['time'], compressed_signal2)
ax[1].set_ylabel('Amplitude')
ax[1].set_xlabel('Time [s]')
ax[1].set_title('Compressed signal channel 3')
plt.show()
#fft of compressed signal
fft_compressed_signal1 = scipy.fft.fft(compressed_signal1)
fft_compressed_signal2 = scipy.fft.fft(compressed_signal2)
fft_chirp = scipy.fft.fft(data_chirp['wave_gen'].to_numpy())
fft_freq_chirp = scipy.fft.fftfreq(len(data_chirp['wave_gen']), 1/SAMPLERATE)
fft_chirp = np.fft.fftshift(fft_chirp)
fft_freq_chirp = np.fft.fftshift(fft_freq_chirp)
fft_freq_compressed_signal1 = scipy.fft.fftfreq(len(compressed_signal1), 1/SAMPLERATE)
fft_freq_compressed_signal2 = scipy.fft.fftfreq(len(compressed_signal2), 1/SAMPLERATE)
fft_compressed_signal1 = np.fft.fftshift(fft_compressed_signal1)
fft_compressed_signal2 = np.fft.fftshift(fft_compressed_signal2)
fft_freq_compressed_signal1 = np.fft.fftshift(fft_freq_compressed_signal1)
fft_freq_compressed_signal2 = np.fft.fftshift(fft_freq_compressed_signal2)

fig, ax = plt.subplots(1, 2)
ax[0].plot(fft_freq_compressed_signal1[fft_freq_compressed_signal1>0], 20*np.log10(np.abs(fft_compressed_signal1[fft_freq_compressed_signal1>0])))
ax[0].set_ylabel('Amplitude [dB]')
ax[0].set_xlabel('Frequency [Hz]')
ax[0].set_title('FFT of compressed signal channel 1')
#for chirp
ax[1].plot(fft_freq_chirp[fft_freq_chirp>0], 20*np.log10(np.abs(fft_chirp[fft_freq_chirp>0])))
ax[1].set_ylabel('Amplitude [dB]')
ax[1].set_xlabel('Frequency [Hz]')
ax[1].set_title('FFT of chirp')
plt.show()

exit()
cut_touch1 = data['channel 1'].iloc[316004:316856]
cut_touch2 = data['channel 3'].iloc[316030:316883]
window1 = np.ones(len(cut_touch1))
window2 = np.ones(len(cut_touch2))
cut_touch1 = cut_touch1 * window1
cut_touch2 = cut_touch2 * window2
#plt.plot(data['time'].iloc[316004:316856], cut_touch1)
#plt.plot(data['time'].iloc[316030:316883], cut_touch2)
fft_touch_full1 = scipy.fft.fft(data['channel 1'].to_numpy())
fft_touch_full2 = scipy.fft.fft(data['channel 3'].to_numpy())
fft_freq_full1 = scipy.fft.fftfreq(len(data['channel 1']), 1/SAMPLERATE)
fft_freq_full2 = scipy.fft.fftfreq(len(data['channel 3']), 1/SAMPLERATE)
fft_touch_full1 = np.fft.fftshift(fft_touch_full1)
fft_touch_full2 = np.fft.fftshift(fft_touch_full2)
fft_freq_full1 = np.fft.fftshift(fft_freq_full1)
fft_freq_full2 = np.fft.fftshift(fft_freq_full2)

corr_1 = correlate(data['channel 1'], cut_touch1, mode='same')
corr_2 = correlate(data['channel 3'], cut_touch2, mode='same')

fft_corr1 = scipy.fft.fft(corr_1)
fft_corr2 = scipy.fft.fft(corr_2)
fft_freq_corr1 = scipy.fft.fftfreq(len(corr_1), 1/SAMPLERATE)
fft_freq_corr2 = scipy.fft.fftfreq(len(corr_2), 1/SAMPLERATE)
fft_corr1 = np.fft.fftshift(fft_corr1)
fft_corr2 = np.fft.fftshift(fft_corr2)
fft_freq_corr1 = np.fft.fftshift(fft_freq_corr1)
fft_freq_corr2 = np.fft.fftshift(fft_freq_corr2)

fig, ax = plt.subplots(1, 2)
ax[0].plot(fft_freq_full1[fft_freq_full1>0], 20*np.log10(np.abs(fft_touch_full1[fft_freq_full1>0])))
ax[0].set_ylabel('Amplitude [dB]')
ax[0].set_xlabel('Frequency [Hz]')
ax[0].set_title('Touch signal FFT')
ax[1].plot(fft_freq_corr1[fft_freq_corr1>0], 20*np.log10(np.abs(fft_corr1[fft_freq_corr1>0])))
ax[1].set_ylabel('Amplitude [dB]')
ax[1].set_xlabel('Frequency [Hz]')
ax[1].set_title('Pulse compression FFT')
plt.show()

#plt.plot(data['time'], corr_2)
#plt.ylabel(ylabel='Amplitude')
#plt.xlabel(xlabel='Time [s]')
#plt.plot(data['time'], data['channel 2'])
#plt.plot(data['time'], data['channel 3'])
#plt.plot(data['time'], data['wave_gen'])
plt.show()
