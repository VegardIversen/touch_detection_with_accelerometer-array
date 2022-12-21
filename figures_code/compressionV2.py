import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
sb.set_theme(style="darkgrid")
# Set the simulation parameters
fs = 1000  # Sample rate (Hz)
T = 1  # Duration of the chirp (s)
f0 = 100  # Starting frequency of the chirp (Hz)
f1 = 200  # Ending frequency of the chirp (Hz)

# Generate the chirp signal
t = np.linspace(0, T, T*fs, endpoint=False)
s = np.cos(2*np.pi*(f0*t + (f1-f0)*t*t/(2*T)))

# Compute the chirp's spectrum
S = np.abs(np.fft.fft(s))
f = np.fft.fftfreq(s.size, 1/fs)

# Compute the matched filter
h = np.conj(s[::-1])

# Convolve the chirp with the matched filter
y = np.convolve(s, h, mode='same')

# Compute the spectrum of the matched filter output
Y = np.abs(np.fft.fft(y))

# Select only the positive frequencies
f_pos = f[f >= 0]
S_pos = S[f >= 0]
Y_pos = Y[f >= 0]

# Plot the results
plt.figure()
#plt.plot(t, s*10, label='Chirp')
plt.plot(t, y, label='Matched filter output')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.figure()
plt.plot(f_pos, 20*np.log10(S_pos), label='Chirp spectrum (dB)')
plt.plot(f_pos, 20*np.log10(Y_pos), label='Matched filter output spectrum (dB)')
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')

plt.show()