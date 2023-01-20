import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


def figure_size_setup(overleaf_size=0.75):
    sb.set(font_scale=12/10)  # font size = 12pt / 10pt/scale = 1.2 times the default size

    # Calculate the column width in inches (assumes page size and margins as specified in the question)
    page_width_mm = 250
    left_margin_mm = 25
    right_margin_mm = 25
    column_width_inches = (page_width_mm - left_margin_mm - right_margin_mm) / 25.4
    # Set the figure height in inches
    figure_height_inches = 6
    # Calculate the figure width in inches as 0.75 of the column width
    
    figure_width_inches = column_width_inches * overleaf_size#0.75

    # Create the figure and set the size
    fig, ax = plt.subplots(figsize=(figure_width_inches, figure_height_inches))

    return fig, ax
#sb.set_theme(style="darkgrid")
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
fig, axs = figure_size_setup(0.33)
#plt.plot(t, s*10, label='Chirp')
axs.plot(t, y, label='Matched filter output')
axs.legend()
axs.set_xlabel('Time (s)')
axs.set_ylabel('Amplitude')
plt.show()
plt.clf()
fig, axs = figure_size_setup(0.33)
axs.plot(f_pos, 20*np.log10(S_pos), label='Chirp spectrum (dB)')
axs.plot(f_pos, 20*np.log10(Y_pos), label='Matched filter output spectrum (dB)')
axs.legend()
axs.set_xlabel('Frequency (Hz)')
axs.set_ylabel('Amplitude (dB)')

plt.show()