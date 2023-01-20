import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.signal import correlate

sb.set_theme(style="darkgrid")
np.random.seed(0)
# Set the duration of the touch
touch_duration = 0.1

# Set the sampling rate
sampling_rate = 1000

# Set the time axis
time = np.linspace(0, touch_duration, int(touch_duration*sampling_rate))

# Set the noise level at the start and end of the signal
noise_level = 0.1

# Generate the signal for the cell

# Add a touch signal in the middle of the data
touch_start = int(len(time) * 0.25)
touch_end = int(len(time) * 0.75)

# Create a figure with 3 rows and 3 columns of subplots
#fig, ax = plt.subplots(1, 1)
count = 0
# Plot the signal in each subplot
for i in range(3):
    for j in range(3):
        # Add some variation to the signal for each subplot
        signal = np.zeros(len(time))

        # Add a touch signal in the middle of the data
        signal[touch_start:touch_end] = 5 * np.exp(-((time[touch_start:touch_end]-(np.random.uniform(0.2,0.8))*touch_duration)/(0.1*touch_duration))**2)
        # Generate a function that increases from 0 to 1 over the duration of the touch signal
        x = np.linspace(0, 1, len(time))
        y = np.interp(np.linspace(0, 1, len(time)), np.linspace(0, 1, len(time)), x)

        # Multiply the oscillations by the function
        oscillations = y * np.sin(2 * np.pi * 100 * time)

        # Add additional oscillations with a different frequency
        oscillations += y * np.sin(2 * np.pi * 200 * time)

        # Add the oscillations to the signal
        signal += oscillations

        # Add noise at the start and end of the signal
        signal[:int(noise_level*len(signal))] += np.random.rand(int(noise_level*len(signal))) * noise_level
        signal[-int(noise_level*len(signal)):] += np.random.rand(int(noise_level*len(signal))) * noise_level
        modified_signal = signal * (i + j + 100) * np.random.uniform(1, 100)*10
        normalized_signal = modified_signal / np.max(modified_signal)
        if i == 0 and j == 0:
            new_sig = normalized_signal
            corr_data = correlate(new_sig, normalized_signal, mode='full')
            max_corr = np.max(corr_data)
            corr_data_norm = corr_data*0.95 / max_corr
        else:
            corr_data = correlate(new_sig, normalized_signal, mode='full')*0.8
            corr_data_norm = corr_data / max_corr
        #x_data for correlation
        x_data = np.linspace(-touch_duration/2, touch_duration/2, len(corr_data))

        #ax[i, j].plot(time, normalized_signal)
        plt.plot(x_data, corr_data_norm, label=f' ({count})')
        plt.ylabel('Correlation coefficient')
        plt.xlabel('Lag')
        # Add y-axis labels to the left column and bottom row
        # if i == 2:
        #     ax[i, j].set_xlabel("Time (s)")
        # else:
        #     ax[i, j].set_xticklabels([])
        # if j == 0:
        #     ax[i, j].set_ylabel("Amplitude")
        # else:
        #     ax[i, j].set_yticklabels([])
        # ax[i, j].set_title(f'({count})')
        count += 1
        

# Show the figure
#plt.title('Cross correaltion between a new signal and signals from database')
plt.legend()
plt.savefig('correlation_technique_res.svg', dpi=300,format='svg')
plt.savefig('correlation_technique_res.png', dpi=300,format='png')
plt.show()