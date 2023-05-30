import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
# Generate input signal

# Generate input pulse signal
t = np.linspace(-10, 10, 1000)  # Time array
input_signal = np.exp(-t**2) * np.cos(10 * np.pi * t)  # Input pulse signal

# Apply dispersion
dispersion_factor = 0.05  # Factor controlling the dispersion effect
propagation_distance = 1.0  # Distance traveled by the pulse

output_signal = np.exp(-(t - propagation_distance)**2 / (1 + dispersion_factor * propagation_distance)**2) * np.cos(10 * np.pi * (t - propagation_distance) / (1 + dispersion_factor * propagation_distance))  # Output pulse signal with dispersion

# Plot the signals
fig, ax = plt.subplots()
ax.plot(t, input_signal, label='Input Pulse')
ax.plot(t, output_signal, label='Output Pulse')
ax.set_xlabel('Time')
ax.set_ylabel('Amplitude')
ax.legend()

# Adjust figure layout for better visualization
plt.tight_layout()

# Show the figure
plt.show()