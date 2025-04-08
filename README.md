# FFT Package

A comprehensive package for FFT analysis and visualization with experiment tracking.

## Features

- Signal generation with multiple frequency components
- Fast Fourier Transform (FFT) analysis
- Inverse Fast Fourier Transform (IFFT) reconstruction
- Automated experiment tracking with timestamps
- Visualization tools for time and frequency domain analysis
- Data export in CSV format

## Installation

```bash
pip install -e .
```

## Usage Example

```python
import numpy as np
from fft_package import (
    generate_signal,
    perform_fft,
    perform_ifft,
    visualize_results,
    save_results
)

# Generate test signal
sampling_rate = 1000.0  # Hz
duration = 1.0  # second
t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
frequencies = [5, 50, 120]  # Hz
amplitudes = [1.0, 0.5, 0.3]

# Generate and analyze signal
signal = generate_signal(t, frequencies, amplitudes)
freq, magnitude, fft_result = perform_fft(signal, sampling_rate)
reconstructed = perform_ifft(fft_result)

# Visualize and save results
vis_path = visualize_results(t, signal, reconstructed, freq, magnitude)
data_path = save_results(t, signal, signal, reconstructed)
```

## Testing

To run the automated tests:

```bash
python -m unittest discover tests
```