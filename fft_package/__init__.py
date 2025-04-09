"""
FFT Package

A comprehensive package for FFT analysis and visualization with experiment tracking.
"""

from .core.signal_processing import (
    generate_signal,
    perform_fft,
    perform_ifft,
    get_experiment_id
)

from .visualization.plotting import (
    visualize_results,
    save_results
)

__version__ = '0.1.0'