#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FFT Visualization Module.

This module provides visualization functionality for FFT analysis results,
including time-domain signals and frequency spectra.
"""

import matplotlib.pyplot as plt
import os
import numpy as np
from ..core.signal_processing import get_experiment_id

def visualize_results(t, original_signal, reconstructed_signal, freq, magnitude, experiment_id=None):
    """
    Visualize the original signal, FFT results, and reconstructed signal.
    
    Args:
        t (numpy.ndarray): Time array.
        original_signal (numpy.ndarray): Original time-domain signal.
        reconstructed_signal (numpy.ndarray): Reconstructed signal after IFFT.
        freq (numpy.ndarray): Frequency array.
        magnitude (numpy.ndarray): FFT magnitude.
        experiment_id (str, optional): Identifier for the experiment.
            If None, one will be generated automatically.
    
    Returns:
        str: Path to the saved visualization file.
    """
    if experiment_id is None:
        experiment_id = get_experiment_id(function_name="visualize_results")

    fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot original signal
    axs[0].plot(t, original_signal)
    axs[0].set_title(f'Original Signal - {experiment_id}')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Amplitude')
    axs[0].grid(True)
    
    # Plot frequency spectrum
    axs[1].stem(freq, magnitude)
    axs[1].set_title(f'Frequency Spectrum - {experiment_id}')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Magnitude')
    axs[1].grid(True)
    
    # Plot reconstructed signal
    axs[2].plot(t, reconstructed_signal)
    axs[2].set_title(f'Reconstructed Signal - {experiment_id}')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Amplitude')
    axs[2].grid(True)
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Save the figure with experiment ID in the filename
    output_path = f'output/fft_visualization_{experiment_id}.png'
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def save_results(t, original_signal, noisy_signal, reconstructed_signal, experiment_id=None):
    """
    Save signal data to CSV file.
    
    Args:
        t (numpy.ndarray): Time array.
        original_signal (numpy.ndarray): Original clean signal.
        noisy_signal (numpy.ndarray): Signal with added noise.
        reconstructed_signal (numpy.ndarray): Reconstructed signal after FFT/IFFT.
        experiment_id (str, optional): Identifier for the experiment.
            If None, one will be generated automatically.
    
    Returns:
        str: Path to the saved data file.
    """
    if experiment_id is None:
        experiment_id = get_experiment_id(function_name="save_results")
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Save results to CSV
    results = np.column_stack((t, original_signal, noisy_signal, reconstructed_signal))
    header = "time,original_signal,noisy_signal,reconstructed_signal"
    output_path = f'output/fft_data_{experiment_id}.csv'
    np.savetxt(output_path, results, delimiter=',', header=header)
    
    return output_path