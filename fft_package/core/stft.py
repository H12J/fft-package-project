#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Short-Time Fourier Transform Module.

This module implements the Short-Time Fourier Transform (STFT) for
time-frequency analysis of signals.
"""

import numpy as np
from typing import Tuple, Optional
from .windowing import apply_window
from .signal_processing import get_experiment_id

def compute_stft(signal: np.ndarray,
                sampling_rate: float,
                window_size: int,
                hop_length: int,
                window_type: str = 'hamming') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Short-Time Fourier Transform of a signal.
    
    Args:
        signal (numpy.ndarray): Input signal to transform.
        sampling_rate (float): Sampling rate of the signal in Hz.
        window_size (int): Size of the window in samples.
        hop_length (int): Number of samples between successive windows.
        window_type (str, optional): Type of window to apply ('hamming', 'hanning', 'blackman').
            Defaults to 'hamming'.
    
    Returns:
        tuple:
            - numpy.ndarray: Time points for each window (in seconds)
            - numpy.ndarray: Frequency bins (in Hz)
            - numpy.ndarray: Complex STFT matrix (time x frequency)
    
    Raises:
        ValueError: If window_size or hop_length is invalid.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if hop_length <= 0:
        raise ValueError("hop_length must be positive")
    if hop_length > window_size:
        raise ValueError("hop_length must be less than or equal to window_size")

    # Calculate number of windows and frequencies
    num_samples = len(signal)
    num_windows = 1 + (num_samples - window_size) // hop_length
    
    # Pre-allocate the STFT matrix
    stft_matrix = np.zeros((num_windows, window_size // 2 + 1), dtype=complex)
    
    # Calculate frequency bins
    freqs = np.fft.rfftfreq(window_size, d=1/sampling_rate)
    
    # Calculate time points
    times = np.arange(num_windows) * hop_length / sampling_rate
    
    # Apply FFT to each window
    for i in range(num_windows):
        # Extract window
        start = i * hop_length
        end = start + window_size
        window = signal[start:end]
        
        # Apply window function
        windowed = apply_window(window, window_type)
        
        # Compute FFT
        stft_matrix[i] = np.fft.rfft(windowed)
    
    return times, freqs, stft_matrix

def compute_spectrogram(times: np.ndarray,
                       freqs: np.ndarray,
                       stft_matrix: np.ndarray,
                       db_range: float = 60.0) -> np.ndarray:
    """
    Compute the power spectrogram from STFT results.
    
    Args:
        times (numpy.ndarray): Time points from STFT.
        freqs (numpy.ndarray): Frequency bins from STFT.
        stft_matrix (numpy.ndarray): Complex STFT matrix.
        db_range (float, optional): Dynamic range for dB scaling.
            Defaults to 60.0.
    
    Returns:
        numpy.ndarray: Power spectrogram in dB, normalized to peak power.
    """
    # Compute power spectrogram
    power = np.abs(stft_matrix) ** 2
    
    # Convert to dB with dynamic range
    power_max = power.max()
    power_min = power_max * 10**(-db_range/10)
    
    # Avoid log of zero
    eps = np.finfo(float).eps
    power_db = 10 * np.log10(np.maximum(power, eps))
    
    # Normalize
    power_db_norm = np.maximum(power_db, power_db.max() - db_range)
    
    return power_db_norm