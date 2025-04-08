#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Signal Processing Core Module.

This module provides core functionality for FFT-based signal processing,
including signal generation, FFT computation, and signal reconstruction.
"""

import numpy as np
from datetime import datetime

def generate_signal(t, frequencies, amplitudes):
    """
    Generate a composite signal from multiple sine waves.
    
    Args:
        t (numpy.ndarray): Time array.
        frequencies (list): List of frequencies in Hz for each component.
        amplitudes (list): List of amplitudes for each component.
    
    Returns:
        numpy.ndarray: The composite signal.
    """
    signal = np.zeros_like(t)
    for freq, amp in zip(frequencies, amplitudes):
        signal += amp * np.sin(2 * np.pi * freq * t)
    return signal

def perform_fft(signal, sampling_rate):
    """
    Perform Fast Fourier Transform on a signal.
    
    Args:
        signal (numpy.ndarray): Input signal to transform.
        sampling_rate (float): Sampling rate of the signal in Hz.
    
    Returns:
        tuple: Frequency array and FFT magnitude.
    """
    n = len(signal)
    fft_result = np.fft.fft(signal)
    magnitude = np.abs(fft_result) / n  # Normalize
    
    # For real signals, the FFT is symmetric, so we take only the first half
    magnitude = magnitude[:n//2]
    
    # Calculate corresponding frequencies
    freq = np.fft.fftfreq(n, d=1/sampling_rate)[:n//2]
    
    return freq, magnitude, fft_result

def perform_ifft(fft_result):
    """
    Perform Inverse Fast Fourier Transform.
    
    Args:
        fft_result (numpy.ndarray): FFT result to transform back.
    
    Returns:
        numpy.ndarray: Time-domain signal.
    """
    return np.fft.ifft(fft_result).real

def get_experiment_id(prefix="EXP", function_name=None):
    """
    Generate a unique experiment identifier with timestamp.
    
    Args:
        prefix (str): Prefix for the experiment ID.
        function_name (str, optional): Name of the function generating the ID.
    
    Returns:
        str: Formatted experiment ID.
    """
    date_str = datetime.now().strftime('%Y%m%d')
    if function_name:
        return f"{prefix}_{date_str}_{function_name}"
    return f"{prefix}_{date_str}"