#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Signal Windowing Module.

This module provides various window functions for spectral analysis,
including Hamming, Hanning, and other commonly used windows.
"""

import numpy as np
from ..core.signal_processing import get_experiment_id

def hamming_window(N):
    """
    Generate a Hamming window.
    
    Args:
        N (int): Length of the window.
    
    Returns:
        numpy.ndarray: Hamming window array.
    """
    n = np.arange(N)
    return 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))

def hanning_window(N):
    """
    Generate a Hanning window.
    
    Args:
        N (int): Length of the window.
    
    Returns:
        numpy.ndarray: Hanning window array.
    """
    n = np.arange(N)
    return 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))

def blackman_window(N):
    """
    Generate a Blackman window.
    
    Args:
        N (int): Length of the window.
    
    Returns:
        numpy.ndarray: Blackman window array.
    """
    n = np.arange(N)
    return (0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 
            0.08 * np.cos(4 * np.pi * n / (N - 1)))

def apply_window(signal, window_type='hamming'):
    """
    Apply a window function to a signal.
    
    Args:
        signal (numpy.ndarray): Input signal to window.
        window_type (str): Type of window to apply ('hamming', 'hanning', or 'blackman').
    
    Returns:
        numpy.ndarray: Windowed signal.
    
    Raises:
        ValueError: If window_type is not recognized.
    """
    N = len(signal)
    
    if window_type == 'hamming':
        window = hamming_window(N)
    elif window_type == 'hanning':
        window = hanning_window(N)
    elif window_type == 'blackman':
        window = blackman_window(N)
    else:
        raise ValueError(f"Unsupported window type: {window_type}")
    
    return signal * window