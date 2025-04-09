#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test module for FFT package STFT functionality.
"""

import unittest
import numpy as np
from fft_package.core import stft, signal_processing

class TestSTFT(unittest.TestCase):
    """Test cases for STFT functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sampling_rate = 1000.0  # Hz
        self.duration = 1.0  # second
        self.t = np.linspace(0, self.duration, 
                           int(self.duration * self.sampling_rate), 
                           endpoint=False)
        
        # Generate test signal with two frequency components
        self.freq1 = 10  # Hz
        self.freq2 = 50  # Hz
        self.signal = signal_processing.generate_signal(
            self.t, 
            frequencies=[self.freq1, self.freq2], 
            amplitudes=[1.0, 0.5]
        )
        
        # STFT parameters
        self.window_size = 256
        self.hop_length = 64
    
    def test_stft_output_shape(self):
        """Test STFT output dimensions."""
        times, freqs, stft_matrix = stft.compute_stft(
            self.signal,
            self.sampling_rate,
            self.window_size,
            self.hop_length
        )
        
        expected_num_windows = 1 + (len(self.signal) - self.window_size) // self.hop_length
        expected_num_freqs = self.window_size // 2 + 1
        
        self.assertEqual(len(times), expected_num_windows)
        self.assertEqual(len(freqs), expected_num_freqs)
        self.assertEqual(stft_matrix.shape, (expected_num_windows, expected_num_freqs))
    
    def test_stft_frequency_detection(self):
        """Test if STFT correctly identifies signal frequencies."""
        times, freqs, stft_matrix = stft.compute_stft(
            self.signal,
            self.sampling_rate,
            self.window_size,
            self.hop_length
        )
        
        spectrogram = stft.compute_spectrogram(times, freqs, stft_matrix)
        mean_power = np.mean(spectrogram, axis=0)
        
        # Find peaks in frequency spectrum
        peaks = []
        for i in range(1, len(mean_power)-1):
            if mean_power[i] > mean_power[i-1] and mean_power[i] > mean_power[i+1]:
                peaks.append(freqs[i])
        
        self.assertTrue(any(abs(peak - self.freq1) < 2 for peak in peaks),
                       f"Frequency {self.freq1} Hz not detected")
        self.assertTrue(any(abs(peak - self.freq2) < 2 for peak in peaks),
                       f"Frequency {self.freq2} Hz not detected")
    
    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        with self.assertRaises(ValueError):
            stft.compute_stft(self.signal, self.sampling_rate, 0, self.hop_length)
        
        with self.assertRaises(ValueError):
            stft.compute_stft(self.signal, self.sampling_rate, self.window_size, 0)
        
        with self.assertRaises(ValueError):
            stft.compute_stft(self.signal, self.sampling_rate, 
                            self.window_size, self.window_size + 1)
    
    def test_spectrogram_range(self):
        """Test spectrogram values are in valid range."""
        times, freqs, stft_matrix = stft.compute_stft(
            self.signal,
            self.sampling_rate,
            self.window_size,
            self.hop_length
        )
        
        spectrogram = stft.compute_spectrogram(times, freqs, stft_matrix)
        max_val = np.max(spectrogram)
        min_val = np.min(spectrogram)
        
        self.assertLessEqual(max_val - min_val, 60.0,
                           "Spectrogram dynamic range exceeds 60 dB")
        self.assertAlmostEqual(max_val, 0, places=5,
                             msg="Spectrogram not properly normalized")

if __name__ == '__main__':
    unittest.main()