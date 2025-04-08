#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test module for FFT package signal processing functionality.
"""

import unittest
import numpy as np
from fft_package.core import signal_processing

class TestSignalProcessing(unittest.TestCase):
    """Test cases for signal processing functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sampling_rate = 1000.0  # Hz
        self.duration = 1.0  # second
        self.t = np.linspace(0, self.duration, 
                           int(self.duration * self.sampling_rate), 
                           endpoint=False)
        self.test_freq = 10  # Hz
        self.test_amp = 1.0
    
    def test_generate_signal(self):
        """Test signal generation with single frequency."""
        signal = signal_processing.generate_signal(
            self.t, 
            frequencies=[self.test_freq], 
            amplitudes=[self.test_amp]
        )
        
        # Check signal length
        self.assertEqual(len(signal), len(self.t))
        
        # Check signal amplitude
        self.assertAlmostEqual(np.max(signal), self.test_amp, places=1)
    
    def test_fft_single_frequency(self):
        """Test FFT with single frequency signal."""
        # Generate test signal
        signal = signal_processing.generate_signal(
            self.t, 
            frequencies=[self.test_freq], 
            amplitudes=[self.test_amp]
        )
        
        # Perform FFT
        freq, mag, _ = signal_processing.perform_fft(signal, self.sampling_rate)
        
        # Find frequency with maximum magnitude
        peak_freq = freq[np.argmax(mag)]
        
        # Check if the detected frequency matches the input
        self.assertAlmostEqual(peak_freq, self.test_freq, places=1)
    
    def test_ifft_reconstruction(self):
        """Test signal reconstruction using IFFT."""
        # Generate original signal
        original = signal_processing.generate_signal(
            self.t, 
            frequencies=[self.test_freq], 
            amplitudes=[self.test_amp]
        )
        
        # Perform FFT
        _, _, fft_result = signal_processing.perform_fft(original, self.sampling_rate)
        
        # Reconstruct signal using IFFT
        reconstructed = signal_processing.perform_ifft(fft_result)
        
        # Check if reconstructed signal matches original
        np.testing.assert_array_almost_equal(original, reconstructed, decimal=10)
    
    def test_experiment_id_format(self):
        """Test experiment ID generation."""
        exp_id = signal_processing.get_experiment_id(
            prefix="TEST",
            function_name="test_function"
        )
        
        # Check ID format
        self.assertRegex(exp_id, r"TEST_\d{8}_test_function")

if __name__ == '__main__':
    unittest.main()