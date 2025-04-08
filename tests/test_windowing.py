#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test module for FFT package windowing functionality.
"""

import unittest
import numpy as np
from fft_package.core import windowing

class TestWindowing(unittest.TestCase):
    """Test cases for windowing functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.N = 1000  # Window length
        self.test_signal = np.ones(self.N)  # Unit signal for testing
    
    def test_hamming_window_symmetry(self):
        """Test Hamming window symmetry."""
        window = windowing.hamming_window(self.N)
        
        # Check window length
        self.assertEqual(len(window), self.N)
        
        # Check symmetry
        np.testing.assert_array_almost_equal(
            window[:self.N//2],
            window[self.N-1:self.N//2-1:-1]
        )
    
    def test_hanning_window_endpoints(self):
        """Test Hanning window endpoint values."""
        window = windowing.hanning_window(self.N)
        
        # Check endpoints (should be zero)
        self.assertAlmostEqual(window[0], 0.0)
        self.assertAlmostEqual(window[-1], 0.0)
    
    def test_blackman_window_range(self):
        """Test Blackman window value range."""
        window = windowing.blackman_window(self.N)
        
        # Check value range (should be between 0 and 1)
        self.assertTrue(np.all(window >= 0))
        self.assertTrue(np.all(window <= 1))
    
    def test_apply_window(self):
        """Test window application to a signal."""
        # Test with each window type
        for window_type in ['hamming', 'hanning', 'blackman']:
            with self.subTest(window_type=window_type):
                windowed = windowing.apply_window(self.test_signal, window_type)
                
                # Check output length
                self.assertEqual(len(windowed), self.N)
                
                # Check if windowing reduces signal at endpoints
                self.assertLess(windowed[0], self.test_signal[0])
                self.assertLess(windowed[-1], self.test_signal[-1])
    
    def test_invalid_window_type(self):
        """Test error handling for invalid window type."""
        with self.assertRaises(ValueError):
            windowing.apply_window(self.test_signal, 'invalid_window')

if __name__ == '__main__':
    unittest.main()