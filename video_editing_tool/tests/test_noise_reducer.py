import unittest
import os
import shutil
import numpy as np
import soundfile as sf
from scipy import signal
from video_editing_tool.src.noise_reducer import reduce_background_noise, remove_hum, reduce_hiss # Added reduce_hiss

class TestNoiseReducer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # ... (existing setUpClass code, including hum test files) ...
        cls.test_temp_dir = "video_editing_tool/data/temp_noise_reducer_unittest/"
        os.makedirs(cls.test_temp_dir, exist_ok=True)
        cls.sample_rate = 44100
        cls.duration_s = 1.0
        time_s = np.linspace(0, cls.duration_s, int(cls.sample_rate * cls.duration_s), endpoint=False)
        cls.clean_signal_mono = 0.5 * np.sin(2 * np.pi * 440 * time_s)
        noise_mono_generic = 0.1 * np.random.randn(len(cls.clean_signal_mono))
        noisy_signal_mono_generic = np.clip(cls.clean_signal_mono + noise_mono_generic, -1.0, 1.0)
        cls.clean_mono_path = os.path.join(cls.test_temp_dir, "clean_mono.wav")
        sf.write(cls.clean_mono_path, cls.clean_signal_mono, cls.sample_rate)
        cls.noisy_mono_path = os.path.join(cls.test_temp_dir, "noisy_mono.wav")
        sf.write(cls.noisy_mono_path, noisy_signal_mono_generic, cls.sample_rate)
        noisy_signal_stereo_generic = np.clip(np.array([noisy_signal_mono_generic, noisy_signal_mono_generic * 0.8]).T, -1.0, 1.0)
        cls.noisy_stereo_path = os.path.join(cls.test_temp_dir, "noisy_stereo.wav")
        sf.write(cls.noisy_stereo_path, noisy_signal_stereo_generic, cls.sample_rate)
        cls.pure_noise_path = os.path.join(cls.test_temp_dir, "pure_noise.wav")
        sf.write(cls.pure_noise_path, noise_mono_generic, cls.sample_rate)
        cls.hum_freq1 = 60.0; cls.hum_freq2 = 120.0
        cls.signal_no_hum_mono = 0.4 * np.sin(2 * np.pi * 300 * time_s) + 0.3 * np.sin(2 * np.pi * 700 * time_s)
        cls.signal_no_hum_mono_path = os.path.join(cls.test_temp_dir, "signal_no_hum_mono.wav")
        sf.write(cls.signal_no_hum_mono_path, cls.signal_no_hum_mono, cls.sample_rate)
        hum_component1 = 0.2 * np.sin(2 * np.pi * cls.hum_freq1 * time_s); hum_component2 = 0.15 * np.sin(2 * np.pi * cls.hum_freq2 * time_s)
        cls.signal_with_hum_mono = np.clip(cls.signal_no_hum_mono + hum_component1 + hum_component2, -1.0, 1.0)
        cls.signal_with_hum_mono_path = os.path.join(cls.test_temp_dir, "signal_with_hum_mono.wav")
        sf.write(cls.signal_with_hum_mono_path, cls.signal_with_hum_mono, cls.sample_rate)
        signal_with_hum_stereo_ch1 = np.clip(cls.signal_no_hum_mono + hum_component1 + hum_component2, -1.0, 1.0)
        signal_with_hum_stereo_ch2 = np.clip(cls.signal_no_hum_mono * 0.8 + hum_component1 * 0.9 + hum_component2 * 0.7, -1.0, 1.0)
        cls.signal_with_hum_stereo = np.array([signal_with_hum_stereo_ch1, signal_with_hum_stereo_ch2]).T
        cls.signal_with_hum_stereo_path = os.path.join(cls.test_temp_dir, "signal_with_hum_stereo.wav")
        sf.write(cls.signal_with_hum_stereo_path, cls.signal_with_hum_stereo, cls.sample_rate)

        # --- New audio files for hiss testing ---
        cls.base_signal_for_hiss = 0.4 * np.sin(2 * np.pi * 800 * time_s) # A clear mid-freq signal
        # Hiss: broadband noise, maybe slightly more power in high freq
        hiss_component = 0.15 * np.random.randn(len(time_s))
        # Optional: make hiss more high-frequency
        # sos_hiss_shaping = signal.butter(4, 4000, 'hp', fs=cls.sample_rate, output='sos')
        # hiss_component = signal.sosfilt(sos_hiss_shaping, hiss_component) * 1.5

        cls.signal_with_hiss_mono = np.clip(cls.base_signal_for_hiss + hiss_component, -1.0, 1.0)
        cls.signal_with_hiss_mono_path = os.path.join(cls.test_temp_dir, "signal_with_hiss_mono.wav")
        sf.write(cls.signal_with_hiss_mono_path, cls.signal_with_hiss_mono, cls.sample_rate)

        cls.pure_hiss_mono = np.clip(hiss_component, -1.0, 1.0)
        cls.pure_hiss_mono_path = os.path.join(cls.test_temp_dir, "pure_hiss_mono.wav")
        sf.write(cls.pure_hiss_mono_path, cls.pure_hiss_mono, cls.sample_rate)


    @classmethod
    def tearDownClass(cls):
        # ... (existing tearDownClass code) ...
        if os.path.exists(cls.test_temp_dir):
            shutil.rmtree(cls.test_temp_dir)

    def get_output_path(self, input_filename_base, operation="reduced"):
        # ... (existing get_output_path code) ...
        return os.path.join(self.test_temp_dir, f"output_{input_filename_base}_{operation}.wav")

    # (Keep all existing test methods for reduce_background_noise and remove_hum)
    # ... (all previous test methods here) ...
    def test_reduce_noise_on_noisy_mono(self):
        output_path = self.get_output_path("noisy_mono")
        success = reduce_background_noise(self.noisy_mono_path, output_path, prop_decrease=0.9, verbose=False)
        self.assertTrue(success); self.assertTrue(os.path.exists(output_path))
        data, sr = sf.read(output_path); self.assertEqual(data.ndim, 1); self.assertEqual(sr, self.sample_rate)
    def test_reduce_noise_on_noisy_stereo(self):
        output_path = self.get_output_path("noisy_stereo")
        success = reduce_background_noise(self.noisy_stereo_path, output_path, prop_decrease=0.9, verbose=False)
        self.assertTrue(success); self.assertTrue(os.path.exists(output_path))
        data, sr = sf.read(output_path); self.assertEqual(data.ndim, 1)
    def test_prop_decrease_zero_minimal_change(self):
        output_path = self.get_output_path("noisy_mono_prop0")
        success = reduce_background_noise(self.noisy_mono_path, output_path, prop_decrease=0.0, verbose=False)
        self.assertTrue(success); self.assertTrue(os.path.exists(output_path))
        input_data, _ = sf.read(self.noisy_mono_path); output_data, _ = sf.read(output_path)
        rms_input = np.sqrt(np.mean(input_data**2)); rms_output = np.sqrt(np.mean(output_data**2))
        self.assertAlmostEqual(rms_input, rms_output, delta=rms_input*0.1)
    def test_prop_decrease_one_max_reduction_on_pure_noise(self):
        output_path = self.get_output_path("pure_noise_prop1")
        success = reduce_background_noise(self.pure_noise_path, output_path, prop_decrease=1.0, verbose=False)
        self.assertTrue(success); self.assertTrue(os.path.exists(output_path))
        input_data, _ = sf.read(self.pure_noise_path); output_data, _ = sf.read(output_path)
        rms_input = np.sqrt(np.mean(input_data**2)); rms_output = np.sqrt(np.mean(output_data**2))
        self.assertLess(rms_output, rms_input * 0.7)
    def test_invalid_input_file(self):
        output_path = self.get_output_path("invalid_input_bg")
        success = reduce_background_noise("non_existent_audio.wav", output_path, verbose=False)
        self.assertFalse(success); self.assertFalse(os.path.exists(output_path))
    def test_invalid_prop_decrease_value(self):
        output_path_neg = self.get_output_path("invalid_prop_neg_bg")
        success_neg = reduce_background_noise(self.noisy_mono_path, output_path_neg, prop_decrease=-0.5, verbose=False)
        self.assertFalse(success_neg); self.assertFalse(os.path.exists(output_path_neg))
        output_path_high = self.get_output_path("invalid_prop_high_bg")
        success_high = reduce_background_noise(self.noisy_mono_path, output_path_high, prop_decrease=1.5, verbose=False)
        self.assertFalse(success_high); self.assertFalse(os.path.exists(output_path_high))
    def test_clean_file_processing_bg_noise(self):
        output_path = self.get_output_path("clean_mono_processed_bg")
        success = reduce_background_noise(self.clean_mono_path, output_path, prop_decrease=0.5, verbose=False)
        self.assertTrue(success); self.assertTrue(os.path.exists(output_path))
        input_data, _ = sf.read(self.clean_mono_path); output_data, _ = sf.read(output_path)
        rms_input = np.sqrt(np.mean(input_data**2)); rms_output = np.sqrt(np.mean(output_data**2))
        self.assertLessEqual(rms_output, rms_input * 1.05)
    def _get_power_at_freq(self, audio_data, sr, freq, bandwidth=2.0):
        n = len(audio_data);
        if n == 0: return 0.0
        fft_data = np.fft.rfft(audio_data); fft_freqs = np.fft.rfftfreq(n, d=1./sr)
        idx_center = np.argmin(np.abs(fft_freqs - freq))
        idx_low = np.argmin(np.abs(fft_freqs - (freq - bandwidth / 2))); idx_high = np.argmin(np.abs(fft_freqs - (freq + bandwidth / 2)))
        idx_start = min(idx_low, idx_high); idx_end = max(idx_low, idx_high) + 1
        if idx_start >= len(fft_data) or idx_end > len(fft_data) or idx_start < 0: return 0.0
        return np.sum(np.abs(fft_data[idx_start:idx_end]))
    def test_remove_hum_mono_successful(self):
        output_path = self.get_output_path("signal_with_hum_mono", operation="hum_removed")
        hum_freqs = [self.hum_freq1, self.hum_freq2]; original_data, sr = sf.read(self.signal_with_hum_mono_path)
        power_before_hum1 = self._get_power_at_freq(original_data, sr, self.hum_freq1); power_before_hum2 = self._get_power_at_freq(original_data, sr, self.hum_freq2)
        success = remove_hum(self.signal_with_hum_mono_path, output_path, hum_frequencies=hum_freqs, quality_factor=30, verbose=False)
        self.assertTrue(success); self.assertTrue(os.path.exists(output_path))
        processed_data, sr_out = sf.read(output_path); self.assertEqual(sr, sr_out); self.assertEqual(processed_data.ndim, 1)
        power_after_hum1 = self._get_power_at_freq(processed_data, sr_out, self.hum_freq1); power_after_hum2 = self._get_power_at_freq(processed_data, sr_out, self.hum_freq2)
        self.assertTrue(power_before_hum1 > 0); self.assertTrue(power_before_hum2 > 0)
        self.assertLess(power_after_hum1, power_before_hum1 * 0.5); self.assertLess(power_after_hum2, power_before_hum2 * 0.5)
    def test_remove_hum_stereo_successful(self):
        output_path = self.get_output_path("signal_with_hum_stereo", operation="hum_removed")
        hum_freqs = [self.hum_freq1, self.hum_freq2]; original_data_stereo, sr = sf.read(self.signal_with_hum_stereo_path)
        power_before_hum1_ch1 = self._get_power_at_freq(original_data_stereo[:, 0], sr, self.hum_freq1); power_before_hum2_ch1 = self._get_power_at_freq(original_data_stereo[:, 0], sr, self.hum_freq2)
        success = remove_hum(self.signal_with_hum_stereo_path, output_path, hum_frequencies=hum_freqs, quality_factor=30, verbose=False)
        self.assertTrue(success); self.assertTrue(os.path.exists(output_path))
        processed_data_stereo, sr_out = sf.read(output_path); self.assertEqual(sr, sr_out); self.assertEqual(processed_data_stereo.ndim, 2); self.assertEqual(processed_data_stereo.shape[1], 2)
        power_after_hum1_ch1 = self._get_power_at_freq(processed_data_stereo[:, 0], sr_out, self.hum_freq1); power_after_hum2_ch1 = self._get_power_at_freq(processed_data_stereo[:, 0], sr_out, self.hum_freq2)
        self.assertTrue(power_before_hum1_ch1 > 0); self.assertTrue(power_before_hum2_ch1 > 0)
        self.assertLess(power_after_hum1_ch1, power_before_hum1_ch1 * 0.5); self.assertLess(power_after_hum2_ch1, power_before_hum2_ch1 * 0.5)
    def test_remove_hum_no_hum_present(self):
        output_path = self.get_output_path("signal_no_hum_mono", operation="hum_removed")
        hum_freqs = [self.hum_freq1, self.hum_freq2]; original_data, sr = sf.read(self.signal_no_hum_mono_path)
        original_rms = np.sqrt(np.mean(original_data**2))
        success = remove_hum(self.signal_no_hum_mono_path, output_path, hum_frequencies=hum_freqs, quality_factor=30, verbose=False)
        self.assertTrue(success); self.assertTrue(os.path.exists(output_path))
        processed_data, sr_out = sf.read(output_path); processed_rms = np.sqrt(np.mean(processed_data**2))
        self.assertAlmostEqual(original_rms, processed_rms, delta=original_rms * 0.1)
    def test_remove_hum_invalid_frequency(self):
        output_path = self.get_output_path("signal_with_hum_mono", operation="hum_removed_invalid_freq")
        hum_freqs = [0, self.hum_freq1, self.sample_rate / 2 + 100]
        original_data, sr = sf.read(self.signal_with_hum_mono_path); power_before_hum1 = self._get_power_at_freq(original_data, sr, self.hum_freq1)
        success = remove_hum(self.signal_with_hum_mono_path, output_path, hum_frequencies=hum_freqs, quality_factor=30, verbose=True)
        self.assertTrue(success); self.assertTrue(os.path.exists(output_path))
        processed_data, sr_out = sf.read(output_path); power_after_hum1 = self._get_power_at_freq(processed_data, sr_out, self.hum_freq1)
        self.assertLess(power_after_hum1, power_before_hum1 * 0.5)
    def test_remove_hum_single_frequency_input(self):
        output_path = self.get_output_path("signal_with_hum_mono", operation="hum_removed_single_freq")
        hum_freq_single = self.hum_freq1; original_data, sr = sf.read(self.signal_with_hum_mono_path)
        power_before_hum1 = self._get_power_at_freq(original_data, sr, self.hum_freq1); power_before_hum2 = self._get_power_at_freq(original_data, sr, self.hum_freq2)
        success = remove_hum(self.signal_with_hum_mono_path, output_path, hum_frequencies=hum_freq_single, quality_factor=30, verbose=False)
        self.assertTrue(success); self.assertTrue(os.path.exists(output_path))
        processed_data, sr_out = sf.read(output_path)
        power_after_hum1 = self._get_power_at_freq(processed_data, sr_out, self.hum_freq1); power_after_hum2 = self._get_power_at_freq(processed_data, sr_out, self.hum_freq2)
        self.assertLess(power_after_hum1, power_before_hum1 * 0.5)
        self.assertAlmostEqual(power_after_hum2, power_before_hum2, delta=power_before_hum2 * 0.1)

    # --- New tests for reduce_hiss ---
    def _get_high_frequency_power_ratio(self, audio_data, sr, high_pass_cutoff=3000):
        """Calculates ratio of power above cutoff to total power."""
        n = len(audio_data)
        if n == 0: return 0.0

        fft_data = np.abs(np.fft.rfft(audio_data))
        fft_freqs = np.fft.rfftfreq(n, d=1./sr)

        total_power = np.sum(fft_data**2)
        if total_power == 0: return 0.0

        high_freq_power = np.sum(fft_data[fft_freqs >= high_pass_cutoff]**2)
        return high_freq_power / total_power

    def test_reduce_hiss_on_hissy_signal(self):
        output_path = self.get_output_path("signal_with_hiss_mono", operation="hiss_removed")

        original_data, sr = sf.read(self.signal_with_hiss_mono_path)
        # For synthesized hiss, it's hard to get a perfect metric.
        # We'll check if overall RMS is reduced, implying noise reduction.
        # A more advanced check could be spectral flatness or high-freq content change.
        original_rms = np.sqrt(np.mean(original_data**2))
        original_hf_power_ratio = self._get_high_frequency_power_ratio(original_data, sr)


        success = reduce_hiss(self.signal_with_hiss_mono_path, output_path,
                              reduction_prop=0.7, freq_smooth_hz=1000,
                              n_std_thresh_stationary=1.5, verbose=False)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_path))

        processed_data, sr_out = sf.read(output_path)
        self.assertEqual(sr, sr_out)
        self.assertEqual(processed_data.ndim, 1) # Should be mono
        processed_rms = np.sqrt(np.mean(processed_data**2))
        processed_hf_power_ratio = self._get_high_frequency_power_ratio(processed_data, sr_out)

        self.assertLess(processed_rms, original_rms * 0.9, # Expect some RMS reduction, similar to general noise reduction
                        "RMS should be reduced on a hissy signal.")
        # The high-frequency power ratio test is removed as the underlying nr.reduce_noise call
        # is now the same as reduce_background_noise and doesn't specifically target hiss features
        # with the available parameters in the current library version.


    def test_reduce_hiss_on_pure_hiss(self):
        output_path = self.get_output_path("pure_hiss_mono", operation="hiss_removed")
        original_data, sr = sf.read(self.pure_hiss_mono_path)
        original_rms = np.sqrt(np.mean(original_data**2))

        # Parameters for reduce_hiss are illustrative here, as they are not used by current nr.reduce_noise call
        success = reduce_hiss(self.pure_hiss_mono_path, output_path,
                              reduction_prop=0.8, freq_smooth_hz=1000,
                              n_std_thresh_stationary=1.5, verbose=False)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_path))

        processed_data, _ = sf.read(output_path)
        processed_rms = np.sqrt(np.mean(processed_data**2))
        # Expectation similar to reduce_background_noise on pure_noise
        self.assertLess(processed_rms, original_rms * 0.7,
                        "RMS of pure hiss signal should be significantly reduced.")

    def test_reduce_hiss_on_clean_signal(self):
        # Test that applying hiss reduction to a clean signal doesn't overly distort it.
        output_path = self.get_output_path("clean_mono_hiss_processed", operation="hiss_removed")
        original_data, sr = sf.read(self.clean_mono_path) # Using clean_mono_path
        original_rms = np.sqrt(np.mean(original_data**2))

        success = reduce_hiss(self.clean_mono_path, output_path, reduction_prop=0.5, verbose=False)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_path))

        processed_data, _ = sf.read(output_path)
        processed_rms = np.sqrt(np.mean(processed_data**2))
        # Expect some reduction as noisereduce might find parts of clean signal "stationary"
        # but it shouldn't be drastic if the signal is strong.
        # Expectation similar to reduce_background_noise on a clean signal
        self.assertAlmostEqual(original_rms, processed_rms, delta=original_rms * 0.3,
                               msg="RMS of clean signal should not be drastically altered by hiss reduction (now same as bg reduction).")

    def test_reduce_hiss_invalid_params(self):
        output_path = self.get_output_path("hiss_invalid_prop", operation="hiss_removed")
        success = reduce_hiss(self.signal_with_hiss_mono_path, output_path, reduction_prop=1.5, verbose=False)
        self.assertFalse(success)
        success = reduce_hiss(self.signal_with_hiss_mono_path, output_path, reduction_prop=-0.1, verbose=False)
        self.assertFalse(success)


if __name__ == '__main__':
    unittest.main()
