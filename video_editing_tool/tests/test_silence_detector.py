import unittest
import os
import numpy as np
import soundfile as sf
import shutil # For cleaning up directories
from video_editing_tool.src.silence_detector import detect_silence

class TestSilenceDetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_audio_dir = "video_editing_tool/data/test_silence_audio/"
        os.makedirs(cls.test_audio_dir, exist_ok=True)
        cls.sample_rate = 44100

        # Test case 1: Clear silence in the middle
        cls.audio_path1 = os.path.join(cls.test_audio_dir, "audio1_middle_silence.wav")
        sound_segment = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, int(cls.sample_rate * 1), endpoint=False))
        silence_segment = np.zeros(int(cls.sample_rate * 1)) # 1s silence
        audio1_signal = np.concatenate((sound_segment, silence_segment, sound_segment))
        sf.write(cls.audio_path1, audio1_signal, cls.sample_rate)

        # Test case 2: No silence (continuous sound)
        cls.audio_path2 = os.path.join(cls.test_audio_dir, "audio2_no_silence.wav")
        audio2_signal = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 3, int(cls.sample_rate * 3), endpoint=False))
        sf.write(cls.audio_path2, audio2_signal, cls.sample_rate)

        # Test case 3: All silence
        cls.audio_path3 = os.path.join(cls.test_audio_dir, "audio3_all_silence.wav")
        audio3_signal = np.zeros(int(cls.sample_rate * 3)) # 3s silence
        sf.write(cls.audio_path3, audio3_signal, cls.sample_rate)

        # Test case 4: Silence shorter than min_duration
        cls.audio_path4 = os.path.join(cls.test_audio_dir, "audio4_short_silence.wav")
        short_silence_segment = np.zeros(int(cls.sample_rate * 0.2)) # 0.2s silence
        audio4_signal = np.concatenate((sound_segment, short_silence_segment, sound_segment))
        sf.write(cls.audio_path4, audio4_signal, cls.sample_rate)

        # Test case 5: Silence at the beginning and end
        cls.audio_path5 = os.path.join(cls.test_audio_dir, "audio5_edge_silences.wav")
        audio5_signal = np.concatenate((silence_segment, sound_segment, silence_segment))
        sf.write(cls.audio_path5, audio5_signal, cls.sample_rate)

        # Test case 6: File not found (will be checked by passing a non-existent path)
        cls.non_existent_audio_path = os.path.join(cls.test_audio_dir, "non_existent.wav")


    @classmethod
    def tearDownClass(cls):
        # Clean up created audio files and directory
        if os.path.exists(cls.test_audio_dir):
            shutil.rmtree(cls.test_audio_dir)

    def assertSilencesAlmostEqual(self, detected_silences, expected_silences, tolerance_sec=0.05):
        self.assertEqual(len(detected_silences), len(expected_silences),
                         f"Expected {len(expected_silences)} silences, got {len(detected_silences)}")
        for detected, expected in zip(detected_silences, expected_silences):
            self.assertAlmostEqual(detected[0], expected[0], delta=tolerance_sec,
                                   msg=f"Start times differ: {detected[0]} vs {expected[0]}")
            self.assertAlmostEqual(detected[1], expected[1], delta=tolerance_sec,
                                   msg=f"End times differ: {detected[1]} vs {expected[1]}")

    def test_middle_silence(self):
        # Expected: silence from ~1.0s to ~2.0s
        silences = detect_silence(self.audio_path1, silence_threshold_db=-30.0, min_silence_duration_ms=500)
        # print(f"Test Middle Silence - Detected: {silences}")
        self.assertSilencesAlmostEqual(silences, [(1.0, 2.0)])

    def test_no_silence(self):
        silences = detect_silence(self.audio_path2, silence_threshold_db=-30.0, min_silence_duration_ms=500)
        # print(f"Test No Silence - Detected: {silences}")
        self.assertEqual(len(silences), 0, "Should detect no silence in a continuous sound file.")

    def test_all_silence(self):
        # Expected: silence from ~0.0s to ~3.0s
        silences = detect_silence(self.audio_path3, silence_threshold_db=-30.0, min_silence_duration_ms=500)
        # print(f"Test All Silence - Detected: {silences}")
        self.assertSilencesAlmostEqual(silences, [(0.0, 3.0)])

    def test_silence_shorter_than_min_duration(self):
        # min_silence_duration_ms is 500ms, the silence is 200ms. So it should not be detected.
        silences = detect_silence(self.audio_path4, silence_threshold_db=-30.0, min_silence_duration_ms=500)
        # print(f"Test Short Silence - Detected: {silences}")
        self.assertEqual(len(silences), 0, "Should not detect silence shorter than min_duration_ms.")

    def test_silence_longer_than_min_duration(self):
        # Same audio as audio_path4, but lower min_silence_duration_ms to detect the 200ms silence
        silences = detect_silence(self.audio_path4, silence_threshold_db=-30.0, min_silence_duration_ms=100) # 100ms min
        # print(f"Test Short Silence (lowered min_duration) - Detected: {silences}")
        # Expected silence is from 1.0s to 1.2s
        self.assertSilencesAlmostEqual(silences, [(1.0, 1.2)])

    def test_edge_silences(self):
        # Expected: silence from ~0.0s to ~1.0s and ~2.0s to ~3.0s
        silences = detect_silence(self.audio_path5, silence_threshold_db=-30.0, min_silence_duration_ms=500)
        # print(f"Test Edge Silences - Detected: {silences}")
        self.assertSilencesAlmostEqual(silences, [(0.0, 1.0), (2.0, 3.0)])

    def test_different_threshold(self):
        # audio1 has sound at 0.5 amplitude (~ -6dBFS peak, RMS ~ -9dBFS)
        # If threshold is -3dBFS, sound should not be mistaken for silence.
        silences = detect_silence(self.audio_path1, silence_threshold_db=-3.0, min_silence_duration_ms=500)
        # print(f"Test Different Threshold (high) - Detected: {silences}")
        # With -3dBFS threshold, the 'sound' part (RMS ~0.35 => ~-9dBFS) is also below threshold.
        # So, the entire file (0s to 3s) should be detected as silence.
        self.assertSilencesAlmostEqual(silences, [(0.0, 3.0)])

        # If threshold is -60dBFS, even the "silence" (zeros) should be detected.
        silences_low_thresh = detect_silence(self.audio_path1, silence_threshold_db=-60.0, min_silence_duration_ms=500)
        # print(f"Test Different Threshold (low) - Detected: {silences_low_thresh}")
        self.assertSilencesAlmostEqual(silences_low_thresh, [(1.0, 2.0)])


    def test_non_existent_file(self):
        silences = detect_silence(self.non_existent_audio_path, silence_threshold_db=-40.0, min_silence_duration_ms=500)
        self.assertEqual(len(silences), 0, "Should return empty list for non-existent file.")


if __name__ == '__main__':
    unittest.main()
