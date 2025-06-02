import unittest
import os
import shutil
import numpy as np
from moviepy.editor import VideoFileClip, ColorClip, concatenate_videoclips
from moviepy.audio.AudioClip import AudioArrayClip # Correct import for moviepy 1.0.3
from video_editing_tool.src.video_splicer import remove_silent_segments

class TestVideoSplicer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_output_dir = "video_editing_tool/data/temp_splicer_test_outputs/"
        os.makedirs(cls.test_output_dir, exist_ok=True)

        cls.fps = 24
        cls.sample_rate = 44100
        cls.clip_size = (100, 100) # Smaller size for faster test video generation

        # Create a reliable source video for testing
        cls.source_video_path = os.path.join(cls.test_output_dir, "test_source_video.mp4")
        cls.source_video_duration = 5.0 # Known duration

        # Segments: 0-1s (Red), 1-2s (Green), 2-3s (Blue), 3-4s (Yellow), 4-5s (Magenta)
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]
        clips = []
        for i, color in enumerate(colors):
            segment_clip = ColorClip(size=cls.clip_size, color=color, duration=1).set_fps(cls.fps)
            # Add simple audio to each segment to make it a valid audio-video clip
            audio_segment = AudioArrayClip(
                0.1 * np.sin(2 * np.pi * (220 + i*50) * np.linspace(0, 1, int(cls.sample_rate * 1), endpoint=False)).reshape(-1,1),
                fps=cls.sample_rate
            )
            segment_clip = segment_clip.set_audio(audio_segment)
            clips.append(segment_clip)

        source_video = concatenate_videoclips(clips)
        source_video.write_videofile(cls.source_video_path, codec="libx264", audio_codec="aac", logger=None)
        source_video.close() # Close the concatenated clip
        for clip_seg in clips: # Close individual segment clips
            clip_seg.close()

        # Load the video using VideoFileClip to pass to the function
        # This is where MoviePy 1.0.3 might have issues with reported duration.
        # However, our remove_silent_segments function was updated to use video_clip.duration
        # which *should* be more reliable for a loaded VideoFileClip than a just-written one.
        # If issues persist, we might need to pass true_duration_override.
        # For now, we rely on video_clip.duration of the loaded clip.
        # UPDATE: Given consistent issues with moviepy 1.0.3 reporting incorrect duration for loaded clips,
        # we will use true_duration_override in tests.
        cls.video_clip_instance = VideoFileClip(cls.source_video_path)
        # Let's check and print the duration MoviePy reports for the loaded clip
        print(f"TestVideoSplicer: Loaded test_source_video.mp4, MoviePy reports duration: {cls.video_clip_instance.duration}s, True duration: {cls.source_video_duration}s")


    @classmethod
    def tearDownClass(cls):
        if cls.video_clip_instance:
            cls.video_clip_instance.close()
        if os.path.exists(cls.test_output_dir):
            shutil.rmtree(cls.test_output_dir)

    def test_no_silences(self):
        result_clip = remove_silent_segments(self.video_clip_instance, [], true_duration_override=self.source_video_duration)
        self.assertIsNotNone(result_clip)
        # In MoviePy 1.0.3, direct comparison of clips is tricky. We check duration.
        # If no silences, original clip is returned. Its duration will be whatever moviepy reported for it.
        self.assertEqual(result_clip, self.video_clip_instance, "Original clip instance should be returned.")
        self.assertAlmostEqual(result_clip.duration, self.video_clip_instance.duration, delta=0.1, msg="Duration should be same as the (potentially faulty) original clip's reported duration.")
        # result_clip.close() # Don't close if it's the original self.video_clip_instance

    def test_remove_middle_segment(self):
        # Original: 0-1(R), 1-2(G), 2-3(B), 3-4(Y), 4-5(M)
        # Remove 2s-3s (Blue)
        silences = [(2.0, 3.0)]
        expected_duration = self.source_video_duration - 1.0

        result_clip = remove_silent_segments(self.video_clip_instance, silences, true_duration_override=self.source_video_duration)
        self.assertIsNotNone(result_clip)
        self.assertAlmostEqual(result_clip.duration, expected_duration, delta=0.1)
        if result_clip != self.video_clip_instance: result_clip.close()

    def test_remove_start_segment(self):
        # Remove 0s-1s (Red)
        silences = [(0.0, 1.0)]
        expected_duration = self.source_video_duration - 1.0

        result_clip = remove_silent_segments(self.video_clip_instance, silences, true_duration_override=self.source_video_duration)
        self.assertIsNotNone(result_clip)
        self.assertAlmostEqual(result_clip.duration, expected_duration, delta=0.1)
        if result_clip != self.video_clip_instance: result_clip.close()

    def test_remove_end_segment(self):
        # Remove 4s-5s (Magenta)
        silences = [(4.0, 5.0)]
        expected_duration = self.source_video_duration - 1.0

        result_clip = remove_silent_segments(self.video_clip_instance, silences, true_duration_override=self.source_video_duration)
        self.assertIsNotNone(result_clip)
        self.assertAlmostEqual(result_clip.duration, expected_duration, delta=0.1)
        if result_clip != self.video_clip_instance: result_clip.close()

    def test_remove_multiple_segments(self):
        # Remove 0-1s (R) and 2-3s (B) and 4-5s (M) -> Keep Green (1-2) and Yellow (3-4)
        silences = [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)]
        expected_duration = self.source_video_duration - 3.0 # Kept 2s

        result_clip = remove_silent_segments(self.video_clip_instance, silences, true_duration_override=self.source_video_duration)
        self.assertIsNotNone(result_clip)
        self.assertAlmostEqual(result_clip.duration, expected_duration, delta=0.1)
        if result_clip != self.video_clip_instance: result_clip.close()

    def test_remove_all_segments(self):
        # Remove everything
        silences = [(0.0, self.source_video_duration + 0.5)] # Ensure it covers the whole duration
        expected_duration = 0.1 # Function returns a 0.1s black clip

        result_clip = remove_silent_segments(self.video_clip_instance, silences, true_duration_override=self.source_video_duration)
        self.assertIsNotNone(result_clip)
        self.assertAlmostEqual(result_clip.duration, expected_duration, delta=0.05) # Tighter delta for the black clip
        # Check if it's a ColorClip (or CompositeVideoClip of one ColorClip)
        self.assertTrue(isinstance(result_clip, ColorClip) or
                        (isinstance(result_clip, CompositeVideoClip) and isinstance(result_clip.clips[0], ColorClip)),
                        "Expected a ColorClip or CompositeVideoClip of ColorClip when all is silent.")
        if result_clip != self.video_clip_instance: result_clip.close()

    def test_silence_timestamps_out_of_order(self):
        # Remove 2s-3s (B) and 0s-1s (R)
        silences = [(2.0, 3.0), (0.0, 1.0)] # Out of order
        expected_duration = self.source_video_duration - 2.0

        result_clip = remove_silent_segments(self.video_clip_instance, silences, true_duration_override=self.source_video_duration)
        self.assertIsNotNone(result_clip)
        self.assertAlmostEqual(result_clip.duration, expected_duration, delta=0.1)
        if result_clip != self.video_clip_instance: result_clip.close()

    def test_silence_timestamps_overlapping(self):
        # Original: 0-1(R), 1-2(G), 2-3(B), 3-4(Y), 4-5(M)
        # Silences: (0.5, 2.5) and (1.5, 3.5)
        # Effective silence: (0.5, 3.5) - removing 3s of content
        # Segments kept: (0.0, 0.5) and (3.5, 5.0) = 0.5 + 1.5 = 2.0s
        # The current implementation processes sorted silences; overlapping ones are handled sequentially.
        # (0.5, 2.5) -> keep (0, 0.5), current_time = 2.5
        # (1.5, 3.5) -> this is sorted after, but it starts before current_time. The logic should handle this.
        # Let's trace current logic:
        # sorted: [(0.5, 2.5), (1.5, 3.5)]
        # 1. (0.5, 2.5): keep (0, 0.5). current_time = 2.5
        # 2. (1.5, 3.5): silence_start (1.5) < current_time (2.5).
        #    This means the code `if current_time < silence_start:` is false.
        #    `current_time` becomes `silence_end` (3.5).
        # Finally, keep (3.5, 5.0).
        # So, kept: (0, 0.5) and (3.5, 5.0). Total duration 0.5 + 1.5 = 2.0s. This is correct.
        silences = [(0.5, 2.5), (1.5, 3.5)]
        expected_duration = 2.0

        result_clip = remove_silent_segments(self.video_clip_instance, silences, true_duration_override=self.source_video_duration)
        self.assertIsNotNone(result_clip)
        self.assertAlmostEqual(result_clip.duration, expected_duration, delta=0.1)
        if result_clip != self.video_clip_instance: result_clip.close()

    def test_invalid_video_clip_input(self):
        result_clip = remove_silent_segments(None, [(0,1)], true_duration_override=self.source_video_duration)
        self.assertIsNone(result_clip)

if __name__ == '__main__':
    unittest.main()
