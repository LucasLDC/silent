import unittest
import os
import shutil
from video_editing_tool.src.video_loader import load_video_and_extract_audio
from moviepy.editor import VideoFileClip

class TestVideoLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_data_dir = "video_editing_tool/data"
        cls.temp_audio_dir = os.path.join(cls.sample_data_dir, "temp_test_audio") # Use a sub-folder for test outputs
        os.makedirs(cls.temp_audio_dir, exist_ok=True)

        # Paths for sample files - these files would need to exist for tests to fully pass
        cls.sample_mp4_path = os.path.join(cls.sample_data_dir, "sample.mp4")
        cls.sample_mkv_path = os.path.join(cls.sample_data_dir, "sample.mkv")
        cls.invalid_file_path = os.path.join(cls.sample_data_dir, "non_existent_video.mp4")
        cls.unsupported_file_path = os.path.join(cls.sample_data_dir, "sample.txt")

        # Create a dummy .txt file for testing unsupported format
        with open(cls.unsupported_file_path, 'w') as f:
            f.write("This is not a video.")

        # NOTE: The following lines attempt to create minimal, valid MP4 and MKV files
        # for testing if they don't exist. This requires FFmpeg to be installed in the
        # environment where tests are run. If FFmpeg is not available or these
        # creations fail, these tests will likely fail or be skipped.
        # For robust testing, provide actual sample.mp4 and sample.mkv files in the data directory.

        try:
            from moviepy.editor import ColorClip # VideoFileClip is already imported at the top
            from moviepy.audio.AudioClip import AudioArrayClip # Correct import for moviepy 1.0.3
            import numpy as np
            if not os.path.exists(cls.sample_mp4_path):
                print(f"Attempting to create dummy {cls.sample_mp4_path} for testing...")
                clip = ColorClip(size=(10,10), color=(0,0,0), duration=1).set_fps(1)
                silent_audio = AudioArrayClip(np.zeros((1 * 44100, 2)), fps=44100) # 1 sec
                clip = clip.set_audio(silent_audio)
                clip.write_videofile(cls.sample_mp4_path, codec="libx264", audio_codec="aac", logger=None)
                print(f"Dummy {cls.sample_mp4_path} created.")

            # Creating a dummy MKV is similar if FFmpeg handles the container.
            # MoviePy might default to MP4 if filename extension is not MKV during write.
            # Forcing container might be needed or use ffmpeg directly.
            # For now, let's try writing with .mkv extension.
            if not os.path.exists(cls.sample_mkv_path):
                print(f"Attempting to create dummy {cls.sample_mkv_path} for testing...")
                clip_mkv = ColorClip(size=(10,10), color=(10,0,0), duration=1).set_fps(1)
                silent_audio_mkv = AudioArrayClip(np.zeros((1 * 44100, 2)), fps=44100)
                clip_mkv = clip_mkv.set_audio(silent_audio_mkv)
                # MoviePy's write_videofile uses the extension to determine container.
                clip_mkv.write_videofile(cls.sample_mkv_path, codec="libx264", audio_codec="aac", logger=None)
                print(f"Dummy {cls.sample_mkv_path} created.")

        except Exception as e:
            print(f"Warning: Could not create dummy video files for testing: {e}. Some tests may fail or be skipped.")
            print("Please ensure FFmpeg is installed and functional, or provide actual sample.mp4 and sample.mkv files.")


    @classmethod
    def tearDownClass(cls):
        # Clean up created dummy files and directories
        if os.path.exists(cls.unsupported_file_path):
            os.remove(cls.unsupported_file_path)
        # Only remove dummy videos if they were likely created by this script (e.g. small size)
        # For now, let's assume if they exist, they might be user-provided, so we won't delete them
        # to avoid accidental data loss. Test-specific temp files should be cleaned.
        if os.path.exists(cls.temp_audio_dir):
            shutil.rmtree(cls.temp_audio_dir)
        # Clean up any temp_audio.wav in the main data dir if created by direct script run
        main_temp_audio = os.path.join(cls.sample_data_dir, "temp_audio.wav")
        if os.path.exists(main_temp_audio):
             os.remove(main_temp_audio)


    def tearDown(self):
        # Clean up any audio files created during a test in temp_audio_dir
        for item in os.listdir(self.temp_audio_dir):
            if item.endswith(".wav"):
                os.remove(os.path.join(self.temp_audio_dir, item))

    def test_load_mp4_and_extract_audio_valid(self):
        if not os.path.exists(self.sample_mp4_path):
            self.skipTest(f"Sample MP4 file not found: {self.sample_mp4_path}")

        audio_path, video_clip = load_video_and_extract_audio(self.sample_mp4_path, self.temp_audio_dir)
        self.assertIsNotNone(audio_path, "Audio path should not be None for valid MP4.")
        self.assertTrue(os.path.exists(audio_path), f"Extracted audio file should exist at {audio_path}.")
        self.assertIsInstance(video_clip, VideoFileClip, "Should return a VideoFileClip object.")
        if video_clip:
            video_clip.close() # Important to close file handles

    def test_load_mkv_and_extract_audio_valid(self):
        if not os.path.exists(self.sample_mkv_path):
            self.skipTest(f"Sample MKV file not found: {self.sample_mkv_path}")

        audio_path, video_clip = load_video_and_extract_audio(self.sample_mkv_path, self.temp_audio_dir)
        self.assertIsNotNone(audio_path, "Audio path should not be None for valid MKV.")
        self.assertTrue(os.path.exists(audio_path), "Extracted audio file should exist.")
        self.assertIsInstance(video_clip, VideoFileClip, "Should return a VideoFileClip object.")
        if video_clip:
            video_clip.close()

    def test_non_existent_file(self):
        audio_path, video_clip = load_video_and_extract_audio(self.invalid_file_path, self.temp_audio_dir)
        self.assertIsNone(audio_path, "Audio path should be None for non-existent file.")
        self.assertIsNone(video_clip, "Video clip should be None for non-existent file.")

    def test_unsupported_file_format(self):
        audio_path, video_clip = load_video_and_extract_audio(self.unsupported_file_path, self.temp_audio_dir)
        self.assertIsNone(audio_path, "Audio path should be None for unsupported file type.")
        self.assertIsNone(video_clip, "Video clip should be None for unsupported file type.")

    # Optional: Test for video without audio (requires a specific sample file)
    # def test_video_without_audio(self):
    #     # Assuming "sample_no_audio.mp4" is a video file without an audio track
    #     sample_no_audio_path = os.path.join(self.sample_data_dir, "sample_no_audio.mp4")
    #     if not os.path.exists(sample_no_audio_path):
    #         self.skipTest(f"Sample video without audio not found: {sample_no_audio_path}")
    #
    #     audio_path, video_clip = load_video_and_extract_audio(sample_no_audio_path, self.temp_audio_dir)
    #     self.assertIsNone(audio_path, "Audio path should be None for video without audio.")
    #     # video_clip might still be loaded, depending on behavior.
    #     # If it's loaded, ensure it's closed.
    #     if video_clip:
    #         self.assertIsNone(video_clip.audio, "Video clip's audio attribute should be None.")
    #         video_clip.close()
    #     else:
    #          # If video_clip is None itself, that's also acceptable if it fails early
    #          self.assertIsNone(video_clip)


if __name__ == '__main__':
    unittest.main()
