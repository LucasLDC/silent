from moviepy.editor import VideoFileClip
import os

def load_video_and_extract_audio(video_path: str, temp_audio_dir: str = "video_editing_tool/data"):
    """
    Loads a video file (MP4 or MKV), extracts its audio, and saves the audio to a temporary WAV file.

    Args:
        video_path (str): The path to the input video file.
        temp_audio_dir (str): The directory where the temporary audio file will be saved.

    Returns:
        tuple: (str, moviepy.editor.VideoFileClip)
               Path to the temporary WAV audio file, and the loaded video clip object.
               Returns (None, None) if loading or audio extraction fails.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None, None

    file_extension = os.path.splitext(video_path)[1].lower()
    if file_extension not in ['.mp4', '.mkv']:
        print(f"Error: Unsupported file format: {file_extension}. Only .mp4 and .mkv are supported.")
        return None, None

    try:
        video_clip = VideoFileClip(video_path)
    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        return None, None

    if video_clip.audio is None:
        print(f"Error: No audio track found in video {video_path}")
        if video_clip:
            video_clip.close()
        return None, None

    # Ensure the temporary directory exists
    os.makedirs(temp_audio_dir, exist_ok=True)

    temp_audio_filename = "temp_audio.wav" # Consider making this more unique later
    temp_audio_path = os.path.join(temp_audio_dir, temp_audio_filename)

    try:
        video_clip.audio.write_audiofile(temp_audio_path, codec='pcm_s16le') # WAV codec
        print(f"Audio extracted successfully to {temp_audio_path}")
        return temp_audio_path, video_clip
    except Exception as e:
        print(f"Error extracting audio to {temp_audio_path}: {e}")
        if video_clip:
            video_clip.close()
        return None, None

if __name__ == '__main__':
    # Basic test placeholder - this will require sample files in data/
    # Create dummy files for initial testing if real samples aren't present.

    # Create a dummy data directory if it doesn't exist for the purpose of this basic test
    if not os.path.exists("video_editing_tool/data"):
        os.makedirs("video_editing_tool/data")

    # Create a dummy MP4 file for testing if it doesn't exist
    # Note: MoviePy will fail if it's not a real video. This is just for path testing.
    # For real testing, actual sample video files are needed.
    sample_mp4_path = "video_editing_tool/data/sample_test.mp4"
    if not os.path.exists(sample_mp4_path):
        # This is not a valid video file, but suffices for a simple path existence test
        # In a real scenario, use actual small video files for testing.
        try:
            # Attempt to create a tiny, valid (but silent) mp4 using moviepy if possible
            # This requires ffmpeg to be installed.
            # Note: AudioArrayClip caused issues with moviepy 1.0.3, so creating a video without audio.
            from moviepy.editor import ColorClip
            # import numpy as np # Not needed without AudioArrayClip
            clip = ColorClip(size=(10,10), color=(0,0,0), duration=1).set_fps(1)
            # # Create a silent audio track - Removed due to compatibility with moviepy 1.0.3
            # silent_audio = AudioArrayClip(np.zeros((44100, 2)), fps=44100) # 1 sec of silence
            # clip = clip.set_audio(silent_audio)
            clip.write_videofile(sample_mp4_path, codec="libx264") # Removed audio_codec
            if clip:
                clip.close() # Close the clip after writing
            print(f"Created dummy video file (silent): {sample_mp4_path}")
        except Exception as e:
            print(f"Could not create dummy mp4 for testing: {e}. Manual sample file needed.")
            with open(sample_mp4_path, 'w') as f:
                f.write("dummy mp4 content") # Placeholder if moviepy fails

    if os.path.exists(sample_mp4_path):
        print(f"Attempting to load: {sample_mp4_path}")
        audio_path, video_obj = load_video_and_extract_audio(sample_mp4_path)
        if audio_path and video_obj:
            print(f"Successfully processed video: {sample_mp4_path}")
            print(f"Audio saved to: {audio_path}")
            video_obj.close() # Important to close the clip
            # Clean up the temp audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
                print(f"Cleaned up temporary audio file: {audio_path}")
        else:
            print(f"Failed to process video: {sample_mp4_path}. This might be due to issues with on-the-fly dummy video generation in this test environment.")
    else:
        print(f"Skipping MP4 test, sample file not found and could not be created: {sample_mp4_path}")

    # Similarly for MKV (though creating a dummy mkv is harder without tools)
    sample_mkv_path = "video_editing_tool/data/sample_test.mkv"
    # if not os.path.exists(sample_mkv_path):
    #     with open(sample_mkv_path, 'w') as f:
    #         f.write("dummy mkv content") # Placeholder

    # if os.path.exists(sample_mkv_path):
    #     print(f"Attempting to load: {sample_mkv_path}")
    #     audio_path_mkv, video_obj_mkv = load_video_and_extract_audio(sample_mkv_path)
    #     if audio_path_mkv and video_obj_mkv:
    #         print(f"Successfully processed video: {sample_mkv_path}")
    #         print(f"Audio saved to: {audio_path_mkv}")
    #         video_obj_mkv.close()
    #         if os.path.exists(audio_path_mkv):
    #             os.remove(audio_path_mkv)
    #     else:
    #         print(f"Failed to process video: {sample_mkv_path}")
    # else:
    #     print(f"Skipping MKV test, sample file not found: {sample_mkv_path}")
