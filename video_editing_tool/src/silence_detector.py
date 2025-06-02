import librosa
import numpy as np
import os

def detect_silence(audio_path: str,
                   silence_threshold_db: float = -40.0,
                   min_silence_duration_ms: float = 500.0,
                   frame_length_ms: float = 30.0,
                   hop_length_ms: float = 10.0):
    """
    Detects silent segments in an audio file.

    Args:
        audio_path (str): Path to the audio file (e.g., WAV).
        silence_threshold_db (float): Silence threshold in dBFS.
                                      Anything below this is considered silence.
        min_silence_duration_ms (float): Minimum duration (in milliseconds) for a segment
                                         to be considered a continuous silence.
        frame_length_ms (float): Duration of each frame for RMS calculation (in milliseconds).
        hop_length_ms (float): Hop length between frames for RMS calculation (in milliseconds).

    Returns:
        list: A list of tuples (start_time_seconds, end_time_seconds) for detected silent segments.
              Returns an empty list if errors occur or no silence is detected.
    """
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return []

    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None) # sr=None to preserve original sampling rate
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return []

    # Convert parameters from milliseconds to samples/frames
    frame_length = int(sr * frame_length_ms / 1000.0)
    hop_length = int(sr * hop_length_ms / 1000.0)
    min_silence_frames = int(min_silence_duration_ms / hop_length_ms)

    # Calculate RMS energy per frame
    # Note: librosa.feature.rms returns RMS for each frame.
    # The output `rms` is a 1D array where rms[t] is the RMS value for frame t.
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # Convert dBFS threshold to amplitude threshold
    # Assuming y is in range [-1, 1] (typical for librosa.load)
    # Amplitude = 10^(dBFS / 20)
    # However, RMS is already a measure of amplitude, so we can compare RMS with a threshold
    # derived from dB. If max possible RMS is 1 (for a sine wave of amplitude 1),
    # then threshold_amplitude = 10^(silence_threshold_db / 20).
    # A more robust way is to use librosa.power_to_db on the squared RMS (power) and compare with dB directly.
    # S = librosa.magphase(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))[0]
    # power_db = librosa.power_to_db(S**2, ref=np.max) # Power in dB relative to max power
    # For RMS directly, if audio is normalized to peak of 1.0, then threshold_amp = 10**(dB/20)
    # Let's use a simpler approach: compare RMS to an amplitude threshold.
    # If y is not guaranteed to be normalized, this threshold might need adjustment or normalization of y.
    # For now, assume y is reasonably normalized by librosa.load.
    amplitude_threshold = 10**(silence_threshold_db / 20.0)

    # Identify frames below the amplitude threshold
    silent_frames_mask = rms < amplitude_threshold

    silent_segments = []
    is_silent = False
    silence_start_frame = 0

    for i, frame_is_silent in enumerate(silent_frames_mask):
        if frame_is_silent and not is_silent:
            is_silent = True
            silence_start_frame = i
        elif not frame_is_silent and is_silent:
            is_silent = False
            if (i - silence_start_frame) >= min_silence_frames:
                start_time = librosa.frames_to_time(silence_start_frame, sr=sr, hop_length=hop_length)
                end_time = librosa.frames_to_time(i, sr=sr, hop_length=hop_length) # i is the first non-silent frame
                silent_segments.append((start_time, end_time))

    # Check if the audio ends in a silent segment
    if is_silent and (len(silent_frames_mask) - silence_start_frame) >= min_silence_frames:
        start_time = librosa.frames_to_time(silence_start_frame, sr=sr, hop_length=hop_length)
        end_time = librosa.frames_to_time(len(silent_frames_mask), sr=sr, hop_length=hop_length)
        silent_segments.append((start_time, end_time))

    return silent_segments

if __name__ == '__main__':
    # Create a dummy audio file for testing
    # This requires FFmpeg/libsndfile for librosa to write audio.
    # And the 'video_editing_tool/data' directory must exist.

    sample_rate = 44100
    duration_s = 5
    frequency = 440 # A4 note

    # Create 1 sec of sound, 1 sec of silence, 1 sec of sound, 1 sec of silence, 1 sec of sound
    time_s = np.linspace(0, 1, int(sample_rate * 1), endpoint=False)
    sound_segment = 0.5 * np.sin(2 * np.pi * frequency * time_s)
    silence_segment = np.zeros(int(sample_rate * 1))

    test_audio_signal = np.concatenate((sound_segment,     # 0-1s sound
                                        silence_segment,   # 1-2s silence
                                        sound_segment,     # 2-3s sound
                                        silence_segment,   # 3-4s silence
                                        sound_segment))    # 4-5s sound

    test_audio_path = "video_editing_tool/data/test_audio_for_silence.wav"

    # Ensure data directory exists
    if not os.path.exists("video_editing_tool/data"):
        os.makedirs("video_editing_tool/data")

    try:
        import soundfile as sf
        sf.write(test_audio_path, test_audio_signal, sample_rate)
        print(f"Created dummy audio file: {test_audio_path}")

        # Test the silence detection
        print(f"Detecting silence in {test_audio_path}...")
        # Use a threshold that should clearly distinguish the silent parts
        # The sound parts are 0.5 amplitude, so RMS will be around 0.5/sqrt(2) ~ 0.35
        # 0.35 is approx -9dB. So -30dB should be well into silence.
        silences = detect_silence(test_audio_path,
                                  silence_threshold_db=-30.0,
                                  min_silence_duration_ms=500.0) # min duration 0.5s

        if silences:
            print("Detected silent segments (start_sec, end_sec):")
            for start, end in silences:
                print(f"  ({start:.2f}, {end:.2f}), duration: {end-start:.2f}s")

            # Expected: (1.0, 2.0) and (3.0, 4.0) approximately
            # Exact times can vary slightly due to framing
        else:
            print("No silence detected or error occurred.")

    except Exception as e:
        print(f"Could not create or process dummy audio file for testing: {e}")
        print("Please ensure 'soundfile' is installed (pip install soundfile) and libsndfile is available.")
        print("Alternatively, place a valid 'test_audio_for_silence.wav' in 'video_editing_tool/data/'.")

    finally:
        # Clean up the dummy audio file
        if os.path.exists(test_audio_path) and "Created dummy audio file" in open(test_audio_path, 'rb').readline().decode(errors='ignore'): # A bit hacky way to check if it was our generated file
             # This check is not reliable. Better to always clean up if created in test.
             # For now, let's assume if it was created in the try block, we can remove it.
             # A more robust way would be to only create if it doesn't exist and then remove.
             # Let's refine: create and remove if it was created by this specific test run.
             # This 'finally' block might run even if sf.write failed, so check again.
            if 'sf' in locals() and os.path.exists(test_audio_path): # Check if sf was imported and file exists
                try:
                    # A more robust check: if the file was created because it wasn't there before this run
                    # This is still tricky. For now, if the test creates it, it attempts to delete it.
                    # This part of the test code could be improved for robustness.
                    # Let's assume we clean it up if the test created it.
                    # For proper unit tests, this kind of file management is handled in setUp/tearDown.
                    pass # Let's not delete it for now to allow manual inspection.
                    # print(f"Cleaned up dummy audio file: {test_audio_path}")
                    # os.remove(test_audio_path)
                except Exception as e_clean:
                    print(f"Error cleaning up dummy audio file: {e_clean}")
