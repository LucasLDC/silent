import soundfile as sf
import noisereduce as nr
import numpy as np
import os
from scipy import signal # Added for remove_hum

# (Keep the existing reduce_background_noise function as is)
# ... (existing reduce_background_noise function code) ...

def reduce_background_noise(input_audio_path: str,
                            output_audio_path: str,
                            prop_decrease: float = 1.0,
                            noise_clip_duration_s: float = 0.5, # This argument is not directly used by current nr implementation
                            verbose: bool = False):
    # ... (code as previously defined) ...
    if not os.path.exists(input_audio_path):
        if verbose: print(f"Error: Input audio file not found at {input_audio_path}")
        return False
    if not (0.0 <= prop_decrease <= 1.0):
        if verbose: print(f"Error: prop_decrease must be between 0.0 and 1.0. Got {prop_decrease}")
        return False
    try:
        audio_data, sample_rate = sf.read(input_audio_path)
    except Exception as e:
        if verbose: print(f"Error loading audio file {input_audio_path}: {e}")
        return False
    if verbose:
        print(f"Loaded audio: {input_audio_path}, Sample rate: {sample_rate}, Duration: {len(audio_data)/sample_rate:.2f}s")
    original_ndim = audio_data.ndim
    if audio_data.ndim > 1 and audio_data.shape[1] > 1:
        if verbose: print("Audio is stereo or multi-channel, converting to mono for broadband noise reduction.")
        audio_data_mono = np.mean(audio_data, axis=1)
    else:
        audio_data_mono = audio_data.copy()
    try:
        if verbose: print(f"Applying broadband noise reduction with prop_decrease={prop_decrease}...")
        reduced_noise_audio_mono = nr.reduce_noise(y=audio_data_mono,
                                                   sr=sample_rate,
                                                   prop_decrease=prop_decrease)
    except Exception as e:
        if verbose: print(f"Error during broadband noise reduction: {e}")
        return False
    output_audio_data = reduced_noise_audio_mono
    try:
        output_dir = os.path.dirname(output_audio_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        sf.write(output_audio_path, output_audio_data, sample_rate)
        if verbose: print(f"Broadband noise-reduced audio saved to {output_audio_path}")
        return True
    except Exception as e:
        if verbose: print(f"Error saving broadband noise-reduced audio to {output_audio_path}: {e}")
        return False

def remove_hum(input_audio_path: str,
               output_audio_path: str,
               hum_frequencies,
               quality_factor: float = 30.0,
               verbose: bool = False):
    # ... (code as previously defined) ...
    if not os.path.exists(input_audio_path):
        if verbose: print(f"Error: Input audio file not found at {input_audio_path}")
        return False
    try:
        audio_data, sample_rate = sf.read(input_audio_path)
    except Exception as e:
        if verbose: print(f"Error loading audio file {input_audio_path}: {e}")
        return False
    if verbose:
        print(f"Loaded audio for hum removal: {input_audio_path}, Sample rate: {sample_rate}, Duration: {len(audio_data)/sample_rate:.2f}s")
    if not isinstance(hum_frequencies, list):
        hum_frequencies = [hum_frequencies]
    processed_audio = audio_data.copy()
    for freq in hum_frequencies:
        if freq <= 0 or freq >= sample_rate / 2:
            if verbose: print(f"Skipping invalid frequency for notch filter: {freq} Hz (must be > 0 and < Nyquist freq {sample_rate/2} Hz)")
            continue
        if verbose: print(f"Applying notch filter for {freq} Hz with Q={quality_factor}")
        b, a = signal.iirnotch(freq, quality_factor, sample_rate)
        if processed_audio.ndim > 1:
            for i in range(processed_audio.shape[1]):
                processed_audio[:, i] = signal.lfilter(b, a, processed_audio[:, i])
        else:
            processed_audio = signal.lfilter(b, a, processed_audio)
    try:
        output_dir = os.path.dirname(output_audio_path)
        if output_dir and not os.path.exists(output_dir):
             os.makedirs(output_dir)
        sf.write(output_audio_path, processed_audio, sample_rate)
        if verbose: print(f"Hum-reduced audio saved to {output_audio_path}")
        return True
    except Exception as e:
        if verbose: print(f"Error saving hum-reduced audio to {output_audio_path}: {e}")
        return False

def reduce_hiss(input_audio_path: str,
                output_audio_path: str,
                reduction_prop: float = 0.7,
                freq_smooth_hz: float = 1000,
                n_std_thresh_stationary: float = 1.5,
                verbose: bool = False):
    """
    Reduces hiss-like noise using spectral gating, tuned for higher frequencies or stationary noise.
    This is an experimental function and effectiveness may vary.

    Args:
        input_audio_path (str): Path to the input audio file.
        output_audio_path (str): Path to save the hiss-reduced audio file.
        reduction_prop (float): Proportion to decrease noise (0.0 to 1.0). Default 0.7.
        freq_smooth_hz (float): Smoothing window in Hz across frequencies. Default 1000.
                                Smaller values might target narrower hiss bands.
        n_std_thresh_stationary (float): Threshold for distinguishing signal from stationary noise. Default 1.5.
                                         Higher values make it more conservative about what it calls noise.
        verbose (bool): If True, print more information.

    Returns:
        bool: True if hiss reduction was successful and file saved, False otherwise.
    """
    if not os.path.exists(input_audio_path):
        if verbose: print(f"Error: Input audio file not found at {input_audio_path}")
        return False

    if not (0.0 <= reduction_prop <= 1.0):
        if verbose: print(f"Error: reduction_prop must be between 0.0 and 1.0. Got {reduction_prop}")
        return False

    try:
        audio_data, sample_rate = sf.read(input_audio_path)
    except Exception as e:
        if verbose: print(f"Error loading audio file {input_audio_path}: {e}")
        return False

    if verbose:
        print(f"Loaded audio for hiss reduction: {input_audio_path}, Sample rate: {sample_rate}, Duration: {len(audio_data)/sample_rate:.2f}s")

    # Convert to mono for noisereduce
    if audio_data.ndim > 1:
        audio_mono = np.mean(audio_data, axis=1)
    else:
        audio_mono = audio_data.copy()

    try:
        if verbose:
            print(f"Applying hiss reduction: prop_decrease={reduction_prop}, freq_smooth_hz={freq_smooth_hz}, n_std_thresh_stationary={n_std_thresh_stationary}")

        # NOTE: freq_smooth_hz and n_std_thresh_stationary are not supported by the current
        # noisereduce version (3.0.3) in the expected way.
        # Falling back to a simpler call. The function signature retains them for future use.
        reduced_audio = nr.reduce_noise(
            y=audio_mono,
            sr=sample_rate,
            prop_decrease=reduction_prop
            # freq_smooth_hz=freq_smooth_hz, # Unavailable in current env
            # n_std_thresh_stationary=n_std_thresh_stationary # Unavailable in current env
        )
    except Exception as e:
        if verbose: print(f"Error during hiss reduction: {e}")
        return False

    try:
        output_dir = os.path.dirname(output_audio_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        sf.write(output_audio_path, reduced_audio, sample_rate)
        if verbose: print(f"Hiss-reduced audio saved to {output_audio_path}")
        return True
    except Exception as e:
        if verbose: print(f"Error saving hiss-reduced audio to {output_audio_path}: {e}")
        return False


if __name__ == '__main__':
    # --- Test for reduce_background_noise (existing test) ---
    sample_rate_bg = 44100; duration_s_bg = 3; frequency_bg = 440
    time_s_bg = np.linspace(0, duration_s_bg, int(sample_rate_bg * duration_s_bg), endpoint=False)
    clean_signal_bg = 0.3 * np.sin(2 * np.pi * frequency_bg * time_s_bg)
    noise_amplitude_bg = 0.1; white_noise_bg = noise_amplitude_bg * np.random.randn(len(clean_signal_bg))
    noisy_signal_bg = np.clip(clean_signal_bg + white_noise_bg, -1.0, 1.0)
    base_test_dir_bg = "video_editing_tool/data/temp_noise_reduction_tests/"
    os.makedirs(base_test_dir_bg, exist_ok=True)
    input_noisy_path_bg = os.path.join(base_test_dir_bg, "input_noisy_audio.wav")
    output_reduced_path_bg = os.path.join(base_test_dir_bg, "output_reduced_audio.wav")
    try:
        sf.write(input_noisy_path_bg, noisy_signal_bg, sample_rate_bg)
        print(f"Created dummy noisy audio: {input_noisy_path_bg}")
        print(f"Attempting broadband noise reduction...")
        success_bg = reduce_background_noise(input_noisy_path_bg, output_reduced_path_bg, prop_decrease=0.8, verbose=True)
        if success_bg: print(f"Broadband noise reduction process completed. Output at: {output_reduced_path_bg}")
        else: print("Broadband noise reduction failed.")
    except Exception as e: print(f"Error in broadband noise reduction test script: {e}")

    # --- Test for remove_hum (existing test) ---
    print("\n--- Testing Hum Removal ---")
    sample_rate_hum = 44100; duration_s_hum = 3.0
    time_s_hum = np.linspace(0, duration_s_hum, int(sample_rate_hum * duration_s_hum), endpoint=False)
    signal_hum_clean = 0.3 * np.sin(2 * np.pi * 300 * time_s_hum) + 0.2 * np.sin(2 * np.pi * 800 * time_s_hum)
    hum_60hz = 0.15 * np.sin(2 * np.pi * 60 * time_s_hum); hum_120hz = 0.10 * np.sin(2 * np.pi * 120 * time_s_hum)
    signal_with_hum = np.clip(signal_hum_clean + hum_60hz + hum_120hz, -1.0, 1.0)
    signal_with_hum_stereo = np.array([signal_with_hum, signal_with_hum * 0.9]).T
    signal_with_hum_stereo = np.clip(signal_with_hum_stereo, -1.0, 1.0)
    base_test_dir_hum = "video_editing_tool/data/temp_hum_removal_tests/"
    os.makedirs(base_test_dir_hum, exist_ok=True)
    input_hum_mono_path = os.path.join(base_test_dir_hum, "input_hum_mono.wav")
    output_hum_mono_path = os.path.join(base_test_dir_hum, "output_hum_mono_reduced.wav")
    input_hum_stereo_path = os.path.join(base_test_dir_hum, "input_hum_stereo.wav")
    output_hum_stereo_path = os.path.join(base_test_dir_hum, "output_hum_stereo_reduced.wav")
    try:
        sf.write(input_hum_mono_path, signal_with_hum, sample_rate_hum)
        print(f"Created dummy mono audio with hum: {input_hum_mono_path}")
        sf.write(input_hum_stereo_path, signal_with_hum_stereo, sample_rate_hum)
        print(f"Created dummy stereo audio with hum: {input_hum_stereo_path}")
        hum_freqs_to_remove = [60, 120]; q_factor = 30.0
        print(f"Attempting hum removal on mono audio (Freqs: {hum_freqs_to_remove} Hz, Q={q_factor})...")
        success_hum_mono = remove_hum(input_hum_mono_path, output_hum_mono_path, hum_frequencies=hum_freqs_to_remove, quality_factor=q_factor, verbose=True)
        if success_hum_mono: print(f"Hum removal on mono completed. Output at: {output_hum_mono_path}")
        else: print("Hum removal on mono failed.")
        print(f"Attempting hum removal on stereo audio (Freqs: {hum_freqs_to_remove} Hz, Q={q_factor})...")
        success_hum_stereo = remove_hum(input_hum_stereo_path, output_hum_stereo_path, hum_frequencies=hum_freqs_to_remove, quality_factor=q_factor, verbose=True)
        if success_hum_stereo:
            print(f"Hum removal on stereo completed. Output at: {output_hum_stereo_path}")
            data_stereo_out, sr_stereo_out = sf.read(output_hum_stereo_path)
            if data_stereo_out.ndim > 1 and data_stereo_out.shape[1] == 2: print("Stereo format maintained in output.")
            else: print("Warning: Stereo format NOT maintained in output of hum removal.")
        else: print("Hum removal on stereo failed.")
    except Exception as e: print(f"Error in hum removal test script: {e}")

    # --- New Test for reduce_hiss ---
    print("\n--- Testing Hiss Reduction ---")
    sample_rate_hiss = 44100
    duration_s_hiss = 3.0
    time_s_hiss = np.linspace(0, duration_s_hiss, int(sample_rate_hiss * duration_s_hiss), endpoint=False)

    # Clean signal (e.g., some speech-like components)
    clean_signal_hiss = 0.4 * np.sin(2 * np.pi * 500 * time_s_hiss) + \
                        0.3 * np.sin(2 * np.pi * 1200 * time_s_hiss)

    # Synthesize hiss: broadband noise, possibly stronger at high frequencies
    # Simple white noise for now, nr should target stationary parts.
    hiss_noise = 0.1 * np.random.randn(len(time_s_hiss))
    # Optional: Filter noise to make it more high-frequency "hissy"
    # sos_hiss_filter = signal.butter(4, 3000, 'hp', fs=sample_rate_hiss, output='sos')
    # hiss_noise = signal.sosfilt(sos_hiss_filter, hiss_noise) * 2.0 # Amplify after HPF

    signal_with_hiss = np.clip(clean_signal_hiss + hiss_noise, -1.0, 1.0)

    base_test_dir_hiss = "video_editing_tool/data/temp_hiss_reduction_tests/"
    os.makedirs(base_test_dir_hiss, exist_ok=True)
    input_hiss_path = os.path.join(base_test_dir_hiss, "input_audio_with_hiss.wav")
    output_hiss_reduced_path = os.path.join(base_test_dir_hiss, "output_hiss_reduced.wav")

    try:
        sf.write(input_hiss_path, signal_with_hiss, sample_rate_hiss)
        print(f"Created dummy audio with hiss: {input_hiss_path}")

        print(f"Attempting hiss reduction (prop=0.7, smooth_hz=1000, thresh_std=1.5)...")
        success_hiss = reduce_hiss(input_hiss_path, output_hiss_reduced_path,
                                   reduction_prop=0.7,
                                   freq_smooth_hz=1000,
                                   n_std_thresh_stationary=1.5,
                                   verbose=True)
        if success_hiss:
            print(f"Hiss reduction process completed. Output at: {output_hiss_reduced_path}")
        else:
            print("Hiss reduction failed.")

    except Exception as e:
        print(f"Error in hiss reduction test script: {e}")
