from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip
import os

def remove_silent_segments(video_clip: VideoFileClip, silent_timestamps: list, true_duration_override: float = None):
    """
    Removes silent segments from a video clip.

    Args:
        video_clip (VideoFileClip): The input video clip object.
        silent_timestamps (list): A list of tuples (start_sec, end_sec)
                                  representing silent segments.
        true_duration_override (float, optional): If provided, this duration is used
                                                  instead of video_clip.duration.

    Returns:
        VideoFileClip: A new video clip with silent segments removed.
                       Returns the original clip if no valid silent_timestamps are provided
                       or if timestamps lead to no valid segments to keep.
    """
    if not video_clip:
        print("Error: Invalid video clip provided.")
        return None

    if not silent_timestamps:
        print("No silent segments provided. Returning original clip.")
        return video_clip

    # Sort timestamps just in case, and merge overlapping/adjacent ones (optional, but good practice)
    # For now, assume timestamps are sorted and non-overlapping as per typical output of silence detector
    # A more robust implementation might add merging logic here.

    video_duration = true_duration_override if true_duration_override is not None else video_clip.duration
    if true_duration_override is not None:
        print(f"Using duration override: {video_duration:.2f}s (Original clip.duration: {video_clip.duration:.2f}s)")


    clips_to_keep = []
    current_time = 0.0

    # Iterate through silent segments to find the segments to keep
    for silence_start, silence_end in sorted(silent_timestamps):
        # Ensure silence times are within video duration
        silence_start = max(0, min(silence_start, video_duration))
        silence_end = max(0, min(silence_end, video_duration))

        if silence_start < silence_end: # Valid silence segment
            if current_time < silence_start:
                # Add the segment before this silence
                print(f"Keeping segment from {current_time:.2f}s to {silence_start:.2f}s")
                clips_to_keep.append(video_clip.subclip(current_time, silence_start))
            current_time = silence_end # Move current time past this silence
        else: # Invalid silence segment (start >= end)
            print(f"Skipping invalid silent segment: ({silence_start}, {silence_end})")


    # Add the final segment after the last silence (if any)
    if current_time < video_duration:
        print(f"Keeping final segment from {current_time:.2f}s to {video_duration:.2f}s")
        clips_to_keep.append(video_clip.subclip(current_time, video_duration))

    if not clips_to_keep:
        print("No segments to keep after processing silences. Returning original clip (or handling as error).")
        # Depending on desired behavior, could return None or an empty clip.
        # For now, returning original clip if silences cover everything or are malformed.
        # A better approach for "silences cover everything" might be to return a very short clip or None.
        # Let's return a 0.1s black clip if all is cut.
        if silent_timestamps and video_duration > 0.1 : # if there were silences, and it's not already tiny
            # Check if the entire duration was marked silent effectively
            is_all_silent = True
            temp_current_time = 0.0
            for s_start, s_end in sorted(silent_timestamps):
                if temp_current_time < s_start: is_all_silent = False; break
                temp_current_time = max(temp_current_time, s_end)
            if temp_current_time < video_duration : is_all_silent = False

            if is_all_silent:
                print("Warning: All content seems to be marked as silent. Returning a 0.1s black clip.")
                from moviepy.editor import ColorClip
                return ColorClip(size=video_clip.size, color=(0,0,0), duration=0.1).set_fps(video_clip.fps if video_clip.fps else 24)


        return video_clip # Fallback to original if something is odd

    if len(clips_to_keep) == 1:
        print("Only one segment to keep.")
        return clips_to_keep[0]
    else:
        print(f"Concatenating {len(clips_to_keep)} segments.")
        try:
            final_clip = concatenate_videoclips(clips_to_keep, method="compose")
            return final_clip
        except Exception as e:
            print(f"Error concatenating video clips: {e}")
            # Fallback or error handling:
            # Could try to return the longest clip, or the original, or None
            return video_clip # Fallback to original clip for now


if __name__ == '__main__':
    from moviepy.editor import ColorClip
    from moviepy.audio.AudioClip import AudioArrayClip # Corrected import for moviepy 1.0.3
    import numpy as np
    from video_editing_tool.src.video_loader import load_video_and_extract_audio # To get a VideoFileClip

    # --- Setup for testing ---
    test_output_dir = "video_editing_tool/data/temp_splicer_outputs/"
    os.makedirs(test_output_dir, exist_ok=True)

    sample_rate = 44100
    fps = 24

    # 1. Create a dummy video file using video_loader's test capability (or use an existing one)
    # This part relies on video_loader.py being able to create a test file or having one.
    # Let's create one directly here for more control in this test.

    source_video_path = os.path.join(test_output_dir, "source_video.mp4")

    # Create a 10-second video with some colored segments
    clip1 = ColorClip(size=(320,240), color=(255,0,0), duration=3).set_fps(fps) # Red 0-3s
    clip2 = ColorClip(size=(320,240), color=(0,255,0), duration=4).set_fps(fps) # Green 3-7s
    clip3 = ColorClip(size=(320,240), color=(0,0,255), duration=3).set_fps(fps) # Blue 7-10s

    # Add a dummy audio track to make it more realistic
    def make_audio_segment(duration, freq):
        return AudioArrayClip(
            0.1 * np.sin(2 * np.pi * freq * np.linspace(0, duration, int(sample_rate * duration), endpoint=False)).reshape(-1,1),
            fps=sample_rate
        )

    audio1 = make_audio_segment(3, 220)
    audio2 = make_audio_segment(4, 440) # Different pitch for the green part
    audio3 = make_audio_segment(3, 660)

    clip1 = clip1.set_audio(audio1)
    clip2 = clip2.set_audio(audio2)
    clip3 = clip3.set_audio(audio3)

    full_video_clip_for_test = concatenate_videoclips([clip1, clip2, clip3])
    full_video_clip_for_test.write_videofile(source_video_path, codec="libx264", audio_codec="aac", logger=None)

    print(f"Created source video: {source_video_path} (Duration: {full_video_clip_for_test.duration}s)")

    # Load the created video file to get a VideoFileClip instance
    # We need to use a VideoFileClip, not the in-memory ColorClip directly,
    # if remove_silent_segments expects a VideoFileClip (which it does by type hint)
    # and to simulate the real workflow.
    try:
        loaded_video_clip = VideoFileClip(source_video_path)
        print(f"Loaded source video duration: {loaded_video_clip.duration:.2f}s") # DEBUG PRINT
    except Exception as e:
        print(f"Failed to load the source video for testing: {e}")
        loaded_video_clip = None # Ensure it's defined

    if loaded_video_clip:
        actual_source_duration = full_video_clip_for_test.duration # This is 10s

        # Test case 1: Remove a segment from the middle
        silences1 = [(3.5, 6.5)] # Remove 3s from the green segment (originally 3s-7s)
        print(f"\nTest Case 1: Removing {silences1}")
        spliced_clip1 = remove_silent_segments(loaded_video_clip, silences1, true_duration_override=actual_source_duration)
        if spliced_clip1:
            output_path1 = os.path.join(test_output_dir, "spliced_video1.mp4")
            spliced_clip1.write_videofile(output_path1, codec="libx264", audio_codec="aac", logger=None)
            print(f"Spliced video 1 saved to {output_path1} (Expected duration: ~7s)")
            print(f"Actual duration: {spliced_clip1.duration:.2f}s")
            spliced_clip1.close()

        # Test case 2: Remove segments from start and end
        silences2 = [(0.0, 1.0), (8.0, 9.5)] # Remove 1s from start, 1.5s from end
        print(f"\nTest Case 2: Removing {silences2}")
        spliced_clip2 = remove_silent_segments(loaded_video_clip, silences2, true_duration_override=actual_source_duration)
        if spliced_clip2:
            output_path2 = os.path.join(test_output_dir, "spliced_video2.mp4")
            spliced_clip2.write_videofile(output_path2, codec="libx264", audio_codec="aac", logger=None)
            print(f"Spliced video 2 saved to {output_path2} (Expected duration: ~7.5s)")
            print(f"Actual duration: {spliced_clip2.duration:.2f}s")
            spliced_clip2.close()

        # Test case 3: No silences
        print(f"\nTest Case 3: No silences")
        spliced_clip3 = remove_silent_segments(loaded_video_clip, [], true_duration_override=actual_source_duration)
        if spliced_clip3: # Should be the original clip
            # Duration will be the (potentially incorrect) loaded_video_clip.duration if it's the original clip returned
            # So we compare with actual_source_duration for expectation.
            print(f"Spliced video 3 duration: {spliced_clip3.duration:.2f}s (Expected: {actual_source_duration:.2f}s if original, or if processed: {loaded_video_clip.duration:.2f}s)")
            # No need to write, it's the original or equivalent. We check duration.
            # spliced_clip3.close() # Careful if it's the same object as loaded_video_clip

        # Test case 4: Silence covers the whole video
        silences4 = [(0.0, actual_source_duration + 1)] # Cover everything based on actual duration
        print(f"\nTest Case 4: Removing all content {silences4}")
        spliced_clip4 = remove_silent_segments(loaded_video_clip, silences4, true_duration_override=actual_source_duration)
        if spliced_clip4:
            output_path4 = os.path.join(test_output_dir, "spliced_video4_all_silent.mp4")
            spliced_clip4.write_videofile(output_path4, codec="libx264", audio_codec="aac", logger=None)
            print(f"Spliced video 4 saved to {output_path4} (Expected duration: 0.1s black clip)")
            print(f"Actual duration: {spliced_clip4.duration:.2f}s")
            spliced_clip4.close()

        # Test case 5: Multiple silences
        silences5 = [(0.5, 1.5), (4.0, 5.0), (8.0, 9.0)] # 1s, 1s, 1s removed = 3s total removed
        print(f"\nTest Case 5: Multiple silences {silences5}")
        spliced_clip5 = remove_silent_segments(loaded_video_clip, silences5, true_duration_override=actual_source_duration)
        if spliced_clip5:
            output_path5 = os.path.join(test_output_dir, "spliced_video5_multiple.mp4")
            spliced_clip5.write_videofile(output_path5, codec="libx264", audio_codec="aac", logger=None)
            print(f"Spliced video 5 saved to {output_path5} (Expected duration: ~7s)")
            print(f"Actual duration: {spliced_clip5.duration:.2f}s")
            spliced_clip5.close()

        # Clean up the main loaded clip
        loaded_video_clip.close()
    else:
        print("Skipping video splicer tests as source video could not be loaded.")

    # Note: For robust cleanup, the test_output_dir could be removed here,
    # but it's left for inspection for now. Proper unit tests would handle this better.
    # import shutil
    # shutil.rmtree(test_output_dir)
    # print(f"Cleaned up test directory: {test_output_dir}")
