# ===========================================================
# START OF MODIFIED FILE: src/utils/videoio.py
# ===========================================================
import shutil
import uuid
import subprocess # Use subprocess for more control over ffmpeg
import platform

import os

import cv2

def load_video_to_cv2(input_path):
    video_stream = cv2.VideoCapture(input_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    full_frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        # Ensure frame is in RGB if needed by downstream enhancers,
        # although cv2 reads in BGR by default.
        # Keep as BGR for internal processing like paste_pic unless specified otherwise.
        # full_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        full_frames.append(frame) # Keep BGR for consistency with cv2.VideoWriter
    return full_frames

def save_video_with_watermark(video_path, audio_path, save_path, watermark=False, output_fps=25):
    """
    Saves the video with optional audio muxing and watermarking.
    Uses a faster ffmpeg preset.
    """
    temp_file = str(uuid.uuid4())+'.mp4'
    temp_no_audio_file = str(uuid.uuid4())+'_noaudio.mp4'

    # Check if input video has audio stream already
    # If yes, we might need to replace it or handle it differently.
    # For simplicity, let's assume input 'video_path' is video-only for now.

    # Base command parts
    ffmpeg_cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error']

    # Input video
    ffmpeg_cmd.extend(['-i', video_path])

    # Input audio (if provided)
    if audio_path and os.path.exists(audio_path):
        ffmpeg_cmd.extend(['-i', audio_path])
        # Map streams: 0:v (video from first input), 1:a (audio from second input)
        ffmpeg_cmd.extend(['-map', '0:v:0', '-map', '1:a:0'])
        # Use AAC for audio, common for MP4
        ffmpeg_cmd.extend(['-c:a', 'aac'])
        # Add shortest flag to stop encoding when the shorter stream ends (typically audio)
        ffmpeg_cmd.extend(['-shortest'])
    else:
        # Map only video stream if no audio
        ffmpeg_cmd.extend(['-map', '0:v:0'])

    # Video codec and preset for speed
    # libx264 is widely compatible. ultrafast is fastest but lowest quality/compression.
    # Consider 'superfast' or 'veryfast' for a better balance if needed.
    ffmpeg_cmd.extend(['-c:v', 'libx264', '-preset', 'ultrafast'])
    # Constant Rate Factor (CRF): Lower values mean better quality, larger file size. 23 is default.
    # Higher values (e.g., 28, 30) for faster encoding, smaller file, lower quality.
    ffmpeg_cmd.extend(['-crf', '28'])
    # Ensure output fps
    ffmpeg_cmd.extend(['-r', str(output_fps)])

    # Output to temporary file
    ffmpeg_cmd.append(temp_file)

    print(f"Running ffmpeg command: {' '.join(ffmpeg_cmd)}")
    try:
        subprocess.run(ffmpeg_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during ffmpeg muxing: {e}")
        # Fallback: copy video only if muxing failed
        print("Attempting to copy video stream only...")
        try:
            ffmpeg_cmd_video_only = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-i', video_path, '-c:v', 'copy', temp_no_audio_file]
            subprocess.run(ffmpeg_cmd_video_only, check=True)
            # If copy succeeds, use this temp file instead
            if os.path.exists(temp_file): os.remove(temp_file) # Remove potentially failed muxed file
            temp_file = temp_no_audio_file
        except subprocess.CalledProcessError as e2:
             print(f"Error copying video stream either: {e2}")
             if os.path.exists(temp_file): os.remove(temp_file) # Clean up
             if os.path.exists(temp_no_audio_file): os.remove(temp_no_audio_file)
             return # Cannot proceed


    if not os.path.exists(temp_file):
        print(f"Error: {temp_file} was not created by ffmpeg.")
        return

    if watermark is False:
        print(f"Moving {temp_file} to {save_path}")
        shutil.move(temp_file, save_path)
    else:
        print("Adding watermark...")
        # Determine watermark path
        try:
            import webui
            from modules import paths
            watermark_path = os.path.join(paths.script_path, "extensions", "SadTalker", "docs", "sadtalker_logo.png")
            if not os.path.exists(watermark_path): # Fallback if extension path incorrect
                 dir_path = os.path.dirname(os.path.realpath(__file__))
                 watermark_path = os.path.join(dir_path, "..", "..", "docs", "sadtalker_logo.png")

        except ImportError:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            watermark_path = os.path.join(dir_path, "..", "..", "docs", "sadtalker_logo.png")

        if not os.path.exists(watermark_path):
             print(f"Warning: Watermark image not found at {watermark_path}. Skipping watermark.")
             shutil.move(temp_file, save_path)
             return

        # Watermark command (re-encodes video, potentially slow depending on preset)
        # Using '-preset ultrafast' again for consistency
        watermark_cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-i', temp_file,
            '-i', watermark_path,
            '-filter_complex', "[1]scale=100:-1[wm];[0][wm]overlay=(main_w-overlay_w)-10:10",
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28', # Apply fast preset here too
            '-c:a', 'copy', # Copy audio stream if it exists
            save_path
        ]
        print(f"Running ffmpeg watermark command: {' '.join(watermark_cmd)}")

        try:
            subprocess.run(watermark_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error applying watermark: {e}")
            print("Saving video without watermark instead.")
            shutil.move(temp_file, save_path) # Save the unwatermarked version
        finally:
             if os.path.exists(temp_file):
                 os.remove(temp_file)

# ===========================================================
# END OF MODIFIED FILE: src/utils/videoio.py
# ===========================================================