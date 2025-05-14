# ===========================================================
# START OF MODIFIED FILE: src/utils/paste_pic.py
# Replaces seamlessClone with Alpha Blending
# ===========================================================
import cv2
import os
import numpy as np
from tqdm import tqdm
import uuid
import multiprocessing # For potential parallelization

from src.utils.videoio import save_video_with_watermark # Use the potentially modified videoio

def paste_pic_alpha_blending(full_img_bgr, crop_frame_bgr, crop_info, extended_crop=False, feather_amount=0.1):
    """
    Pastes the crop_frame onto full_img using alpha blending with a feathered mask.

    Args:
        full_img_bgr: The original background image (NumPy array, BGR).
        crop_frame_bgr: The generated face frame (NumPy array, BGR).
        crop_info: Tuple containing ((crop_w, crop_h), crop_coords, quad_coords).
        extended_crop: Boolean indicating if the extended crop region should be used.
        feather_amount: Float between 0 and 0.5 controlling the feathering size relative to the face box.

    Returns:
        The blended image (NumPy array, BGR).
    """
    if len(crop_info) != 3 or crop_info[1] is None or crop_info[2] is None:
        print("Warning: Invalid crop_info provided for pasting. Returning original image.")
        return full_img_bgr

    # Extract coordinates
    _, crop_coords, quad_coords = crop_info
    clx, cly, crx, cry = crop_coords
    lx, ly, rx, ry = quad_coords
    lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)

    # Determine the paste region based on extended_crop flag
    if extended_crop:
        # Use the full crop region for pasting (less precise blending at edges)
        paste_x1, paste_y1, paste_x2, paste_y2 = clx, cly, crx, cry
    else:
        # Use the tighter quad region for pasting (better for alpha blending)
        paste_x1, paste_y1, paste_x2, paste_y2 = clx + lx, cly + ly, clx + rx, cly + ry

    # Ensure paste coordinates are within the full image bounds
    frame_h, frame_w = full_img_bgr.shape[:2]
    paste_x1 = max(0, paste_x1)
    paste_y1 = max(0, paste_y1)
    paste_x2 = min(frame_w, paste_x2)
    paste_y2 = min(frame_h, paste_y2)

    paste_w = paste_x2 - paste_x1
    paste_h = paste_y2 - paste_y1

    if paste_w <= 0 or paste_h <= 0:
        print("Warning: Invalid paste dimensions calculated. Returning original image.")
        return full_img_bgr

    # Resize the generated frame to fit the paste region
    try:
        resized_crop_frame = cv2.resize(crop_frame_bgr, (paste_w, paste_h), interpolation=cv2.INTER_LANCZOS4)
    except Exception as e:
        print(f"Error resizing crop_frame: {e}. Returning original image.")
        return full_img_bgr

    # --- Create Feathered Alpha Mask ---
    mask = np.zeros((paste_h, paste_w), dtype=np.float32)

    # Define ellipse parameters (slightly smaller than the paste box)
    center_x = paste_w // 2
    center_y = paste_h // 2
    axis_x = int(center_x * (1.0 - feather_amount))
    axis_y = int(center_y * (1.0 - feather_amount))

    # Draw filled white ellipse
    cv2.ellipse(mask, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, (1.0,), -1)

    # Feather the mask using Gaussian blur
    feather_ksize = int(max(paste_w, paste_h) * feather_amount)
    # Ensure kernel size is odd
    if feather_ksize % 2 == 0:
        feather_ksize += 1
    feather_ksize = max(3, feather_ksize) # Minimum kernel size of 3

    try:
        mask = cv2.GaussianBlur(mask, (feather_ksize, feather_ksize), 0)
    except Exception as e:
        print(f"Warning: GaussianBlur failed for mask feathering: {e}. Using hard mask.")
        # Fallback to hard mask if blur fails
        mask.fill(0)
        cv2.ellipse(mask, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, (1.0,), -1)


    mask = mask[..., np.newaxis] # Add channel dimension for broadcasting

    # --- Alpha Blending ---
    # Extract the region of interest (ROI) from the background
    roi = full_img_bgr[paste_y1:paste_y2, paste_x1:paste_x2]

    # Ensure ROI and resized frame have the same shape (can happen with edge cases)
    if roi.shape != resized_crop_frame.shape:
        print(f"Warning: ROI shape {roi.shape} differs from resized crop frame {resized_crop_frame.shape}. Attempting resize.")
        try:
            # Resize the smaller one to match the larger one
            if roi.shape[0] * roi.shape[1] < resized_crop_frame.shape[0] * resized_crop_frame.shape[1]:
                 roi = cv2.resize(roi, (resized_crop_frame.shape[1], resized_crop_frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                 mask = cv2.resize(mask, (resized_crop_frame.shape[1], resized_crop_frame.shape[0]), interpolation=cv2.INTER_LINEAR)[..., np.newaxis]
            else:
                 resized_crop_frame = cv2.resize(resized_crop_frame, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                 mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)[..., np.newaxis]

        except Exception as e:
             print(f"Error during fallback resize: {e}. Returning original image.")
             return full_img_bgr


    # Blend: foreground * alpha + background * (1 - alpha)
    blended_roi = resized_crop_frame.astype(np.float32) * mask + roi.astype(np.float32) * (1 - mask)

    # Create a copy of the full image to avoid modifying the original
    output_img = full_img_bgr.copy()
    output_img[paste_y1:paste_y2, paste_x1:paste_x2] = blended_roi.astype(np.uint8)

    return output_img


def paste_pic(video_path, pic_path, crop_info, new_audio_path, full_video_path, extended_crop=False):
    """
    Pastes generated face frames onto the original picture background using alpha blending.
    """
    if not os.path.isfile(pic_path):
        raise ValueError('pic_path must be a valid path to video/image file')
    elif pic_path.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
        full_img_bgr = cv2.imread(pic_path)
        if full_img_bgr is None:
             raise ValueError(f"Could not read image file: {pic_path}")
    else:
        # Handle video source - read only the first frame for the background
        video_stream = cv2.VideoCapture(pic_path)
        if not video_stream.isOpened():
             raise ValueError(f"Could not open video file: {pic_path}")
        still_reading, frame = video_stream.read()
        video_stream.release()
        if not still_reading or frame is None:
             raise ValueError(f"Could not read first frame from video: {pic_path}")
        full_img_bgr = frame

    frame_h, frame_w = full_img_bgr.shape[:2]

    # Read generated face video frames
    video_stream = cv2.VideoCapture(video_path)
    if not video_stream.isOpened():
        print(f"Error: Could not open generated video {video_path}")
        return
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    crop_frames = []
    while True:
        still_reading, frame = video_stream.read()
        if not still_reading:
            break
        crop_frames.append(frame)
    video_stream.release()

    if not crop_frames:
        print("Error: No frames read from generated video.")
        return

    # --- Prepare for saving the output video ---
    # Use a temporary path for writing to avoid issues with existing files
    tmp_path = os.path.join(os.path.dirname(full_video_path), str(uuid.uuid4()) + '_paste.mp4')
    out_tmp = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))
    if not out_tmp.isOpened():
         print(f"Error: Could not open VideoWriter for {tmp_path}")
         return

    # --- Process frames ---
    print("Pasting frames using Alpha Blending...")
    for crop_frame in tqdm(crop_frames, 'Pasting:'):
        if crop_frame is None:
            print("Warning: Encountered None frame in crop_frames list.")
            continue
        # Apply alpha blending
        blended_img = paste_pic_alpha_blending(full_img_bgr, crop_frame, crop_info, extended_crop)
        out_tmp.write(blended_img)

    out_tmp.release()

    # --- Mux audio (if available) and move to final path ---
    if new_audio_path and os.path.exists(new_audio_path):
        print(f"Muxing audio and saving to {full_video_path}")
        save_video_with_watermark(tmp_path, new_audio_path, full_video_path, watermark=False)
    else:
        print(f"Warning: No audio path provided or file not found. Saving video without audio to {full_video_path}")
        shutil.move(tmp_path, full_video_path)

    # Clean up intermediate video file if it still exists (might have been moved)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

# ===========================================================
# END OF MODIFIED FILE: src/utils/paste_pic.py
# ===========================================================