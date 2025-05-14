# ===========================================================
# START OF MODIFIED FILE: inference.py
# Adds --render_batch_size, --batch_size=1 default, Caching logic
# FP16 removed
# ===========================================================
from glob import glob
import shutil
import torch
from time import strftime, time
import os, sys
from argparse import ArgumentParser
import hashlib # For caching
import pickle # For caching

# Removed: from torch.cuda.amp import autocast

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
from scipy.io import loadmat, savemat # For loading cached mat content

# --- Caching Helper Functions ---
def get_file_sha256(filepath):
    """Calculates the SHA256 hash of a file."""
    hasher = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(4096) # Read in chunks
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"Warning: Could not hash file {filepath}: {e}")
        return None

def save_cache(data, cache_path):
    """Saves data to a cache file using pickle."""
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved cache to: {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

def load_cache(cache_path):
    """Loads data from a cache file using pickle."""
    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded cache from: {cache_path}")
        return data
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Warning: Could not load cache from {cache_path}: {e}")
        return None
# --- End Caching Helpers ---


def main(args):
    # --- Argument Parsing & Setup ---
    pic_path = args.source_image
    audio_path = args.driven_audio
    save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)
    pose_style = args.pose_style
    device = args.device
    batch_size_face_render = args.batch_size
    render_batch_size = args.render_batch_size
    input_yaw_list = args.input_yaw
    input_pitch_list = args.input_pitch
    input_roll_list = args.input_roll
    ref_eyeblink = args.ref_eyeblink
    ref_pose = args.ref_pose

    # Caching Flags
    use_cache = args.use_cache
    cache_dir = args.cache_dir

    current_root_path = os.path.split(sys.argv[0])[0]
    sadtalker_paths = init_path(args.checkpoint_dir, os.path.join(current_root_path, 'src/config'), args.size, args.old_version, args.preprocess)

    # --- Model Initialization ---
    print("Initializing models...")
    init_start_time = time()
    try:
        preprocess_model = CropAndExtract(sadtalker_paths, device)
        audio_to_coeff = Audio2Coeff(sadtalker_paths, device)
        animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device, render_batch_size=render_batch_size)
    except Exception as e:
        print(f"Error during model initialization: {e}")
        return
    print(f"Model initialization time: {time() - init_start_time:.2f}s")

    # --- 1. Preprocessing (3DMM Extraction) with Caching ---
    print("Starting 3DMM extraction...")
    dmm_start_time = time()
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir') # Temp dir for this run
    os.makedirs(first_frame_dir, exist_ok=True)

    first_coeff_path, crop_pic_path, crop_info = None, None, None
    cache_data_source = None
    source_hash = None
    source_cache_path = None

    if use_cache:
        source_hash = get_file_sha256(pic_path)
        if source_hash:
            source_cache_path = os.path.join(cache_dir, f"source_{source_hash}_sz{args.size}_pp{args.preprocess}.pkl")
            cache_data_source = load_cache(source_cache_path)

    if cache_data_source is not None:
        print("Using cached 3DMM data for source image.")
        first_coeff_path = cache_data_source['coeff_path'] # This is the path within the *cache*
        crop_pic_path = cache_data_source['png_path']     # This is the path within the *cache*
        crop_info = cache_data_source['crop_info']
        # We need the actual coefficient data, load it from the cached .mat file
        if not os.path.exists(first_coeff_path):
             print(f"Warning: Cached .mat file {first_coeff_path} not found. Recomputing.")
             cache_data_source = None # Force recomputation
        else:
             # Copy cached png to current run's dir if needed downstream (optional)
             os.makedirs(os.path.dirname(os.path.join(first_frame_dir, os.path.basename(crop_pic_path))), exist_ok=True)
             shutil.copy2(crop_pic_path, os.path.join(first_frame_dir, os.path.basename(crop_pic_path)))
             crop_pic_path = os.path.join(first_frame_dir, os.path.basename(crop_pic_path)) # Update path

    if cache_data_source is None:
        print("Computing 3DMM data for source image...")
        # No autocast here
        first_coeff_path_temp, crop_pic_path_temp, crop_info_temp = preprocess_model.generate(
            pic_path, first_frame_dir, args.preprocess,
            source_image_flag=True, pic_size=args.size
        )
        if first_coeff_path_temp is None:
            print("Error: Can't get 3DMM coefficients from the input image.")
            if os.path.exists(save_dir): shutil.rmtree(save_dir) # Clean up temp dir
            return
        first_coeff_path = first_coeff_path_temp
        crop_pic_path = crop_pic_path_temp
        crop_info = crop_info_temp

        # Save to cache if enabled and hashing worked
        if use_cache and source_cache_path:
            # Save the actual .mat and .png file content to cache dir
            cache_mat_path = os.path.join(cache_dir, os.path.basename(first_coeff_path))
            cache_png_path = os.path.join(cache_dir, os.path.basename(crop_pic_path))
            try:
                os.makedirs(cache_dir, exist_ok=True)
                shutil.copy2(first_coeff_path, cache_mat_path)
                shutil.copy2(crop_pic_path, cache_png_path)
                cache_data_to_save = {
                    'coeff_path': cache_mat_path, # Store path within cache
                    'png_path': cache_png_path,   # Store path within cache
                    'crop_info': crop_info
                }
                save_cache(cache_data_to_save, source_cache_path)
            except Exception as e:
                print(f"Warning: Failed to save source cache: {e}")

    print(f"3DMM Extraction time: {time() - dmm_start_time:.2f}s")


    # --- 2. Reference Video Processing (Optional) with Caching ---
    ref_proc_start_time = time()
    ref_eyeblink_coeff_path = None
    ref_pose_coeff_path = None

    # Function to handle reference video processing and caching
    def process_reference(ref_video_path, ref_type_str):
        ref_coeff_path_cached = None
        ref_cache_path = None
        ref_hash = None

        if use_cache:
            ref_hash = get_file_sha256(ref_video_path)
            if ref_hash:
                ref_cache_path = os.path.join(cache_dir, f"ref_{ref_hash}_pp{args.preprocess}.mat") # Cache the .mat content directly
                ref_coeff_data_cached = load_cache(ref_cache_path)
                if ref_coeff_data_cached is not None:
                    # Save cached data to a temporary .mat file for this run
                    temp_mat_filename = f"cached_{ref_type_str}_{ref_hash}.mat"
                    ref_coeff_path_cached = os.path.join(save_dir, temp_mat_filename) # Use temp dir for this run
                    try:
                        savemat(ref_coeff_path_cached, ref_coeff_data_cached)
                        print(f"Using cached coefficients for {ref_type_str} reference.")
                        return ref_coeff_path_cached
                    except Exception as e:
                        print(f"Warning: Failed to write cached .mat file {ref_coeff_path_cached}: {e}")
                        ref_coeff_path_cached = None # Force recomputation

        # If cache not used or failed
        print(f'3DMM Extraction for {ref_type_str} Reference')
        ref_videoname = os.path.splitext(os.path.split(ref_video_path)[-1])[0]
        ref_frame_dir = os.path.join(save_dir, ref_videoname) # Temp dir for this run
        os.makedirs(ref_frame_dir, exist_ok=True)

        # No autocast here
        ref_coeff_path_generated, _, _ = preprocess_model.generate(
            ref_video_path, ref_frame_dir, args.preprocess, source_image_flag=False
        )

        # Save to cache if enabled
        if use_cache and ref_cache_path and ref_coeff_path_generated and os.path.exists(ref_coeff_path_generated):
            try:
                ref_coeff_data_to_cache = loadmat(ref_coeff_path_generated)
                save_cache(ref_coeff_data_to_cache, ref_cache_path)
            except Exception as e:
                print(f"Warning: Failed to save reference cache for {ref_type_str}: {e}")

        return ref_coeff_path_generated

    if ref_eyeblink is not None:
        ref_eyeblink_coeff_path = process_reference(ref_eyeblink, "eyeblink")

    if ref_pose is not None:
        if ref_pose == ref_eyeblink and ref_eyeblink_coeff_path is not None:
            ref_pose_coeff_path = ref_eyeblink_coeff_path
            print('Using same reference coefficients for Pose and Eye Blink')
        else:
            ref_pose_coeff_path = process_reference(ref_pose, "pose")

    if ref_eyeblink or ref_pose:
        print(f"Reference Video Processing time: {time() - ref_proc_start_time:.2f}s")


    # --- 3. Audio to Coefficients ---
    print("Starting Audio to Coefficients conversion...")
    audio2coeff_start_time = time()
    batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=args.still)
    # No autocast here
    coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)
    print(f"Audio2Coeff time: {time() - audio2coeff_start_time:.2f}s")

    # --- 4. Face Rendering ---
    print("Starting Face Rendering...")
    render_start_time = time()
    # 3dface render (optional)
    if args.face3dvis:
        print("Generating 3D face visualization...")
        vis_start_time = time()
        from src.face3d.visualize import gen_composed_video
        gen_composed_video(args, device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))
        print(f"3D Face Visualization time: {time() - vis_start_time:.2f}s")

    # Prepare data for the face render generator
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path,
                                batch_size_face_render, # Use original batch_size here
                                input_yaw_list, input_pitch_list, input_roll_list,
                                expression_scale=args.expression_scale, still_mode=args.still,
                                preprocess=args.preprocess, size=args.size)

    # Call generate WITHOUT the use_fp16 flag
    # render_batch_size is handled inside AnimateFromCoeff
    result = animate_from_coeff.generate(data, save_dir, pic_path, crop_info,
                                         enhancer=args.enhancer, background_enhancer=args.background_enhancer,
                                         preprocess=args.preprocess, img_size=args.size)
    print(f"Face Rendering (make_animation) time: {time() - render_start_time:.2f}s")


    # --- 5. Post-processing and Saving ---
    print("Starting Post-processing...")
    post_start_time = time()
    if result is None:
         print("Face rendering failed. Exiting.")
         if os.path.exists(save_dir): shutil.rmtree(save_dir)
         return

    temp_final_path = os.path.join(save_dir, f"{os.path.basename(save_dir)}_final.mp4")
    shutil.move(result, temp_final_path)
    final_output_path = save_dir + '.mp4'
    shutil.move(temp_final_path, final_output_path)

    print(f'>>> The generated video is named: {final_output_path}')

    if not args.verbose:
        print("Cleaning up intermediate files...")
        shutil.rmtree(save_dir)
    print(f"Post-processing & Cleanup time: {time() - post_start_time:.2f}s")


if __name__ == '__main__':
    parser = ArgumentParser("SadTalker Inference Logic with Batching and Caching Optimization (FP32)")

    # --- Input/Output ---
    parser.add_argument("--driven_audio", default='./examples/driven_audio/bus_chinese.wav', help="path to driven audio (.wav, .mp3)")
    parser.add_argument("--source_image", default='./examples/source_image/full_body_1.png', help="path to source image/video")
    parser.add_argument("--ref_eyeblink", default=None, help="path to reference video providing eye blinking")
    parser.add_argument("--ref_pose", default=None, help="path to reference video providing pose")
    parser.add_argument("--checkpoint_dir", default='./checkpoints', help="path to model checkpoints")
    parser.add_argument("--result_dir", default='./results', help="path to save generated videos")
    parser.add_argument("--config_dir", default='./src/config', help="path to config files")

    # --- Core Animation Parameters ---
    parser.add_argument("--pose_style", type=int, default=0,  help="Pose style index (0-46)")
    parser.add_argument("--size", type=int, default=256,  help="Face model resolution (256 or 512)")
    parser.add_argument("--expression_scale", type=float, default=1.,  help="Scale factor for expression intensity")
    parser.add_argument("--preprocess", default='full', choices=['crop', 'extcrop', 'resize', 'full', 'extfull'], help="Image preprocess mode")
    parser.add_argument("--still", action="store_true", help="Enable still mode (fewer head movements, works best with preprocess 'full')")

    # --- Enhancement ---
    parser.add_argument('--enhancer',  type=str, default=None, help="Face enhancer (gfpgan, RestoreFormer) or None to disable")
    parser.add_argument('--background_enhancer',  type=str, default=None, help="Background enhancer (realesrgan) or None to disable")

    # --- Pose Control (Optional) ---
    parser.add_argument('--input_yaw', nargs='+', type=int, default=None, help="Explicit yaw control sequence (degrees)")
    parser.add_argument('--input_pitch', nargs='+', type=int, default=None, help="Explicit pitch control sequence (degrees)")
    parser.add_argument('--input_roll', nargs='+', type=int, default=None, help="Explicit roll control sequence (degrees)")

    # --- Speed/Resource Optimization ---
    parser.add_argument("--render_batch_size", type=int, default=4, help="Batch size for the face rendering step (tune based on VRAM)")
    parser.add_argument("--batch_size", type=int, default=1,  help="Batch size for data loading (set to 1 when using render_batch_size > 1)")
    parser.add_argument("--use_cache", action="store_true", help="Enable caching for 3DMM coefficients.")
    parser.add_argument("--cache_dir", type=str, default="./.cache/sadtalker_coeffs/", help="Directory to store cached 3DMM coefficients.")

    # --- Misc/Debugging ---
    parser.add_argument("--cpu", action="store_true", help="Use CPU for inference (slow)")
    parser.add_argument("--face3dvis", action="store_true", help="Generate 3d face visualization video")
    parser.add_argument("--verbose", action="store_true", help="Save intermediate outputs")
    parser.add_argument("--old_version", action="store_true", help="Use legacy pth checkpoints instead of safetensors")

    # --- 3DMM Model Parameters (Keep defaults) ---
    parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='Face recon network (internal use)')
    parser.add_argument('--init_path', type=str, default=None, help='Init path for recon net (internal use)')
    parser.add_argument('--use_last_fc', default=False, help='Use last FC in recon net (internal use)')
    parser.add_argument('--bfm_folder', type=str, default='./checkpoints/BFM_Fitting/')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='BFM model file')
    parser.add_argument('--focal', type=float, default=1015.)
    parser.add_argument('--center', type=float, default=112.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)

    args = parser.parse_args()

    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda"
    else:
        args.device = "cpu"
        print("Warning: Running on CPU. Inference will be slow.")

    # Correct BFM folder based on config_dir if necessary
    if not os.path.exists(args.bfm_folder):
         args.bfm_folder = args.config_dir

    if args.batch_size > 1 and args.render_batch_size > 1:
        print(f"Warning: --batch_size is {args.batch_size} but --render_batch_size is {args.render_batch_size}. "
              "The current batching implementation works best when --batch_size is 1. "
              "Consider setting --batch_size 1 for potentially more correct behavior.")

    main(args)
# ===========================================================
# END OF MODIFIED FILE: inference.py
# ===========================================================