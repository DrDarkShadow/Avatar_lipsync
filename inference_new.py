# ===========================================================
# START OF MODIFIED FILE: inference.py
# ===========================================================
from glob import glob
import shutil
import torch
from time import strftime
import os, sys, time
from argparse import ArgumentParser

# Added for FP16
from torch.cuda.amp import autocast

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path

def main(args):
    # torch.backends.cudnn.enabled = False # Potentially disable for determinism, but might slow down; test impact
    # torch.backends.cudnn.benchmark = True # Might speed up if input sizes are consistent

    pic_path = args.source_image
    audio_path = args.driven_audio
    save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)
    pose_style = args.pose_style
    device = args.device
    batch_size = args.batch_size
    input_yaw_list = args.input_yaw
    input_pitch_list = args.input_pitch
    input_roll_list = args.input_roll
    ref_eyeblink = args.ref_eyeblink
    ref_pose = args.ref_pose

    # Speed Optimization Flags
    use_fp16 = args.use_fp16
    use_jit = args.use_jit

    current_root_path = os.path.split(sys.argv[0])[0]

    sadtalker_paths = init_path(args.checkpoint_dir, os.path.join(current_root_path, 'src/config'), args.size, args.old_version, args.preprocess)

    # init models with optimization flags
    try:
        # Attempt JIT compilation if enabled
        preprocess_model = CropAndExtract(sadtalker_paths, device, use_jit=use_jit)
        audio_to_coeff = Audio2Coeff(sadtalker_paths, device, use_jit=use_jit)
        animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device, use_jit=use_jit)
    except Exception as e:
        print(f"Error during model initialization or JIT compilation: {e}")
        print("Falling back to standard model loading.")
        use_jit = False # Disable JIT if it failed
        preprocess_model = CropAndExtract(sadtalker_paths, device, use_jit=False)
        audio_to_coeff = Audio2Coeff(sadtalker_paths, device, use_jit=False)
        animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device, use_jit=False)


    # --- Preprocessing ---
    start_time = time.time()
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    # Wrap 3DMM extraction in autocast for potential FP16 speedup
    with autocast(enabled=use_fp16):
        first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(pic_path, first_frame_dir, args.preprocess,
                                                                             source_image_flag=True, pic_size=args.size)
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return
    print(f"3DMM Extraction time: {time.time() - start_time:.2f}s")

    # --- Reference Video Processing (if used) ---
    start_time = time.time()
    if ref_eyeblink is not None:
        ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing eye blinking')
        with autocast(enabled=use_fp16):
            ref_eyeblink_coeff_path, _, _ = preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir, args.preprocess, source_image_flag=False)
    else:
        ref_eyeblink_coeff_path = None

    if ref_pose is not None:
        if ref_pose == ref_eyeblink:
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            with autocast(enabled=use_fp16):
                ref_pose_coeff_path, _, _ = preprocess_model.generate(ref_pose, ref_pose_frame_dir, args.preprocess, source_image_flag=False)
    else:
        ref_pose_coeff_path = None
    print(f"Reference Video Processing time: {time.time() - start_time:.2f}s")

    # --- Audio to Coefficients ---
    start_time = time.time()
    batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=args.still)
    # Wrap audio->coeff generation in autocast
    with autocast(enabled=use_fp16):
        coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)
    print(f"Audio2Coeff time: {time.time() - start_time:.2f}s")


    # --- Face Rendering ---
    start_time = time.time()
    # 3dface render (optional)
    if args.face3dvis:
        from src.face3d.visualize import gen_composed_video
        gen_composed_video(args, device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))

    # coeff2video
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path,
                                batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                expression_scale=args.expression_scale, still_mode=args.still, preprocess=args.preprocess, size=args.size)

    # Pass FP16 flag to the generate method
    result = animate_from_coeff.generate(data, save_dir, pic_path, crop_info,
                                         enhancer=args.enhancer, background_enhancer=args.background_enhancer,
                                         preprocess=args.preprocess, img_size=args.size, use_fp16=use_fp16) # Pass use_fp16
    print(f"Face Rendering time: {time.time() - start_time:.2f}s")

    # --- Final Output ---
    start_time = time.time()
    # Use a unique filename to avoid potential conflicts if multiple runs save to the same base directory before moving
    temp_final_path = os.path.join(save_dir, f"{os.path.basename(save_dir)}_final.mp4")
    shutil.move(result, temp_final_path)
    # Final rename/move to the desired output name
    final_output_path = save_dir + '.mp4'
    shutil.move(temp_final_path, final_output_path)

    print('The generated video is named:', final_output_path)

    if not args.verbose:
        print("Removing intermediate files...")
        shutil.rmtree(save_dir)
    print(f"Postprocessing & Cleanup time: {time.time() - start_time:.2f}s")


if __name__ == '__main__':
    parser = ArgumentParser("SadTalker Inference Logic for Speed Optimization")
    parser.add_argument("--driven_audio", default='./examples/driven_audio/bus_chinese.wav', help="path to driven audio")
    parser.add_argument("--source_image", default='./examples/source_image/full_body_1.png', help="path to source image")
    parser.add_argument("--ref_eyeblink", default=None, help="path to reference video providing eye blinking")
    parser.add_argument("--ref_pose", default=None, help="path to reference video providing pose")
    parser.add_argument("--checkpoint_dir", default='./checkpoints', help="path to output")
    parser.add_argument("--result_dir", default='./results', help="path to output")
    parser.add_argument("--pose_style", type=int, default=0,  help="input pose style from [0, 46)")
    parser.add_argument("--batch_size", type=int, default=2,  help="the batch size of facerender")
    # Default to 256 for speed
    parser.add_argument("--size", type=int, default=256,  help="the image size of the facerender")
    parser.add_argument("--expression_scale", type=float, default=1.,  help="the batch size of facerender")
    parser.add_argument('--input_yaw', nargs='+', type=int, default=None, help="the input yaw degree of the user ")
    parser.add_argument('--input_pitch', nargs='+', type=int, default=None, help="the input pitch degree of the user")
    parser.add_argument('--input_roll', nargs='+', type=int, default=None, help="the input roll degree of the user")
    # Default enhancers to None for speed
    parser.add_argument('--enhancer',  type=str, default=None, help="Face enhancer, [gfpgan, RestoreFormer], None to disable")
    parser.add_argument('--background_enhancer',  type=str, default=None, help="background enhancer, [realesrgan], None to disable")
    parser.add_argument("--cpu", dest="cpu", action="store_true")
    parser.add_argument("--face3dvis", action="store_true", help="generate 3d face and 3d landmarks")
    parser.add_argument("--still", action="store_true", help="can crop back to the original videos for the full body animation")
    # Keep 'full' as default, but enhancers=None makes it faster
    parser.add_argument("--preprocess", default='full', choices=['crop', 'extcrop', 'resize', 'full', 'extfull'], help="how to preprocess the images" )
    parser.add_argument("--verbose",action="store_true", help="saving the intermediate output or not" )
    parser.add_argument("--old_version",action="store_true", help="use the pth other than safetensor version" )

    # Speed optimization arguments
    parser.add_argument("--use_fp16", action="store_true", help="Enable FP16 (Mixed Precision) inference. Requires compatible GPU.")
    parser.add_argument("--use_jit", action="store_true", help="Enable TorchScript (JIT compilation) for compatible models. Experimental.")

    # net structure and parameters (keep defaults, unlikely bottleneck for inference speed)
    parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='useless')
    parser.add_argument('--init_path', type=str, default=None, help='Useless')
    parser.add_argument('--use_last_fc', default=False, help='zero initialize the last fc')
    parser.add_argument('--bfm_folder', type=str, default='./checkpoints/BFM_Fitting/')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

    # default renderer parameters (keep defaults)
    parser.add_argument('--focal', type=float, default=1015.)
    parser.add_argument('--center', type=float, default=112.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)

    args = parser.parse_args()

    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda"
        if args.use_fp16:
             print("FP16 (Mixed Precision) enabled.")
        if args.use_jit:
            print("TorchScript (JIT) enabled. Note: May fail for some models.")
    else:
        args.device = "cpu"
        if args.use_fp16 or args.use_jit:
            print("Warning: FP16 and JIT are only effective on CUDA GPUs. Disabling them.")
            args.use_fp16 = False
            args.use_jit = False

    if args.enhancer == 'None': args.enhancer = None
    if args.background_enhancer == 'None': args.background_enhancer = None

    total_start_time = time.time()
    main(args)
    print(f"Total execution time: {time.time() - total_start_time:.2f}s")
# ===========================================================
# END OF MODIFIED FILE: inference.py
# ===========================================================