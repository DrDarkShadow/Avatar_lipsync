# ===========================================================
# START OF MODIFIED FILE: src/facerender/animate.py
# Uses Batched Frame Rendering ONLY (FP16 removed)
# Includes fix for even frame dimensions
# ===========================================================
import os
import cv2
import yaml
import numpy as np
import warnings
from skimage import img_as_ubyte
import safetensors
import safetensors.torch
warnings.filterwarnings('ignore')

import imageio
import torch
import torchvision
import shutil

# Import the MODIFIED make_animation (with batching only)
from src.facerender.modules.make_animation import make_animation
# Import other necessary components
from src.facerender.modules.keypoint_detector import HEEstimator, KPDetector
from src.facerender.modules.mapping import MappingNet
from src.facerender.modules.generator import OcclusionAwareSPADEGenerator # Assuming SPADE

from pydub import AudioSegment
from src.utils.face_enhancer import enhancer_generator_with_len, enhancer_list
from src.utils.paste_pic import paste_pic
from src.utils.videoio import save_video_with_watermark # Use the potentially modified videoio

try:
    import webui  # in webui
    in_webui = True
except:
    in_webui = False

import random
# --- make_blink_schedule function (if you added it) remains the same ---
def make_blink_schedule(n_frames, avg_interval=80, blink_len=4):
    schedule = set()
    t = 0
    while t<n_frames:
        interval = random.randint(max(10, avg_interval - 20), avg_interval + 20)
        t += interval
        for b in range(blink_len):
            frame_index = min(t + b, n_frames - 1)
            if frame_index >= 0:
                 schedule.add(frame_index)
    return schedule


class AnimateFromCoeff():

    # Add render_batch_size to __init__
    def __init__(self, sadtalker_path, device, render_batch_size=4): # Added default

        with open(sadtalker_path['facerender_yaml']) as f:
            config = yaml.safe_load(f)

        # Instantiate the OcclusionAwareSPADEGenerator (or correct one)
        generator_type = config['model_params'].get('generator_params', {}).get('type', 'OcclusionAwareSPADEGenerator')
        if generator_type == 'OcclusionAwareSPADEGenerator':
             print("Using OcclusionAwareSPADEGenerator.")
             generator = OcclusionAwareSPADEGenerator(**config['model_params']['generator_params'],
                                                     **config['model_params']['common_params'])
        else:
             print(f"Using {generator_type} (or fallback OcclusionAwareGenerator).")
             from src.facerender.modules.generator import OcclusionAwareGenerator
             generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                                 **config['model_params']['common_params'])

        kp_extractor = KPDetector(**config['model_params']['kp_detector_params'],
                                    **config['model_params']['common_params'])
        he_estimator = HEEstimator(**config['model_params']['he_estimator_params'],
                               **config['model_params']['common_params'])
        mapping = MappingNet(**config['model_params']['mapping_params'])

        generator.to(device)
        kp_extractor.to(device)
        he_estimator.to(device)
        mapping.to(device)

        # Set requires_grad = False
        for param in generator.parameters():
            param.requires_grad_(False)
        for param in kp_extractor.parameters():
            param.requires_grad_(False)
        for param in he_estimator.parameters():
            param.requires_grad_(False)
        for param in mapping.parameters():
            param.requires_grad_(False)

        # Load checkpoints (keep robust loading logic)
        if sadtalker_path is not None:
            if 'checkpoint' in sadtalker_path and sadtalker_path.get('use_safetensor', False):
                print("Loading safetensors checkpoint:", sadtalker_path['checkpoint'])
                self.load_cpk_facevid2vid_safetensor(sadtalker_path['checkpoint'], kp_detector=kp_extractor, generator=generator, he_estimator=he_estimator)
            elif 'free_view_checkpoint' in sadtalker_path:
                 print("Loading pth.tar checkpoint:", sadtalker_path['free_view_checkpoint'])
                 self.load_cpk_facevid2vid(sadtalker_path['free_view_checkpoint'], kp_detector=kp_extractor, generator=generator, he_estimator=he_estimator)
            else:
                 raise AttributeError("Cannot find valid checkpoint path for generator/kp_extractor in sadtalker_paths.")
        else:
            raise AttributeError("sadtalker_path dictionary is None.")

        # --- MappingNet Loading Logic (Keep as before) ---
        if 'mappingnet_checkpoint' in sadtalker_path and sadtalker_path['mappingnet_checkpoint'] is not None and os.path.exists(sadtalker_path['mappingnet_checkpoint']):
             print("Loading MappingNet from separate file:", sadtalker_path['mappingnet_checkpoint'])
             self.load_cpk_mapping(sadtalker_path['mappingnet_checkpoint'], mapping=mapping)
        elif 'checkpoint' in sadtalker_path and sadtalker_path.get('use_safetensor', False):
             print("Attempting to load MappingNet from main safetensor checkpoint.")
             try:
                  checkpoint_map = safetensors.torch.load_file(sadtalker_path['checkpoint'])
                  mapping_state_dict = {}
                  prefixes_to_try = ['mapping.', 'map_net.']
                  loaded_from_main = False
                  for prefix in prefixes_to_try:
                       mapping_state_dict = {k.replace(prefix, ''): v for k, v in checkpoint_map.items() if k.startswith(prefix)}
                       if mapping_state_dict:
                            mapping.load_state_dict(mapping_state_dict)
                            print(f"Loaded MappingNet from main checkpoint using prefix '{prefix}'.")
                            loaded_from_main = True
                            break
                  if not loaded_from_main:
                       raise KeyError("MappingNet keys not found in main safetensor with known prefixes.")
             except Exception as e:
                  print(f"Error loading MappingNet from main checkpoint: {e}")
                  raise AttributeError("MappingNet checkpoint is required but failed to load.")
        else:
             raise AttributeError("MappingNet checkpoint path is required but could not be found or loaded.")
        # --- End MappingNet Loading ---

        self.kp_extractor = kp_extractor
        self.generator = generator
        self.he_estimator = he_estimator
        self.mapping = mapping

        self.kp_extractor.eval()
        self.generator.eval()
        self.he_estimator.eval()
        self.mapping.eval()

        self.device = device
        self.render_batch_size = render_batch_size # Store the batch size

    # --- load_cpk methods (keep robust versions) ---
    def load_cpk_facevid2vid_safetensor(self, checkpoint_path, generator=None,
                        kp_detector=None, he_estimator=None,
                        device="cpu"):
        # ... (keep robust loading from previous step) ...
        try:
            checkpoint = safetensors.torch.load_file(checkpoint_path)
        except Exception as e:
            print(f"Error loading safetensor file {checkpoint_path}: {e}")
            return None

        generator_sd = {k.replace('generator.', ''): v for k, v in checkpoint.items() if k.startswith('generator.')}
        kp_sd = {k.replace('kp_extractor.', ''): v for k, v in checkpoint.items() if k.startswith('kp_extractor.')}
        he_sd = {k.replace('he_estimator.', ''): v for k, v in checkpoint.items() if k.startswith('he_estimator.')}

        if generator is not None and generator_sd:
            try:
                generator.load_state_dict(generator_sd, strict=True)
                print("Generator loaded successfully from safetensor.")
            except RuntimeError as e:
                print(f"Warning: Error loading generator state_dict (strict=True): {e}. Trying strict=False.")
                try: generator.load_state_dict(generator_sd, strict=False)
                except Exception as e2: print(f"Error loading generator state_dict (strict=False): {e2}")

        if kp_detector is not None and kp_sd:
             try:
                kp_detector.load_state_dict(kp_sd, strict=True)
                print("KP Detector loaded successfully from safetensor.")
             except RuntimeError as e:
                print(f"Warning: Error loading kp_detector state_dict (strict=True): {e}. Trying strict=False.")
                try: kp_detector.load_state_dict(kp_sd, strict=False)
                except Exception as e2: print(f"Error loading kp_detector state_dict (strict=False): {e2}")

        if he_estimator is not None and he_sd:
             try:
                he_estimator.load_state_dict(he_sd, strict=True)
                print("HE Estimator loaded successfully from safetensor.")
             except RuntimeError as e:
                print(f"Warning: Error loading he_estimator state_dict (strict=True): {e}. Trying strict=False.")
                try: he_estimator.load_state_dict(he_sd, strict=False)
                except Exception as e2: print(f"Error loading he_estimator state_dict (strict=False): {e2}")
        elif he_estimator is not None:
             print("Warning: he_estimator keys not found in safetensor checkpoint.")
        return None

    def load_cpk_facevid2vid(self, checkpoint_path, generator=None, discriminator=None,
                        kp_detector=None, he_estimator=None, optimizer_generator=None,
                        optimizer_discriminator=None, optimizer_kp_detector=None,
                        optimizer_he_estimator=None, device="cpu"):
        # ... (keep robust loading from previous step) ...
        try:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        except Exception as e:
            print(f"Error loading checkpoint file {checkpoint_path}: {e}")
            return 0

        if generator is not None and 'generator' in checkpoint:
            try: generator.load_state_dict(checkpoint['generator'], strict=True)
            except RuntimeError as e:
                print(f"Warning: Error loading generator from pth.tar (strict=True): {e}. Trying strict=False.")
                try: generator.load_state_dict(checkpoint['generator'], strict=False)
                except Exception as e2: print(f"Error loading generator state_dict (strict=False): {e2}")

        if kp_detector is not None and 'kp_detector' in checkpoint:
             try: kp_detector.load_state_dict(checkpoint['kp_detector'], strict=True)
             except RuntimeError as e:
                print(f"Warning: Error loading kp_detector from pth.tar (strict=True): {e}. Trying strict=False.")
                try: kp_detector.load_state_dict(checkpoint['kp_detector'], strict=False)
                except Exception as e2: print(f"Error loading kp_detector state_dict (strict=False): {e2}")

        if he_estimator is not None and 'he_estimator' in checkpoint:
             try: he_estimator.load_state_dict(checkpoint['he_estimator'], strict=True)
             except RuntimeError as e:
                print(f"Warning: Error loading he_estimator from pth.tar (strict=True): {e}. Trying strict=False.")
                try: he_estimator.load_state_dict(checkpoint['he_estimator'], strict=False)
                except Exception as e2: print(f"Error loading he_estimator state_dict (strict=False): {e2}")
        elif he_estimator is not None:
             print("Warning: he_estimator key not found in the .pth.tar checkpoint.")

        # --- Load optimizers and discriminator if they exist ---
        if discriminator is not None and 'discriminator' in checkpoint:
            try: discriminator.load_state_dict(checkpoint['discriminator'])
            except: print ('No discriminator in the state-dict. Dicriminator will be randomly initialized')
        if optimizer_generator is not None and 'optimizer_generator' in checkpoint:
            optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
        if optimizer_discriminator is not None and 'optimizer_discriminator' in checkpoint:
            try: optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
            except RuntimeError as e: print ('No discriminator optimizer in the state-dict. Optimizer will be not initialized')
        if optimizer_kp_detector is not None and 'optimizer_kp_detector' in checkpoint:
            optimizer_kp_detector.load_state_dict(checkpoint['optimizer_kp_detector'])
        if optimizer_he_estimator is not None and 'optimizer_he_estimator' in checkpoint:
            optimizer_he_estimator.load_state_dict(checkpoint['optimizer_he_estimator'])

        return checkpoint.get('epoch', 0)

    def load_cpk_mapping(self, checkpoint_path, mapping=None, discriminator=None,
                 optimizer_mapping=None, optimizer_discriminator=None, device='cpu'):
        # ... (keep robust loading from previous step) ...
        try:
            checkpoint = torch.load(checkpoint_path,  map_location=torch.device(device))
        except Exception as e:
            print(f"Error loading mapping checkpoint file {checkpoint_path}: {e}")
            return 0

        if mapping is not None and 'mapping' in checkpoint:
            try: mapping.load_state_dict(checkpoint['mapping'], strict=True)
            except RuntimeError as e:
                print(f"Warning: Error loading mapping net from pth.tar (strict=True): {e}. Trying strict=False.")
                try: mapping.load_state_dict(checkpoint['mapping'], strict=False)
                except Exception as e2: print(f"Error loading mapping net state_dict (strict=False): {e2}")

        if discriminator is not None and 'discriminator' in checkpoint:
            discriminator.load_state_dict(checkpoint['discriminator'])
        if optimizer_mapping is not None and 'optimizer_mapping' in checkpoint:
            optimizer_mapping.load_state_dict(checkpoint['optimizer_mapping'])
        if optimizer_discriminator is not None and 'optimizer_discriminator' in checkpoint:
            optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])

        return checkpoint.get('epoch', 0)

    # Removed use_fp16 parameter from generate
    def generate(self, x, video_save_dir, pic_path, crop_info, enhancer=None, background_enhancer=None, preprocess='crop', img_size=256):

        source_image=x['source_image'].type(torch.FloatTensor)
        source_semantics=x['source_semantics'].type(torch.FloatTensor)
        target_semantics=x['target_semantics_list'].type(torch.FloatTensor)
        source_image=source_image.to(self.device)
        source_semantics=source_semantics.to(self.device)
        target_semantics=target_semantics.to(self.device)

        yaw_c_seq = x.get('yaw_c_seq')
        if yaw_c_seq is not None:
            yaw_c_seq = yaw_c_seq.type(torch.FloatTensor).to(self.device)
        pitch_c_seq = x.get('pitch_c_seq')
        if pitch_c_seq is not None:
            pitch_c_seq = pitch_c_seq.type(torch.FloatTensor).to(self.device)
        roll_c_seq = x.get('roll_c_seq')
        if roll_c_seq is not None:
            roll_c_seq = roll_c_seq.type(torch.FloatTensor).to(self.device)

        frame_num = x['frame_num']

        # Call make_animation with batching, but WITHOUT FP16 flag
        predictions_video = make_animation(source_image, source_semantics, target_semantics,
                                        self.generator, self.kp_extractor, self.he_estimator, self.mapping,
                                        yaw_c_seq, pitch_c_seq, roll_c_seq, use_exp = True,
                                        render_batch_size=self.render_batch_size) # Pass batch size

        predictions_video = predictions_video.reshape((-1,)+predictions_video.shape[2:])
        predictions_video = predictions_video[:frame_num]

        video = []
        for idx in range(predictions_video.shape[0]):
            image = predictions_video[idx]
            image = np.transpose(image.data.cpu().numpy(), [1, 2, 0]).astype(np.float32)
            video.append(image)
        result = img_as_ubyte(video)

        # --- Resize and ensure even dimensions (Keep this fix) ---
        original_size = crop_info[0]
        target_width = img_size

        if original_size and original_size[0] > 0 and original_size[1] > 0:
            aspect_ratio = original_size[1] / original_size[0]
            target_height = int(target_width * aspect_ratio)
            if target_width % 2 != 0: target_width += 1
            if target_height % 2 != 0: target_height += 1
            print(f"Resizing generated frames to: ({target_width}, {target_height})")
            try:
                result = [ cv2.resize(result_i, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4) for result_i in result ]
            except Exception as e:
                print(f"Error during cv2.resize: {e}. Using original size.")
        else:
             print("Warning: Invalid original_size in crop_info. Skipping resize.")
        # --- End of resize modification ---

        video_name = x['video_name']  + '.mp4'
        path = os.path.join(video_save_dir, 'temp_'+video_name)

        try:
            imageio.mimsave(path, result, fps=float(25), macro_block_size=1)
        except Exception as e:
            print(f"Error saving intermediate video with imageio: {e}")
            return None

        av_path = os.path.join(video_save_dir, video_name)
        return_path = av_path

        # --- Audio Processing (Keep robust version) ---
        audio_path =  x['audio_path']
        audio_name = os.path.splitext(os.path.split(audio_path)[-1])[0]
        new_audio_path = os.path.join(video_save_dir, audio_name+'.wav')
        new_audio_exists = False
        try:
             sound = AudioSegment.from_file(audio_path)
             frames = frame_num
             end_time = frames * (1000 / 25) # Calculate end time in ms
             word1=sound.set_frame_rate(16000)
             word = word1[:end_time] # Slice audio to match video length
             word.export(new_audio_path, format="wav")
             new_audio_exists = True
        except Exception as e:
             print(f"Error processing audio {audio_path}: {e}")
             new_audio_path = None

        # --- Mux Video and Audio ---
        if new_audio_exists and os.path.exists(new_audio_path):
             save_video_with_watermark(path, new_audio_path, av_path, watermark= False)
             print(f'Generated video with audio: {av_path}')
        else:
             print(f"Warning: Audio processing failed or missing. Saving video without audio at {av_path}")
             if os.path.exists(path): # Ensure temp video exists before moving
                  shutil.move(path, av_path)
             else:
                  print(f"Error: Temp video file {path} not found for moving.")
                  return None # Indicate failure

        # --- Paste Back for 'full' mode (Keep robust version) ---
        if 'full' in preprocess.lower():
            video_name_full = x['video_name']  + '_full.mp4'
            full_video_path = os.path.join(video_save_dir, video_name_full)
            return_path = full_video_path
            if new_audio_exists and os.path.exists(new_audio_path):
                 paste_pic(av_path, pic_path, crop_info, new_audio_path, full_video_path, extended_crop= True if 'ext' in preprocess.lower() else False)
                 print(f'Generated full video with audio: {full_video_path}')
            else:
                 print("Pasting picture back without audio...")
                 paste_pic(av_path, pic_path, crop_info, None, full_video_path, extended_crop= True if 'ext' in preprocess.lower() else False)
                 print(f'Generated full video (no audio muxed in paste_pic): {full_video_path}')
        else:
            full_video_path = av_path

        # --- Enhancement (Keep robust version) ---
        if enhancer and os.path.exists(full_video_path):
            video_name_enhancer = x['video_name']  + '_enhanced.mp4'
            enhanced_path = os.path.join(video_save_dir, 'temp_'+video_name_enhancer)
            av_path_enhancer = os.path.join(video_save_dir, video_name_enhancer)
            return_path = av_path_enhancer

            print(f"Running enhancer {enhancer}...")
            try:
                enhanced_images_gen_with_len = enhancer_generator_with_len(full_video_path, method=enhancer, bg_upsampler=background_enhancer)
                imageio.mimsave(enhanced_path, enhanced_images_gen_with_len, fps=float(25), macro_block_size=1)
            except Exception as e:
                 print(f"Error during enhancement: {e}. Trying list-based enhancement.")
                 enhanced_images_list = enhancer_list(full_video_path, method=enhancer, bg_upsampler=background_enhancer)
                 if enhanced_images_list:
                      imageio.mimsave(enhanced_path, enhanced_images_list, fps=float(25), macro_block_size=1)
                 else:
                      print("Enhancement failed to produce frames.")
                      enhanced_path = None

            if enhanced_path and os.path.exists(enhanced_path):
                 if new_audio_exists and os.path.exists(new_audio_path):
                      save_video_with_watermark(enhanced_path, new_audio_path, av_path_enhancer, watermark= False)
                      print(f'Generated enhanced video with audio: {av_path_enhancer}')
                 else:
                      print(f"Warning: Audio missing. Saving enhanced video without audio at {av_path_enhancer}")
                      shutil.move(enhanced_path, av_path_enhancer)
                 os.remove(enhanced_path)
            else:
                 print("Enhancement failed, returning non-enhanced video path.")
                 return_path = full_video_path

        # --- Cleanup ---
        if os.path.exists(path):
             os.remove(path)
        if new_audio_exists and os.path.exists(new_audio_path):
             os.remove(new_audio_path)

        return return_path
# ===========================================================
# END OF MODIFIED FILE: src/facerender/animate.py
# ===========================================================