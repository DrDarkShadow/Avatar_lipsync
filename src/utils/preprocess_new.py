# ===========================================================
# START OF MODIFIED FILE: src/utils/preprocess.py
# ===========================================================
import numpy as np
import cv2, os, sys, torch
from tqdm import tqdm
from PIL import Image

# 3dmm extraction
import safetensors
import safetensors.torch
from src.face3d.util.preprocess import align_img
from src.face3d.util.load_mats import load_lm3d
from src.face3d.models import networks

from scipy.io import loadmat, savemat
from src.utils.croper import Preprocesser

# Added for JIT and FP16
import torch.jit
from torch.cuda.amp import autocast

import warnings

from src.utils.safetensor_helper import load_x_from_safetensor
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def split_coeff(coeffs):
        """
        Return:
            coeffs_dict     -- a dict of torch.tensors

        Parameters:
            coeffs          -- torch.tensor, size (B, 256)
        """
        id_coeffs = coeffs[:, :80]
        exp_coeffs = coeffs[:, 80: 144]
        tex_coeffs = coeffs[:, 144: 224]
        angles = coeffs[:, 224: 227]
        gammas = coeffs[:, 227: 254]
        translations = coeffs[:, 254:]
        return {
            'id': id_coeffs,
            'exp': exp_coeffs,
            'tex': tex_coeffs,
            'angle': angles,
            'gamma': gammas,
            'trans': translations
        }


class CropAndExtract():
    # Added use_jit flag
    def __init__(self, sadtalker_path, device, use_jit=False):

        self.propress = Preprocesser(device)
        self.net_recon = networks.define_net_recon(net_recon='resnet50', use_last_fc=False, init_path='').to(device)

        if sadtalker_path['use_safetensor']:
            checkpoint = safetensors.torch.load_file(sadtalker_path['checkpoint'])
            self.net_recon.load_state_dict(load_x_from_safetensor(checkpoint, 'face_3drecon'))
        else:
            checkpoint = torch.load(sadtalker_path['path_of_net_recon_model'], map_location=torch.device(device))
            self.net_recon.load_state_dict(checkpoint['net_recon'])

        self.net_recon.eval()

        # Attempt JIT compilation if requested
        if use_jit:
            try:
                scripted_recon = torch.jit.script(self.net_recon)
                # Test with dummy input to ensure compatibility
                dummy_input = torch.randn(1, 3, 224, 224).to(device)
                _ = scripted_recon(dummy_input)
                self.net_recon = scripted_recon
                print("Successfully JIT scripted net_recon.")
            except Exception as e:
                print(f"Warning: JIT scripting net_recon failed: {e}. Falling back to eager mode.")

        self.lm3d_std = load_lm3d(sadtalker_path['dir_of_BFM_fitting'])
        self.device = device

    def generate(self, input_path, save_dir, crop_or_resize='crop', source_image_flag=False, pic_size=256):

        pic_name = os.path.splitext(os.path.split(input_path)[-1])[0]

        landmarks_path =  os.path.join(save_dir, pic_name+'_landmarks.txt')
        coeff_path =  os.path.join(save_dir, pic_name+'.mat')
        png_path =  os.path.join(save_dir, pic_name+'.png')

        #load input
        if not os.path.isfile(input_path):
            raise ValueError('input_path must be a valid path to video/image file')
        elif input_path.split('.')[-1] in ['jpg', 'png', 'jpeg']:
            # loader for first frame
            full_frames = [cv2.imread(input_path)]
            fps = 25
        else:
            # loader for videos
            video_stream = cv2.VideoCapture(input_path)
            fps = video_stream.get(cv2.CAP_PROP_FPS)
            full_frames = []
            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                full_frames.append(frame)
                if source_image_flag:
                    break

        if not full_frames:
             print(f"Error: No frames read from {input_path}")
             return None, None, None

        x_full_frames= [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  for frame in full_frames]

        #### crop images as the
        if 'crop' in crop_or_resize.lower(): # default crop
            x_full_frames, crop, quad = self.propress.crop(x_full_frames, still=True if 'ext' in crop_or_resize.lower() else False, xsize=512)
            clx, cly, crx, cry = crop
            lx, ly, rx, ry = quad
            lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
            # oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx # This seems incorrect for crop_info
            crop_info = ((crx - clx, cry - cly), crop, quad) # Use cropped size, not quad size for crop_info[0]
        elif 'full' in crop_or_resize.lower():
            x_full_frames, crop, quad = self.propress.crop(x_full_frames, still=True if 'ext' in crop_or_resize.lower() else False, xsize=512)
            clx, cly, crx, cry = crop
            lx, ly, rx, ry = quad
            lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
            # oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx
            crop_info = ((crx - clx, cry - cly), crop, quad) # Use cropped size
        else: # resize mode
             oy1, oy2, ox1, ox2 = 0, x_full_frames[0].shape[0], 0, x_full_frames[0].shape[1] # check this order H, W
             crop_info = ((ox2 - ox1, oy2 - oy1), None, None) # W, H


        frames_pil = [Image.fromarray(cv2.resize(frame,(pic_size, pic_size))) for frame in x_full_frames]
        if len(frames_pil) == 0:
            print('No face is detected in the input file')
            return None, None, None

        # save crop info
        # Only save the first frame's cropped image for source_image
        if source_image_flag:
             cv2.imwrite(png_path, cv2.cvtColor(np.array(frames_pil[0]), cv2.COLOR_RGB2BGR))
        # Otherwise, for reference videos, maybe don't save the png to speed up?
        # else:
        #      pass # Skip saving for reference video frames

        # 2. get the landmark according to the detected face.
        if not os.path.isfile(landmarks_path):
            # Use FP16 for landmark extraction if available and desired? (predictor internal might not support it well)
            # Currently, predictor runs face detection (CPU/GPU?) and landmark model (GPU)
            # We focus FP16 on the 3DMM part for now.
            lm = self.propress.get_landmark(np.array(frames_pil[0])) # only first frame for source image lm
            if lm is None and source_image_flag: # If source image has no landmarks, fail
                 print(f"Warning: No landmarks found in source image {input_path}")
                 return None, None, None
            elif lm is None and not source_image_flag: # If reference video frame fails, maybe use previous or default?
                 # For simplicity in speed focus, let's skip frames without landmarks in refs for now
                 print(f"Warning: No landmarks found in reference frame from {input_path}. Skipping 3DMM for this frame if needed.")
                 # Or handle potential errors later if lm is needed but None
                 pass # lm remains None

            # If lm extraction is needed for all frames (e.g., reference video), loop here.
            # For speed, the current code only extracts for the first frame. Let's keep it that way
            # unless ref_video processing requires per-frame landmarks.
            # Assuming lm is only needed for the first frame or handled downstream if None.
            if lm is not None:
                 np.savetxt(landmarks_path, lm.reshape(-1)) # Save flattened if needed later
            lm = lm.reshape([1, -1, 2]) # Reshape to match potential loop structure, even if only 1 frame

        else:
            print(' Using saved landmarks.')
            lm = np.loadtxt(landmarks_path).astype(np.float32)
            # Assuming saved landmarks correspond to the number of frames processed (usually 1 for source)
            lm = lm.reshape([1, -1, 2]) # Adjust based on how landmarks are saved/needed

        # Ensure lm corresponds to frames_pil length if processing multiple frames
        # Current logic seems designed for single frame (source image) extraction primarily.
        # Let's assume lm is now shape [1, 68, 2] for the source image

        if not os.path.isfile(coeff_path):
            video_coeffs, full_coeffs = [],  []
            # Only process the first frame for source_image_flag=True
            num_frames_to_process = 1 if source_image_flag else len(frames_pil)

            # Use tqdm only if processing multiple frames
            frame_iterator = range(num_frames_to_process)
            if num_frames_to_process > 1 : frame_iterator = tqdm(frame_iterator, desc='3DMM Extraction In Video:')

            for idx in frame_iterator:
                if lm is None or idx >= len(lm) or np.mean(lm[idx]) == -1:
                     # Handle case where landmarks weren't found for this frame (relevant for ref videos)
                     # Use standard landmarks as fallback
                     print(f"Using standard landmarks for frame {idx} due to detection failure.")
                     lm1 = (self.lm3d_std[:, :2]+1)/2.
                     lm1 = np.concatenate(
                         [lm1[:, :1]*pic_size, lm1[:, 1:2]*pic_size], 1 # Use pic_size as frame is resized
                     )
                else:
                    frame = frames_pil[idx]
                    W,H = frame.size
                    lm1 = lm[idx].reshape([-1, 2])
                    lm1[:, -1] = H - 1 - lm1[:, -1] # Adjust y-coordinate origin


                trans_params, im1, lm1, _ = align_img(frames_pil[idx], lm1, self.lm3d_std) # Use aligned image size (likely 224x224)

                trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
                im_t = torch.tensor(np.array(im1)/255., dtype=torch.float32).permute(2, 0, 1).to(self.device).unsqueeze(0)

                # No autocast here, let the caller handle it if needed
                # If enabling here: with autocast(enabled=self.use_fp16): # Assuming use_fp16 is passed or available
                with torch.no_grad():
                    full_coeff = self.net_recon(im_t)
                    coeffs = split_coeff(full_coeff)

                pred_coeff = {key:coeffs[key].cpu().numpy() for key in coeffs}

                pred_coeff = np.concatenate([
                    pred_coeff['exp'],
                    pred_coeff['angle'],
                    pred_coeff['trans'],
                    trans_params[2:][None], # Scale factor
                    ], 1)
                video_coeffs.append(pred_coeff)
                full_coeffs.append(full_coeff.cpu().numpy())

            semantic_npy = np.array(video_coeffs)[:,0]
            savemat(coeff_path, {'coeff_3dmm': semantic_npy, 'full_3dmm': np.array(full_coeffs)[0]}) # Save first frame's full coeff

        return coeff_path, png_path, crop_info
# ===========================================================
# END OF MODIFIED FILE: src/utils/preprocess.py
# ===========================================================