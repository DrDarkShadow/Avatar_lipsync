# ===========================================================
# START OF MODIFIED FILE: src/test_audio2coeff.py
# ===========================================================
import os
import torch
import numpy as np
from scipy.io import savemat, loadmat
from yacs.config import CfgNode as CN
from scipy.signal import savgol_filter

import safetensors
import safetensors.torch

# Added for JIT and FP16
import torch.jit
from torch.cuda.amp import autocast

from src.audio2pose_models.audio2pose import Audio2Pose
from src.audio2exp_models.networks import SimpleWrapperV2
from src.audio2exp_models.audio2exp import Audio2Exp
from src.utils.safetensor_helper import load_x_from_safetensor

# load_cpk remains unchanged
def load_cpk(checkpoint_path, model=None, optimizer=None, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    if model is not None:
        # Handle potential DataParallel wrapper if saved that way
        state_dict = checkpoint['model']
        if isinstance(model, torch.nn.DataParallel) and not list(state_dict.keys())[0].startswith('module.'):
             state_dict = {'module.' + k: v for k, v in state_dict.items()}
        elif not isinstance(model, torch.nn.DataParallel) and list(state_dict.keys())[0].startswith('module.'):
             state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint['epoch']


class Audio2Coeff():
    # Added use_jit flag
    def __init__(self, sadtalker_path, device, use_jit=False):
        #load config
        fcfg_pose = open(sadtalker_path['audio2pose_yaml_path'])
        cfg_pose = CN.load_cfg(fcfg_pose)
        cfg_pose.freeze()
        fcfg_exp = open(sadtalker_path['audio2exp_yaml_path'])
        cfg_exp = CN.load_cfg(fcfg_exp)
        cfg_exp.freeze()

        # load audio2pose_model
        # Assuming Audio2Pose internal structure might be complex for JIT, try scripting its components if needed
        self.audio2pose_model = Audio2Pose(cfg_pose, None, device=device) # Wav2lip checkpoint not needed if loaded below
        self.audio2pose_model = self.audio2pose_model.to(device)
        self.audio2pose_model.eval()
        # No need to set requires_grad=False explicitly if using torch.no_grad() later

        try:
            if sadtalker_path['use_safetensor']:
                checkpoints = safetensors.torch.load_file(sadtalker_path['checkpoint'])
                self.audio2pose_model.load_state_dict(load_x_from_safetensor(checkpoints, 'audio2pose'))
            else:
                load_cpk(sadtalker_path['audio2pose_checkpoint'], model=self.audio2pose_model, device=device)
        except Exception as e:
             print(f"Failed in loading audio2pose_checkpoint: {e}")
             raise Exception("Failed in loading audio2pose_checkpoint")


        # load audio2exp_model
        netG = SimpleWrapperV2()
        netG = netG.to(device)
        netG.eval()

        try:
            if sadtalker_path['use_safetensor']:
                checkpoints = safetensors.torch.load_file(sadtalker_path['checkpoint'])
                netG.load_state_dict(load_x_from_safetensor(checkpoints, 'audio2exp'))
            else:
                load_cpk(sadtalker_path['audio2exp_checkpoint'], model=netG, device=device)
        except Exception as e:
             print(f"Failed in loading audio2exp_checkpoint: {e}")
             raise Exception("Failed in loading audio2exp_checkpoint")

        self.audio2exp_model = Audio2Exp(netG, cfg_exp, device=device, prepare_training_loss=False)
        self.audio2exp_model = self.audio2exp_model.to(device)
        self.audio2exp_model.eval()

        # Attempt JIT compilation if requested
        if use_jit:
            try:
                # Script the core network of audio2exp
                self.audio2exp_model.netG = torch.jit.script(self.audio2exp_model.netG)
                print("Successfully JIT scripted audio2exp_model.netG.")
            except Exception as e:
                print(f"Warning: JIT scripting audio2exp_model.netG failed: {e}. Falling back.")
            # Scripting the whole Audio2Pose might be tricky due to its structure (CVAE inside).
            # Let's try scripting the audio encoder within it if possible.
            try:
                self.audio2pose_model.audio_encoder = torch.jit.script(self.audio2pose_model.audio_encoder)
                print("Successfully JIT scripted audio2pose_model.audio_encoder.")
            except Exception as e:
                print(f"Warning: JIT scripting audio2pose_model.audio_encoder failed: {e}. Falling back.")
            # Scripting self.audio2pose_model.netG (the CVAE) is likely problematic.

        self.device = device

    # generate method now accepts use_fp16 flag passed from inference.py
    def generate(self, batch, coeff_save_dir, pose_style, ref_pose_coeff_path=None):

        # No autocast needed here if the caller (inference.py) wraps this call
        with torch.no_grad():
            #test
            results_dict_exp= self.audio2exp_model.test(batch)
            exp_pred = results_dict_exp['exp_coeff_pred'] #bs T 64

            batch['class'] = torch.LongTensor([pose_style]).to(self.device)
            results_dict_pose = self.audio2pose_model.test(batch)
            pose_pred = results_dict_pose['pose_pred'] #bs T 6

            pose_len = pose_pred.shape[1]
            # Apply Savitzky-Golay filter (CPU operation, won't benefit from FP16/JIT here)
            if pose_len > 1: # Need at least 3 points for order 2 polynomial
                 window_length = min(13, pose_len if pose_len % 2 != 0 else pose_len -1) # Ensure odd window length <= 13
                 if window_length >= 3: # Polyorder must be less than window_length
                     polyorder = 2
                     try:
                          pose_pred = torch.Tensor(savgol_filter(np.array(pose_pred.cpu()), window_length, polyorder, axis=1)).to(self.device)
                     except ValueError as e:
                          print(f"savgol_filter failed: {e}. Skipping smoothing.")
                          # Keep original pose_pred if filtering fails
                 else:
                     print(f"Skipping smoothing, pose length ({pose_len}) too short for window size 3.")


            coeffs_pred = torch.cat((exp_pred, pose_pred), dim=-1) #bs T 70
            coeffs_pred_numpy = coeffs_pred[0].clone().detach().cpu().numpy()

            if ref_pose_coeff_path is not None:
                 coeffs_pred_numpy = self.using_refpose(coeffs_pred_numpy, ref_pose_coeff_path)

        savemat(os.path.join(coeff_save_dir, '%s##%s.mat'%(batch['pic_name'], batch['audio_name'])),
                {'coeff_3dmm': coeffs_pred_numpy})

        return os.path.join(coeff_save_dir, '%s##%s.mat'%(batch['pic_name'], batch['audio_name']))

    def using_refpose(self, coeffs_pred_numpy, ref_pose_coeff_path):
        num_frames = coeffs_pred_numpy.shape[0]
        refpose_coeff_dict = loadmat(ref_pose_coeff_path)
        refpose_coeff = refpose_coeff_dict['coeff_3dmm'][:,64:70]
        refpose_num_frames = refpose_coeff.shape[0]
        if refpose_num_frames<num_frames:
            div = num_frames//refpose_num_frames
            re = num_frames%refpose_num_frames
            refpose_coeff_list = [refpose_coeff for i in range(div)]
            refpose_coeff_list.append(refpose_coeff[:re, :])
            refpose_coeff = np.concatenate(refpose_coeff_list, axis=0)

        #### relative head pose
        coeffs_pred_numpy[:, 64:70] = coeffs_pred_numpy[:, 64:70] + ( refpose_coeff[:num_frames, :] - refpose_coeff[0:1, :] )
        return coeffs_pred_numpy
# ===========================================================
# END OF MODIFIED FILE: src/test_audio2coeff.py
# ===========================================================