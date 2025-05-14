# ===========================================================
# START OF MODIFIED FILE: src/facerender/modules/make_animation.py
# Implements Batched Frame Rendering ONLY (FP16 removed)
# ===========================================================
from scipy.spatial import ConvexHull
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
# Removed: from torch.cuda.amp import autocast

# --- Helper Functions (Standard FP32 versions) ---

def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        with torch.no_grad():
             source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
             driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        driving_area = max(driving_area, 1e-6) # Avoid division by zero
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian and 'jacobian' in kp_driving and kp_driving.get('jacobian') is not None \
           and 'jacobian' in kp_driving_initial and kp_driving_initial.get('jacobian') is not None \
           and 'jacobian' in kp_source and kp_source.get('jacobian') is not None:
            try:
                dev = kp_driving['jacobian'].device
                inv_jacobian_initial = torch.inverse(kp_driving_initial['jacobian'].to(dev))
                jacobian_diff = torch.matmul(kp_driving['jacobian'].to(dev), inv_jacobian_initial)
                kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'].to(dev))
            except Exception as e:
                 kp_new['jacobian'] = kp_driving.get('jacobian')
        elif 'jacobian' in kp_driving:
             kp_new['jacobian'] = kp_driving.get('jacobian')

    return kp_new

def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).type_as(pred).to(device)
    if pred.ndim > 1 and pred.shape[1] == 66:
        pred = F.softmax(pred, dim=1)
    else:
        pred = pred / torch.sum(pred, dim=1, keepdim=True).clamp(min=1e-8)
    degree = torch.sum(pred*idx_tensor, 1) * 3 - 99
    return degree

def get_rotation_matrix(yaw, pitch, roll):
    device = yaw.device
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    ones = torch.ones_like(pitch)
    zeros = torch.zeros_like(pitch)

    pitch_mat = torch.cat([ones, zeros, zeros,
                          zeros, torch.cos(pitch), -torch.sin(pitch),
                          zeros, torch.sin(pitch), torch.cos(pitch)], dim=1)
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat([torch.cos(yaw), zeros, torch.sin(yaw),
                           zeros, ones, zeros,
                           -torch.sin(yaw), zeros, torch.cos(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    roll_mat = torch.cat([torch.cos(roll), -torch.sin(roll), zeros,
                         torch.sin(roll), torch.cos(roll), zeros,
                         zeros, zeros, ones], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    rot_mat = torch.einsum('bij,bjk,bkm->bim', pitch_mat.to(device), yaw_mat.to(device), roll_mat.to(device))

    return rot_mat

def keypoint_transformation(kp_canonical, he, wo_exp=False):
    kp = kp_canonical['value']
    device = kp.device

    yaw, pitch, roll = he['yaw'].to(device), he['pitch'].to(device), he['roll'].to(device)
    t, exp = he['t'].to(device), he['exp'].to(device)

    yaw = headpose_pred_to_degree(yaw)
    pitch = headpose_pred_to_degree(pitch)
    roll = headpose_pred_to_degree(roll)

    if 'yaw_in' in he:
        yaw = he['yaw_in'].to(device)
    if 'pitch_in' in he:
        pitch = he['pitch_in'].to(device)
    if 'roll_in' in he:
        roll = he['roll_in'].to(device)

    rot_mat = get_rotation_matrix(yaw, pitch, roll)

    if wo_exp:
        exp = exp * 0

    kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp)

    if t.dim() == 2:
       t = t.unsqueeze(1)
    if t.shape[1] != 1 or t.shape[2] != 3:
         t = he['t'].to(device).unsqueeze(1)

    kp_t = kp_rotated + t

    exp = exp.view(exp.shape[0], -1, 3)
    kp_transformed = kp_t + exp

    kp_result = {'value': kp_transformed}
    if 'jacobian' in kp_canonical and kp_canonical.get('jacobian') is not None:
         jacobian = kp_canonical['jacobian'].to(device)
         jacobian_transformed = torch.einsum('bmp,bkpq->bkmq', rot_mat, jacobian)
         kp_result['jacobian'] = jacobian_transformed

    return kp_result


# --- MODIFIED make_animation function with BATCHING ONLY ---
def make_animation(source_image, source_semantics, target_semantics,
                   generator, kp_detector, he_estimator, mapping,
                   yaw_c_seq=None, pitch_c_seq=None, roll_c_seq=None,
                   use_exp=True, use_half=False, # use_half not used
                   render_batch_size=4): # <-- Batching parameter

    bs = source_image.shape[0]
    if bs != 1 and render_batch_size > 1:
        print(f"Warning: Batching animation generation (render_batch_size>1) is implemented assuming input batch size (bs) is 1, but got bs={bs}. Results might be incorrect.")

    predictions_list = []

    with torch.no_grad():
        # --- Initial Calculations (run once) ---
        # No autocast here
        kp_canonical = kp_detector(source_image)
        he_source = mapping(source_semantics)
        kp_source = keypoint_transformation(kp_canonical, he_source, wo_exp=(not use_exp))

        num_frames = target_semantics.shape[1]

        # --- Loop through frames in batches ---
        for i in tqdm(range(0, num_frames, render_batch_size), 'Face Renderer Batched:'):
            start_idx = i
            end_idx = min(i + render_batch_size, num_frames)
            current_render_batch_size = end_idx - start_idx

            target_semantics_batch = target_semantics[:, start_idx:end_idx]

            # --- Prepare driving keypoints for the batch ---
            kp_driving_value_list = []
            kp_driving_jacobian_list = []

            for frame_offset in range(current_render_batch_size):
                frame_abs_idx = start_idx + frame_offset
                target_semantics_frame = target_semantics_batch[:, frame_offset]

                # No autocast here
                he_driving = mapping(target_semantics_frame)

                if yaw_c_seq is not None:
                    he_driving['yaw_in'] = yaw_c_seq[:, frame_abs_idx]
                if pitch_c_seq is not None:
                    he_driving['pitch_in'] = pitch_c_seq[:, frame_abs_idx]
                if roll_c_seq is not None:
                    he_driving['roll_in'] = roll_c_seq[:, frame_abs_idx]

                kp_driving = keypoint_transformation(kp_canonical, he_driving, wo_exp=(not use_exp))

                kp_driving_value_list.append(kp_driving['value'])
                if 'jacobian' in kp_driving:
                    kp_driving_jacobian_list.append(kp_driving['jacobian'])

            # --- Stack/Expand inputs for the generator batch ---
            kp_driving_batch_value = torch.cat(kp_driving_value_list, dim=0)
            kp_driving_for_gen = {'value': kp_driving_batch_value}

            if kp_driving_jacobian_list:
                kp_driving_batch_jacobian = torch.cat(kp_driving_jacobian_list, dim=0)
                kp_driving_for_gen['jacobian'] = kp_driving_batch_jacobian

            source_image_batch = source_image.repeat(current_render_batch_size, 1, 1, 1)
            kp_source_batch = {
                'value': kp_source['value'].repeat(current_render_batch_size, 1, 1)
            }
            if 'jacobian' in kp_source and kp_source.get('jacobian') is not None:
                kp_source_batch['jacobian'] = kp_source['jacobian'].repeat(current_render_batch_size, 1, 1, 1)

            # --- Run Generator on the batch ---
            # No autocast here
            out_batch = generator(source_image_batch, kp_source=kp_source_batch, kp_driving=kp_driving_for_gen)

            # Store the batch of predicted frames
            predictions_list.append(out_batch['prediction']) # No .float() needed

        # Concatenate predictions from all batches
        if predictions_list:
             predictions_ts = torch.cat(predictions_list, dim=0)
        else:
             return torch.empty(0)

        # Reshape back to (bs, total_frames, C, H, W)
        if bs == 1:
            predictions_ts = predictions_ts.unsqueeze(0)
        else:
            try:
                predictions_ts = predictions_ts.view(bs, num_frames, *predictions_ts.shape[1:])
            except RuntimeError as e:
                 print(f"Error reshaping batched output: {e}. Output shape: {predictions_ts.shape}, Target: ({bs}, {num_frames}, C, H, W)")
                 predictions_ts = predictions_ts.unsqueeze(0) # Fallback

    return predictions_ts

# --- AnimateModel class (No changes needed here) ---
class AnimateModel(torch.nn.Module):
    def __init__(self, generator, kp_extractor, mapping):
        super(AnimateModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.mapping = mapping

        self.kp_extractor.eval()
        self.generator.eval()
        self.mapping.eval()

    def forward(self, x):
        render_batch_size = x.get('render_batch_size', 4) # Default to 4 if not provided

        source_image = x['source_image']
        source_semantics = x['source_semantics']
        target_semantics = x['target_semantics']
        yaw_c_seq = x.get('yaw_c_seq', None)
        pitch_c_seq = x.get('pitch_c_seq', None)
        roll_c_seq = x.get('roll_c_seq', None)

        predictions_video = make_animation(
            source_image, source_semantics, target_semantics,
            self.generator, self.kp_extractor, None,
            self.mapping, use_exp=True,
            yaw_c_seq=yaw_c_seq, pitch_c_seq=pitch_c_seq, roll_c_seq=roll_c_seq,
            render_batch_size=render_batch_size # Pass the batch size
            # Removed use_fp16=use_fp16
        )

        return predictions_video
# ===========================================================
# END OF MODIFIED FILE: src/facerender/modules/make_animation.py
# ===========================================================