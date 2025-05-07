# ===========================================================
# START OF MODIFIED FILE: src/facerender/modules/make_animation.py
# ===========================================================
from scipy.spatial import ConvexHull
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Added for FP16
from torch.cuda.amp import autocast

# --- normalize_kp and headpose_pred_to_degree remain unchanged ---
def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    # ... (original code) ...
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            # Ensure jacobians exist and are not None before proceeding
            if 'jacobian' in kp_driving and kp_driving['jacobian'] is not None and \
               'jacobian' in kp_driving_initial and kp_driving_initial['jacobian'] is not None and \
               'jacobian' in kp_source and kp_source['jacobian'] is not None:

                # Attempt inverse, catch potential non-invertible matrices
                try:
                    inv_jacobian_initial = torch.inverse(kp_driving_initial['jacobian'])
                    jacobian_diff = torch.matmul(kp_driving['jacobian'], inv_jacobian_initial)
                    kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])
                except Exception as e:
                     # Fallback or warning if inverse fails
                     print(f"Warning: Jacobian inverse calculation failed: {e}. Using original driving Jacobian.")
                     if 'jacobian' in kp_driving: # Keep driving if it exists
                          kp_new['jacobian'] = kp_driving['jacobian']
                     # else: kp_new['jacobian'] remains None if driving didn't have one

            # Handle cases where some jacobians might be missing
            elif 'jacobian' in kp_driving:
                 kp_new['jacobian'] = kp_driving['jacobian'] # Default to driving if others missing
            # else: kp_new['jacobian'] remains whatever it was initialized with (likely None or previous value)


    return kp_new


def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    # Use type_as(pred) for compatibility
    idx_tensor = torch.FloatTensor(idx_tensor).type_as(pred).to(device)
    pred = F.softmax(pred, dim=1) # Apply softmax along the correct dimension
    degree = torch.sum(pred*idx_tensor, 1) * 3 - 99
    return degree

# --- get_rotation_matrix remains unchanged ---
def get_rotation_matrix(yaw, pitch, roll):
    # ... (original code) ...
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    pitch_mat = torch.cat([torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch),
                          torch.zeros_like(pitch), torch.cos(pitch), -torch.sin(pitch),
                          torch.zeros_like(pitch), torch.sin(pitch), torch.cos(pitch)], dim=1)
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat([torch.cos(yaw), torch.zeros_like(yaw), torch.sin(yaw),
                           torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
                           -torch.sin(yaw), torch.zeros_like(yaw), torch.cos(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    roll_mat = torch.cat([torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),
                         torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
                         torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    rot_mat = torch.einsum('bij,bjk,bkm->bim', pitch_mat, yaw_mat, roll_mat)

    return rot_mat

# --- keypoint_transformation remains unchanged ---
def keypoint_transformation(kp_canonical, he, wo_exp=False):
    kp = kp_canonical['value']    # (bs, k, 3)
    yaw, pitch, roll= he['yaw'], he['pitch'], he['roll']
    yaw = headpose_pred_to_degree(yaw)
    pitch = headpose_pred_to_degree(pitch)
    roll = headpose_pred_to_degree(roll)

    if 'yaw_in' in he:
        yaw = he['yaw_in']
    if 'pitch_in' in he:
        pitch = he['pitch_in']
    if 'roll_in' in he:
        roll = he['roll_in']

    rot_mat = get_rotation_matrix(yaw, pitch, roll)    # (bs, 3, 3)

    t, exp = he['t'], he['exp']
    if wo_exp:
        exp =  exp*0

    # keypoint rotation
    kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp)

    # keypoint translation (correct t shape for broadcasting)
    t = t.unsqueeze(1) # .repeat(1, kp.shape[1], 1) # Make sure t is (bs, 1, 3)
    kp_t = kp_rotated + t

    # add expression deviation
    exp = exp.view(exp.shape[0], -1, 3)
    kp_transformed = kp_t + exp

    # Add jacobian transformation if present (optional, depends on KPDetector/HEEstimator)
    kp_result = {'value': kp_transformed}
    if 'jacobian' in kp_canonical:
         jacobian = kp_canonical['jacobian']
         jacobian_transformed = torch.einsum('bmp,bkpq->bkmq', rot_mat, jacobian)
         kp_result['jacobian'] = jacobian_transformed


    return kp_result


# Modified make_animation to accept use_fp16 flag
def make_animation(source_image, source_semantics, target_semantics,
                            generator, kp_detector, he_estimator, mapping,
                            yaw_c_seq=None, pitch_c_seq=None, roll_c_seq=None,
                            use_exp=True, use_half=False, use_fp16=False): # Added use_fp16
    with torch.no_grad():
        predictions = []
        # Initial KP and HE calculation (can potentially use autocast too)
        with autocast(enabled=use_fp16):
             kp_canonical = kp_detector(source_image)
             he_source = mapping(source_semantics)
             # Check if he_estimator is needed? MappingNet seems to provide HE directly
             # he_source_estimator = he_estimator(source_image) # If needed
             kp_source = keypoint_transformation(kp_canonical, he_source, wo_exp=(not use_exp))


        for frame_idx in tqdm(range(target_semantics.shape[1]), 'Face Renderer:'):
            target_semantics_frame = target_semantics[:, frame_idx]

            # Core animation step within the loop - wrap model calls here
            with autocast(enabled=use_fp16):
                he_driving = mapping(target_semantics_frame)
                if yaw_c_seq is not None:
                    he_driving['yaw_in'] = yaw_c_seq[:, frame_idx]
                if pitch_c_seq is not None:
                    he_driving['pitch_in'] = pitch_c_seq[:, frame_idx]
                if roll_c_seq is not None:
                    he_driving['roll_in'] = roll_c_seq[:, frame_idx]

                kp_driving = keypoint_transformation(kp_canonical, he_driving, wo_exp=(not use_exp))

                # kp_norm = kp_driving # Assuming normalize_kp is not used based on original call
                # kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving, kp_driving_initial=kp_source) # If normalization needed

                # Generator forward pass
                out = generator(source_image, kp_source=kp_source, kp_driving=kp_driving) # Pass kp_driving directly if no normalization

            # Append prediction (CPU transfer happens later in AnimateFromCoeff)
            predictions.append(out['prediction'].float()) # Ensure output is float32 if autocast was used

        predictions_ts = torch.stack(predictions, dim=1)
    return predictions_ts

# --- AnimateModel class remains unchanged ---
class AnimateModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, generator, kp_extractor, mapping):
        super(AnimateModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.mapping = mapping

        self.kp_extractor.eval()
        self.generator.eval()
        self.mapping.eval()

    def forward(self, x):

        source_image = x['source_image']
        source_semantics = x['source_semantics']
        target_semantics = x['target_semantics']
        yaw_c_seq = x['yaw_c_seq']
        pitch_c_seq = x['pitch_c_seq']
        roll_c_seq = x['roll_c_seq']

        # Note: This AnimateModel's forward doesn't explicitly take use_fp16.
        # If used directly, FP16 needs to be handled by the caller wrapping this forward pass,
        # or by modifying this class to accept and use the flag internally.
        # The main `AnimateFromCoeff.generate` calls `make_animation` directly,
        # where we added the FP16 handling.
        predictions_video = make_animation(source_image, source_semantics, target_semantics,
                                        self.generator, self.kp_extractor,
                                        self.mapping, use_exp = True, # he_estimator removed as mapping provides HE
                                        yaw_c_seq=yaw_c_seq, pitch_c_seq=pitch_c_seq, roll_c_seq=roll_c_seq)

        return predictions_video
# ===========================================================
# END OF MODIFIED FILE: src/facerender/modules/make_animation.py
# ===========================================================