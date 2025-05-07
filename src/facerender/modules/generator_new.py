# ===========================================================
# START OF CORRECTED FILE: src/facerender/modules/generator.py
# ===========================================================
import torch
from torch import nn
import torch.nn.functional as F
# Assuming these imports are correct relative to this file's location
from .util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d, ResBlock3d, SPADEResnetBlock
from .dense_motion import DenseMotionNetwork


class OcclusionAwareGenerator(nn.Module):
    """
    Generator follows NVIDIA architecture.
    """

    def __init__(self, image_channel, feature_channel, num_kp, block_expansion, max_features, num_down_blocks, reshape_channel, reshape_depth,
                 num_resblocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False):
        super(OcclusionAwareGenerator, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, feature_channel=feature_channel,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        # CORRECTED LINE from previous step: Changed kernel_size to (3, 3) and padding to (1, 1)
        self.first = SameBlock2d(image_channel, block_expansion, kernel_size=(3, 3), padding=(1, 1)) # Corrected

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        out_features_encoder = min(max_features, block_expansion * (2 ** num_down_blocks)) # Output features from encoder
        # Projection layer - ensure this matches the input channels needed later
        self.second = nn.Conv2d(in_channels=out_features_encoder, out_channels=max_features, kernel_size=1, stride=1)

        self.reshape_channel = reshape_channel
        self.reshape_depth = reshape_depth

        self.resblocks_3d = torch.nn.Sequential()
        for i in range(num_resblocks):
            # Input to 3D resblocks should match self.second output if reshaping
            # Requires max_features == reshape_channel * reshape_depth
            # For now, assume input is reshape_channel as per original DenseMotion example
            self.resblocks_3d.add_module('3dr' + str(i), ResBlock3d(reshape_channel, kernel_size=3, padding=1))


        in_features_2d = min(max_features, block_expansion * (2 ** num_down_blocks))
        # Input channels for third layer should match the flattened output of 3D blocks (c*d) or the output of self.second if 3D blocks removed
        # Assuming flattened 3D block output: reshape_channel * reshape_depth = max_features
        self.third = SameBlock2d(max_features, in_features_2d, kernel_size=(3, 3), padding=(1, 1), lrelu=True)
        self.fourth = nn.Conv2d(in_channels=in_features_2d, out_channels=in_features_2d, kernel_size=1, stride=1)

        self.resblocks_2d = torch.nn.Sequential()
        for i in range(num_resblocks):
            self.resblocks_2d.add_module('2dr' + str(i), ResBlock2d(in_features_2d, kernel_size=3, padding=1))

        up_blocks = []
        for i in range(num_down_blocks):
            in_features_up = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features_up = max(block_expansion, block_expansion * (2 ** (num_down_blocks - i - 1)))
            out_features_up = max(block_expansion, out_features_up)
            up_blocks.append(UpBlock2d(in_features_up, out_features_up, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.final = nn.Conv2d(block_expansion, image_channel, kernel_size=(7, 7), padding=(3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map
        self.image_channel = image_channel

    def deform_input(self, inp, deformation):
        _, d_old, h_old, w_old, _ = deformation.shape
        _, _, d, h, w = inp.shape
        if d_old != d or h_old != h or w_old != w:
            deformation = deformation.permute(0, 4, 1, 2, 3)
            deformation = F.interpolate(deformation, size=(d, h, w), mode='trilinear', align_corners=False)
            deformation = deformation.permute(0, 2, 3, 4, 1)
        return F.grid_sample(inp, deformation, align_corners=False)

    def forward(self, source_image, kp_driving, kp_source):
        # Encoding (downsampling) part
        out = self.first(source_image)
        # encoder_features = [out] # Store features for potential skip connections if needed by decoder
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
            # encoder_features.append(out)

        out = self.second(out) # Project to max_features
        bs, c, h, w = out.shape

        # Reshape and 3D processing
        try:
             # Ensure c == self.reshape_channel * self.reshape_depth
             feature_3d = out.view(bs, self.reshape_channel, self.reshape_depth, h ,w)
        except RuntimeError as e:
             print(f"Error reshaping encoder output for 3D ResBlocks: {e}")
             print(f"Shape was {out.shape}, trying to reshape to ({bs}, {self.reshape_channel}, {self.reshape_depth}, {h}, {w})")
             print("Ensure max_features == reshape_channel * reshape_depth in config.")
             raise e
        feature_3d = self.resblocks_3d(feature_3d)

        # Transforming feature representation
        output_dict = {}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(feature=feature_3d, kp_driving=kp_driving,
                                                     kp_source=kp_source)
            output_dict['mask'] = dense_motion['mask']
            # ***************************************************************
            # REMOVED LINE: 'sparse_deformed' key doesn't exist in dense_motion dict
            # output_dict['sparse_deformed'] = dense_motion['sparse_deformed']
            # ***************************************************************

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None
            deformation = dense_motion['deformation']
            output_dict['deformation'] = deformation # Pass deformation if needed

            out = self.deform_input(feature_3d, deformation) # Deform the 3D features

            bs, c, d, h, w = out.shape
            out = out.view(bs, c*d, h, w) # Flatten depth into channels
            out = self.third(out)
            out = self.fourth(out)

            if occlusion_map is not None:
                if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear', align_corners=False)
                out = out * occlusion_map # Apply occlusion

        else: # Handle case without dense motion network if needed
            bs, c, d, h, w = feature_3d.shape
            out = feature_3d.view(bs, c*d, h, w) # Flatten depth into channels
            out = self.third(out)
            out = self.fourth(out)
            occlusion_map = None # No occlusion if no dense motion

        # Decoding part
        out = self.resblocks_2d(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)

        out = self.final(out)
        out = torch.sigmoid(out) # Use torch.sigmoid

        output_dict["prediction"] = out

        return output_dict


# --- SPADEDecoder remains the same ---
class SPADEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        ic = 256
        oc = 64
        norm_G = 'spadespectralinstance'
        label_nc = 256

        self.fc = nn.Conv2d(ic, 2 * ic, 3, padding=1)
        self.G_middle_0 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_1 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_2 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_3 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_4 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_5 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.up_0 = SPADEResnetBlock(2 * ic, ic, norm_G, label_nc)
        self.up_1 = SPADEResnetBlock(ic, oc, norm_G, label_nc)
        self.conv_img = nn.Conv2d(oc, 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) # Use bilinear and align_corners=False

    def forward(self, feature):
        seg = feature
        x = self.fc(feature)
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        x = self.G_middle_2(x, seg)
        x = self.G_middle_3(x, seg)
        x = self.G_middle_4(x, seg)
        x = self.G_middle_5(x, seg)
        x = self.up(x)
        x = self.up_0(x, seg)         # 256, 128, 128
        x = self.up(x)
        x = self.up_1(x, seg)         # 64, 256, 256

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.sigmoid(x) # Use torch.sigmoid

        return x


class OcclusionAwareSPADEGenerator(nn.Module):
    """
    Generator using SPADE architecture in the decoder.
    """
    def __init__(self, image_channel, feature_channel, num_kp, block_expansion, max_features, num_down_blocks, reshape_channel, reshape_depth,
                 num_resblocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False):
        super(OcclusionAwareSPADEGenerator, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, feature_channel=feature_channel,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        # CORRECTED LINE from previous step: Changed kernel_size to (3, 3) and padding to (1, 1)
        self.first = SameBlock2d(image_channel, block_expansion, kernel_size=(3, 3), padding=(1, 1)) # Corrected

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        out_features_encoder = min(max_features, block_expansion * (2 ** num_down_blocks))
        decoder_input_channels = 256 # Example value, should match SPADEDecoder's expected input
        self.second = nn.Conv2d(in_channels=out_features_encoder, out_channels=decoder_input_channels, kernel_size=1, stride=1)

        self.reshape_channel = reshape_channel
        self.reshape_depth = reshape_depth

        self.estimate_occlusion_map = estimate_occlusion_map
        self.image_channel = image_channel

        self.decoder = SPADEDecoder()

    def deform_input(self, inp, deformation):
        # (Same as in OcclusionAwareGenerator)
        _, d_old, h_old, w_old, _ = deformation.shape
        _, _, d, h, w = inp.shape
        if d_old != d or h_old != h or w_old != w:
            deformation = deformation.permute(0, 4, 1, 2, 3)
            deformation = F.interpolate(deformation, size=(d, h, w), mode='trilinear', align_corners=False)
            deformation = deformation.permute(0, 2, 3, 4, 1)
        return F.grid_sample(inp, deformation, align_corners=False)

    def forward(self, source_image, kp_driving, kp_source):
        # Encoding (downsampling) part
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)

        out = self.second(out)
        bs, c, h, w = out.shape

        # --- Motion and Deformation ---
        output_dict = {}
        if self.dense_motion_network is not None:
            try:
                 feature_3d_for_motion = out.view(bs, self.reshape_channel, self.reshape_depth, h ,w)
            except RuntimeError:
                 print(f"Warning: Cannot reshape feature map {out.shape} for dense motion in SPADE Gen. Ensure channel compatibility.")
                 feature_3d_for_motion = out # Placeholder

            dense_motion = self.dense_motion_network(feature=feature_3d_for_motion, kp_driving=kp_driving, kp_source=kp_source)
            output_dict['mask'] = dense_motion['mask']
            # ***************************************************************
            # REMOVED LINE: 'sparse_deformed' key doesn't exist in dense_motion dict
            # output_dict['sparse_deformed'] = dense_motion['sparse_deformed']
            # ***************************************************************

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None

            deformation = dense_motion['deformation']
            output_dict['deformation'] = deformation

            # Placeholder for applying deformation or using motion info with SPADE
            if occlusion_map is not None:
                if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear', align_corners=False)
                deformed_features = out * occlusion_map # Apply occlusion
            else:
                deformed_features = out
            # --- End Placeholder ---

        else: # No dense motion network
            deformed_features = out
            occlusion_map = None

        # --- Decoding part using SPADE ---
        out_decoder = self.decoder(deformed_features)

        output_dict["prediction"] = out_decoder

        return output_dict
# ===========================================================
# END OF CORRECTED FILE: src/facerender/modules/generator.py
# ===========================================================