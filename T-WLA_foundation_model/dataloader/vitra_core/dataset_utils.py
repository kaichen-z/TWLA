"""Vendored from VITRA (`vitra/datasets/dataset_utils.py`): feature slot layout, field of view
and intrinsics helpers."""
import numpy as np

class ActionFeature(object):
    """(start, end) slots of each feature in the unified 192-D action vector. The HUMAN_*
    ranges are those of the joint-angle parameterisation, 51 per hand."""

    ALL_FEATURES = (0, 192)

    HUMAN_LEFT_HAND = (0, 51)
    HUMAN_RIGHT_HAND = (51, 102)
    HUMAN_LEFT_TRANS = (0, 3)
    HUMAN_LEFT_ROT = (3, 6)
    HUMAN_LEFT_6D = (0, 6)
    HUMAN_LEFT_JOINTS = (6, 51)
    HUMAN_RIGHT_TRANS = (51, 54)
    HUMAN_RIGHT_ROT = (54, 57)
    HUMAN_RIGHT_6D = (51, 57)
    HUMAN_RIGHT_JOINTS = (57, 102)
    PADDING_FEATURES = (102, 192)    # unused
    @classmethod
    def get_concatenated_action_feature_from_dict(cls, action_feature_dict):
        """Scatter {feature name: (B, k) array} into a (B, 192) feature array and a bool mask
        of the filled slots."""
        batch_size = next(iter(action_feature_dict.values())).shape[0]
        features = np.zeros((batch_size, cls.ALL_FEATURES[1]), dtype=np.float32)
        feature_mask = np.zeros((batch_size, cls.ALL_FEATURES[1]), dtype=bool)
        
        for key, value in action_feature_dict.items():
            assert len(value.shape) == 2
            start, end = getattr(cls, key)
            k = value.shape[1]
            features[:, start:start + k] = value
            feature_mask[:, start:start + k] = True
        return features, feature_mask
    
    @classmethod
    def get_dict_from_concatenated_action_feature(cls, feature, feature_mask):
        """Inverse of `get_concatenated_action_feature_from_dict`."""
        action_feature_dict = {}
        consts = {
            name: getattr(cls, name)
            for name in dir(cls)
            if name.isupper() and "ALL" not in name
        }
        for key, (start, end) in consts.items():
            k = np.sum(feature_mask[0, start:end])
            if k == 0:
                continue
            action_feature_dict[key] = feature[:, start:start + k]
        return action_feature_dict
    
    @classmethod
    def get_loss_components(cls, action_type='angle'):
        """{component name: (start, end, weight)} for action_type 'angle' or 'keypoints'."""
        if action_type == 'angle':
            return {
                'left_hand_6d': (*cls.HUMAN_LEFT_6D, 1.0),
                'left_hand_joints': (*cls.HUMAN_LEFT_JOINTS, 1.0),
                'right_hand_6d': (*cls.HUMAN_RIGHT_6D, 1.0),
                'right_hand_joints': (*cls.HUMAN_RIGHT_JOINTS, 1.0),
            }
        elif action_type == 'keypoints':
            # per hand: 6 wrist dims, then 21 x 3 keypoints
            hand_dim = 6 + 21 * 3
            left_6d_start = 0
            left_6d_end = 6
            left_joints_start = left_6d_end
            left_joints_end = hand_dim
            right_6d_start = hand_dim
            right_6d_end = right_6d_start + 6
            right_joints_start = right_6d_end
            right_joints_end = 2 * hand_dim

            return {
                'left_6d': (left_6d_start, left_6d_end, 1.0),
                'left_joints': (left_joints_start, left_joints_end, 1.0),
                'right_6d': (right_6d_start, right_6d_end, 1.0),
                'right_joints': (right_joints_start, right_joints_end, 1.0),
            }
        else:
            raise ValueError(f"Unknown action type: {action_type}")
    
    @classmethod
    def get_hand_group_mapping(cls, action_type='angle'):
        """{hand group: [component names]}, for averaging the losses per hand."""
        return {
            'left_hand': ['left_trans', 'left_rot', 'left_joints'],
            'right_hand': ['right_trans', 'right_rot', 'right_joints'],
        }
    
    @classmethod
    def get_xhand_loss_components(cls):
        """Get loss components specific to XHand dataset."""
        return {
            'left_hand_6d': (*cls.HUMAN_LEFT_6D, 1.6),
            'left_hand_joints': (*cls.HUMAN_LEFT_JOINTS, 0.4),
            'right_hand_6d': (*cls.HUMAN_RIGHT_6D, 1.6),
            'right_hand_joints': (*cls.HUMAN_RIGHT_JOINTS, 0.4),
        }

class StateFeature(ActionFeature):
    """Slots of the 212-D state vector: the action slots plus the MANO shape (beta)."""
    ALL_FEATURES = (0, 212)
    HUMAN_LEFT_BETA = (192, 202)  # unused
    HUMAN_RIGHT_BETA = (202, 212)  # unused

def calculate_fov(h, w, intrinsics):
    """(horizontal, vertical) field of view in radians, from the image size and the 3x3 intrinsics."""

    hfov = 2 * np.arctan(w / (2 * intrinsics[0][0]))
    vfov = 2 * np.arctan(h / (2 * intrinsics[1][1]))
    fov = np.array([hfov, vfov], dtype=np.float32)

    return fov

def compute_new_intrinsics_crop(original_intrinsics, original_size, crop_size, resize_size):
    """Intrinsics after a centred square crop of `crop_size` out of a square image of
    `original_size`, followed by a resize to `resize_size`."""
    original_fx = original_intrinsics[0][0]
    original_fy = original_intrinsics[1][1]
    original_cx = original_intrinsics[0][2]
    original_cy = original_intrinsics[1][2]
    
    crop_offset = (original_size - crop_size) / 2
    
    cropped_cx = original_cx - crop_offset
    cropped_cy = original_cy - crop_offset
    
    scale = resize_size / crop_size
    
    new_fx = original_fx * scale
    new_fy = original_fy * scale
    new_cx = cropped_cx * scale
    new_cy = cropped_cy * scale
    
    intrinsics_matrix = np.array([
        [new_fx, 0, new_cx],
        [0, new_fy, new_cy],
        [0, 0, 1]
    ])
    return intrinsics_matrix

def compute_new_intrinsics_resize(original_intrinsics, resize_size):
    """Intrinsics after a resize to `resize_size` = (H, W). The original image size is taken
    to be twice the principal point."""
    original_fx = original_intrinsics[0][0]
    original_fy = original_intrinsics[1][1]
    original_cx = original_intrinsics[0][2]
    original_cy = original_intrinsics[1][2]

    H, W = resize_size

    scale_x = W / (2*original_cx)
    scale_y = H / (2*original_cy)

    new_fx = original_fx * scale_x
    new_fy = original_fy * scale_y
    new_cx = original_cx * scale_x
    new_cy = original_cy * scale_y

    intrinsics_matrix = np.array([
        [new_fx, 0, new_cx],
        [0, new_fy, new_cy],
        [0, 0, 1]
    ])

    return intrinsics_matrix
