"""Vendored from VITRA (`vitra/datasets/augment_utils.py`): geometric and colour augmentation.

`_image_uv` and `_uv_to_pixel` are local copies of two utils3d helpers.
"""
import re
import cv2
import copy
import numpy as np
from typing import *
from PIL import Image
from scipy.spatial.transform import Rotation as R
# utils3d is pinned to a commit in requirements.txt: its API changes between versions.
import utils3d  # pip install git+https://github.com/EasternJournalist/utils3d.git@d790d33#egg=utils3d


def _image_uv(width: int, height: int, dtype=np.float32) -> np.ndarray:
    """Pixel-centre UV grid, (height, width, 2), values in (0, 1): width 4 gives 0.125 .. 0.875.

    Local copy of `utils3d.numpy.image_uv`, so that the grid stays centre-aligned whichever
    utils3d version is installed.
    """
    u = np.linspace(0.5 / width, (width - 0.5) / width, width, dtype=dtype)
    v = np.linspace(0.5 / height, (height - 0.5) / height, height, dtype=dtype)
    u, v = np.meshgrid(u, v, indexing="xy")
    return np.stack([u, v], axis=2)


def _uv_to_pixel(uv: np.ndarray, width: int, height: int) -> np.ndarray:
    """UV in (0, 1) -> pixel coordinates. Inverse of `_image_uv`, hence the -0.5."""
    return uv * np.array([width, height], dtype=uv.dtype) - 0.5

def sample_perspective_rot_flip_with_traj_constraint(
    src_intrinsics: np.ndarray,                         # [3, 3] normalized camera intrinsics matrix of source image
    tgt_aspect: float,                                  # target aspect ratio (width / height)
    trajectory_uv: np.ndarray,                          # [N, 2] trajectory points in normalized [0, 1] coordinates. None means no constraint.
    margin_ratio: float,                                # margin ratio for trajectory bounding box expansion/shrinkage
    center_augmentation: float,                         # 0.0 means no augmentation, 1.0 means random center within 50% of fov
    fov_range_absolute: Tuple[float, float],            # (min_fov, max_fov) in degrees
    fov_range_relative: Tuple[float, float],            # (min_fov, max_fov) relative to source image fov
    inplane_range: Tuple[float, float] = (0.0, 0.0),    # (min_angle, max_angle) in radians
    min_overlap: float = 0.75,                          # minimum required intersection ratio after cropping relative to bbox_uv area
    flip_augmentation: float = 0.0,                     # 0.0 means no flip, 1.0 means random flip with 50% chance
    rng: np.random.Generator = None
):  
    "Compute target intrinsics, rotation matrix, and optional flip for perspective warping augmentation with trajectory constraints."
    
    if rng is None:
        rng = np.random.default_rng()

    raw_horizontal, raw_vertical = abs(1.0 / src_intrinsics[0, 0]), abs(1.0 / src_intrinsics[1, 1])
    raw_fov_x, raw_fov_y = utils3d.numpy.intrinsics_to_fov(src_intrinsics)

    # ------- 1. set target fov -------
    fov_range_absolute_min, fov_range_absolute_max = fov_range_absolute
    fov_range_relative_min, fov_range_relative_max = fov_range_relative
    tgt_fov_x_min = min(fov_range_relative_min * raw_fov_x,
                        utils3d.focal_to_fov(utils3d.fov_to_focal(fov_range_relative_min * raw_fov_y) / tgt_aspect))
    tgt_fov_x_max = min(fov_range_relative_max * raw_fov_x,
                        utils3d.focal_to_fov(utils3d.fov_to_focal(fov_range_relative_max * raw_fov_y) / tgt_aspect))
    tgt_fov_x_min = max(np.deg2rad(fov_range_absolute_min), tgt_fov_x_min)
    tgt_fov_x_max = min(np.deg2rad(fov_range_absolute_max), tgt_fov_x_max)

    # trajectory constraint on fov
    if trajectory_uv is not None:
        bbox_uv = np.array([trajectory_uv[:, 0].min(), trajectory_uv[:, 1].min(),
                            trajectory_uv[:, 0].max(), trajectory_uv[:, 1].max()], dtype=np.float32)
        bbox_uv = shrink_or_expand_bbox_uv(bbox_uv, margin_ratio)

        traj_x_range = bbox_uv[2] - bbox_uv[0]
        traj_y_range = bbox_uv[3] - bbox_uv[1]
        traj_fov_x = 2 * np.arctan(0.5 * traj_x_range * raw_horizontal)
        traj_fov_x = np.clip(traj_fov_x, 1e-2, None)
        traj_fov_y = 2 * np.arctan(0.5 * traj_y_range * raw_vertical)
        traj_fov_y = np.clip(traj_fov_y, 1e-2, None)
        traj_fov_needed = max(traj_fov_x, utils3d.focal_to_fov(utils3d.fov_to_focal(traj_fov_y) / tgt_aspect))
        tgt_fov_x_min = max(tgt_fov_x_min, traj_fov_needed)

    tgt_fov_x = rng.uniform(min(tgt_fov_x_min, tgt_fov_x_max), tgt_fov_x_max)
    tgt_fov_y = utils3d.focal_to_fov(utils3d.numpy.fov_to_focal(tgt_fov_x) * tgt_aspect)

    # ------- 2. set target image center -------
    valid_center_dtheta_range = center_augmentation * np.array([-0.5, 0.5]) * (raw_fov_x - tgt_fov_x)
    valid_center_dphi_range   = center_augmentation * np.array([-0.5, 0.5]) * (raw_fov_y - tgt_fov_y)

    valid_center_x_range = 0.5 + 0.5 * np.tan(valid_center_dtheta_range) / np.tan(raw_fov_x / 2)
    valid_center_y_range = 0.5 + 0.5 * np.tan(valid_center_dphi_range) / np.tan(raw_fov_y / 2)

    crop_box_size_x = 2 * np.tan(tgt_fov_x * 0.5) / raw_horizontal
    crop_box_size_y = 2 * np.tan(tgt_fov_y * 0.5) / raw_vertical

    # ensure the crop box position contains the trajectory bounding box
    if trajectory_uv is not None:
        cx_min = bbox_uv[2] - crop_box_size_x / 2
        cx_max = bbox_uv[0] + crop_box_size_x / 2
        cy_min = bbox_uv[3] - crop_box_size_y / 2
        cy_max = bbox_uv[1] + crop_box_size_y / 2

        valid_center_x_range = resolve_valid_range(cx_min, cx_max,
                                                valid_center_x_range[0], valid_center_x_range[1])
        valid_center_y_range = resolve_valid_range(cy_min, cy_max,
                                                valid_center_y_range[0], valid_center_y_range[1])

    cu = rng.uniform(valid_center_x_range[0], valid_center_x_range[1])
    cv = rng.uniform(valid_center_y_range[0], valid_center_y_range[1])

    # ------- 3. initial camera transformation for target view -------
    direction = utils3d.unproject_cv(
        np.array([[cu, cv]], dtype=np.float32),
        np.array([1.0], dtype=np.float32), intrinsics=src_intrinsics
    )[0]

    R_trans = utils3d.rotation_matrix_from_vectors(direction, np.array([0, 0, 1], dtype=np.float32))

    # ------- 4. shrink the target view to fit into the original image range -------
    corners = np.array([[0,0],[0,1],[1,1],[1,0]], dtype=np.float32)
    corners = np.concatenate([corners, np.ones((4,1),dtype=np.float32)], axis=1)
    corners = corners @ (np.linalg.inv(src_intrinsics).T @ R_trans.T)
    corners = corners[:,:2] / corners[:,2:3]
    tgt_horizontal = float(2 * np.tan(tgt_fov_x * 0.5))
    tgt_vertical   = float(2 * np.tan(tgt_fov_y * 0.5))
    warp_h, warp_v = float('inf'), float('inf')
    for i in range(4):
        inter, _ = utils3d.numpy.ray_intersection(
            np.array([0.,0.]), np.array([[tgt_aspect,1.0],[tgt_aspect,-1.0]]),
            corners[i-1], corners[i]-corners[i-1]
        )
        warp_h = min(warp_h, 2 * abs(inter[:,0]).min())
        warp_v = min(warp_v, 2 * abs(inter[:,1]).min())
    tgt_horizontal = min(tgt_horizontal, warp_h)
    tgt_vertical   = min(tgt_vertical, warp_v)

    # ------- 5. finalize target intrinsics -------
    fx, fy = 1 / tgt_horizontal, 1 / tgt_vertical
    tgt_intrinsics = utils3d.numpy.intrinsics_from_focal_center(fx, fy, 0.5, 0.5).astype(np.float32)

    # ------- 6. compute continuous valid in-plane rotation range via binary search -------

    # define crop rectangle corners relative to center
    crop_box_size_x, crop_box_size_y = tgt_horizontal / raw_horizontal, tgt_vertical / raw_vertical # update crop box size after shrinking
    half_w, half_h = crop_box_size_x/2, crop_box_size_y/2
    rect = np.array([[-half_w, -half_h],[-half_w, half_h],[half_w, half_h],[half_w, -half_h]])

    # area of bbox_uv for overlap normalization
    if trajectory_uv is not None:
        bbox_area = (bbox_uv[2] - bbox_uv[0]) * (bbox_uv[3] - bbox_uv[1])

    def is_valid(ang: float) -> bool:
        R2 = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], dtype=np.float32)
        pts = (rect @ R2.T) + np.array([cu, cv])
        if pts.min() < 0 or pts.max() > 1:
            return False
        if trajectory_uv is None:
            return True
        x0, y0 = pts[:,0].min(), pts[:,1].min()
        x1, y1 = pts[:,0].max(), pts[:,1].max()

        # degenerate bbox?
        if bbox_area <= 0:
            # ensure crop bounding box contains bbox_uv extents
            return (x0 <= bbox_uv[0] <= x1) and (x0 <= bbox_uv[2] <= x1) and \
                   (y0 <= bbox_uv[1] <= y1) and (y0 <= bbox_uv[3] <= y1)

        ix0, iy0 = max(x0, bbox_uv[0]), max(y0, bbox_uv[1])
        ix1, iy1 = min(x1, bbox_uv[2]), min(y1, bbox_uv[3])
        if ix1 <= ix0 or iy1 <= iy0:
            return False
        inter_area = (ix1 - ix0) * (iy1 - iy0)
        return (inter_area / bbox_area) >= min_overlap

    # binary search for max positive angle
    lo_p, hi_p = 0.0, inplane_range[1]
    for _ in range(20):
        mid = (lo_p + hi_p) / 2
        if is_valid(mid): lo_p = mid
        else: hi_p = mid
    max_valid = lo_p

    # binary search for max negative angle
    lo_n, hi_n = inplane_range[0], 0.0
    for _ in range(20):
        mid = (lo_n + hi_n) / 2
        if is_valid(mid): hi_n = mid
        else: lo_n = mid
    min_valid = hi_n

    if min_valid > max_valid:
        rot_angle = 0.0
    else:
        rot_angle = float(rng.uniform(min_valid, max_valid))

    R_inplane = np.array([[np.cos(rot_angle), -np.sin(rot_angle), 0],
                          [np.sin(rot_angle),  np.cos(rot_angle), 0],
                          [0,                   0,                  1]], dtype=np.float32)
    R_final = R_inplane @ R_trans

    # ------- 7. apply optional horizontal flip -------
    flip_prob = flip_augmentation * 0.5
    do_flip = rng.random() < flip_prob
    if do_flip:
        # reflect principal point u around center
        tgt_intrinsics[0, 2] = 1.0 - tgt_intrinsics[0, 2]
        # mirror the x axis
        M_flip = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
    else:
        M_flip = np.eye(3, dtype=np.float32)
    
    R_final = M_flip @ R_final

    return tgt_intrinsics, R_final, M_flip

def warp_perspective(
    src_image: np.ndarray = None,           # [H, W, C] source image to be warped
    src_intrinsics: np.ndarray = None,      # [3, 3] normalized camera intrinsics matrix of source image
    tgt_intrinsics: np.ndarray = None,      # [3, 3] normalized camera intrinsics matrix of target image
    R: np.ndarray = None,                   # [3, 3] rotation matrix from source to target view
    tgt_width: int = None,                  # target image width in pixels
    tgt_height: int = None,                 # target image height in pixels
):
    "Perspective warping with careful resampling."
    # Resize the source to about the target's pixel size first (antialiased), then warp.
    src_horizontal, src_vertical = 1 / src_intrinsics[0, 0], 1 / src_intrinsics[1, 1]
    tgt_horizontal, tgt_vertical = 1 / tgt_intrinsics[0, 0], 1 / tgt_intrinsics[1, 1]
    tgt_pixel_w, tgt_pixel_h = tgt_horizontal / tgt_width, tgt_vertical / tgt_height        # (should be exactly the same for x and y axes)
    resized_w, resized_h = int(src_horizontal / tgt_pixel_w), int(src_vertical / tgt_pixel_h)
    resized_image = np.array(Image.fromarray(src_image).resize((resized_w, resized_h), Image.Resampling.LANCZOS))

    transform = src_intrinsics @ np.linalg.inv(R) @ np.linalg.inv(tgt_intrinsics)
    uv_tgt = _image_uv(width=tgt_width, height=tgt_height)
    pts = np.concatenate([uv_tgt, np.ones((tgt_height, tgt_width, 1), dtype=np.float32)], axis=-1) @ transform.T
    uv_remap = pts[:, :, :2] / (pts[:, :, 2:3] + 1e-12)
    pixel_remap = _uv_to_pixel(uv_remap, width=resized_w, height=resized_h).astype(np.float32)
    try:
        tgt_image = cv2.remap(resized_image, pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_LANCZOS4)
    except:
        print("cv2.remap error, using nearest instead of lanczos4")
        print(pixel_remap[:, :, 0])
        print(pixel_remap[:, :, 1])
        breakpoint()

    return tgt_image

def center_crop_short_side(
    img: np.ndarray                                # [H, W, C] image to be center cropped
):

    h, w = img.shape[:2]
    short_side = min(h, w)

    top = (h - short_side) // 2
    left = (w - short_side) // 2

    return img[top:top+short_side, left:left+short_side]

def apply_color_augmentation(
    src_image: np.ndarray,                          # [H, W, C] source image to be augmented
    brightness: float = 0.3,                        # additive brightness shift, +/-
    contrast: float = 0.3,                          # contrast factor, 1 +/-
    saturation: float = 0.4,                        # saturation factor, 1 +/-
    hue: float = 0.3,                               # hue shift, +/- (used when preserve_hue is False)
    p: float = 0.8,                                 # probability of applying augmentation
    preserve_hue: bool = True,                      # if True, the hue is shifted only slightly
    rng: np.random.Generator = None
):
    """Colour jitter on a uint8 RGB image [H, W, C]. Returns uint8."""
    if rng is None:
        rng = np.random.default_rng()

    img = src_image.astype(np.float32) / 255.0

    if rng.random() < p:
        delta_brightness = rng.uniform(-brightness, brightness)
        img += delta_brightness
        img = np.clip(img, 0.0, 1.0)

        delta_contrast = rng.uniform(1 - contrast, 1 + contrast)
        img = (img - 0.5) * delta_contrast + 0.5
        img = np.clip(img, 0.0, 1.0)

        img_hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)

        delta_saturation = rng.uniform(1 - saturation, 1 + saturation)
        img_hsv[..., 1] *= delta_saturation
        img_hsv[..., 1] = np.clip(img_hsv[..., 1], 0, 255)

        if not preserve_hue:
            delta_hue = rng.uniform(-hue, hue) * 180  # OpenCV hue in [0,180]
            img_hsv[..., 0] = (img_hsv[..., 0] + delta_hue) % 180
        else:
            delta_hue = rng.uniform(-0.04, 0.04) * 180
            img_hsv[..., 0] = (img_hsv[..., 0] + delta_hue) % 180

        img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    return (img * 255).astype(np.uint8)

def apply_transform_to_rot(
    src_rotation: np.ndarray = None,     # [N, 3, 3] rotation matrix
    aug_transforms: tuple = None         # (tgt_intrinsics, R, M_flip)
):
    
    "Apply perspective transformation and YZ-plane flipping to rotation matrix."

    src_rotation = src_rotation.copy()
    if src_rotation.ndim == 2:
        src_rotation = src_rotation.reshape(1, 3, 3)

    _, R_trans, M_flip = aug_transforms

    N = len(src_rotation)

    R_trans = R_trans.reshape(1, 3, 3).repeat(N, axis=0)  # [N, 3, 3]
    M_flip = M_flip.reshape(1, 3, 3).repeat(N, axis=0)  # [N, 3, 3]

    tgt_rotation = R_trans @ src_rotation @ M_flip

    return tgt_rotation

def apply_transform_to_delta_rot(
    src_delta_rotation: np.ndarray = None,  # [N, 3, 3] delta rotation matrix
    aug_transforms: tuple = None            # (tgt_intrinsics, R, M_flip)
):          
    "Apply perspective transformation and YZ-plane flipping to delta rotation matrix."
    
    src_delta_rotation = src_delta_rotation.copy()
    if src_delta_rotation.ndim == 2:
        src_delta_rotation = src_delta_rotation.reshape(1, 3, 3)

    _, R_trans, _ = aug_transforms

    N = len(src_delta_rotation)

    R_trans = R_trans.reshape(1, 3, 3).repeat(N, axis=0)  # [N, 3, 3]

    tgt_delta_rotation = R_trans @ src_delta_rotation @ R_trans.transpose(0, 2, 1)

    return tgt_delta_rotation

def apply_transform_to_t(
    src_t: np.ndarray = None,               # [N, 3] translation vector
    aug_transforms: tuple = None            # (tgt_intrinsics, R, M_flip)
):    
    "Apply perspective transformation and YZ-plane flipping to position(translation)."

    src_t = src_t.copy()
    if src_t.ndim == 1:
        src_t = src_t.reshape(1, 3)

    _, R_trans, _ = aug_transforms

    tgt_t = (R_trans @ src_t.T).T

    return tgt_t

def apply_text_augmentation(
    src_text: str = None,   
    set_none_ratio: float = 0.3,  # probability of setting text to None
    sub_type: str = None,  # 'left' or 'right'
    rng: np.random.Generator = None
):
    "Set text to None with a certain probability for sub_type"
    if rng is None:
        rng = np.random.default_rng()

    tgt_text = copy.deepcopy(src_text)

    if rng.random() < set_none_ratio:

        left_start = tgt_text.index("Left hand:")
        right_start = tgt_text.index("Right hand:")
        left_part = tgt_text[left_start:right_start].strip()
        right_part = tgt_text[right_start:].strip()
        if sub_type == 'left':
            left_part = "Left hand: None."
        else:
            right_part = "Right hand: None."
        
        tgt_text = f"{left_part} {right_part}"
    
    return tgt_text

def apply_transform_to_text(
    src_text: str = None,                   # source text to be transformed
    aug_transforms: tuple = None            # (tgt_intrinsics, R, M_flip)
):          
    "Adjust the text for horizontal flips."

    _, _, M_flip = aug_transforms
    tgt_text = copy.deepcopy(src_text)
    
    if M_flip[0, 0] < 0:  # horizontal flip
        
        tgt_text = tgt_text.replace("upright", "<<placeholder1>>")
        tgt_text = tgt_text.replace("leftover", "<<placeholder2>>")

        tgt_text = tgt_text.replace("Left", "<<TEMP>>")
        tgt_text = tgt_text.replace("Right", "Left")
        tgt_text = tgt_text.replace("<<TEMP>>", "Right")

        tgt_text = tgt_text.replace("left", "<<TEMP>>")
        tgt_text = tgt_text.replace("right", "left")
        tgt_text = tgt_text.replace("<<TEMP>>", "right")

        left_start = tgt_text.index("Left hand:")
        right_start = tgt_text.index("Right hand:")

        if left_start < right_start:
            left_part = tgt_text[left_start:right_start].strip()
            right_part = tgt_text[right_start:].strip()
        else:
            right_part = tgt_text[right_start:left_start].strip()
            left_part = tgt_text[left_start:].strip()

        tgt_text = f"{left_part} {right_part}"
        tgt_text = tgt_text.replace("<<placeholder1>>", "upright")
        tgt_text = tgt_text.replace("<<placeholder2>>", "leftover")

    return tgt_text

def project_to_image_space(
    joints: np.ndarray,             # shape [N, M, 3] where N is number of samples and M is number of joints
    intrinsics: np.ndarray,         # shape [3, 3] normalized camera intrinsics
    render_size: Tuple[int, int]    # (height, width) of the target image
):
    "Project 3D joints to 2D image space using camera intrinsics."
    x = joints[..., 0]  # shape [N, M]
    y = joints[..., 1]
    z = joints[..., 2]
    z = np.clip(z, 0.05, None)  # Avoid division by zero

    ones = np.ones_like(z)
    points_normalized = np.stack([x / z, y / z, ones], axis=-1)  # shape [N, M, 3]

    points_normalized_flat = points_normalized.reshape(-1, 3)
    points_2d_flat = (intrinsics @ points_normalized_flat.T).T  # shape [N*M, 3]

    points_2d = points_2d_flat[:, :2].reshape(joints.shape[0], joints.shape[1], 2)  # shape [N, M, 2]
    
    points_2d[..., 0] *= render_size[1]  # width
    points_2d[..., 1] *= render_size[0]  # height
    uv_coords = np.round(points_2d).astype(np.int32)

    return uv_coords  # shape [N, M, 2]

def shrink_or_expand_bbox_uv(
    bbox_uv: np.ndarray,    # Array of shape (4,) with [x_min, y_min, x_max, y_max] in normalized [0, 1] coordinates
    margin_ratio = 0.0      # Expansion/shrink ratio (positive = expand, negative = shrink)
):
    """Scale a UV bbox about its centre by 1 + 2 * margin_ratio and clip it to [0, 1]."""
    x_min, y_min, x_max, y_max = bbox_uv
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    orig_w = x_max - x_min
    orig_h = y_max - y_min

    scale = 1.0 + 2.0 * margin_ratio
    new_w = orig_w * scale
    new_h = orig_h * scale

    new_x_min = cx - new_w / 2.0
    new_x_max = cx + new_w / 2.0
    new_y_min = cy - new_h / 2.0
    new_y_max = cy + new_h / 2.0

    new_x_min = max(0.0, new_x_min)
    new_y_min = max(0.0, new_y_min)
    new_x_max = min(1.0, new_x_max)
    new_y_max = min(1.0, new_y_max)

    return np.array([new_x_min, new_y_min, new_x_max, new_y_max], dtype=np.float32)

def resolve_valid_range(min_req, max_req, valid_min, valid_max):
    "Intersect [min_req, max_req] with [valid_min, valid_max]; if they are disjoint, return the nearer valid bound twice."
    if max_req < valid_min:
        return valid_min, valid_min
    elif min_req > valid_max:
        return valid_max, valid_max
    else:
        return max(min_req, valid_min), min(max_req, valid_max)

def contains_color_word(text: str) -> bool:
    color_words = [
        'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink',
        'black', 'white', 'gray', 'grey', 'brown', 'cyan', 'magenta',
        'gold', 'silver', 'beige', 'maroon', 'violet', 'indigo', 'turquoise',
        'navy', 'olive', 'teal', 'lime', 'ivory', 'bluish', 'reddish'
    ]

    pattern = r'\b(' + '|'.join(color_words) + r')\b'
    return re.search(pattern, text, flags=re.IGNORECASE) is not None

def augmentation_func(
        image, 
        intrinsics,
        actions,
        states,
        captions,
        uv_traj,
        target_size = (224, 224),
        augment_params=None,
        sub_type=None,
    ):
    """Warp the image (perspective, in-plane rotation, optional flip), apply the same transform
    to the intrinsics, actions, states and instruction, then colour-jitter the image.

    actions = (action_abs, action_rel, action_mask), states = (current_state, current_state_mask).
    Returns (image, intrinsics, actions, states, captions) in the same structure.
    """
    if image is not None:
        image = image.copy()
    intrinsics = intrinsics.copy() # (3,3)
    actions = copy.deepcopy(actions)
    states = copy.deepcopy(states)
    captions = copy.deepcopy(captions)

    # normalise the intrinsics by the image size, taken as twice the principal point
    intrinsics[0] /= intrinsics[0,2]*2
    intrinsics[1] /= intrinsics[1,2]*2

    tgt_aspect = augment_params.get('tgt_aspect', 1.0)
    margin_ratio = augment_params.get('margin_ratio', 0.05)
    center_augmentation = augment_params.get('center_augmentation', 1.0)
    fov_range_absolute = augment_params.get('fov_range_absolute', (45, 150))
    fov_range_relative = augment_params.get('fov_range_relative', (0.05, 1.0))
    inplane_range = augment_params.get('inplane_range', (-np.pi / 6, np.pi / 6))
    min_overlap = augment_params.get('min_overlap', 0.95)
    flip_augmentation = augment_params.get('flip_augmentation', 1.0)
    set_none_ratio = augment_params.get('set_none_ratio', 0.0)
    rng = augment_params.get('rng', np.random)

    aug_transforms = sample_perspective_rot_flip_with_traj_constraint(
        intrinsics,
        trajectory_uv = uv_traj,
        margin_ratio = margin_ratio,
        tgt_aspect = tgt_aspect,
        center_augmentation = center_augmentation,
        fov_range_absolute = fov_range_absolute,
        fov_range_relative = fov_range_relative,
        inplane_range = inplane_range,
        min_overlap = min_overlap,
        flip_augmentation = flip_augmentation,
        rng = rng,
    )

    new_intrinsics, R_trans, M_flip = aug_transforms

    tgt_width, tgt_height = target_size

    if image is not None:
        if len(image.shape) == 4:
            image = image.squeeze(0)
        new_image = warp_perspective(image,
            src_intrinsics=intrinsics,
            tgt_intrinsics=new_intrinsics,
            R = R_trans,
            tgt_width=tgt_width, 
            tgt_height=tgt_height,
        )
    else:
        new_image = None

    # unnormalize the intrinsics
    new_intrinsics[0] *= tgt_width
    new_intrinsics[1] *= tgt_height

    # apply the augmentation transform to the actions
    action_abs, action_rel, action_mask = actions

    action_abs_dim = action_abs.shape[1]
    action_rel_dim = action_rel.shape[1]
    abs_L = action_abs[:, :action_abs_dim//2]
    abs_R = action_abs[:, action_abs_dim//2:]
    rel_L = action_rel[:, :action_rel_dim//2]
    rel_R = action_rel[:, action_rel_dim//2:]
    msk_L = action_mask[:, 0]
    msk_R = action_mask[:, 1]

    abs_L_t = abs_L[:,:3]
    abs_R_t = abs_R[:,:3]
    rel_L_t = rel_L[:,:3]
    rel_R_t = rel_R[:,:3]

    abs_L_rot = R.from_euler('xyz', abs_L[:,3:6]).as_matrix()
    abs_R_rot = R.from_euler('xyz', abs_R[:,3:6]).as_matrix()
    rel_L_rot = R.from_euler('xyz', rel_L[:,3:6]).as_matrix()
    rel_R_rot = R.from_euler('xyz', rel_R[:,3:6]).as_matrix()

    abs_L_hand_pose = abs_L[:,6:]
    abs_R_hand_pose = abs_R[:,6:]
    rel_L_hand_pose = rel_L[:,6:]
    rel_R_hand_pose = rel_R[:,6:]

    if abs_L_hand_pose.shape[-1] != 45: # keypoints (T, N*3) instead of 45 joint angles: only the flip applies
        pose_dim = abs_L_hand_pose.shape[-1]
        abs_L_hand_pose = abs_L_hand_pose.copy().reshape(-1, 3)
        abs_L_hand_pose = (M_flip @ abs_L_hand_pose.T).T
        abs_L_hand_pose = abs_L_hand_pose.reshape(-1, pose_dim)
    if abs_R_hand_pose.shape[-1] != 45:
        pose_dim = abs_R_hand_pose.shape[-1]
        abs_R_hand_pose = abs_R_hand_pose.copy().reshape(-1, 3)
        abs_R_hand_pose = (M_flip @ abs_R_hand_pose.T).T
        abs_R_hand_pose = abs_R_hand_pose.reshape(-1, pose_dim)
    if rel_L_hand_pose.shape[-1] != 45:
        pose_dim = rel_L_hand_pose.shape[-1]
        rel_L_hand_pose = rel_L_hand_pose.copy().reshape(-1, 3)
        rel_L_hand_pose = (M_flip @ rel_L_hand_pose.T).T
        rel_L_hand_pose = rel_L_hand_pose.reshape(-1, pose_dim)
    if rel_R_hand_pose.shape[-1] != 45:
        pose_dim = rel_R_hand_pose.shape[-1]
        rel_R_hand_pose = rel_R_hand_pose.copy().reshape(-1, 3)
        rel_R_hand_pose = (M_flip @ rel_R_hand_pose.T).T
        rel_R_hand_pose = rel_R_hand_pose.reshape(-1, pose_dim)

    abs_L_t = apply_transform_to_t(abs_L_t, aug_transforms)
    abs_R_t = apply_transform_to_t(abs_R_t, aug_transforms)
    rel_L_t = apply_transform_to_t(rel_L_t, aug_transforms)
    rel_R_t = apply_transform_to_t(rel_R_t, aug_transforms)

    abs_L_rot = apply_transform_to_rot(abs_L_rot, aug_transforms)
    abs_R_rot = apply_transform_to_rot(abs_R_rot, aug_transforms)
    rel_L_rot = apply_transform_to_delta_rot(rel_L_rot, aug_transforms)
    rel_R_rot = apply_transform_to_delta_rot(rel_R_rot, aug_transforms)

    abs_L_rot_xyz = R.from_matrix(abs_L_rot).as_euler('xyz', degrees=False)
    abs_R_rot_xyz = R.from_matrix(abs_R_rot).as_euler('xyz', degrees=False)
    rel_L_rot_xyz = R.from_matrix(rel_L_rot).as_euler('xyz', degrees=False)
    rel_R_rot_xyz = R.from_matrix(rel_R_rot).as_euler('xyz', degrees=False)

    new_abs_L = np.concatenate([abs_L_t, abs_L_rot_xyz, abs_L_hand_pose], axis=1)
    new_abs_R = np.concatenate([abs_R_t, abs_R_rot_xyz, abs_R_hand_pose], axis=1)
    new_rel_L = np.concatenate([rel_L_t, rel_L_rot_xyz, rel_L_hand_pose], axis=1)
    new_rel_R = np.concatenate([rel_R_t, rel_R_rot_xyz, rel_R_hand_pose], axis=1)

    if M_flip[0,0] < 0:
        # flip the left hand to right hand
        new_abs_L, new_abs_R = new_abs_R, new_abs_L
        new_rel_L, new_rel_R = new_rel_R, new_rel_L
        msk_L, msk_R = msk_R, msk_L

    new_action_abs = np.concatenate([new_abs_L, new_abs_R], axis=1)
    new_action_rel = np.concatenate([new_rel_L, new_rel_R], axis=1)
    new_action_mask = np.stack([msk_L, msk_R], axis=1)  # (W,2)

    # randomly set sub_type hand text to None for single hand training
    captions = apply_text_augmentation(captions, set_none_ratio=set_none_ratio, sub_type=sub_type, rng=rng)

    new_captions = apply_transform_to_text(captions, aug_transforms)

    # keep the hue nearly fixed when the caption names a colour
    if contains_color_word(captions):
        preserve_hue = True
    else:
        preserve_hue = False

    if new_image is not None:
        new_image = apply_color_augmentation(new_image, preserve_hue=preserve_hue)
        new_image = new_image[None,...]

    # apply the augmentation transform to the states
    current_state, current_state_mask = states
    state_dim = current_state.shape[0]
    cur_L = current_state[:state_dim//2]
    cur_R = current_state[state_dim//2:]
    msk_L = current_state_mask[0]
    msk_R = current_state_mask[1]

    cur_L_t = cur_L[:3]
    cur_R_t = cur_R[:3]
    cur_L_rot = R.from_euler('xyz', cur_L[3:6]).as_matrix()
    cur_R_rot = R.from_euler('xyz', cur_R[3:6]).as_matrix()
    cur_L_hand_pose = cur_L[6:-10]
    cur_R_hand_pose = cur_R[6:-10]
    cur_L_beta = cur_L[-10:]
    cur_R_beta = cur_R[-10:]

    if cur_L_hand_pose.shape[-1] != 45: # keypoints: only the flip applies
        cur_L_hand_pose = cur_L_hand_pose.copy().reshape(-1, 3)
        cur_L_hand_pose = (M_flip @ cur_L_hand_pose.T).T
        cur_L_hand_pose = cur_L_hand_pose.reshape(-1)
    if cur_R_hand_pose.shape[-1] != 45:
        cur_R_hand_pose = cur_R_hand_pose.copy().reshape(-1, 3)
        cur_R_hand_pose = (M_flip @ cur_R_hand_pose.T).T
        cur_R_hand_pose = cur_R_hand_pose.reshape(-1)

    cur_L_t = apply_transform_to_t(cur_L_t, aug_transforms).squeeze(0)
    cur_R_t = apply_transform_to_t(cur_R_t, aug_transforms).squeeze(0)
    cur_L_rot = apply_transform_to_rot(cur_L_rot, aug_transforms).squeeze(0)
    cur_R_rot = apply_transform_to_rot(cur_R_rot, aug_transforms).squeeze(0)

    cur_L_rot_xyz = R.from_matrix(cur_L_rot).as_euler('xyz', degrees=False)
    cur_R_rot_xyz = R.from_matrix(cur_R_rot).as_euler('xyz', degrees=False)

    new_cur_L = np.concatenate([cur_L_t, cur_L_rot_xyz, cur_L_hand_pose, cur_L_beta], axis=0)
    new_cur_R = np.concatenate([cur_R_t, cur_R_rot_xyz, cur_R_hand_pose, cur_R_beta], axis=0) 

    if M_flip[0,0] < 0:
        # flip the left hand to right hand
        new_cur_L, new_cur_R = new_cur_R, new_cur_L
        msk_L, msk_R = msk_R, msk_L

    new_current_state = np.concatenate([new_cur_L, new_cur_R])
    new_current_state_mask = np.array([msk_L, msk_R])   

    return new_image, \
            new_intrinsics, \
            (new_action_abs, new_action_rel, new_action_mask), \
            (new_current_state, new_current_state_mask), \
            new_captions
