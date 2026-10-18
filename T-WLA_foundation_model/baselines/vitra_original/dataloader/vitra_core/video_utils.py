"""Vendored from VITRA (`vitra/datasets/video_utils.py`), pruned to `load_video_decord`.

`_load_frames_opencv` is a local addition: a fallback for containers decord cannot read.
None of this runs when a JPEG frame store is configured.
"""
import ffmpeg
import decord
from skimage.color import gray2rgb
import numpy as np
import torch
from PIL import Image
import random

def _load_frames_opencv(name, frame_index):
    """Fallback decoder for containers decord cannot read (e.g. VP9 .webm): a sequential read.

    Returns (N, H, W, 3) RGB in the order of `frame_index` (duplicates allowed). A frame that
    could not be read is replaced by the one returned before it.
    """
    import cv2
    idx = [int(i) for i in frame_index]
    want = set(idx)
    grabbed = {}
    cap = cv2.VideoCapture(name)
    i, last = 0, max(idx)
    while i <= last:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            grabbed[i] = fr[:, :, ::-1].copy()   # BGR -> RGB
        i += 1
    cap.release()
    if not grabbed:
        raise RuntimeError("opencv fallback decoded no frames from %s" % name)
    last_good = None
    out = []
    for k in idx:
        f = grabbed.get(k, last_good)
        if f is None:
            f = grabbed[min(grabbed)]
        last_good = f
        out.append(f)
    return np.stack(out, axis=0)

def load_video_decord(name, 
                      frame_index=None, 
                      num_random=2, 
                      load_full_video=False, 
                      sampling_step=1, 
                      max_frame_cnt=5,
                      is_continuous=False,
                      rotation=False,
                      st_list=None,
                      crop_size=None,
                      ):
    video_reader = decord.VideoReader(name)
    num_frames = len(video_reader)
    if frame_index is None:
        if load_full_video:
            if sampling_step > 0:
                frame_index = list(range(0, num_frames, sampling_step))[:max_frame_cnt]
            else:
                # max_frame_cnt indices spread uniformly over the video
                frame_index = np.linspace(0, num_frames, max_frame_cnt + 1, endpoint=False)[1:]
                frame_index = list(np.round(frame_index).astype(np.int32))
        else:
            if is_continuous:
                if st_list is not None:
                    st = np.random.choice(st_list)
                else:
                    st = np.random.randint(0, num_frames - num_random)
                frame_index = list(range(st, st + num_random))
            else:
                if st_list is not None:
                    frame_index = np.random.choice(st_list, replace=False, size=num_random)
                else:
                    frame_index = np.random.choice(num_frames, replace=False, size=num_random)
    try:
        video = video_reader.get_batch(frame_index).asnumpy()
    except Exception:
        video = _load_frames_opencv(name, frame_index)
    if len(video.shape) == 3:
        video = np.array([gray2rgb(frame) for frame in video])
    if video.shape[-1] == 4:
        video = video[..., :3]
    if rotation:
        video = np.flip(np.transpose(video, (0, 2, 1, 3)), axis=2)
    if crop_size is not None:
        video = center_crop_video(video, crop_size=(crop_size, crop_size))
    del video_reader
    return video, frame_index

def center_crop_video(video, crop_size= (256, 256) ):
    """Centre-crop (num_frames, H, W, C) frames to `crop_size` = (height, width)."""
    num_frames, height, width, channels = video.shape

    if height < crop_size[0] or width < crop_size[1]:
        raise ValueError("Video dimensions must be at least the cropped size.")

    start_y = (height - crop_size[0]) // 2
    start_x = (width - crop_size[1]) // 2

    cropped_video = video[:, start_y:start_y + crop_size[0], start_x:start_x + crop_size[1], :]

    return cropped_video
