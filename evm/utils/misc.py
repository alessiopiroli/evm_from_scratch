import subprocess

import cv2
import numpy as np
import scipy.signal as signal
import yaml
from easydict import EasyDict as edict


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return edict(config)


def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        frames.append(ycrcb_frame)

    cap.release()
    return np.array(frames, dtype=np.float32), fps


def get_pyramids(frames, n_levels):
    pyramids = []
    organized_pyramids = []

    for frame in frames:
        pyramids.append(build_pyramid(frame, n_levels))

    for level in range(n_levels + 1):
        level_frames = np.array([pyramid[level] for pyramid in pyramids])
        organized_pyramids.append(level_frames)

    return organized_pyramids


def build_pyramid(frame, n_levels):
    gaussian_pyramid = [frame]
    laplacian_pyramid = []

    for _ in range(n_levels):
        frame = cv2.pyrDown(frame)
        gaussian_pyramid.append(frame)

    for i in range(n_levels):
        h, w = gaussian_pyramid[i].shape[:2]
        expanded = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=(w, h))
        laplacian_pyramid.append(gaussian_pyramid[i] - expanded)

    laplacian_pyramid.append(gaussian_pyramid[-1])

    return laplacian_pyramid


def isolate_frequencies(pyramids, lowcut, range, fps):
    nyquist_frequency = fps / 2.0
    low = lowcut / nyquist_frequency
    high = (lowcut + range) / nyquist_frequency

    b, a = signal.butter(1, [low, high], btype="band")
    filtered_pyramids = []

    for level in pyramids:
        filtered_level = signal.filtfilt(b, a, level, axis=0)
        filtered_pyramids.append(filtered_level)

    return filtered_pyramids


def magnificate(original_pyramids, filtered_pyramids, alpha, chrom_attenuation=0.1):
    amplified_pyramid = []
    last_level = len(original_pyramids) - 1

    for level_idx, (original_level, filtered_level) in enumerate(zip(original_pyramids, filtered_pyramids)):
        if level_idx == last_level:
            amplified_pyramid.append(original_level.copy())
            continue

        boost = filtered_level.copy()
        boost[..., 1] *= chrom_attenuation
        boost[..., 2] *= chrom_attenuation
        amplified_level = original_level + (boost * alpha)
        amplified_pyramid.append(amplified_level)

    return amplified_pyramid


def create_magnified_video(amplified, output_path, fps, original_frames):
    height, width = original_frames[0].shape[:2]

    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps}",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            output_path,
        ],
        stdin=subprocess.PIPE,
    )

    for frame_idx in range(len(original_frames)):
        frame_pyramid = [amplified[level_idx][frame_idx] for level_idx in range(len(amplified))]
        reconstructed_ycrcb = reconstruct_frame(frame_pyramid)
        reconstructed_ycrcb = np.clip(reconstructed_ycrcb, 0, 255).astype(np.uint8)
        bgr_frame = cv2.cvtColor(reconstructed_ycrcb, cv2.COLOR_YCrCb2BGR)
        ffmpeg.stdin.write(bgr_frame.tobytes())

    ffmpeg.stdin.close()
    ffmpeg.wait()


def reconstruct_frame(pyramid_levels):
    current = pyramid_levels[-1]

    for i in range(len(pyramid_levels) - 2, -1, -1):
        h, w = pyramid_levels[i].shape[:2]
        current = cv2.pyrUp(current, dstsize=(w, h)) + pyramid_levels[i]

    return current
