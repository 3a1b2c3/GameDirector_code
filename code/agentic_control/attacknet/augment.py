"""Clip-level augmentation for AttackNet.

Design constraint that drives everything here: the label is "did HP drop during
this 0.25s cell", and the only visual evidence is a hit flash / impact animation
spanning the three frames of a cell. So every photometric transform must be
applied *identically to all frames of the clip* -- a per-frame brightness jitter
would manufacture exactly the temporal luminance step the model is supposed to
read as a hit, and would poison the labels.

The baseline recipe (hflip + -/+8 brightness + 0.9-1.1 contrast) is far too weak
to bridge the two capture locations: Frozen Wasteland is dark and blue, Temple
Arena is bright and yellow, and a model trained on that recipe saturates
player_hit to 1.0 on every Temple Arena cell. The `strong` recipe below jitters
hue/saturation/value/gamma hard enough that scene palette stops being a usable
shortcut feature.
"""

from __future__ import annotations

import random

import numpy as np


def _as_pixel_block(frames: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """[cells, frames, 3, H, W] uint8 -> contiguous [(cells*frames*H), W, 3] for cv2."""
    cells, per_cell, channels, height, width = frames.shape
    block = frames.transpose(0, 1, 3, 4, 2).reshape(cells * per_cell * height, width, channels)
    return np.ascontiguousarray(block), (cells, per_cell, channels, height, width)


def _from_pixel_block(block: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    cells, per_cell, channels, height, width = shape
    restored = block.reshape(cells, per_cell, height, width, channels)
    return np.ascontiguousarray(restored.transpose(0, 1, 4, 2, 3))


def _gamma_lut(gamma: float) -> np.ndarray:
    grid = np.arange(256, dtype=np.float32) / 255.0
    return np.clip(np.power(grid, gamma) * 255.0, 0, 255).astype(np.uint8)


def photometric_clip(
    frames: np.ndarray,
    rng: random.Random,
    hue_degrees: float,
    saturation: float,
    value: float,
    contrast: float,
    gamma: float,
    gray_probability: float,
    channel_shuffle_probability: float = 0.0,
) -> np.ndarray:
    """Apply one clip-wide photometric transform to ``[cells, frames, 3, H, W]`` uint8."""
    import cv2

    if not (hue_degrees or saturation or value or contrast or gamma or gray_probability
            or channel_shuffle_probability):
        return frames

    block, shape = _as_pixel_block(frames)

    if gray_probability and rng.random() < gray_probability:
        gray = cv2.cvtColor(block, cv2.COLOR_RGB2GRAY)
        block = np.repeat(gray[:, :, None], 3, axis=2)

    if channel_shuffle_probability and rng.random() < channel_shuffle_probability:
        order = [0, 1, 2]
        rng.shuffle(order)
        block = np.ascontiguousarray(block[:, :, order])

    if hue_degrees or saturation or value:
        hsv = cv2.cvtColor(block, cv2.COLOR_RGB2HSV).astype(np.int16)
        if hue_degrees:
            # OpenCV packs hue into 0..179, i.e. half-degrees.
            shift = int(round(rng.uniform(-hue_degrees, hue_degrees) / 2.0))
            hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
        if saturation:
            scale = 1.0 + rng.uniform(-saturation, saturation)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)
        if value:
            scale = 1.0 + rng.uniform(-value, value)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * scale, 0, 255)
        block = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    if contrast:
        scale = 1.0 + rng.uniform(-contrast, contrast)
        mean = float(block.mean())
        block = np.clip((block.astype(np.float32) - mean) * scale + mean, 0, 255).astype(np.uint8)

    if gamma:
        exponent = float(np.exp(rng.uniform(-gamma, gamma)))
        block = cv2.LUT(block, _gamma_lut(exponent))

    return _from_pixel_block(block, shape)


def geometric_clip(
    frames: np.ndarray,
    rng: random.Random,
    horizontal_flip: bool,
    scale_jitter: float,
    translate: float,
) -> np.ndarray:
    """Flip / random-resized-crop the whole clip with one shared transform."""
    import cv2

    if horizontal_flip and rng.random() < 0.5:
        frames = frames[..., ::-1]

    if scale_jitter or translate:
        cells, per_cell, channels, height, width = frames.shape
        keep = 1.0 - rng.uniform(0.0, scale_jitter) if scale_jitter else 1.0
        crop_h, crop_w = max(int(height * keep), 8), max(int(width * keep), 8)
        span_y, span_x = height - crop_h, width - crop_w
        if translate:
            top = int(rng.uniform(0, 1) * span_y)
            left = int(rng.uniform(0, 1) * span_x)
        else:
            top, left = span_y // 2, span_x // 2
        cropped = frames[..., top : top + crop_h, left : left + crop_w]
        flat = cropped.transpose(0, 1, 3, 4, 2).reshape(cells * per_cell, crop_h, crop_w, channels)
        resized = np.empty((cells * per_cell, height, width, channels), dtype=np.uint8)
        for index in range(flat.shape[0]):
            resized[index] = cv2.resize(flat[index], (width, height), interpolation=cv2.INTER_LINEAR)
        frames = resized.reshape(cells, per_cell, height, width, channels).transpose(0, 1, 4, 2, 3)

    return np.ascontiguousarray(frames)


def to_grayscale(frames: np.ndarray) -> np.ndarray:
    """Deterministic RGB->gray->RGB, used as an *input mode* (train and eval alike)."""
    import cv2

    block, shape = _as_pixel_block(frames)
    gray = cv2.cvtColor(block, cv2.COLOR_RGB2GRAY)
    return _from_pixel_block(np.repeat(gray[:, :, None], 3, axis=2), shape)


# Named recipes: (hue_degrees, saturation, value, contrast, gamma, gray_p, shuffle_p,
#                 scale_jitter, translate)
RECIPES: dict[str, dict[str, float]] = {
    # The historical baseline, kept bit-comparable for reference runs.
    "baseline": dict(
        hue_degrees=0.0, saturation=0.0, value=0.0, contrast=0.1, gamma=0.0,
        gray_probability=0.0, channel_shuffle_probability=0.0,
        scale_jitter=0.0, translate=0.0, brightness_shift=8.0,
    ),
    "medium": dict(
        hue_degrees=15.0, saturation=0.3, value=0.25, contrast=0.25, gamma=0.25,
        gray_probability=0.1, channel_shuffle_probability=0.0,
        scale_jitter=0.10, translate=1.0, brightness_shift=0.0,
    ),
    # Aimed squarely at the snow/temple palette gap.
    "strong": dict(
        hue_degrees=40.0, saturation=0.6, value=0.4, contrast=0.4, gamma=0.4,
        gray_probability=0.25, channel_shuffle_probability=0.1,
        scale_jitter=0.15, translate=1.0, brightness_shift=0.0,
    ),
    "extreme": dict(
        hue_degrees=90.0, saturation=0.8, value=0.5, contrast=0.5, gamma=0.5,
        gray_probability=0.4, channel_shuffle_probability=0.25,
        scale_jitter=0.20, translate=1.0, brightness_shift=0.0,
    ),
}


def augment_clip(frames: np.ndarray, recipe: str, rng: random.Random) -> np.ndarray:
    """Apply the named augmentation recipe to one ``[cells, frames, 3, H, W]`` clip."""
    if recipe == "none":
        return frames
    if recipe not in RECIPES:
        raise ValueError(f"unknown augment recipe: {recipe!r}")
    config = RECIPES[recipe]
    frames = geometric_clip(
        frames,
        rng,
        horizontal_flip=True,
        scale_jitter=float(config["scale_jitter"]),
        translate=float(config["translate"]),
    )
    frames = photometric_clip(
        frames,
        rng,
        hue_degrees=float(config["hue_degrees"]),
        saturation=float(config["saturation"]),
        value=float(config["value"]),
        contrast=float(config["contrast"]),
        gamma=float(config["gamma"]),
        gray_probability=float(config["gray_probability"]),
        channel_shuffle_probability=float(config["channel_shuffle_probability"]),
    )
    shift = float(config["brightness_shift"])
    if shift:
        offset = rng.uniform(-shift, shift)
        frames = np.clip(frames.astype(np.float32) + offset, 0, 255).astype(np.uint8)
    return frames
