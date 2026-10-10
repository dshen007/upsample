"""Shared grayscale I/O. Float arrays always use the range [0, 1]."""
from pathlib import Path
import json
import numpy as np
from PIL import Image


def read_gray(path, bits=8):
    if not 8 <= bits <= 16:
        raise ValueError('bits must be between 8 and 16')
    with Image.open(path) as image:
        if getattr(image, 'n_frames', 1) != 1:
            raise ValueError('Multi-frame images must be split into individual grayscale frames')
        if image.mode == 'P':
            raise ValueError('Palette images are not raw grayscale inputs')
        data = np.asarray(image)
    if data.ndim != 2 or data.dtype.kind not in 'ui':
        raise ValueError(f'{path}: expected integer single-channel image, got {data.shape}/{data.dtype}')
    maximum = (1 << bits) - 1
    if data.min() < 0 or data.max() > maximum:
        raise ValueError(f'{path}: samples outside 0..{maximum}; check --bits')
    if bits > 8 and data.dtype.itemsize == 1:
        raise ValueError(f'{path}: 8-bit storage supplied for --bits {bits}')
    return np.ascontiguousarray(data.astype(np.float32) / maximum)


def resize_gray(data, width, height):
    if width < 1 or height < 1:
        raise ValueError('image dimensions must be positive')
    result = Image.fromarray(data.astype(np.float32)).resize(
        (width, height), Image.Resampling.BICUBIC)
    return np.clip(np.asarray(result, dtype=np.float32), 0, 1).copy()


def image_paths(folder):
    paths = sorted(p.resolve() for p in Path(folder).rglob('*')
                   if p.suffix.lower() in {'.png', '.tif', '.tiff', '.bmp'} and p.is_file())
    if not paths:
        raise ValueError(f'No PNG/TIFF/BMP images in {folder}')
    return paths


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def quality(prediction, reference):
    if prediction.shape != reference.shape:
        raise ValueError('Prediction and reference shapes differ')
    error = prediction.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(error ** 2))
    return {'mae': float(np.mean(np.abs(error))), 'max_abs': float(np.max(np.abs(error))),
            'psnr_db': float(-10 * np.log10(mse)) if mse > 0 else None,
            'exact_match': mse == 0}


def save_result(folder, low, prediction, reference=None, scale=2):
    if scale not in (2, 3):
        raise ValueError('Supported SR scales are 2 and 3')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    expected = (low.shape[0] * scale, low.shape[1] * scale)
    if prediction.shape != expected or not np.isfinite(prediction).all():
        raise ValueError(f'Invalid SR result: expected {expected}, got {prediction.shape}')
    prediction = np.asarray(prediction, dtype=np.float32)
    np.save(folder / 'sr_float.npy', prediction)
    sr = np.clip(prediction, 0, 1)
    baseline = resize_gray(low, expected[1], expected[0])
    nearest = np.repeat(np.repeat(low, scale, axis=0), scale, axis=1)
    for name, data in [('sr', sr), ('bicubic', baseline),
                       ('comparison', np.concatenate([nearest, baseline, sr], axis=1))]:
        Image.fromarray(np.rint(data * 255).astype(np.uint8)).save(folder / f'{name}.png')
    # These are normalized display intensities, not calibrated temperatures.
    Image.fromarray(np.rint(sr * 65535).astype(np.uint16)).save(folder / 'sr_16bit.png')
    metrics = {'comparison_left_to_right': ['nearest', 'bicubic', 'model'],
               'raw_output_range': [float(prediction.min()), float(prediction.max())]}
    if reference is not None:
        metrics['model_vs_hr'] = quality(sr, reference)
        metrics['bicubic_vs_hr'] = quality(baseline, reference)
    write_json(folder / 'metrics.json', metrics)
    return metrics
