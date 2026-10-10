"""Model-independent infrared display I/O, geometry and evaluation helpers.
Extracted unchanged from the retired experimental pipelines on 2026-10-09.
SIFT geometry is an evaluation proxy, not a VINS feature pipeline.
"""
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from ir_sr import read_gray, quality
from runtime_v5 import sha, atomic_json

def resize(x, size, method='bicubic'):
    filters = {'bicubic': Image.Resampling.BICUBIC,
               'lanczos': Image.Resampling.LANCZOS, 'nearest': Image.Resampling.NEAREST}
    if method not in filters or min(size) < 1:
        raise ValueError('Invalid resampling method or output size')
    return np.asarray(Image.fromarray(np.asarray(x, np.float32)).resize(
        tuple(size), filters[method]), np.float32).clip(0, 1).copy()


def center_roi(shape, scale):
    if scale not in (2, 3):
        raise ValueError('Zoom must be 2 or 3')
    h, w = shape
    rw, rh = int(round(w/scale)), int(round(h/scale))
    return ((w-rw)//2, (h-rh)//2, rw, rh)


def geometry(shape, roi, size):
    h, w = shape
    x, y, rw, rh = roi
    if len(size) != 2 or min(size) < 1 or min(rw, rh) < 2 or x < 0 or y < 0 or x+rw > w or y+rh > h:
        raise ValueError('ROI is outside the source or output dimensions are invalid')
    sx, sy = size[0]/rw, size[1]/rh
    transform = np.array([[sx, 0, sx*(.5-x)-.5],
                          [0, sy, sy*(.5-y)-.5], [0, 0, 1]], np.float64)
    return {'roi_xywh': list(roi), 'source_size_wh': [w, h], 'output_size_wh': list(size),
            'actual_scale_xy': [sx, sy], 'source_to_output': transform.tolist(),
            'output_to_source': np.linalg.inv(transform).tolist(),
            'pixel_center_rule': 'q_out = s * (q_source - roi_origin + 0.5) - 0.5'}


def transform_intrinsics(k, metadata):
    matrix = np.asarray(k, np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError('Expected a finite 3x3 camera matrix')
    return np.asarray(metadata['source_to_output']) @ matrix


def prepare(raw, scale, scenario='crop', roi=None, size=None):
    x = np.asarray(raw, np.float32)
    if x.ndim != 2 or not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError('Expected finite normalized grayscale')
    size = tuple(size or x.shape[::-1])
    roi = tuple(roi or center_roi(x.shape, scale))
    metadata = geometry(x.shape, roi, size)
    x0, y0, rw, rh = roi
    if scenario == 'crop':
        low = x[y0:y0+rh, x0:x0+rw].copy()
        reference = None
    elif scenario == 'sampling':
        if roi != center_roi(x.shape, scale) or size != x.shape[::-1]:
            raise ValueError('Sampling simulation uses the full reference and default LR dimensions')
        reference = x.copy()
        low = resize(reference, (rw, rh))
        # The full reference footprint is compressed into LR; it is NOT a crop.
        metadata = geometry(x.shape, (0, 0, x.shape[1], x.shape[0]), size)
        metadata['reference_to_lr'] = geometry(x.shape, (0, 0, x.shape[1], x.shape[0]), (rw, rh))['source_to_output']
        metadata['low_to_output'] = geometry(low.shape, (0, 0, rw, rh), size)['source_to_output']
        metadata['actual_scale_xy'] = [size[0]/rw, size[1]/rh]
    else:
        raise ValueError('Scenario must be crop or sampling')
    metadata.update(scenario=scenario, nominal_scale=scale, low_size_wh=list(low.shape[::-1]),
                    reference_is_true_hr=False)
    return low, reference, metadata


def save_image(path, x):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(x, 0, 1)*255).astype(np.uint8)).save(path)


def checked_read(root, row):
    path = root/row['relative_path']
    if sha(path) != row['png_sha256']:
        raise ValueError(f'Input hash mismatch: {path}')
    image = read_gray(path)
    if image.shape != (512, 640):
        raise ValueError('Expected verified 640x512 thermal frame')
    return image


def symmetric_matches(descriptors_a, descriptors_b, norm):
    if descriptors_a is None or descriptors_b is None or len(descriptors_a) < 2 or len(descriptors_b) < 2:
        return []
    matcher = cv2.BFMatcher(norm)
    forward = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    reverse = matcher.knnMatch(descriptors_b, descriptors_a, k=2)
    accepted_reverse = {(m.trainIdx, m.queryIdx) for pair in reverse
                        if len(pair) == 2 for m, other in [pair]
                        if m.distance < 0.8 * other.distance}
    return [m for pair in forward if len(pair) == 2
            for m, other in [pair]
            if m.distance < 0.8 * other.distance
            and (m.queryIdx, m.trainIdx) in accepted_reverse]


def points(keypoints_a, keypoints_b, matches):
    a = np.float32([keypoints_a[m.queryIdx].pt for m in matches])
    b = np.float32([keypoints_b[m.trainIdx].pt for m in matches])
    return a, b


def source_geometry(sift_a, sift_b):
    ka, da = sift_a
    kb, db = sift_b
    matches = symmetric_matches(da, db, cv2.NORM_L2)
    if len(matches) < 30:
        return None, len(matches), 0
    pa, pb = points(ka, kb, matches)
    cv2.setRNGSeed(20260929)
    h, inlier_mask = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0,
                                        maxIters=5000, confidence=0.995)
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    if h is None or inliers < 25 or inliers / len(matches) < 0.55:
        return None, len(matches), inliers
    return h, len(matches), inliers


def to_u8(x):
    return np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8)


def write_table(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ref_metrics(prediction, reference):
    a, b = prediction[8:-8, 8:-8], reference[8:-8, 8:-8]
    result = quality(a, b)
    result['gradient_mae_dn'] = float(.5*(np.abs(np.diff(a, axis=0)-np.diff(b, axis=0)).mean()+
                                               np.abs(np.diff(a, axis=1)-np.diff(b, axis=1)).mean())*255)
    ea = cv2.Canny(to_u8(a), 40, 80) > 0
    eb = cv2.Canny(to_u8(b), 40, 80) > 0
    if ea.any() and eb.any():
        da = cv2.distanceTransform((~ea).astype(np.uint8), cv2.DIST_L2, 5)
        db = cv2.distanceTransform((~eb).astype(np.uint8), cv2.DIST_L2, 5)
        precision, recall = float((db[ea] <= 2).mean()), float((da[eb] <= 2).mean())
        result.update(edge_precision_2px=precision, edge_recall_2px=recall,
                      edge_f1_2px=2*precision*recall/(precision+recall) if precision+recall else 0.)
    else:
        result.update(edge_precision_2px=None, edge_recall_2px=None, edge_f1_2px=None)
    return result


def temporal_metrics(previous, current, h, reference_previous=None, reference_current=None):
    if h is None or previous is None:
        return {}
    size = current.shape[::-1]
    warped = cv2.warpPerspective(previous, h, size, flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.ones(previous.shape, np.uint8), h, size, flags=cv2.INTER_NEAREST) != 0
    mask[:8] = mask[-8:] = False
    mask[:, :8] = mask[:, -8:] = False
    if mask.sum() < 100:
        return {}
    residual = current-warped
    result = {'warp_difference_mae_dn': float(np.abs(residual[mask]).mean()*255)}
    if reference_previous is not None:
        warped_ref = cv2.warpPerspective(reference_previous, h, size, flags=cv2.INTER_LINEAR)
        true_residual = reference_current-warped_ref
        result['temporal_reference_error_dn'] = float(np.abs((residual-true_residual)[mask]).mean()*255)
    return result


def load_geometry(out, bag, clip, sources, hashes, existing=None):
    cache = out/'geometry'/f'{bag}_{clip}.json'
    if existing is not None:
        import json
        source_cache = existing/'geometry'/cache.name
        if source_cache.exists():
            value = json.loads(source_cache.read_text(encoding='utf-8'))
            if value['source_sha256'] == hashes:
                atomic_json(cache, value)
                return ([None]+[np.array(r['H']) if r['H'] is not None else None for r in value['pairs']],
                        [np.eye(3)]+[np.array(r['anchor_H']) if r['anchor_H'] is not None else None for r in value['pairs']])
    sift = cv2.SIFT_create(nfeatures=1800, contrastThreshold=.015)
    features = [sift.detectAndCompute(to_u8(x), None) for x in sources]
    pair_h, anchors, rows = [None], [np.eye(3)], []
    for i in range(1, len(sources)):
        h, count, inliers = source_geometry(features[i-1], features[i])
        pair_h.append(h)
        # Direct frame-0 audit avoids compounding an estimated per-frame homography.
        anchor, _, _ = source_geometry(features[0], features[i])
        anchors.append(anchor)
        rows.append({'frame': i, 'candidate_matches': count, 'inliers': inliers,
                     'H': h.tolist() if h is not None else None,
                     'anchor_H': anchor.tolist() if anchor is not None else None})
    atomic_json(cache, {'source_sha256': hashes, 'pairs': rows, 'proxy_not_ground_truth': True})
    return pair_h, anchors


def checked(x):
    x = np.asarray(x, np.float32)
    if x.ndim != 2 or min(x.shape) < 16 or not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError('Expected finite normalized native grayscale')
    return x


def gradient(x):
    smooth = cv2.GaussianBlur(x, (0, 0), .7)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3, scale=1/8)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3, scale=1/8)
    return gx, gy, np.hypot(gx, gy)


def adaptive_denoise(x, sigma_dn=1.5, strength=.75):
    """Noise-aware local Wiener smoothing with a conservative edge gate."""
    x = checked(x)
    if sigma_dn <= 0 or strength <= 0:
        return x.copy()
    mean = cv2.GaussianBlur(x, (0, 0), .8)
    variance = np.maximum(cv2.GaussianBlur(x*x, (0, 0), .8)-mean*mean, 0)
    noise2 = (sigma_dn*strength/255)**2
    mix = np.clip(noise2/(variance+1e-10), 0, .85)
    _, _, g = gradient(x)
    mix *= 1/(1+(g*255/4.)**4)
    return np.clip(x+(mean-x)*mix, 0, 1).astype(np.float32)


def mad(x):
    v = np.asarray(x)
    return float(1.4826*np.median(np.abs(v-np.median(v)))) if v.size else None


def flat_mask(x):
    _, _, g = gradient(x)
    local = cv2.GaussianBlur((x-cv2.GaussianBlur(x, (0, 0), 1.2))**2, (0, 0), 3)
    mask = (g*255 < 1.5) & (local < np.quantile(local, .35)) & (x > .03) & (x < .97)
    mask = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    mask[:8] = mask[-8:] = False
    mask[:, :8] = mask[:, -8:] = False
    return mask


def regions(x):
    mask = flat_mask(x)
    boxes = []
    candidates = []
    for y in range(8, x.shape[0]-24, 16):
        for xx in range(8, x.shape[1]-24, 16):
            if mask[y:y+16, xx:xx+16].mean() > .8:
                patch = x[y:y+16, xx:xx+16]
                candidates.append((float(patch.var()), xx, y))
    for _, xx, y in sorted(candidates):
        if all(np.hypot(xx-b[0], y-b[1]) > 32 for b in boxes):
            boxes.append([xx, y, 16, 16])
        if len(boxes) >= 6:
            break
    gx, gy, g = gradient(x)
    edges = cv2.Canny(np.rint(x*255).astype(np.uint8), 40, 80) > 0
    valid = edges & (g*255 > 6)
    valid[:8] = valid[-8:] = False
    valid[:, :8] = valid[:, -8:] = False
    ys, xs = np.where(valid)
    selected = []
    for k in np.argsort(-g[ys, xs]):
        xx, y = int(xs[k]), int(ys[k])
        if all(np.hypot(xx-p[0], y-p[1]) > 14 for p in selected):
            selected.append([xx, y, float(gx[y, xx]/g[y, xx]), float(gy[y, xx]/g[y, xx])])
        if len(selected) >= 80:
            break
    return {'flat_boxes': boxes, 'edge_samples_xy_normal': selected,
            'selection': 'automatic source-only low-gradient patches and strong Canny edges; not manual semantic labels'}
