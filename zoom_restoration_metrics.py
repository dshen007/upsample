"""Shared source-selected grain and contour proxies, not HR ground truth."""
import cv2
import numpy as np

from display_support import mad, flat_mask, regions
from display_support import resize


def image_metrics(image, source_crop, baseline=None):
    baseline = resize(source_crop, image.shape[::-1]) if baseline is None else baseline
    lowmask = flat_mask(source_crop)
    mask = cv2.resize(lowmask.astype(np.uint8), image.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
    highpass = image-cv2.GaussianBlur(image, (0, 0), image.shape[1]/source_crop.shape[1]*1.2)
    result = {'flat_output_pixels': int(mask.sum()),
              'flat_grain_mad_dn': mad(highpass[mask]*255) if mask.sum() >= 400 else None,
              'change_mae_dn': float(np.abs(image-baseline).mean()*255)}
    descriptors = regions(source_crop)['edge_samples_xy_normal']
    if not descriptors:
        return {**result, 'edge_count': 0}
    d = np.asarray(descriptors, np.float32)
    t = np.linspace(-5, 5, 101, dtype=np.float32)
    tangent = np.arange(-2, 3, dtype=np.float32)
    sx, sy = image.shape[1]/source_crop.shape[1], image.shape[0]/source_crop.shape[0]
    qx = d[:, 0, None, None]+d[:, 2, None, None]*t[None, :, None]-d[:, 3, None, None]*tangent
    qy = d[:, 1, None, None]+d[:, 3, None, None]*t[None, :, None]+d[:, 2, None, None]*tangent
    mx, my = ((qx+.5)*sx-.5).reshape(len(d), -1), ((qy+.5)*sy-.5).reshape(len(d), -1)
    def profiles(x):
        return cv2.remap(x, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(len(d),101,5).mean(-1)
    bp, yp = profiles(baseline), profiles(image)
    widths, shifts, overshoots = [], [], []
    for b, y in zip(bp, yp):
        low, high = float(np.median(b[:10])), float(np.median(b[-10:]))
        if high-low < 15/255:
            continue
        # Reject complex or nonmonotone source profiles independently of candidates.
        if np.maximum(-np.diff(b), 0).sum() > .25*(high-low):
            continue
        def crossings(p):
            out=[]
            for level in (.1, .5, .9):
                v=low+level*(high-low)
                k=np.where((p[:-1] <= v)&(p[1:] >= v))[0]
                if not len(k):
                    return None
                i=k[np.argmin(np.abs(t[k]))]
                out.append(float(t[i]+.1*(v-p[i])/(p[i+1]-p[i]+1e-12)))
            return out if out[0] < out[1] < out[2] else None
        bc, yc = crossings(b), crossings(y)
        if bc is None or yc is None or bc[2]-bc[0] < .2:
            continue
        widths.append((yc[2]-yc[0])/(bc[2]-bc[0]))
        shifts.append(abs(yc[1]-bc[1]))
        overshoots.append(float(max(0, y.max()-high, low-y.min())*255))
    result.update(edge_count=len(widths), edge_width_ratio_median=float(np.median(widths)) if widths else None,
                  edge_shift_median_source_px=float(np.median(shifts)) if shifts else None,
                  edge_shift_p90_source_px=float(np.percentile(shifts,90)) if shifts else None,
                  edge_overshoot_median_dn=float(np.median(overshoots)) if widths else None)
    return result
