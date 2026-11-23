"""
Comprehensive Image Fusion Evaluation Metrics
Includes: EN, SD, SF, AG, MI, Qabf, SSIM, VIF, CC, SCD, FMI, PSNR, MSE
"""
import numpy as np
import cv2
import math
from scipy import signal, ndimage
from scipy.ndimage import sobel, uniform_filter
from skimage.metrics import structural_similarity as ssim
from skimage.feature import graycomatrix, graycoprops
from sklearn.feature_extraction.image import extract_patches_2d
import pywt

def normalize_img(img):
    """Normalize image to [0, 255] uint8"""
    if img.max() != img.min():
        img = (img - img.min()) / (img.max() - img.min()) * 255
    return img.astype(np.uint8)

def to_gray(img):
    """Convert to grayscale if needed"""
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif len(img.shape) == 3 and img.shape[2] == 1:
        return img.squeeze(2)
    return img

# ========== Traditional Fusion Metrics ==========

def entropy(img):
    """Calculate Entropy (EN) - measures information content"""
    img = to_gray(img)
    img = normalize_img(img)
    hist, _ = np.histogram(img, bins=256, range=(0, 255))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def std_deviation(img):
    """Calculate Standard Deviation (SD) - measures contrast"""
    img = to_gray(img)
    return np.std(img)

def spatial_frequency(img):
    """Calculate Spatial Frequency (SF), float64-safe.

    Original implementation subtracted uint8 arrays, which wraps around.
    This version casts to float64 before differencing.
    """
    img = to_gray(img).astype(np.float64)
    RF = np.sqrt(np.mean((img[:, 1:] - img[:, :-1]) ** 2))
    CF = np.sqrt(np.mean((img[1:, :] - img[:-1, :]) ** 2))
    return float(np.sqrt(RF ** 2 + CF ** 2))

def spatial_frequency_legacy(img):
    """Original SF implementation, kept only for legacy comparison."""
    img = to_gray(img)
    RF = np.sqrt(np.mean((img[:, 1:] - img[:, :-1]) ** 2))
    CF = np.sqrt(np.mean((img[1:, :] - img[:-1, :]) ** 2))
    return float(np.sqrt(RF ** 2 + CF ** 2))

def average_gradient(img):
    """Calculate Average Gradient (AG)"""
    img = to_gray(img).astype(np.float64)
    dx = img[:, 1:] - img[:, :-1]
    dy = img[1:, :] - img[:-1, :]
    gx = np.mean(np.abs(dx))
    gy = np.mean(np.abs(dy))
    return (gx + gy) / 2

def mutual_information(img1, img2):
    """Calculate Mutual Information (MI) between two images"""
    img1 = to_gray(img1)
    img2 = to_gray(img2)
    img1 = normalize_img(img1)
    img2 = normalize_img(img2)
    hist_2d, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=256)
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    mi = np.sum(pxy[nzs] * np.log2(pxy[nzs] / px_py[nzs]))
    return mi

def correlation_coefficient(img1, img2):
    """Calculate Correlation Coefficient (CC)"""
    img1 = to_gray(img1).astype(np.float64)
    img2 = to_gray(img2).astype(np.float64)
    mean1, mean2 = np.mean(img1), np.mean(img2)
    num = np.sum((img1 - mean1) * (img2 - mean2))
    den = np.sqrt(np.sum((img1 - mean1)**2) * np.sum((img2 - mean2)**2))
    return num / den if den != 0 else 0

def psnr(img1, img2):
    """Calculate PSNR between two images"""
    img1 = to_gray(img1).astype(np.float64)
    img2 = to_gray(img2).astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def mse(img1, img2):
    """Mean Squared Error"""
    img1 = to_gray(img1).astype(np.float64)
    img2 = to_gray(img2).astype(np.float64)
    return np.mean((img1 - img2) ** 2)

def ssim_metric(img1, img2):
    """Calculate SSIM"""
    img1 = normalize_img(to_gray(img1))
    img2 = normalize_img(to_gray(img2))
    return ssim(img1, img2, data_range=255)

def vif_metric(img_ref, img_dist):
    """Visual Information Fidelity (VIF) - simplified implementation"""
    img_ref = to_gray(img_ref).astype(np.float64)
    img_dist = to_gray(img_dist).astype(np.float64)
    sigma_nsq = 2.0
    eps = 1e-10
    num, den = 0, 0
    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        win = np.outer(signal.gaussian(N, N/5), signal.gaussian(N, N/5))
        win = win / np.sum(win)
        if scale > 1:
            img_ref = img_ref[::2, ::2]
            img_dist = img_dist[::2, ::2]
        mu1 = signal.correlate2d(img_ref, win, mode='valid')
        mu2 = signal.correlate2d(img_dist, win, mode='valid')
        sigma1_sq = signal.correlate2d(img_ref**2, win, mode='valid') - mu1**2
        sigma2_sq = signal.correlate2d(img_dist**2, win, mode='valid') - mu2**2
        sigma12 = signal.correlate2d(img_ref * img_dist, win, mode='valid') - mu1 * mu2
        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12
        sigma1_sq[sigma1_sq < eps] = eps
        g[sigma1_sq < eps] = 0
        sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
        sigma1_sq[sigma1_sq < eps] = eps
        num += np.sum(np.log2(1 + g**2 * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log2(1 + sigma1_sq / sigma_nsq))
    return num / den if den != 0 else 0

def scd_metric(img_fused, img_ir, img_vis):
    """Sum of Correlations of Differences (SCD)"""
    img_fused = to_gray(img_fused).astype(np.float64)
    img_ir = to_gray(img_ir).astype(np.float64)
    img_vis = to_gray(img_vis).astype(np.float64)
    D1 = img_fused - img_vis
    D2 = img_fused - img_ir
    r1 = np.corrcoef(D1.flatten(), img_ir.flatten())[0, 1] if np.std(D1) > 0 and np.std(img_ir) > 0 else 0
    r2 = np.corrcoef(D2.flatten(), img_vis.flatten())[0, 1] if np.std(D2) > 0 and np.std(img_vis) > 0 else 0
    return r1 + r2

def fmi_metric(img_fused, img_ir, img_vis):
    """Feature Mutual Information (FMI) using wavelet decomposition"""
    img_fused = to_gray(img_fused).astype(np.float64)
    img_ir = to_gray(img_ir).astype(np.float64)
    img_vis = to_gray(img_vis).astype(np.float64)

    def get_fmi(a, b):
        coeffs_a = pywt.dwt2(a, 'db1')
        coeffs_b = pywt.dwt2(b, 'db1')
        mi_approx = mutual_information(coeffs_a[0], coeffs_b[0])
        details_a = np.concatenate([c.flatten() for c in coeffs_a[1]])
        details_b = np.concatenate([c.flatten() for c in coeffs_b[1]])
        mi_detail = mutual_information(details_a.reshape(-1, 1), details_b.reshape(-1, 1))
        return mi_approx + mi_detail

    return get_fmi(img_fused, img_ir) + get_fmi(img_fused, img_vis)

def qabf_metric(img_fused, img_ir, img_vis):
    """Qabf metric - quality assessment based on edge information"""
    img_fused = to_gray(img_fused).astype(np.float64)
    img_ir = to_gray(img_ir).astype(np.float64)
    img_vis = to_gray(img_vis).astype(np.float64)
    img_fused = normalize_img(img_fused)
    img_ir = normalize_img(img_ir)
    img_vis = normalize_img(img_vis)

    def sobel_edge(img):
        gx = sobel(img, axis=1)
        gy = sobel(img, axis=0)
        return np.sqrt(gx**2 + gy**2), np.arctan2(gy, gx)

    g_f, a_f = sobel_edge(img_fused)
    g_ir, a_ir = sobel_edge(img_ir)
    g_vis, a_vis = sobel_edge(img_vis)

    G_ir = g_ir / (g_ir.max() + 1e-10)
    G_vis = g_vis / (g_vis.max() + 1e-10)
    g_ir_safe = np.maximum(g_ir, 1e-10)
    g_f_safe = np.maximum(g_f, 1e-10)
    g_vis_safe = np.maximum(g_vis, 1e-10)
    G_f_ir = np.where(g_ir > g_f, g_f_safe / g_ir_safe, g_ir_safe / g_f_safe)
    G_f_vis = np.where(g_vis > g_f, g_f_safe / g_vis_safe, g_vis_safe / g_f_safe)

    A_ir = 1 - np.abs(a_ir - a_f) / (math.pi / 2)
    A_vis = 1 - np.abs(a_vis - a_f) / (math.pi / 2)

    Q_ir = G_f_ir * A_ir
    Q_vis = G_f_vis * A_vis

    w_ir = G_ir
    w_vis = G_vis

    Q = np.sum(Q_ir * w_ir + Q_vis * w_vis) / np.sum(w_ir + w_vis)
    return Q

def ncie_metric(img_fused, img_ir, img_vis):
    """Nonlinear Correlation Information Entropy"""
    return (mutual_information(img_fused, img_ir) + mutual_information(img_fused, img_vis)) / 2

# ========== Comprehensive Metric Calculator ==========

def compute_all_metrics(img_fused, img_ir, img_vis):
    """Compute all fusion metrics for a single fused image"""
    # Ensure consistent sizing
    fused = to_gray(img_fused)
    ir = to_gray(img_ir)
    vis = to_gray(img_vis)

    # Resize if needed
    if fused.shape != ir.shape:
        fused = cv2.resize(fused, (ir.shape[1], ir.shape[0]))
    if fused.shape != vis.shape:
        fused = cv2.resize(fused, (vis.shape[1], vis.shape[0]))

    results = {}
    try:
        results['EN'] = entropy(fused)
    except: results['EN'] = 0
    try:
        results['SD'] = std_deviation(fused)
    except: results['SD'] = 0
    try:
        results['SF'] = spatial_frequency(fused)
        results['SF_legacy'] = spatial_frequency_legacy(fused)
    except:
        results['SF'] = 0
        results['SF_legacy'] = 0
    try:
        results['AG'] = average_gradient(fused)
    except: results['AG'] = 0
    try:
        results['MI'] = (mutual_information(fused, ir) + mutual_information(fused, vis)) / 2
    except: results['MI'] = 0
    try:
        results['SSIM'] = (ssim_metric(fused, ir) + ssim_metric(fused, vis)) / 2
    except: results['SSIM'] = 0
    try:
        results['CC'] = (correlation_coefficient(fused, ir) + correlation_coefficient(fused, vis)) / 2
    except: results['CC'] = 0
    try:
        results['VIF'] = (vif_metric(fused, ir) + vif_metric(fused, vis)) / 2
    except: results['VIF'] = 0
    try:
        results['SCD'] = scd_metric(fused, ir, vis)
    except: results['SCD'] = 0
    try:
        results['FMI'] = fmi_metric(fused, ir, vis)
    except: results['FMI'] = 0
    try:
        results['Qabf'] = qabf_metric(fused, ir, vis)
    except: results['Qabf'] = 0
    try:
        results['PSNR_IR'] = psnr(fused, ir)
        results['PSNR_VIS'] = psnr(fused, vis)
    except:
        results['PSNR_IR'] = 0
        results['PSNR_VIS'] = 0
    try:
        results['NCIE'] = ncie_metric(fused, ir, vis)
    except: results['NCIE'] = 0

    return results

def compute_metrics_batch(fused_dir, ir_dir, vis_dir, save_csv=None):
    """Compute all metrics for a batch of fusion results"""
    import os, pandas as pd
    from pathlib import Path

    fused_files = sorted([f for f in os.listdir(fused_dir) if f.endswith(('.png', '.jpg', '.bmp', '.tif'))])
    ir_files = sorted([f for f in os.listdir(ir_dir) if f.endswith(('.png', '.jpg', '.bmp', '.tif'))])
    vis_files = sorted([f for f in os.listdir(vis_dir) if f.endswith(('.png', '.jpg', '.bmp', '.tif'))])

    all_results = []
    for fused_name in fused_files:
        # Find matching IR/VIS
        base = os.path.splitext(fused_name)[0]
        ir_match = None
        vis_match = None
        for ir_f in ir_files:
            if os.path.splitext(ir_f)[0] == base or os.path.splitext(ir_f)[0].replace('.','') == base.replace('.',''):
                ir_match = ir_f
                break
        for vis_f in vis_files:
            if os.path.splitext(vis_f)[0] == base or os.path.splitext(vis_f)[0].replace('.','') == base.replace('.',''):
                vis_match = vis_f
                break

        if ir_match is None or vis_match is None:
            # Try sequential matching
            idx = fused_files.index(fused_name)
            if idx < len(ir_files):
                ir_match = ir_files[idx]
            if idx < len(vis_files):
                vis_match = vis_files[idx]

        if ir_match and vis_match:
            fused = cv2.imread(os.path.join(fused_dir, fused_name), cv2.IMREAD_GRAYSCALE)
            ir = cv2.imread(os.path.join(ir_dir, ir_match), cv2.IMREAD_GRAYSCALE)
            vis = cv2.imread(os.path.join(vis_dir, vis_match), cv2.IMREAD_GRAYSCALE)

            if fused is not None and ir is not None and vis is not None:
                metrics = compute_all_metrics(fused, ir, vis)
                metrics['Image'] = fused_name
                all_results.append(metrics)

    if all_results:
        df = pd.DataFrame(all_results)
        # Compute means
        mean_row = {col: df[col].mean() if col != 'Image' else 'MEAN' for col in df.columns}
        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

        if save_csv:
            df.to_csv(save_csv, index=False)

        return df

    return None


MAIN8 = ['EN', 'SD', 'SF', 'AG', 'MI', 'SSIM', 'CC', 'SCD']

def compute_main8(img_fused, img_ir, img_vis):
    """Compute the eight main metrics plus legacy SF for audit."""
    m = compute_all_metrics(img_fused, img_ir, img_vis)
    out = {k: m[k] for k in MAIN8}
    out['SF_legacy'] = m.get('SF_legacy', 0)
    return out
