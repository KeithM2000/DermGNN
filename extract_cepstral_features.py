import numpy as np
import mahotas
from scipy.stats import skew, kurtosis
from skimage.exposure import rescale_intensity
from skimage.feature import graycomatrix

def extract_cepstrum_features(cepstrum, glcm_distance=1):
    features = {}

    # Normalize and convert to uint8 for GLCM-based methods
    norm_cep = rescale_intensity(cepstrum, out_range=(0, 255)).astype(np.uint8)

    # Global statistics
    features['mean'] = np.mean(cepstrum)
    features['std'] = np.std(cepstrum)
    features['skew'] = skew(cepstrum.ravel())
    features['kurtosis'] = kurtosis(cepstrum.ravel())
    p = cepstrum / (np.sum(cepstrum) + 1e-8)
    features['cepstral_entropy'] = -np.sum(p * np.log2(p + 1e-8))

    # Radial statistics
    h, w = cepstrum.shape
    y, x = np.indices((h, w))
    center = np.array([h // 2, w // 2])
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2).astype(int)
    r_bin = np.bincount(r.ravel(), weights=cepstrum.ravel())
    r_count = np.bincount(r.ravel())
    radial_profile = r_bin / (r_count + 1e-8)

    features['radial_peak_val'] = np.max(radial_profile)
    features['radial_peak_pos'] = np.argmax(radial_profile)
    features['radial_AUC'] = np.sum(radial_profile)

    print(np.sum(norm_cep))

    # 13 Haralick features from mahotas
    haralick_feats_ = mahotas.features.haralick(norm_cep, distance=glcm_distance, ignore_zeros=True, return_mean=False)
    haralick_feats_mean = haralick_feats_.mean(axis=0)
    haralick_names = [
        'Har_Cep_ASM', 'Har_Cep_contrast', 'Har_Cep_correlation', 'Har_Cep_variance', 'Har_Cep_inverse_diff_moment',
        'Har_Cep_sum_average', 'Har_Cep_sum_variance', 'Har_Cep_sum_entropy', 'Har_Cep_glcm_entropy',
        'Har_Cep_diff_variance', 'Har_Cep_diff_entropy', 'Har_Cep_IMC1', 'Har_Cep_IMC2'
    ]
    for name, value in zip(haralick_names, haralick_feats_mean):
        features[name] = value

    glcm = graycomatrix(
        norm_cep,
        distances=[glcm_distance],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )
    # Calculate trace by summing the diagonal across all angles and averaging
    diagonals_mean = [np.trace(glcm[:, :, 0, i]) for i in range(glcm.shape[-1])]
    features['Cep_glcm_trace'] = np.mean(diagonals_mean)

    haralick_feats_max = haralick_feats_.max(axis=0)
    haralick_feats_dir = haralick_feats_max/haralick_feats_mean
    haralick_names_dir = [
        'Har_Cep_ASM_Dir', 'Har_Cep_contrast_Dir', 'Har_Cep_correlation_Dir', 'Har_Cep_variance_Dir', 'Har_Cep_inverse_diff_moment_Dir',
        'Har_Cep_sum_average_Dir', 'Har_Cep_sum_variance_Dir', 'Har_Cep_sum_entropy_Dir', 'Har_Cep_glcm_entropy_Dir',
        'Har_Cep_diff_variance_Dir', 'Har_Cep_diff_entropy_Dir', 'Har_Cep_IMC1_Dir', 'Har_Cep_IMC2_Dir'
    ]
    for name, value in zip(haralick_names_dir, haralick_feats_dir):
        features[name] = value

    # Calculate trace by summing the diagonal across all angles and averaging
    diagonals_max = [np.trace(glcm[:, :, 0, i]) for i in range(glcm.shape[-1])]
    features['Cep_glcm_trace_Dir'] = np.max(diagonals_max)/features['Cep_glcm_trace']

    return features

