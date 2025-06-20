from trainingImageLoader import ImageLoader
import cv2
import numpy as np
import pandas as pd
from extract_cepstral_features import extract_cepstrum_features

cepstral_feature_list = []
benign_feature_list = []
malignant_feature_list = []


training_data_list = []
training_folder_ = 'Data\ISIC-images_2019_training'
mask_folder_ = 'Data\isic2019_borders'
trainig_metadata_ = 'Data\challenge-2019-training_metadata_2025-06-01.csv'
training_data_loader = ImageLoader(training_folder_, mask_folder_)
training_data_loader.load_metadata(trainig_metadata_)


for i in range(training_data_loader.num_images):
    image_, malignant_, mask_, diagnosis_ = training_data_loader.iterate(cv2.IMREAD_GRAYSCALE, ['diagnosis_2', 'diagnosis_3'])
    if len(mask_) > 0:
        haralick_features = []
        width = image_.shape[1]
        height = image_.shape[0]
        dim = (width, height)
        mask_ = ((cv2.resize(mask_, dim, interpolation = cv2.INTER_AREA))//255)

        masked_image = np.multiply(mask_,image_)

# Optional: apply a mask if needed
# masked_image = np.multiply(image, mask)
# For now, let's use the image directly
        masked_image_fr = masked_image.astype(np.float32) / 255.0

# Compute FFT
        f_transform = np.fft.fft2(masked_image_fr)

# Take log of magnitude spectrum (no fftshift here)
        log_magnitude = np.log(np.abs(f_transform) + 1e-8)

# Compute inverse FFT to get the cepstrum
        cepstrum = np.fft.ifft2(log_magnitude)
        cepstrum = np.abs(cepstrum)

# Shift to center for visualization only (after inverse FFT)
        cepstrum_display = np.fft.fftshift(cepstrum)

        features = extract_cepstrum_features(cepstrum_display)
        features['diagnosis_2'] = diagnosis_[0]
        features['diagnosis_3'] = diagnosis_[1]

        if malignant_ == 1:
            malignant_feature_list.append(features)
        else: benign_feature_list.append(features)
        


        cepstral_feature_list.append(features)
        print(i)
cep_malignant_data = pd.DataFrame(malignant_feature_list)
cep_benign_data = pd.DataFrame(benign_feature_list)
cepstral_data = pd.DataFrame(cepstral_feature_list)