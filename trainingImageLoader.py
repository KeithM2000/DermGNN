import pandas as pd
import numpy as np
import os
import cv2

class ImageLoader:
    def __init__(self, image_repo, mask_repo = None):
        self.num_images = 0
        self.current_index = 0
        self.image_repository = image_repo
        self.mask_repository = mask_repo
        self.diagnosis_map = {"benign": 0, "malignant": 1}
    
    def load_metadata(self, file_path, column_selection = []):
        metadataframe_init = pd.read_csv(file_path)
        metadataframe = pd.DataFrame()
        if column_selection == []:
            column_selection = metadataframe_init.columns.to_list()
        for column_name in column_selection:
            metadataframe[column_name] = metadataframe_init[column_name]
        self.metadataframe = metadataframe
        self.num_images = len(self.metadataframe)

    def iterate(self, return_type):
        next_file_ = self.metadataframe.loc[self.current_index, "isic_id"]
        next_file_name = next_file_ + ".jpg"
        file_path = os.path.join(self.image_repository, next_file_name)
        image = cv2.imread(file_path,return_type)
        type_ = self.metadataframe.loc[self.current_index, "benign_malignant"]

        if type_ in ["benign", "malignant"]:
            label = self.diagnosis_map[type_]
        else: 
            label = 0
            type_ = "benign"
        self.current_index += 1
        if self.current_index == self.num_images:
            self.current_index = 0
        if self.mask_repository is not None:
            mask_file_ = [entry for entry in os.listdir(self.mask_repository) if entry.startswith(next_file_) and os.path.isfile(os.path.join(self.mask_repository, entry))]
            if len(mask_file_) > 0:
                mask_file_ = os.path.join(self.mask_repository, mask_file_[0])
                mask_ = cv2.imread(mask_file_,return_type)    
            else:
                mask_ = []
            return image, label, mask_
        return image, label