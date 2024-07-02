'''
Name: Kathryn Chen
Date: June 23, 2023
'''

from functions import make_directories

'''
Run this file first to make the necessary folders,
then fill the original_bee_masks and bee_original
directories with the necessary images.
Comment out the directories you do not want to make.
'''

def make_folders_main():
    directory_list = [
        "analysis_results",
        "artificial_bees",
        "augmented_masks",
        "bee_crops",
        "bee_original",
        "entropy_images",
        "entropy_analysis",
        "image_regression",
        "hair_original",
        "original_bee_masks",
        "original_hair_masks",
        "predicted_bee_masks",
        "predicted_hair_masks",
        "removed_background_bees",
        "segmented_hair_final"
    ]

    make_directories(directory_list)
