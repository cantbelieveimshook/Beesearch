'''
Name: Kathryn Chen
Date: June 23, 2023
'''

import os
from functions import make_fake_bees

'''
Creates "artificial bees" and "artificial hair" by multiplying the predicted bee and hair masks with the original images,
which is then saved as an image that shows only what is segmented as bee or bee hair by the machine learning models.
For background_bees_directory, change removed_background_bees to bee_original if not using the images with manually removed backgrounds.
Change artificial_bees_directory or artificial_hair_directory if you want to make multiple folders of artificial bees/hair.
'''

def artificial_bees_main(background_removed = False):
    root = os.getcwd()
    if background_removed:
        bees_directory = os.path.join(root, 'removed_background_bees')
    else:
        bees_directory = os.path.join(root, 'bee_original')
    bee_masks_directory = os.path.join(root, 'predicted_bee_masks')
    artificial_bees_directory = os.path.join(root, 'artificial_bees')
    masks = os.listdir(bee_masks_directory)

    make_fake_bees(bees_directory, masks, bee_masks_directory, save_path = artificial_bees_directory)

def artificial_hair_main():
    root = os.getcwd()
    bees_directory = os.path.join(root, 'artificial_bees')
    bee_masks_directory = os.path.join(root, 'predicted_hair_masks')
    artificial_hair_directory = os.path.join(root, 'segmented_hair_final')
    masks = os.listdir(bee_masks_directory)

    make_fake_bees(bees_directory, masks, bee_masks_directory, save_path = artificial_hair_directory)
