'''
Name: Kathryn Chen
Date: June 23, 2023
'''

import os
import torch
from functions import make_fake_bees

def artificial_bees_main():
    root = os.getcwd()
    background_bees_directory = os.path.join(root, 'removed_background_bees')
    bee_masks_directory = os.path.join(root, 'predicted_bee_masks')
    artificial_bees_directory = os.path.join(root, 'artificial_bees')
    masks = os.listdir(bee_masks_directory)

    make_fake_bees(background_bees_directory, bee_masks_directory, masks, artificial_bees_directory)