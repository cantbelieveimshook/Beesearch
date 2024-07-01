import os
import torch
from functions import background_removal_updated


def remove_background_main():
    root = os.getcwd()
    background_bees_directory = os.path.join(root, 'removed_background_bees')
    bee_images_directory = os.path.join(root, 'bee_original')
    bee_images = os.listdir(bee_images_directory)
    background_removed = os.listdir(background_bees_directory)

    # This ensures the background removal is only performed on images not already in the removed background folder
    to_remove = [i for i in bee_images if i not in background_removed]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    background_removal_updated(bee_images_directory, background_bees_directory, to_remove = to_remove)
