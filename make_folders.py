from paths import *

# run this file first to make the necessary folders, then fill the original_bee_masks_directory and bee_images_directory with the necessary images
folders = [original_bee_masks_directory,
           background_bees_directory,
           bee_images_directory,
           bee_masks_directory,
           artificial_bees_directory,
           original_hair_masks_directory,
           hair_images_directory,
           hair_masks_directory]

for i in folders:
    if not os.path.exists(i):
        os.mkdir(i)