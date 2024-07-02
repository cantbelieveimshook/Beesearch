'''
Name: Kathryn Chen
Date: June 23, 2023
'''

import os

root = os.getcwd()

original_bee_masks_directory = os.path.join(root, 'original_bee_masks')
background_bees_directory = os.path.join(root, 'removed_background_bees')
bee_images_directory = os.path.join(root, 'bee_original')
bee_masks_directory = os.path.join(root, 'predicted_bee_masks')
artificial_bees_directory = os.path.join(root, 'artificial_bees')
original_hair_masks_directory = os.path.join(root, 'original_hair_masks')
hair_images_directory = os.path.join(root, 'hair_original')
hair_masks_directory = os.path.join(root, 'predicted_hair_masks')

bee_images = os.listdir(bee_images_directory)
original_bee_masks = os.listdir(original_bee_masks_directory)
bee_masks = os.listdir(bee_masks_directory)
background_bees = os.listdir(background_bees_directory)
artificial_bees = os.listdir(artificial_bees_directory)
hair_images = os.listdir(hair_images_directory)
hair_masks = os.listdir(hair_masks_directory)

to_remove = [i for i in bee_images if i not in background_bees]

aug_im_dir = root + 'augmented_images/'
aug_mask_dir = root + 'augmented_masks/'
blur_im_dir = aug_im_dir + 'blurred/'
blur_mask_dir = aug_mask_dir + 'blurred/'
bright_im_dir = aug_im_dir + 'brightness/'
bright_mask_dir = aug_mask_dir + 'brightness/'
flipped_im_dir = aug_im_dir + 'flipped/'
flipped_mask_dir = aug_mask_dir + 'flipped/'
horizontal_im_dir = aug_im_dir + 'horizontal_shifts/'
horizontal_mask_dir = aug_mask_dir + 'horizontal_shifts/'
inverted_im_dir = aug_im_dir + 'inverted/'
inverted_mask_dir = aug_mask_dir + 'inverted/'
noisy_im_dir = aug_im_dir + 'noisy/'
noisy_mask_dir = aug_mask_dir + 'noisy/'
rotated_im_dir = aug_im_dir + 'rotated/'
rotated_mask_dir = aug_mask_dir + 'rotated/'
vertical_im_dir = aug_im_dir + 'vertical_shifts/'
vertical_mask_dir = aug_mask_dir + 'vertical_shifts/'
zoom_im_dir = aug_im_dir + 'zoom/'
zoom_mask_dir = aug_mask_dir + 'zoom/'

'''
Make sure the images and masks directories for these augmentations actually exist.
Otherwise, keep everything in the master list as is.
If you want to use certain augmentations but not others, comment/uncomment the lines 
based on which augmentations you want to add to your dataset.
I would always recommend keeping the original images and masks directories.
'''

masterlist = [(background_bees_directory, root + 'whole_bee_mask/'),
 # (blur_im_dir, blur_mask_dir),
  # (bright_im_dir, bright_mask_dir),
   # (flipped_im_dir, flipped_mask_dir),
 # (horizontal_im_dir, horizontal_mask_dir),
  # (inverted_im_dir, inverted_mask_dir),
   # (noisy_im_dir, noisy_mask_dir),
   # (rotated_im_dir, rotated_mask_dir),
   # (vertical_im_dir, vertical_mask_dir),
   # (zoom_im_dir, zoom_mask_dir)
 ]
