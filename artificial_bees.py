import os
import torch
from functions import make_fake_bees

root = os.getcwd()
background_bees_directory = os.path.join(root, 'removed_background_bees')
bee_masks_directory = os.path.join(root, 'predicted_bee_masks')
artificial_bees_directory = os.path.join(root, 'artificial_bees')

artificial_bees_path = root + 'full_segmentation_pipeline/' + 'artificial_bees/'
masks = os.listdir(bee_masks_directory)
images = os.listdir(background_bees_directory)

device = "cuda" if torch.cuda.is_available() else torch.device('cpu')

make_fake_bees(background_bees_directory, bee_masks_directory, masks, artificial_bees_directory)