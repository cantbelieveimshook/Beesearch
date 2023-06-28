'''
Name: Kathryn Chen
Date: June 23, 2023
'''

import random
import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from paths import *
from classes import BeeInferenceDataset
from functions import make_augs, train, predict, resize_predictions, display_image_grid, display_bees, count_surface_area, calculate_accuracy

# to_train determines if the model will be trained on a set of images and masks before the model outputs predicted segmentations.
# Only set to True if you have a set of images and corresponding masks for the model to train on.
to_train = False
if len(sys.argv) > 1:
    if sys.argv[1] == True or sys.argv[1] == False:
        to_train = sys.argv[1]

root = os.getcwd()
images_directory = os.path.join(root, 'removed_background_bees')
masks_directory = root + 'whole_bee_mask/'
images = os.listdir(images_directory)
masks = os.listdir(masks_directory)

# shuffle images to increase likelihood of balanced train, val, and test datasets
random.shuffle(images)

train_count = int(0.6 * len(images))
test_count = int(0.2 * len(images))
val_count = len(images) - train_count - test_count
train_images_filenames = images[:train_count]
val_images_filenames = images[train_count:train_count + val_count]
test_images_filenames = images[train_count + val_count:]

device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.load(root + 'models/Model_300').to(device)

params = {
    "device": device,
    "lr": 0.001,
    "batch_size": 2,
    "num_workers": 4,
    "epochs": 10,
}

# The transformations done on the image dataset before training and prediction.
image_transform = A.Compose(
    [A.Resize(256, 256), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
)

# The transformations done on the mask dataset, if it exists, before training and prediction.
mask_transform = A.Compose(
    [A.Resize(256, 256), ToTensorV2()]
)

loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = params['lr'], weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 4)

# Creates a Pytorch dataset of the test images.
test_dataset = BeeInferenceDataset(test_images_filenames, images_directory, masks_directory,
                                   image_transform=image_transform, mask_transform = mask_transform)

if to_train:
    # Creates Pytorch datasets of the training and validation images.
    train_dataset = make_augs(train_images_filenames, masterlist, image_transform, mask_transform)
    val_dataset = make_augs(val_images_filenames, masterlist, image_transform, mask_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

    train(model = model, batch_size = params['batch_size'], loss_fn = loss_fn,
          train_dataloader = train_dataloader, val_dataloader = val_dataloader,
          scheduler = scheduler, optimizer = optimizer, device = device, train_images = images, epochs = 15)

# Outputs the model's predicted segmentations of input bee images, with the aim of removing the eyes, wings, and antennae.
predictions = predict(model, params, test_dataset, batch_size=16)

# Resizes the 256 x 256 segmentations to their original size.
predicted_masks = resize_predictions(predictions, test_dataset)

# Slightly different grids are displayed based on whether there exist ground truth masks.
if len(masks) == 0:
    display_bees(test_images_filenames[:10], images_directory, predicted_masks[:10])
else:
    display_image_grid(test_images_filenames[:10], images_directory, masks_directory, predicted_masks=predicted_masks[:10])

# If you want the surface area and accuracy metrics
count_surface_area(predicted_masks, test_dataset, path=root + 'bee_surface_areas.csv')
calculate_accuracy(predicted_masks, masks_directory = masks_directory, filenames = test_images_filenames,
                       csv_path = root + 'bee_prediction_accuracies.csv')
