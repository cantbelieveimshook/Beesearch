import math
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
from functions import make_augs, train, predict, resize_predictions, display_image_grid, count_surface_area, calculate_accuracy

to_train = True
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

train_count = 0.6 * math.ceil(len(images))
test_count = 0.2 * math.ceil(len(images))
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

image_transform = A.Compose(
    [A.Resize(256, 256), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
)

mask_transform = A.Compose(
    [A.Resize(256, 256), ToTensorV2()]
)

loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = params['lr'], weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 4)

test_dataset = BeeInferenceDataset(test_images_filenames, images_directory, masks_directory,
                                   image_transform=image_transform, mask_transform = mask_transform)

if to_train:
    train_dataset = make_augs(train_images_filenames, masterlist)
    val_dataset = make_augs(val_images_filenames, masterlist)

    train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

    train(epochs = 15)

predictions = predict(model, params, test_dataset, batch_size=16)
predicted_masks = resize_predictions(predictions, test_dataset)
# display_image_grid(test_images_filenames[:10], images_directory, masks_directory, predicted_masks=predicted_masks[:10])

# If you want the surface area and accuracy metrics
count_surface_area(predicted_masks, test_dataset, path=root + 'bee_surface_areas.csv')
calculate_accuracy(predicted_masks, masks_directory = masks_directory, filenames = test_images_filenames,
                       csv_path = root + 'bee_prediction_accuracies.csv')
