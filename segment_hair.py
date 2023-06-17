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
from classes import BeeInferenceDataset, CroppedDataset
from functions import train, predict, resize_predictions, display_image_grid, display_bees, count_surface_area, calculate_accuracy, predict_crops, restitch_predictions

to_crop = True
to_train = True
if len(sys.argv) > 1:
    if sys.argv[1] == True or sys.argv[1] == False:
        to_train = sys.argv[1]
if len(sys.argv) > 2:
    if sys.argv[2] == True or sys.argv[2] == False:
        to_crop = sys.argv[2]

root = os.getcwd()
images_directory = os.path.join(root, 'artificial_bees/')
masks_directory = root + 'original_hair_masks/'
images = os.listdir(images_directory)
masks = os.listdir(masks_directory)

random.shuffle(images)

train_count = 0.6 * math.ceil(len(images))
test_count = 0.2 * math.ceil(len(images))
val_count = len(images) - train_count - test_count
train_images_filenames = images[:train_count]
val_images_filenames = images[train_count:train_count + val_count]
test_images_filenames = images[train_count + val_count:]

device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.load('/content/drive/MyDrive/Beesearch/Model Scripts/Hair Best Pipeline/model_98.pth')

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

test_dataset = BeeInferenceDataset(test_images_filenames, images_directory, masks_directory, image_transform=image_transform, mask_transform = mask_transform)

if to_train:
    train_dataset = BeeInferenceDataset(train_images_filenames, images_directory, masks_directory,
                                        image_transform=image_transform, mask_transform=mask_transform)
    val_dataset = BeeInferenceDataset(val_images_filenames, images_directory, masks_directory,
                                      image_transform=image_transform, mask_transform=mask_transform)

    train(epochs=15)

if to_crop:
    bee_dataset = CroppedDataset(artificial_bees[:16], artificial_bees_directory, image_transform=image_transform)
    test_loader = DataLoader(bee_dataset, batch_size=16, shuffle=False, collate_fn=bee_dataset.collate_fn,
                             num_workers=params["num_workers"], pin_memory=True)
    restitched_predictions = predict_crops(model, params, bee_dataset, test_loader)
    restitched_predicted_masks = restitch_predictions(restitched_predictions, bee_dataset)
    # display_bees(artificial_bees[:10], artificial_bees_directory, restitched_predicted_masks[:10])

else:
    test_dataset = BeeInferenceDataset(test_images_filenames, images_directory, masks_directory,
                                       image_transform=image_transform, mask_transform=mask_transform)
    predictions = predict(model, params, test_dataset, batch_size=16)
    predicted_masks = resize_predictions(predictions)
    # display_image_grid(test_images_filenames[:10], images_directory, masks_directory, predicted_masks=predicted_masks[:10])
