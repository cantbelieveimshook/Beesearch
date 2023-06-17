import math
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from classes import *
from paths import *
from carvekit.web.schemas.config import MLConfig
from carvekit.web.utils.init_utils import init_interface

def make_fake_bees(images_path, masks_path, masks, save_path):
  for i in masks:
    im = plt.imread(images_path + i)
    mask = plt.imread(masks_path + i)
    mask = preprocess_mask(mask)
    if np.shape(im) != np.shape(mask):
      raise ShapeException("Images and masks are mismatched shapes: file " + i)
    fake_bee = im * mask
    fake_bee = fake_bee.astype('uint8')
    bee_reborn = cv2.cvtColor(fake_bee, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_path + i, bee_reborn)

def convert_to_black(image, pixel=130):
  for i in range(len(image)):
    for j in range(len(image[i])):
      if (image[i][j] == pixel).all():
        image[i][j] = [0, 0, 0]
      j += 1
    i += 1
  return image

def make_augs(filenames, augs):
  masterlist = []
  for x, y in augs:
    dset = BeeInferenceDataset(filenames, x, y, image_transform=image_transform, mask_transform = mask_transform)
    masterlist.append(dset)
  datasets = torch.utils.data.ConcatDataset(masterlist)
  return datasets

def train_one_epoch(model, loss_fn):
    running_loss = 0
    last_loss = 0
    batches = math.ceil(len(train_images_filenames) / params['batch_size'])

    for i, data in enumerate(train_dataloader):
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)
      optimizer.zero_grad()

      outputs = model(inputs)
      outputs = torch.sigmoid(outputs)

      loss = loss_fn(outputs, labels)
      loss.backward()

      optimizer.step()
      running_loss += loss.item()

    last_loss = running_loss / batches

    return last_loss

def train(model, epochs=20):
  epoch_number = 0
  for epoch in range(epochs):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch()

    # We don't need gradients on to do reporting
    model.eval()

    running_vloss = 0.0

    with torch.no_grad():
      # batch = 0
      for i, vdata in enumerate(val_dataloader):
        # print(i)
        vinputs, vlabels = vdata
        vinputs, vlabels = vinputs.to(device), vlabels.to(device)
        voutputs = model(vinputs)
        voutputs = torch.sigmoid(voutputs)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss
        # batch += 1

    scheduler.step(running_vloss)

    avg_vloss = running_vloss / (i)
    print('LOSS train {} val {}'.format(avg_loss, avg_vloss))

    epoch_number += 1

def predict(model, params, test_dataset, batch_size):
  test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=params["num_workers"], pin_memory=True,
  )
  model.eval()
  predictions = []
  with torch.no_grad():
    i = 0
    for images, _ in test_loader:
      images = images.to(params["device"], non_blocking=True)
      output = model(images)
      probabilities = torch.sigmoid(output.squeeze(1))
      predicted_masks = (probabilities >= 0.5).float() * 1
      predicted_masks = predicted_masks.cpu().numpy()
      for predicted_mask in zip(
              predicted_masks
      ):
        original_heights, original_widths = test_dataset.getsize(i)
        predictions.append((predicted_mask, original_heights, original_widths))
        i += 1
  return predictions

def predict_crops(model, params, test_dataset, test_loader):
  model.eval()
  predictions = []
  with torch.no_grad():
    for crops in test_loader:  # change to for crops, _ in test_loader if the test_loader has masks
      for i in crops:  # if batch_size is 1, this is the one crop list in the batch
        predicted_crops = []
        for j in i:  # one row in the crop list
          row = []
          for k in j:  # one image in a row in the crop list
            images = k.unsqueeze(0)
            images = images.to(params["device"], non_blocking=True)
            output = model(images)
            probabilities = torch.sigmoid(output.squeeze(1))
            predicted_mask = (probabilities >= 0.5).float() * 1
            predicted_mask = predicted_mask.cpu().numpy()
            row.append(predicted_mask)
          predicted_crops.append(row)
        predictions.append(predicted_crops)

  return predictions

def resize_predictions(predictions, test_dataset, save=True, save_path=root + 'predicted_bee_masks/'):
  i = 0
  predicted_masks = []
  for predicted_256x256_mask, original_height, original_width in predictions:
    mask = np.transpose(predicted_256x256_mask, (1, 2, 0))
    full_sized_mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    predicted_masks.append(full_sized_mask)
    if save:
      name = test_dataset.getname(i)
      save_name = save_path + name
      cv2.imwrite(save_name, full_sized_mask)
    i += 1

  return predicted_masks


def restitch_predictions(predictions, test_dataset, save=True,
                         save_path=root + 'full_segmentation_pipeline/' + 'predicted_hair_masks/'):
  restitched_images = []
  idx = 0
  for image in predictions:
    height, width = test_dataset.getsize(idx)
    restitch = Image.new('L', (width, height))

    starth = 0
    for i in range(len(image)):
      startw = 0
      croph = height - 300 * (len(image) - 1) if i == len(image) - 1 else 300
      for j in range(len(image[i])):
        cropw = width - 300 * (len(image[i]) - 1) if j == len(image[i]) - 1 else 300
        mask = np.transpose(image[i][j], (1, 2, 0))
        mask = cv2.resize(mask, (cropw, croph), interpolation=cv2.INTER_NEAREST)
        mask_image = Image.fromarray(mask)
        restitch.paste(mask_image, (startw, starth, startw + cropw, starth + croph))
        startw += cropw
      starth += croph

    restitched_images.append(restitch)

    if save:
      name = test_dataset.getname(idx)
      save_name = save_path + name
      cv2.imwrite(save_name, restitch)

    idx += 1

  return restitched_images

def display_image_grid(images_filenames, images_directory, masks_directory,
                       predicted_masks=None, save = False, filetype='.png'):
  cols = 3 if predicted_masks else 2
  rows = len(images_filenames)
  figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 24))
  for i, image_filename in enumerate(images_filenames):
    image = cv2.imread(os.path.join(images_directory, image_filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(os.path.join(masks_directory, image_filename))
    mask = preprocess_mask(mask)
    mask = mask[:, :, 0]

    ax[i, 0].imshow(image)
    ax[i, 1].imshow(mask, interpolation="nearest")

    ax[i, 0].set_title("Image")
    ax[i, 1].set_title("Ground truth mask")

    ax[i, 0].set_axis_off()
    ax[i, 1].set_axis_off()

    if predicted_masks:
      predicted_mask = predicted_masks[i]
      ax[i, 2].imshow(predicted_mask, interpolation="nearest")
      ax[i, 2].set_title("Predicted mask")
      ax[i, 2].set_axis_off()
  plt.tight_layout()
  if save:
    root = os.getcwd()
    filename = os.path.join(root, 'whole_bee_predictions')
    plt.savefig(filename + filetype, bbox_inches='tight')
    plt.close(figure)
  else:
    plt.show()

def display_bees(images_filenames, images_directory, predicted_masks):
  cols = 2
  rows = len(images_filenames)
  figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 24))
  for i, image_filename in enumerate(images_filenames):
    image = cv2.imread(os.path.join(images_directory, image_filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predicted_mask = predicted_masks[i]
    ax[i, 0].imshow(image)
    ax[i, 1].imshow(predicted_mask, interpolation="nearest")

    ax[i, 0].set_title("Image")
    ax[i, 1].set_title("Predicted mask")

    ax[i, 0].set_axis_off()
    ax[i, 1].set_axis_off()

  plt.tight_layout()
  # plt.savefig(filename + filetype, bbox_inches='tight')
  # plt.close(figure)
  plt.show()

def count_surface_area(masks, dataset, path):
  df = pd.DataFrame(columns=['name', 'surface area', 'percentage of pixels'])
  for i in range(len(masks)):
    name = dataset.images_filenames[i]
    unique = np.unique(masks[i], return_counts=True)
    if unique[1][0] == np.size(masks[i]):
      df.loc[len(df)] = [name, 0, 0]
    elif len(unique[1]) == 2:
      bee_pixels = unique[1][1]
      total_pixels = np.shape(masks[i])[0] * np.shape(masks[i])[1]
      percentage = bee_pixels / total_pixels
      percentage = f'{str(percentage * 100)}%'
      df.loc[len(df)] = [name, bee_pixels, percentage]
  df.to_csv(path, index=False)

def jaccard(y_true, y_pred):
  intersection = (y_true * y_pred).sum()
  union = y_true.sum() + y_pred.sum() - intersection
  return (intersection + 1e-15) / (union + 1e-15)


def dice(y_true, y_pred):
  return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)


def accuracy(y_true, y_pred):
  return (((y_pred + y_true) == 2).sum() + ((y_pred + y_true) == 0).sum()) / y_true.size

def calculate_accuracy(predicted_masks, masks_directory, filenames, csv_path):
  results = pd.DataFrame(columns=['Name', 'Accuracy', 'F1', 'IoU'])

  for i, mask in enumerate(filenames):
    mask_orig = cv2.imread(os.path.join(masks_directory, mask))
    mask_orig = preprocess_mask(mask_orig)
    mask_orig = mask_orig[:, :, 0]
    mask_pred = predicted_masks[i]

    acc = accuracy(mask_orig, mask_pred)
    iou = jaccard(mask_orig, mask_pred)
    f1 = dice(mask_orig, mask_pred)

    sample = {'Name': mask, 'Accuracy': acc, 'F1': f1, 'IoU': iou}

    results = pd.concat([results, pd.DataFrame(data=sample, index=[mask])])
  results.to_csv(csv_path, index = False)

def background_removal_updated(input_dir, output_dir, input_extension=".png", to_remove=to_remove):
  SEGMENTATION_NETWORK = "tracer_b7"
  PREPROCESSING_METHOD = "stub"
  POSTPROCESSING_METHOD = "fba"
  SEGMENTATION_MASK_SIZE = 512
  TRIMAP_DILATION = 8
  TRIMAP_EROSION = 8
  DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
  BACKGROUND_COLOR = (0, 0, 0)  # Black background

  config = MLConfig(
    segmentation_network=SEGMENTATION_NETWORK,
    preprocessing_method=PREPROCESSING_METHOD,
    postprocessing_method=POSTPROCESSING_METHOD,
    seg_mask_size=SEGMENTATION_MASK_SIZE,
    trimap_dilation=TRIMAP_DILATION,
    trimap_erosion=TRIMAP_EROSION,
    device=DEVICE,
    background_color=BACKGROUND_COLOR
  )

  interface = init_interface(config)

  os.makedirs(output_dir, exist_ok=True)
  for file_name in to_remove:
    if file_name.endswith(input_extension):
      image_path = os.path.join(input_dir, file_name)
      image = Image.open(image_path)

      if image.mode != "RGB":
        image = image.convert("RGB")

      print(f"Beginning removal on {file_name}...")

      result = interface([image])
      result = convert_to_black(result)
      result_file_name = file_name.replace(input_extension, '.png')
      result_path = os.path.join(output_dir, result_file_name)
      result[0].save(result_path)
      new_path = result_path.replace('.png', '.jpg')
      os.rename(result_path, new_path)

      print(f"Saved {file_name} to {new_path}")