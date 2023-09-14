'''
Name: Kathryn Chen
Name: Luning Ding
Date: June 23, 2023
'''

from classes import *
from paths import *
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from carvekit.web.schemas.config import MLConfig
from carvekit.web.utils.init_utils import init_interface
import copy
import time
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import math
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
from skimage.io import imread, imshow, imsave
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from csv import writer
from collections import defaultdict
import glob
import tqdm.notebook as tqdm
from torch.autograd import Variable


'''
This file contains most of the functions that are used in this repository.
The functions that make the augmented images are located in make_augment_functions.py
'''

root = os.getcwd()

# Checks if each directory from the list of directories exists, and if not, creates the directory.
def make_directories():
  directory_list = [
    "artificial_bees",
    "augmented_masks",
    "bee_crops",
    "bee_original",
    "entropy_images_bee",
    "entropy_images_hair",
    "hair_original",
    "original_bee_masks",
    "original_hair_masks",
    "predicted_bee_masks",
    "predicted_hair_masks",
    "removed_background_bees",
    "segmented_hair_final"
    ]

  for i in directory_list:
    path = os.path.join(root, i)
    if not os.path.isdir(path):
      os.mkdir(path)

# Uses the predicted bee masks to artificially remove the eyes, wings, and antennae
# from the original bee images, then saves these modified images in a new folder.
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


# Converts the transparent pixels in the bees with artificially removed backgrounds
# into black pixels, which are brings the images closer to what the model was pre-trained on.
def convert_to_black(image, pixel=130, degree=0.05):
  image = np.array(image)[:, :, :3]
  lower = pixel - pixel * degree
  upper = pixel + pixel * degree
  for i in range(len(image)):
    for j in range(len(image[i])):
      if all(lower <= k <= upper for k in image[i][j]):
        image[i][j] = [0, 0, 0]
      j += 1
    i += 1

  image = Image.fromarray(np.uint8(image))
  return image

# Creates a concatenated Pytorch Dataset of augmented values.
def make_augs(filenames, augs, image_transform, mask_transform):
  masterlist = []
  for x, y in augs:
    dset = BeeInferenceDataset(filenames, x, y, image_transform = image_transform, mask_transform = mask_transform)
    masterlist.append(dset)
  datasets = torch.utils.data.ConcatDataset(masterlist)
  return datasets


# Trains one epoch of a supervised segmentation model using the typical Pytorch training loop.
# Original code found here: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
def train_one_epoch(model, loss_fn, train_dataloader, optimizer, device, batch_size, train_images):
    running_loss = 0
    batches = math.ceil(len(train_images) / batch_size)

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


# Trains a supervised segmentation model.
# Original code found here: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
def train(model, batch_size, loss_fn, train_dataloader, val_dataloader, scheduler, optimizer, device, train_images, epochs=20):
  epoch_number = 0
  for epoch in range(epochs):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(model, loss_fn, train_dataloader, optimizer, device, batch_size, train_images)

    model.eval()

    running_vloss = 0.0

    with torch.no_grad():
      for i, vdata in enumerate(val_dataloader):
        vinputs, vlabels = vdata
        vinputs, vlabels = vinputs.to(device), vlabels.to(device)
        voutputs = model(vinputs)
        voutputs = torch.sigmoid(voutputs)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss

    scheduler.step(running_vloss)

    avg_vloss = running_vloss / (i)
    print('LOSS train {} val {}'.format(avg_loss, avg_vloss))

    epoch_number += 1


'''
Predicts the images of a test dataset using a model.
Zips the original image height and width along with the predicted crop, 
then adds that to a list of predictions that is returned at the end.
'''
def predict(model, params, dataset, dataloader = None):
  if dataloader == None:
    dataloader = DataLoader(dataset,
                            batch_size=params["batch_size"],
                            shuffle=False,
                            num_workers=params["num_workers"],
                            pin_memory=True)

  model.eval()
  predictions = []
  with torch.no_grad():
    i = 0
    for images, _ in dataloader:
      images = images.to(params["device"], non_blocking=True)
      output = model(images)
      probabilities = torch.sigmoid(output.squeeze(1))
      predicted_masks = (probabilities >= 0.5).float() * 1
      predicted_masks = predicted_masks.cpu().numpy()
      for predicted_mask in zip(
              predicted_masks
      ):
        original_heights, original_widths = dataset.getsize(i)
        predictions.append((predicted_mask, original_heights, original_widths))
        i += 1
  return predictions


'''
Only used if the hair segmentation process is done by cropping and restitching the images.
Predicts the images of a test dataset by cropping those images and segmenting the crops,
then re-appending those crops into a nested list in a way that allows the predicted crops
to be reconstructed into the original image.
'''
def predict_crops(model, params, dataloader):
  model.eval()
  predictions = []
  with torch.no_grad():
    for crops in dataloader:  # change to for crops, _ in test_loader if the test_loader has masks
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


# Used to resize the predicted images back to their original sizes.
# If save = True, it also saves the predicted images to the designated save path.
def resize_predictions(predictions, dataset, save=True, save_path=root + 'predicted_bee_masks/'):
  i = 0
  predicted_masks = []
  for predicted_256x256_mask, original_height, original_width in predictions:
    mask = np.transpose(predicted_256x256_mask, (1, 2, 0))
    full_sized_mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    predicted_masks.append(full_sized_mask)
    if save:
      name = dataset.getname(i)
      save_name = save_path + name
      cv2.imwrite(save_name, full_sized_mask)
    i += 1

  return predicted_masks

'''
Only used if the hair segmentation process is done by cropping and restitching the images.
Takes a nested list of predicted crops and restitches them back into a full mask of the original image.
If save = True, it also saves the restitched predicted images to the designated save path.
'''
def restitch_predictions(predictions, test_dataset, save=True,
                         save_path=root + 'predicted_hair_masks/'):
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


'''Used to display the original image, original mask, and predicted mask in a grid.
Displays two columns if no ground truth masks.
Otherwise displays three: one for original images, one for ground truth masks, one for predicted masks.
Ideally is used to display ten sets of images/masks.
If save = True, this image grid will be saved to the save_path.
'''
def display_image_grid(images_filenames, images_directory, masks_directory = None, predicted_masks=None,
                       save = False, save_path = 'whole_bee_predictions', filetype='.png'):
  cols = 3 if masks_directory else 2
  rows = len(images_filenames)
  figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 24))
  for i, image_filename in enumerate(images_filenames):
    image = cv2.imread(os.path.join(images_directory, image_filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax[i, 0].imshow(image)
    ax[i, 0].set_title("Image")
    ax[i, 0].set_axis_off()

    if masks_directory:
      mask = cv2.imread(os.path.join(masks_directory, image_filename))
      mask = preprocess_mask(mask)
      mask = mask[:, :, 0]

      ax[i, 1].imshow(mask, interpolation="nearest")

      ax[i, 1].set_title("Ground truth mask")

      ax[i, 1].set_axis_off()

    if predicted_masks:
      predicted_mask = predicted_masks[i]
      ax[i, cols - 1].imshow(predicted_mask, interpolation="nearest")
      ax[i, cols - 1].set_title("Predicted mask")
      ax[i, cols - 1].set_axis_off()
  plt.tight_layout()
  if save:
    root = os.getcwd()
    filename = os.path.join(root, save_path)
    plt.savefig(filename + filetype, bbox_inches='tight')
    plt.close(figure)
  else:
    plt.show()


'''
Same as display_image_grid, but does not display the ground truth masks.
Use this function if you don't have ground truth masks.
Ideally is used to display ten sets of images/masks.
If save = True, this image grid will be saved to the save_path.
'''
def display_bees(images_filenames, images_directory, predicted_masks,
                 save = False, save_path = 'whole_bee_predictions', filetype = '.png'):
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
  if save:
    root = os.getcwd()
    filename = os.path.join(root, save_path)
    plt.savefig(filename + filetype, bbox_inches='tight')
    plt.close(figure)
  else:
    plt.show()


# For each predicted mask, counts the number of predicted bee pixels and the percentage
# those pixels comprise the total image, then saves both values of each image into a csv.
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


# Calculates the jaccard (intersection over union) value.
def jaccard(y_true, y_pred):
  intersection = (y_true * y_pred).sum()
  union = y_true.sum() + y_pred.sum() - intersection
  return (intersection + 1e-15) / (union + 1e-15)


# Calculates the dice coefficient.
def dice(y_true, y_pred):
  return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)


# Calculates the raw accuracy score of accurate pixels divided by the total pixels.
def accuracy(y_true, y_pred):
  return (((y_pred + y_true) == 2).sum() + ((y_pred + y_true) == 0).sum()) / y_true.size


# Calculates the accuracy, dice, and IoU values, then saves them to a csv file.
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

'''
This is a modified version of the background removal function.
The original function can be found in the background_removal_helper.py file of this github:
https://github.com/Schachte/Background-Removal-Utility/blob/master/background_removal_helper.py
'''
def background_removal_updated(input_dir, output_dir, to_remove):
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
    image_path = os.path.join(input_dir, file_name)
    image = Image.open(image_path)

    if image.mode != "RGB":
      image = image.convert("RGB")

    print(f"Beginning removal on {file_name}...")

    result = interface([image])
    result = convert_to_black(result[0])
    result_path = os.path.join(output_dir, file_name)
    result.save(result_path)

    print(f"Saved {file_name} to {result_path}")


'''
Crops every image in a folder into squares of the crop height and crop width, with any remaining pixels
at the end of a row/column of images added to the last image in that row/column. The crops are then saved
to the path specified by crop_folder.
'''
def make_crops(folder, crop_folder, crop_height, crop_width):

  if not os.path.exists(crop_folder):
    os.makedirs(crop_folder)

  for img in os.listdir(folder):
    img_path = os.path.join(folder, img)
    if img.endswith(".jpg") or img.endswith(".JPG"):
      image = cv2.imread(img_path)
      height, width = np.shape(image)[0], np.shape(image)[1]
      hcrop = height // crop_height
      wcrop = width // crop_width

      for i in range(hcrop):
        for j in range(wcrop):
          crop = image[i * crop_height:(i + 1) * crop_height, j * crop_width:(j + 1) * crop_width]
          flat = np.ndarray.flatten(crop)
          sum = np.sum(flat)
          if sum > 0:
            name = img[:-4] + str(i) + str(j) + img[-4:]
            cv2.imwrite(os.path.join(crop_folder, name), crop)


'''
Picks random crops from the folder_path to save to the destination folder.
The number of crops is determined by size.
clear is set to True if you want to clear the random_crop folder first before adding a new set of random crops.
'''
def pick_random_crops(folder_path, destination_folder, size=250, clear=False):
  if clear:
    files = glob.glob(destination_folder)
    for f in files:
      os.remove(f)

  if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

  folder = os.listdir(folder_path)
  numbers = np.random.choice(len(folder), size=size, replace=False).tolist()
  for i in numbers:
    img_name = os.path.join(folder_path, folder[i])
    if not os.path.isfile(destination_folder + '/' + img_name):
      image = cv2.imread(img_name)
      cv2.imwrite(os.path.join(destination_folder, folder[i]), image)
  print('done')


# Counts how many images and their corresponding masks have mismatched sizes, and saves the names of any mismatched pairs,
# along with their respective sizes, to a csv. Then prints the numbers of pairs with matching and not matching shapes.
def entire_folder_check(folder, image_path, mask_path, csv='unmatched_shapes.csv'):
  root = os.getcwd()
  same_shape = 0
  different_shape = 0
  df = pd.DataFrame(columns=['Path', 'Image shape', 'Mask shape'])
  for i in folder:
    im_path = os.path.join(image_path, i)
    masks_path = os.path.join(mask_path, i)
    im = plt.imread(im_path)
    masks = plt.imread(masks_path)
    if np.shape(im) != np.shape(masks):
      different_shape += 1
      df.loc[len(df)] = [i, np.shape(im), np.shape(masks)]
    else:
      same_shape += 1

  df.to_csv(root + csv)

  print(same_shape, different_shape)


'''
Divides an image into 300 x 300 crops, with the remaining ends 256 x 256 pixels at the smallest,
and appends the crops into a nested list that creates a new list for each row of crops, in which all
the crops in that row are stored in that list.
'''
def crop(image, crop_height = 300, crop_width = 300):
  crops = []
  crop_height = crop_height
  crop_width = crop_width

  height, width = np.shape(image)[0], np.shape(image)[1]
  hcrop = height // crop_height
  wcrop = width // crop_width

  lasth = height - crop_height * (hcrop - 1)
  if lasth >= 556:
    hcrop += 1

  lastw = width - crop_width * (wcrop - 1)
  if lastw >= 556:
    wcrop += 1

  for i in range(hcrop):
    hcrops = []
    starth = i * crop_height
    endh = height if i == hcrop - 1 else (i + 1) * crop_height

    for j in range(wcrop):
      startw = j * crop_width
      endw = width if j == wcrop - 1 else (j + 1) * crop_width

      crop = image[starth:endh, startw:endw]
      hcrops.append(crop)

    crops.append(hcrops)

  return crops


'''
Name: Luning Ding
Date: July 3, 2023
'''

'''
Entropy Anlaysis
'''

'''
Changing the initial radius of the disk:
1 ---  sharpness
9 ---  dullness
'''
def disk_iterations(image):
  image_gray = rgb2gray(image)
  f_size = 20
  radi = list(range(1, 10))
  fig, ax = plt.subplots(3, 3, figsize=(15, 15))
  for n, ax in enumerate(ax.flatten()):
    ax.set_title(f'Radius at {radi[n]}', fontsize=f_size)
    ax.imshow(entropy(image_gray, disk(radi[n])), cmap='magma')
    ax.set_axis_off()
  fig.tight_layout()


# This returns a figure of the input image at various entropy thresholds.
def threshold_checker(image):
  thresholds = np.arange(0.1, 1.1, 0.1)
  image_gray = rgb2gray(image)
  entropy_image = entropy(image_gray, disk(6))
  scaled_entropy = entropy_image / entropy_image.max()
  fig, ax = plt.subplots(2, 5, figsize=(17, 10))
  for n, ax in enumerate(ax.flatten()):
    ax.set_title(f'Threshold  : {round(thresholds[n], 2)}',
                   fontsize=16)
    threshold = scaled_entropy > thresholds[n]
    ax.imshow(threshold, cmap='gist_stern_r')
    ax.axis('off')
  fig.tight_layout()


# Tests on single image and choose the radius of the disk for entropy analysis
def entropy_analysis_test(image_path):
  im = imread(image_path)
  plt.figure(num=None, figsize=(8, 6), dpi=80)
  imshow(im)
  # convert image to grayscale
  im_gray = rgb2gray(im)
  plt.figure(num=None, figsize=(8, 6), dpi=80)
  imshow(im_gray)
  disk_iterations(im_gray)
  threshold_checker(im_gray)


'''
Performs entropy analysis on the input image, then turns two images:
a version of the image with only the entropy values greater than 0.8, and
a version of the image with only the entropy values less than 0.8.
'''
def entropy_mask_viz(image):
    image_gray = rgb2gray(image)
    entropy_image = entropy(image_gray, disk(6))
    scaled_entropy = entropy_image / entropy_image.max()
    f_size = 24
    fig, ax = plt.subplots(1, 2, figsize=(17, 10))

    ax[0].set_title('Greater Than Threshold',
                    fontsize=f_size)
    threshold = scaled_entropy > 0.8
    image_a = np.dstack([image[:, :, 0] * threshold,
                         image[:, :, 1] * threshold,
                         image[:, :, 2] * threshold])
    ax[0].imshow(image_a)
    ax[0].axis('off')

    ax[1].set_title('Less Than Threshold',
                    fontsize=f_size)
    threshold = scaled_entropy < 0.8
    image_b = np.dstack([image[:, :, 0] * threshold,
                         image[:, :, 1] * threshold,
                         image[:, :, 2] * threshold])
    ax[1].imshow(image_b)
    ax[1].axis('off')
    fig.tight_layout()
    return [image_a, image_b]


'''
Loop over the entire folder, perform entropy analysis on each image in the folder using disk 6,
find the mean, median, and standard deviation of the image's entropy values,
save all of those values to a csv file, and save the entropy analysis images to another folder.
'''
def entropy_analysis_images(masked_image_path, entropy_output_path, csv_path = 'Entropy_Analysis_for_Bees.csv', save = True):
  for image_name in os.listdir(masked_image_path):
    im = imread(masked_image_path + image_name)
    # convert image to grayscale
    im_gray = rgb2gray(im)
    entropy_image = entropy(im_gray, disk(6))
    # exclude zeros in the matrix
    entropy_image_1 = entropy_image[entropy_image != 0]
    # write mean/median/standard deviations to csv file
    with open(csv_path, 'a') as f_object:
      # Pass this file object to csv.writer() and get a writer object
      writer_object = writer(f_object)

      # Pass the list as an argument into the writerow()
      writer_object.writerow(
        [image_name, np.mean(entropy_image_1), np.median(entropy_image_1), np.std(entropy_image_1)])

      # Close the file object
      f_object.close()

      if save:
        # save entropy images to output_path
        plt.imsave(entropy_output_path + image_name, entropy_image, cmap = 'magma')



'''
image regression
'''

# Helper function to show a batch and check if images are successfully loaded
def show_landmarks_batch(sample_batched, landmarks_batch):
    """Show image with landmarks for a batch of samples."""
    images_batch = \
      sample_batched['image']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch, padding=5, normalize=True, pad_value=1)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    print('rating:', [float(rating) for rating in sample_batched['rating'].numpy()])
    print('rating bin class', [float(class_label) for class_label in sample_batched['class'].numpy()])
    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                   landmarks_batch[i, :, 1].numpy() + grid_border_size,
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')



# train the model
def train_hairiness_model(model, device, dataloaders, rank_tensor, dataset_sizes,
                          criterion, optimizer, scheduler, num_epochs=25):
  since = time.time()

  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0
  best_mse = 10000
  result_dict = defaultdict(list)
  softmax = nn.Softmax().to(device)

  for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
      print('start', phase, 'phase')
      if phase == 'train':
        model.train()  # Set model to training mode
      else:
        model.eval()  # Set model to evaluate mode
      running_loss = 0.0
      epoch_CE_loss = 0.0
      running_corrects = 0
      running_rmse_loss = 0.0
      # Iterate over data.
      for data in tqdm.tqdm(dataloaders[phase]):
        images, ratings, labels = data['image'], data['rating'], data['class']
        inputs = images.to(device)
        labels = labels.to(device)
        ratings = ratings.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
          outputs = model(inputs)
          _, preds = torch.max(outputs, 1)

          loss1 = criterion(outputs, labels)

          # revised loss function
          softmaxed_output = softmax(outputs)
          expected_output = torch.sum(softmaxed_output * rank_tensor, 1)
          expected_output = expected_output[None]
          expected_output = torch.transpose(expected_output, 0, 1)

          # RMSE loss
          reg_criterion = nn.MSELoss().to(device)
          loss2 = torch.sqrt(reg_criterion(expected_output, ratings))

          loss = loss1 + loss2

          # backward + optimize only if in training phase
          if phase == 'train':
            loss.backward()
            optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        running_rmse_loss += loss2.item() * inputs.size(0)

      if phase == 'train':
        scheduler.step()

      epoch_loss = running_loss / dataset_sizes[phase]
      epoch_rmse_loss = running_rmse_loss / dataset_sizes[phase]
      epoch_acc = running_corrects.double() / dataset_sizes[phase]
      result_dict['epoch_loss'].append(epoch_loss)
      result_dict['epoch_rmse_loss'].append(epoch_rmse_loss)
      result_dict['epoch_acc'].append(epoch_acc.item())

      print(f'{phase} RMSE Loss: {epoch_rmse_loss:.2f} Total Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

      # deep copy the model
      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
      if phase == 'val' and epoch_rmse_loss < best_mse:
        best_mse = epoch_rmse_loss
        best_model_wts = copy.deepcopy(model.state_dict())

    print()

  time_elapsed = time.time() - since
  print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
  print(f'Best val Acc: {best_acc:4f}')
  print(f'Best val RMSE: {best_mse:4f}')

  # load best model weights
  model.load_state_dict(best_model_wts)
  return model, result_dict



def hairiness_rating(root_dir, model, device, rank_dict, data_transform):
  idx_tensor = [idx for idx in range(20)]
  idx_tensor = Variable(torch.FloatTensor(idx_tensor)).to(device)
  rank_list = []
  for i in idx_tensor:
    rank_list.append(rank_dict[i.item()])
  rank_tensor = Variable(torch.FloatTensor(rank_list)).to(device)

  surface_area_df = pd.read_csv(
    '/content/drive/MyDrive/Pilosity ML Project/2023Updated/Segmentation results/bee_surface_areas_no_new_images.csv')
  imgname_df = surface_area_df['name']
  surface_area_df = surface_area_df['percentage of pixels']

  whole_bee_dict = defaultdict(list)
  softmax = nn.Softmax().to(device)

  with torch.no_grad():
    for imgname, surf_percent in zip(imgname_df, surface_area_df):
      figure(figsize=(4, 3), dpi=80)
      print(imgname)
      im = imread(root_dir + imgname)
      plt.imshow(im)
      plt.show()
      image = Image.open(root_dir + imgname)
      image = data_transform(image).to(device)
      image = image[None, :]
      outputs = model(image)
      _, preds = torch.max(outputs, 1)
      softmaxed_output = softmax(outputs)
      expected_output = torch.sum(softmaxed_output * rank_tensor, 1)
      expected_output = expected_output[None]
      expected_output = torch.transpose(expected_output, 0, 1)
      # print('Predicted rating w/o surface area: ', expected_output.item())
      whole_bee_dict['predicted_score'].append(expected_output.item() / float(surf_percent.strip('%')) * 100)
      print('Predicted_rating w/ surface area: ', expected_output.item() / float(surf_percent.strip('%')) * 100)
      print()
      print()

      # convert image to grayscale
      im_gray = rgb2gray(im)
      entropy_image = entropy(im_gray, disk(6))
      # exclude zeros in the matrix
      entropy_image_1 = entropy_image[entropy_image != 0]
      whole_bee_dict['Entropy_mean'].append(np.mean(entropy_image_1) / float(surf_percent.strip('%')) * 100)


