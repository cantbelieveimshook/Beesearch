import numpy as np
import cv2
import os
from torch.utils.data import Dataset
from paths import *

def preprocess_mask(input_mask):
  test_mask = input_mask / 255
  test_mask[test_mask > 0.5] = 1
  test_mask[test_mask <= 0.5] = 0
  return test_mask

class ShapeException(Exception):
  pass

class BeeInferenceDataset(Dataset):
  def __init__(self, images_filenames, images_directory, masks_directory, image_transform=None, mask_transform=None):
    self.images_filenames = images_filenames
    self.images_directory = images_directory
    self.masks_directory = masks_directory
    self.image_transform = image_transform
    self.mask_transform = mask_transform

  def __len__(self):
    return len(self.images_filenames)

  def getsize(self, idx):
    image_filename = self.images_filenames[idx]
    image = cv2.imread(os.path.join(self.images_directory, image_filename))
    original_size = tuple(image.shape[:2])
    return original_size

  def getname(self, idx):
    image_filename = self.images_filenames[idx]
    return image_filename

  def __getitem__(self, idx):
    image_filename = self.images_filenames[idx]
    image = cv2.imread(os.path.join(self.images_directory, image_filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(os.path.join(self.masks_directory, image_filename))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if self.image_transform is not None:
      transformed_img = self.image_transform(image=image)
      image = transformed_img["image"]
    if self.mask_transform is not None:
      transformed_mask = self.mask_transform(image=mask)
      mask = preprocess_mask(transformed_mask["image"])
      mask = mask.float()
    return image, mask

# to be used to create test datasets only
class CroppedDataset(Dataset):
  def __init__(self, images_filenames, images_directory, masks_directory=None, image_transform=None,
               mask_transform=None, has_masks=False, crop_height=300, crop_width=300):
    self.images_filenames = images_filenames
    self.images_directory = images_directory
    self.masks_directory = masks_directory
    self.image_transform = image_transform
    self.mask_transform = mask_transform
    self.has_masks = has_masks
    self.crop_height = crop_height
    self.crop_width = crop_width

  def __len__(self):
    return len(self.images_filenames)

  def getsize(self, idx):
    image_filename = self.images_filenames[idx]
    image = cv2.imread(os.path.join(self.images_directory, image_filename))
    original_size = tuple(image.shape[:2])
    return original_size

  def getname(self, idx):
    image_filename = self.images_filenames[idx]
    return image_filename

  def collate_fn(self, batch):
    return batch

  def crop(self, image):
    crops = []
    crop_height = self.crop_height
    crop_width = self.crop_width

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

  def __getitem__(self, idx):
    image_filename = self.images_filenames[idx]
    image = cv2.imread(os.path.join(self.images_directory, image_filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    crops = self.crop(image)
    croplist = []

    for i in crops:
      row = []
      for j in i:
        if self.image_transform is not None:
          transformed_img = self.image_transform(image=j)
          j = transformed_img["image"]
        row.append(j)
      croplist.append(row)

    if self.has_masks:
      mask = cv2.imread(os.path.join(self.masks_directory, image_filename))
      mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
      maskcrops = self.crop(mask)
      maskcroplist = []

      for i in maskcrops:
        maskrow = []
        for j in i:
          if self.mask_transform is not None:
            transformed_mask = self.mask_transform(image=j)
            mask = preprocess_mask(transformed_mask["image"])
            j = mask.float()
          maskrow.append(j)
        maskcroplist.append(maskrow)

      return croplist, maskcroplist

    else:
      return croplist