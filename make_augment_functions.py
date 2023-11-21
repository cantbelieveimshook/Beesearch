'''
Name: Kathryn Chen
Date: June 23, 2023
'''

import cv2
from skimage import util
import random
import numpy as np
from paths import *

# This is used to ensure that the image and mask directories exist. It also includes the code for clearing the folders.
def housekeeping(new_image_directory, new_mask_directory, clear):
    if not os.path.exists(new_image_directory):
        os.makedirs(new_image_directory)
    if not os.path.exists(new_mask_directory):
        os.makedirs(new_mask_directory)

    if clear:
        files_im = os.listdir(new_image_directory)
        for f in files_im:
            os.remove(os.path.join(new_image_directory, f))
        files_mask = os.listdir(new_mask_directory)
        for f in files_mask:
            os.remove(os.path.join(new_mask_directory, f))

'''
The following functions have been modified from a source.
The original functions are written by the authors of this paper:
https://escholarship.org/uc/item/0h07635j
'''

# Creates a set of images that have been flipped vertically.
def create_flip_set(image_names=original_bee_masks, images_directory=bee_images_directory, masks_directory=original_bee_masks_directory,
                    new_image_directory=aug_im_dir + 'flipped/', new_mask_directory=aug_mask_dir + 'flipped/',
                    flag = True, clear=False):

    housekeeping(new_image_directory, new_mask_directory, clear)

    if flag:
        for image in image_names:
            original = cv2.imread(f'{images_directory}/{image}')
            mask = cv2.imread(f'{masks_directory}/{image}')

            if original is None or mask is None:
                print("whoops")
                continue

            cv2.imwrite(f'{new_image_directory}/{image}', cv2.flip(original, -1))
            cv2.imwrite(f'{new_mask_directory}/{image}', cv2.flip(mask, -1))
    else:
        return "Nothing needs to be done"


# Creates a set of images rotated 90, 180, and 270 degrees.
def create_rotated_set(image_names=original_bee_masks, images_directory=bee_images_directory, masks_directory=original_bee_masks_directory,
                       new_image_directory=aug_im_dir + 'rotated/', new_mask_directory=aug_mask_dir + 'rotated/',
                       flag = True, clear=False):
    housekeeping(new_image_directory, new_mask_directory, clear)

    rotations = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]

    if flag:
        for image in image_names:
            original = cv2.imread(f'{images_directory}/{image}')
            mask = cv2.imread(f'{masks_directory}/{image}')

            if original is None or mask is None: continue

            c = random.randint(0, 2)
            original = cv2.rotate(original, rotations[c])
            mask = cv2.rotate(mask, rotations[c])

            cv2.imwrite(f'{new_image_directory}/{image}', original)
            cv2.imwrite(f'{new_mask_directory}/{image}', mask)
    else:
        return "Nothing needs to be done"


# Creates a set of images that have been inverted on the RGB color scale.
def invert_images(image_names=original_bee_masks, images_directory=bee_images_directory, masks_directory=original_bee_masks_directory,
                  new_image_directory=aug_im_dir + 'inverted/', new_mask_directory=aug_mask_dir + 'inverted/',
                  flag = True, clear=False):
    housekeeping(new_image_directory, new_mask_directory, clear)

    if flag:
        for image_name in image_names:
            image = cv2.imread(os.path.join(images_directory, image_name))
            mask = cv2.imread(os.path.join(masks_directory, image_name))
            if image is None or mask is None: continue
            inv_image = 255 - image
            cv2.imwrite(f'{new_image_directory}/{image_name}', inv_image)
            cv2.imwrite(f'{new_mask_directory}/{image_name}', mask)
    else:
        return "Nothing needs to be done"


# Creates a set of images that have had some amount of Gaussian blur added.
def blur_image(image_names=original_bee_masks, images_directory=bee_images_directory, masks_directory=original_bee_masks_directory,
               new_image_directory=aug_im_dir + 'blurred/', new_mask_directory=aug_mask_dir + 'blurred/',
               flag = True, clear=False):
    housekeeping(new_image_directory, new_mask_directory, clear)

    if flag:
        for image_name in image_names:
            image = cv2.imread(os.path.join(images_directory, image_name))
            mask = cv2.imread(os.path.join(masks_directory, image_name))

            if image is None or mask is None: continue

            gausBlur = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)

            cv2.imwrite(f'{new_image_directory}/{image_name}', gausBlur)
            cv2.imwrite(f'{new_mask_directory}/{image_name}', mask)
    else:
        return "Nothing needs to be done"


# Creates a set of images with some random noise added.
def add_noise(image_names=original_bee_masks, images_directory=bee_images_directory, masks_directory=original_bee_masks_directory,
              new_image_directory=aug_im_dir + 'noisy/', new_mask_directory=aug_mask_dir + 'noisy/',
              flag=True, clear=False):
    housekeeping(new_image_directory, new_mask_directory, clear)

    if flag:
        for image_name in image_names:
            image = cv2.imread(os.path.join(images_directory, image_name))
            mask = cv2.imread(os.path.join(masks_directory, image_name))

            if image is None or mask is None: continue

            noisy = util.random_noise(image, mode='s&p', amount=(np.random.rand() * .1)) * 255

            cv2.imwrite(os.path.join(new_image_directory, image_name), noisy)
            cv2.imwrite(os.path.join(new_mask_directory, image_name), mask)
    else:
        return "Nothing needs to be done"


# The following functions have been modified from a source.
# The original functions are from https://towardsdatascience.com/complete-image-augmentation-in-opencv-31a6b02694f5.

# Resizes the image based on the h (height) and w (weight) arguments.
def fill(img, h, w):
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img


# Creates a set of images that are horizontally shifted.
def horizontal_shift(ratio, image_names=original_bee_masks, images_directory=bee_images_directory, masks_directory=original_bee_masks_directory,
                     new_image_directory=aug_im_dir + 'horizontal_shifts/',
                     new_mask_directory=aug_mask_dir + 'horizontal_shifts/', flag=True, clear=False):
    housekeeping(new_image_directory, new_mask_directory, clear)

    if flag:
        for path in image_names:
            image = cv2.imread(os.path.join(images_directory, path))
            mask = cv2.imread(os.path.join(masks_directory, path))

            if image is None or mask is None: continue

            if ratio > 1 or ratio < -1:
                print('Value should be less than 1 and greater than -1')
                return image
            ratio = random.uniform(-ratio, ratio)
            h, w = image.shape[:2]
            to_shift = w * ratio
            if ratio > 0:
                new_img = image[:, :int(w - to_shift), :]
                new_mask = mask[:, :int(w - to_shift), :]
            if ratio < 0:
                new_img = image[:, int(-1 * to_shift):, :]
                new_mask = mask[:, int(-1 * to_shift):, :]
            new_img = fill(new_img, h, w)
            new_mask = fill(new_mask, h, w)
            cv2.imwrite(os.path.join(new_image_directory, path), new_img)
            cv2.imwrite(os.path.join(new_mask_directory, path), new_mask)
    else:
        return "Nothing needs to be done"


# Creates a set of images that are vertically shifted.
def vertical_shift(ratio, image_names=original_bee_masks, images_directory=bee_images_directory, masks_directory=original_bee_masks_directory,
                   new_image_directory=aug_im_dir + 'vertical_shifts/',
                   new_mask_directory=aug_mask_dir + 'vertical_shifts/', flag=True, clear=False):
    housekeeping(new_image_directory, new_mask_directory, clear)

    if flag:
        for path in image_names:
            image = cv2.imread(os.path.join(images_directory, path))
            mask = cv2.imread(os.path.join(masks_directory, path))

            if image is None or mask is None: continue

            if ratio > 1 or ratio < -1:
                print('Value should be less than 1 and greater than -1')
                return image
            ratio = random.uniform(-ratio, ratio)
            h, w = image.shape[:2]
            to_shift = w * ratio
            if ratio > 0:
                new_img = image[:int(h - to_shift), :, :]
                new_mask = mask[:int(h - to_shift), :, :]
            if ratio < 0:
                new_img = image[int(-1 * to_shift):, :, :]
                new_mask = mask[int(-1 * to_shift):, :, :]
            new_img = fill(new_img, h, w)
            new_mask = fill(new_mask, h, w)
            cv2.imwrite(os.path.join(new_image_directory, path), new_img)
            cv2.imwrite(os.path.join(new_mask_directory, path), new_mask)
    else:
        return "Nothing needs to be done"


# Creates a set of images that are zoomed in by a certain value.
def zoom(value, image_names=original_bee_masks, images_directory=bee_images_directory, masks_directory=original_bee_masks_directory,
         new_image_directory=aug_im_dir + 'zoom/', new_mask_directory=aug_mask_dir + 'zoom/', flag=True,
         clear=False):
    housekeeping(new_image_directory, new_mask_directory, clear)

    if flag:
        for path in image_names:
            image = cv2.imread(os.path.join(images_directory, path))
            mask = cv2.imread(os.path.join(masks_directory, path))

            if image is None or mask is None: continue

            if value > 1 or value < 0:
                print('Value for zoom should be less than 1 and greater than 0')
                return image
            value = random.uniform(value, 1)
            h, w = image.shape[:2]
            h_taken = int(value * h)
            w_taken = int(value * w)
            h_start = random.randint(0, h - h_taken)
            w_start = random.randint(0, w - w_taken)
            new_img = image[h_start:h_start + h_taken, w_start:w_start + w_taken, :]
            new_mask = mask[h_start:h_start + h_taken, w_start:w_start + w_taken, :]
            new_img = fill(new_img, h, w)
            new_mask = fill(new_mask, h, w)
            cv2.imwrite(os.path.join(new_image_directory, path), new_img)
            cv2.imwrite(os.path.join(new_mask_directory, path), new_mask)
    else:
        return "Nothing needs to be done"


# Converts the image to the HSV color representation.
def hsv_convert(image, value):
    hsv = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), dtype=np.float64)
    hsv[:, :, 1] = hsv[:, :, 1] * value
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
    hsv[:, :, 2] = hsv[:, :, 2] * value
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    return hsv


# Creates a set of images that are brightened by a certain value.
def brightness(low, high, image_names=original_bee_masks, images_directory=bee_images_directory, masks_directory=original_bee_masks_directory,
               new_image_directory=aug_im_dir + 'brightness/',
               new_mask_directory=aug_mask_dir + 'brightness/', flag=True, clear=False):
    housekeeping(new_image_directory, new_mask_directory, clear)

    if flag:
        for path in image_names:
            image = cv2.imread(os.path.join(images_directory, path))
            mask = cv2.imread(os.path.join(masks_directory, path))

            if image is None or mask is None: continue

            value = random.uniform(low, high)
            hsv_img = hsv_convert(image, value)
            hsv_mask = hsv_convert(mask, value)
            new_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
            new_mask = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
            cv2.imwrite(os.path.join(new_image_directory, path), new_img)
            cv2.imwrite(os.path.join(new_mask_directory, path), mask)
    else:
        return "Nothing needs to be done"


# Redirects images and masks from their old directories to new image and mask directories.
def redirect_images(image_filenames, old_original_dir, old_mask_dir, new_original_dir, new_mask_dir):
    for image in image_filenames:
        original = cv2.imread(os.path.join(old_original_dir, image))
        if original is None: continue
        mask = cv2.imread(os.path.join(old_mask_dir, image))

        os.chdir(new_original_dir)
        cv2.imwrite(image, original)

        os.chdir(new_mask_dir)
        cv2.imwrite(image, mask)