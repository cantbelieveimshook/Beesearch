# README

# In order to run everything:

$python main.py

How to upload images:

There are many differnt image folders with a wide variety of purposes. You can either run make_folders.py or manually create the necessary folders based on your needs. If you wish to change any of the filenames, the filenames of each folder can either be manually changed or changed in paths.py before the folders are created.

The following is an explanation for every folder name and what they should be used for:
root: The path of the directory where everything is located. Called by getting os.cwd(). For organization purposes, every subfolder and script should be located in the root folder. All other folders will be named assuming they are located in the root folder.

**bee_images_directory**

path: root + 'bee_original/'

This is the file path for the original full bee images. Use this folder to store the images of bees you wish to analyze.

**original_bee_masks_directory**

path: root + 'original_bee_masks/'

If you have manually created bee masks you would like to use, upload them to this folder. This folder is not for storing predicted bee masks created by machine learning models.

**bee_masks_directory**

path = root + 'predicted_bee_masks/'

This is the file path for the predicted bee masks, which are segmented by a machine learning model. If you run a script to segment out the eyes, wings, and antennae from bee images, the resulting masks will automatically be saved here, unless you set save = False in resize_predictions.

**artificial_bees_directory**

path: root + 'artificial_bees/'

This is the file path for the artificial bees, which are created by multiplying the predicted bee masks with the original bee images.
These are images created to artificially remove the eyes, wings, and antennae. Used for hair segmentation.

**hair_images_directory**

path: root + 'hair_original/'

This is the file path for the original full bee images. Use this folder to store the images of bees you wish to analyze.

**original_hair_masks_directory**

path: root + 'original_hair_masks/'

If you have manually created hair masks you would like to use, upload them to this folder. This folder is not for storing predicted hair masks created by machine learning models.

**hair_masks_directory**

path = root + 'predicted_hair_masks/'

This is the file path for the predicted hair masks, which are segmented by another machine learning model. If you run a script to segment out the hair from bee or artificial bee images, the resulting masks will automatically be saved here, unless you set save = False in restitch_predictions.

**artificial_hair_directory**

path - root + 'segmented_hair_final/'

This is the file path for the folder that will contain the final "artificial hair", which are created by multiplying the artificial bee images with the predicted hair masks. The resulting images are the result of two machine learning models working to artificially remove anything that is not hair from the original bee images, so the resulted hair can be used for a variety of image analysis.

**background_bees_directory**

path = root + 'removed_background_bees/'

If you choose to artificially remove the backgrounds of your bee images by running remove_background.py, which may help produce more accurate bee segmentations, the removed background images should automatically appear in this folder after running the script.

**aug_im_dir**

path = root + 'augmented_images/'

This is the folder that will contain all folders of augmented images. If you intend to use any augmentations during model training, run make_augmentations.py and the resulting augmented images will appear in subdirectories in this folder. If you want to use augmentations for both bee and hair model training, make another copy of this folder.

**aug_mask_dir**

path = root + 'augmented_mask/'

This is the folder that will contain all folders of augmented masks. If you intend to use any augmentations during model training, run make_augmentations.py and the resulting masks for the corresponding augmented images will appear in subdirectories in this folder. If you want to use augmentations for both bee and hair model training, make another copy of this folder.

**crop_path**

path = root + 'bee_crops/'

This is a folder that will contain rectangular crops of artificial bees, which will range in size anywhere from 256 x 256 to 556 x 556. You likely will not need to use this folder unless you are using crops to make more hair masks.

**entropy_output_path**

path = root + 'entropy_images/'

This folder contains the outputs of entropy analysis on images. Only use if you intend to do entropy analysis on bees or bee hair. There is currently no pair of subfolders for entropy analysis on bees and bee hair, so if you wish to do entropy analysis on both, create another similarly named folder and change the folder names in the entropy analysis code in main.py.

**entropy_values**

path = root + 'entropy_analysis/'

This folder contains the results of the entropy analysis, such as entropy values for a dataset.

**image_regression**

path = root + 'image_regression/'

This folder contains the data and models necessary for the hairiness rating scripts. Use if you plan on calculating hairiness scores for bee images.


