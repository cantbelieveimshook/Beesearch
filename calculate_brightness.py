'''
Name: Kathryn Chen
Date: September 14, 2023
'''

from classes import *
from functions import predict, resize_predictions, calculate_brightness
import albumentations as A
from albumentations.pytorch import ToTensorV2

'''
Calculates the brightness, which is measured by the average pixel value, of every pixel in an image that is considered to be a "bee" pixel.
This may mean that some pixels are inaccurately included or excluded from the calculations, but this is to help ensure that the background
pixels are overall left out of the calculations. 
'''
def calculate_brightness_main():
    device = "cuda" if torch.cuda.is_available() else torch.device('cpu')
    root = '/content/drive/MyDrive/Beesearch/2023Updated/'
    original_images_directory = root + 'whole_bee_original/'
    images = os.listdir(original_images_directory)
    model = torch.load('/content/drive/MyDrive/Beesearch/Models/Whole bee final/Model_303').to(device)
    params = {
        "device": device,
        "batch_size": 16,
        "num_workers": 4,
    }

    image_transform = A.Compose(
        [A.Resize(256, 256),
         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
         ToTensorV2()]
    )

    test_dataset = BeeInferenceDataset(images, original_images_directory, image_transform=image_transform)
    predictions = predict(model, params, test_dataset)
    predicted_masks = resize_predictions(predictions, test_dataset, save=True, save_path=root + 'predicted_bees_with_eyes_wings_antennae/')

    calculate_brightness(images, predicted_masks, original_images_directory, save = True, load = False)