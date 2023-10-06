'''
Name: Kathryn Chen
Date: October 6, 2023
'''

import albumentations as A
from albumentations.pytorch import ToTensorV2
from classes import *
from functions import preprocess_mask, predict, resize_predictions, count_surface_area



def calculate_surface_area_main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = os.getcwd()
    bee_images_directory = os.path.join(root, 'bee_original')
    artificial_bees_directory = os.path.join(root, 'artificial_bees')
    bee_masks_directory = os.path.join(root, 'predicted_bee_masks')
    hair_masks_directory = os.path.join(root, 'predicted_hair_masks')

    bee_images = os.listdir(bee_images_directory)
    artificial_bees = os.listdir(artificial_bees_directory)
    bee_masks = os.listdir(bee_masks_directory)
    hair_masks = os.listdir(hair_masks_directory)

    bee_model = model = torch.load(root + 'models/New_Bee_Model').to(device)
    hair_model = torch.load(root + 'models/model_98.pth').to(device)

    params = {
        "device": device,
        "lr": 0.001,
        "batch_size": 16,
        "num_workers": 4,
        "epochs": 15,
    }

    image_transform = A.Compose(
        [A.Resize(256, 256),
         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
         ToTensorV2()]
    )

    # If there are no bee masks already made, then make a dataset to create a list of predicted masks.
    if len(bee_masks) == 0:
        bee_dataset = BeeInferenceDataset(bee_images, bee_images_directory, image_transform=image_transform)
        bee_predictions = predict(bee_model, params, bee_dataset)
        bee_predicted_masks = resize_predictions(bee_predictions, bee_dataset)
    else:
        bee_predicted_masks = []
        for i in bee_masks:
            path = os.path.join(bee_masks_directory, i)
            image = cv2.imread(path)
            mask = preprocess_mask(image)
            bee_predicted_masks.append(mask)


    # If there are no hair masks already made, then make a dataset to create a list of predicted masks.
    if len(hair_masks) == 0:
        hair_dataset = BeeInferenceDataset(artificial_bees, artificial_bees_directory, image_transform=image_transform)
        hair_predictions = predict(hair_model, params, hair_dataset)
        hair_predicted_masks = resize_predictions(hair_predictions, hair_dataset)
    else:
        hair_predicted_masks = []
        for i in hair_masks:
            path = os.path.join(hair_masks_directory, i)
            image = cv2.imread(path)
            mask = preprocess_mask(image)
            hair_predicted_masks.append(mask)

    # Set load = True if you want to add to an existing csv.
    count_surface_area(bee_predicted_masks, hair_predicted_masks, dataset = bee_dataset,
                       save_path = root + 'surface_areas.csv',
                       load_path = root + 'surface_areas.csv',
                       load=False)