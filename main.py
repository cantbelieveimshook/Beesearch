'''
Name: Kathryn Chen
Name: Luning Ding
Date: June 23, 2023
'''

import os
import sys
from remove_background import remove_background_main
from segment_bee import segment_bee_main
from artificial_bees import artificial_bees_main
from segment_hair import segment_hair_main
from entropy_analysis import entropy_analysis
from image_regression import image_regression, predicted_rating_entropy_values, predicted_rating_entropy_surfacec_area


remove_background = False
if len(sys.argv) > 1:
    if sys.argv[1] == True or sys.argv[1] == False:
        remove_background = sys.argv[1]

if remove_background:
    remove_background_main()

segment_bee_main(background_removed = remove_background, to_train = False)
artificial_bees_main()
segment_hair_main(to_crop = True, to_train = False)



root = os.getcwd()
image_folder_path = os.path.join(root, 'artificial_bees/')
entropy_output_path = os.path.join(root, 'entropy_images/')
entropy_values = os.path.join(root, 'entropy_analysis/', 'entropy_values_for_artificial_bees.csv')
ground_truth_hairiness_rating = os.path.join(root, 'image_regression/', 'groud_truth_rating.csv')
model_save = os.path.join(root, 'image_regression/')
predicted_rating = os.path.join(root, 'image_regression/', 'predicted_rating.csv')
surface_area = os.path.join(root, 'artificial_bees/', 'surface_area.csv')
# Entropy Anaysis
'''
Optional functions to test on a single image:
entropy_mask_viz(image)
entropy_analysis_test(image_path)
'''
entropy_analysis(image_folder_path, entropy_output_path, entropy_values)

# Image regression
image_regression(ground_truth_hairiness_rating, image_folder_path, model_save)
predicted_rating_entropy_values(ground_truth_hairiness_rating, image_folder_path, model_save, predicted_rating)
predicted_rating_entropy_surfacec_area(ground_truth_hairiness_rating, model_save, image_folder_path, surface_area)


