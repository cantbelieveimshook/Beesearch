'''
Name: Kathryn Chen
Date: June 23, 2023
'''

import torch
from make_augment_functions import *

device = "cuda" if torch.cuda.is_available() else "cpu"

'''
For any of these functions, set clear = True to clear out the current 
directory of its existing augmented images before adding the new one.
Clear is set to False by default.
Set flag = False for any augmentations you do not want to make before 
running the .py file. "flag" is set to True by default.
'''

create_flip_set(flag = True)

create_rotated_set(flag = True)

invert_images(flag = True)

blur_image(flag = True)

add_noise(flag = True)

# The argument is the degree of horizontal shift.
horizontal_shift(0.25, flag = True)

# The argument is the degree of vertical shift.
vertical_shift(0.25, flag = True)

# The argument is the amount the image is zoomed in.
zoom(0.75, flag = True)

# First argument is the lowest degree of brightness, second is the highest.
brightness(0.5, 2.5, flag = True)
