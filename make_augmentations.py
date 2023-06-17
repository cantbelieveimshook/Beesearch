import torch
from make_augment_functions import *

device = "cuda" if torch.cuda.is_available() else "cpu"

# For any of these functions, set clear = True to clear out the current directory of its existing augmented images before adding the new one
create_flip_set()

create_rotated_set()

invert_images()

blur_image()

add_noise()

# the argument is the degree of horizontal shift
horizontal_shift(0.25)

# the argument is the degree of vertical shift
vertical_shift(0.25)

# the argument is the amount the image is zoomed in
zoom(0.75)

# first argument is the lowest degree of brightness, second is the highest
brightness(0.5, 2.5)