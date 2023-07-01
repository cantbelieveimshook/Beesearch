import os
import sys
from remove_background import remove_background_main
from segment_bee import segment_bee_main
from artificial_bees import artificial_bees_main
from segment_hair import segment_hair_main


remove_background = False
if len(sys.argv) > 1:
    if sys.argv[1] == True or sys.argv[1] == False:
        remove_background = sys.argv[1]

if remove_background:
    remove_background_main()

segment_bee_main(background_removed = remove_background, to_train = False)
artificial_bees_main()
segment_hair_main(to_crop = True, to_train = False)
