import os
import sys
import random
import string

import numpy as np
import pandas as pd
from PIL import Image
import cv2 as cv
import shutil

# This script will produce a set of patches from unlabelized wafer artificial_images
# To be then sent to the brainseg model (to generates inferences, segmentations and detection)

#To use it : python create_wafer_patches.py <DatasetWafer number> <section size>
#It will then look at the specified path here ""../augmented_dataset/newWafersBIOP/Wafer_"+str(DATASET_NUMBER)" to be changed in purpose
# Extract the image and produce the crops

def generate_batches(file_wafersBIOP, path_image, section_size):

    wafer = Image.open(path_image)

    coef_x = (wafer.size[0] // section_size) -1
    left_x = wafer.size[0] - (coef_x * section_size)

    coef_y = (wafer.size[1] // section_size) -1
    left_y = wafer.size[1] - (coef_y * section_size)

    for i in range(coef_x):

        for j in range(coef_y):
        # random crop coordinates (top-left point of the cropped area)
            start_x = i*section_size
            start_y = j*section_size

            # cropping the wafer image
            cropped_image = wafer.crop((start_x,start_y,start_x+section_size,start_y+section_size))

            image_folder = str(file_wafersBIOP)+"/grid/wafer_crop"+str(i)+"_"+str(j)+"/image/"
            os.makedirs(os.path.dirname(image_folder), exist_ok=True)

            # saving the cropped image
            cropped_image.save(image_folder+"wafer_crop"+str(i)+"_"+str(j)+".tif")





def main():
	DATASET_NUMBER = int(sys.argv[1])
	section_size = int(sys.argv[2])
	# Wafer file to crop
	file_wafersBIOP = "../augmented_dataset/newWafersBIOP/Wafer_"+str(DATASET_NUMBER)
	path_image = file_wafersBIOP+"/stitched_RGB_small.tif"
	wafer = Image.open(path_image)


	generate_batches(file_wafersBIOP, path_image, section_size)

if __name__ == "__main__":
    main()
