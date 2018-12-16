import os
import random
import string

import numpy as np
import pandas as pd
from PIL import Image
import cv2 as cv
import shutil

DATASET_NUMBER = 11

# Wafer file to crop
file_wafersBIOP = "../augmented_dataset/newWafersBIOP/Wafer_"+str(DATASET_NUMBER)
path_image = file_wafersBIOP+"/stitched_RGB_small.tif"
wafer = Image.open(path_image)

# Height and width of the resulting section in pixels
section_size = 512

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



    generate_batches(file_wafersBIOP, path_image , section_size)
