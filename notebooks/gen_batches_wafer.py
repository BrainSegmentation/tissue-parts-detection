import os
import sys
import random
import string

import numpy as np
import pandas as pd

from PIL import Image
import cv2 as cv
import shutil


def generate_mask(image, points):
    # size of the image
    height = image.size[1]
    width = image.size[0]

    # init mask
    img_mask = np.zeros([height,width],dtype=np.uint8)
    img_mask.fill(0)

    cv.fillConvexPoly(img_mask, points, 255)

    return img_mask


def generate_batches(DATASET_NUMBER, number_of_sections, section_size, rgb):

    index = pd.MultiIndex.from_tuples([('point_1', 'x'), ('point_1', 'y'), ('point_2', 'x'), ('point_2', 'y'),
                                   ('point_3', 'x'), ('point_3', 'y'), ('point_4', 'x'), ('point_4', 'y')])

    # Wafer file to crop
    if(rgb==True):
        file = "../augmented_dataset/wafer_with_fluo_RGB_"+str(DATASET_NUMBER)+".tif"
    else:
        file = "../dataset/silicon_wafer_"+str(DATASET_NUMBER)+"/wafer_"+str(DATASET_NUMBER)"_downsized_3_intensityCorrected.tif"
    wafer = Image.open(file)
    seg_tissues = pd.read_csv("../dataset/silicon_wafer_"+str(DATASET_NUMBER)+"/source_sections_tissue_scale3.txt", sep="\t|,", header=None, names=index, engine='python')
    seg_mag = pd.read_csv("../dataset/silicon_wafer_"+str(DATASET_NUMBER)+"/source_sections_mag_scale3.txt", sep="\t|,", header=None, names=index, engine='python')



    for i in range(1,number_of_sections+1):

        # random crop coordinates (top-left point of the cropped area)
        start_x = random.randint(0, wafer.size[0] - section_size)
        start_y = random.randint(0, wafer.size[1] - section_size)

        # cropping the wafer image
        cropped_image = wafer.crop((start_x,start_y,start_x+section_size,start_y+section_size))


        # index of all tissue part within the cropped area
        tissue_indicies = list()
        for index, row in seg_tissues.iterrows(): # iterating over sections
            points_within = 0
            for j in range(0,8,2): # iterating over the 4 points for each section
                if (start_x < row[j]-5) &  (row[j] < start_x+section_size-5) & (start_y < row[j+1]-5) &  (row[j+1]< start_y+section_size-5):
                    points_within += 1
            if(points_within >= 1):
                tissue_indicies.append(index)

        # index of all magnetic part within the cropped area
        mag_indicies = list()
        for index, row in seg_mag.iterrows(): # iterating over sections
            points_within = 0
            for j in range(0,8,2): # iterating over the 4 points for each section
                if (start_x < row[j]-5) &  (row[j] < start_x+section_size-5) & (start_y < row[j+1]-5) &  (row[j+1]< start_y+section_size-5):
                    points_within += 1
            if(points_within >= 1):
                mag_indicies.append(index)


        if( (len(tissue_indicies) >=3) & (len(mag_indicies) >=3) ):

            # creating directories to store results
            image_folder = f"../augmented_dataset/stage1/wafer{str(DATASET_NUMBER)}_crop"+str(i)+"/image/"
            os.makedirs(os.path.dirname(image_folder), exist_ok=True)
            tissue_masks_folder = f"../augmented_dataset/stage1/wafer{str(DATASET_NUMBER)}_crop"+str(i)+"/tissue_masks/"
            os.makedirs(os.path.dirname(tissue_masks_folder), exist_ok=True)
            magnetic_masks_folder = f"../augmented_dataset/stage1/wafer{str(DATASET_NUMBER)}_crop"+str(i)+"/magnetic_masks/"
            os.makedirs(os.path.dirname(magnetic_masks_folder), exist_ok=True)

            # creating tissue part mask
            ind_img = 0
            for section_index in tissue_indicies:
                vertices = np.array(seg_tissues.loc[[section_index]]).reshape((4, 2))
                #print(vertices)

                mask = generate_mask(wafer,vertices)
                _mask = Image.fromarray(np.uint8(mask))
                _mask = _mask.crop((start_x,start_y,start_x+section_size,start_y+section_size))


                # saving the cropped mask
                _mask.save(tissue_masks_folder+str(section_index)+".tif")
                ind_img = ind_img +1


            # creating magnetic part mask
            ind_img = 0
            for section_index in mag_indicies:
                vertices = np.array(seg_mag.loc[[section_index]], 'int32').reshape((4, 2))
                #print(vertices)

                mask = generate_mask(wafer,vertices)
                _mask = Image.fromarray(np.uint8(mask))
                _mask = _mask.crop((start_x,start_y,start_x+section_size,start_y+section_size))

                # saving the cropped mask
                _mask.save(magnetic_masks_folder+str(section_index)+".tif")
                ind_img = ind_img +1

            # saving the cropped image
            cropped_image.save(image_folder+"wafer"+str(DATASET_NUMBER)+"_crop"+str(i)+".tif")


# ----


def main():

    if len(sys.argv) < 5:
        if len(sys.argv) == 2 and sys.argv[1] == "help":
            print("""format: python3 labelimage.py <dataset number> <'rgb' / 'grayscale'> <nb artificial images> <nb batches per images> <section size> """)
            sys.exit()
        else:
            raise ValueError("Invalid number of arguments! Type help as arguments")

    if (sys.argv[1] == "1") or (sys.argv[1] == "2") or (sys.argv[1] == "3"):
        DATASET_NUMBER = int(sys.argv[1])
    else:
        raise ValueError("arg1 : Choose dataset number between 1, 2 or 3")

    if (sys.argv[2] == "rgb") :
        rgb = True
    elif(sys.argv[2] == "grayscale"):
        rgb = False
    else:
        raise ValueError("arg2 : Choose between 'rgb' or 'grayscale'")

    nb_artificial_images =  int(sys.argv[3])
    number_batches = int(sys.argv[4])

    if(len(sys.argv) == 6):
        section_size = int(sys.argv[5])
    else:
        section_size = 512



    if(rgb==True):
        path_rgb = "rgb"
    else:
        path_rgb="intensity"

    AUGMENTED_DATASET_PATH = "../augmented_dataset"
    WAFER_CROPPED_PATH = str(AUGMENTED_DATASET_PATH)+"/wafer_"+str(path_rgb)+"_cropped_"+str(DATASET_NUMBER)

    index = pd.MultiIndex.from_tuples([('point_1', 'x'), ('point_1', 'y'), ('point_2', 'x'), ('point_2', 'y'),
                                       ('point_3', 'x'), ('point_3', 'y'), ('point_4', 'x'), ('point_4', 'y')])

    seg_coord_tissues = pd.read_csv(str(WAFER_CROPPED_PATH)+"/boxes_tissues.txt", sep="\t|,", header=None, names=index, engine='python')
    seg_coord_mag = pd.read_csv(str(WAFER_CROPPED_PATH)+"/boxes_mag.txt", sep="\t|,", header=None, names=index, engine='python')

    path_background= str(AUGMENTED_DATASET_PATH)+"/background_"+str(path_rgb)+"_wafer"+str(DATASET_NUMBER)+".tif"
    if(rgb==True):
        backgnd = cv.imread(path_background)
    else:
        backgnd = cv.imread(path_background,0)



    if(DATASET_NUMBER == 1):
        max_num_section = 100
    elif(DATASET_NUMBER == 2):
        max_num_section = 35
    else:
        max_num_section = 69


    # Size of artificial image (grid_background*size of background) for height and width
    grid_background = 4


    temp_path_seg_tissues = str(AUGMENTED_DATASET_PATH)+"/artificial_images/seg_tissues_artif.txt"
    temp_path_seg_mag = str(AUGMENTED_DATASET_PATH)+"/artificial_images/seg_mag_artif.txt"

    random.seed(22) # Initialize random for the next methods in artifial images generation
    for index_artificial_image in range(nb_artificial_images):
        print("Generating Artificial Image "+str(index_artificial_image))
        res = create_artificial_images(index_artificial_image, DATASET_NUMBER, backgnd, grid_background, seg_coord_tissues, seg_coord_mag, rgb, max_num_section, temp_path_seg_tissues, temp_path_seg_mag )

        # Generate artificial batches
        if(rgb==True):
            wafer = Image.fromarray(np.uint8(res),'RGB')
        else:
            wafer = Image.fromarray(np.uint8(res))
        print("Create  "+str(number_batches)+ " batches from Artificial Image "+str(index_artificial_image))
        generate_artificial_batches(wafer, number_batches, index_artificial_image, DATASET_NUMBER, section_size, temp_path_seg_tissues, temp_path_seg_mag )


if __name__ == "__main__":
    main()
