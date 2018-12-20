"""
gen_artificial_img.py
---
Script to automaticly generate an artificial dataset of patches with their associated masks to train the model on

- Kevin Pelletier
- Eliott Joulot
- Jelena Banjac
"""

import os
import sys
import numpy as np
import pandas as pd
import random
import cv2 as cv
from PIL import Image

## How to run the script:
# python gen_artificial_img.py <Dataset_Number> <'rgb' or 'grayscale'> <nb of artificial images> <nb of batches> <size of produced batches images>
# Nb of artificial images will generate the number of images (not saved)
# and for each images will generate the nb of batches (of size 512 by default) associated with the masks


"""
Function to create a collision box around a section
This is done by slicing our section image in N (number of boxes), and for each of them, draw a box that best fits the sliced section.
Higher N will thus allow more precision on the section representation.
"""
def get_collision_boxes(orig_section,nb_boxes = 4, rgb=False, draw = False):
    #Divide the section into 5 block
    if(rgb == True):
        section = np.mean(orig_section,axis=2)
    else:
        section = orig_section

    height_section = section.shape[0]
    width_section = section.shape[1]


    limit_boxes = np.zeros([nb_boxes,4],dtype=np.uint16)

    cut_height = (height_section//nb_boxes)+1 # Slicing the section in N boxes to create a specific number of boxes
    for cuts in range(nb_boxes):
        # for each slice, we will find the smallest box that fits it
        i=0

        if((cuts)*cut_height + 4 >= height_section):
            limit_boxes[cuts] = limit_boxes[cuts-1]

        else:
            ymax_box = (cuts+1)*cut_height
            if(ymax_box >= height_section):
                ymax_box = height_section-1
                temp_cut_height = height_section - cut_height*(cuts)
            else:
                temp_cut_height = cut_height


            # y Min
            for i in range(temp_cut_height):
                if( (np.mean( section[i + cuts*cut_height] ) != 0)):
                    limit_boxes[cuts,0] = i+cuts*cut_height
                    break
            # y Max

            for i in range(cut_height):
                if(np.mean(section[ymax_box-i-1]) != 0):
                    limit_boxes[cuts,1] = ymax_box-i
                    break
            # x Min
            for i in range(width_section):
                if( np.mean( section[ cuts*cut_height+1 : ymax_box , i] ) != 0):
                    limit_boxes[cuts,2] = i
                    break
            # x Max
            for i in range(width_section-1,0,-1):
                if(np.mean(section[cuts*cut_height+1:ymax_box,i]) != 0):
                    limit_boxes[cuts,3] = i
                    break

            # Prevent empty slices
            if(np.mean(limit_boxes[cuts]) == 0 ):
                limit_boxes[cuts] = limit_boxes[cuts - 1]
            if(limit_boxes[cuts,2] == limit_boxes[cuts,3] ):
                limit_boxes[cuts,3] = limit_boxes[cuts,3]+1

            for i in range(min(cuts,5)):
                if(np.mean(limit_boxes[cuts-(i+1)]) <= 1 ):
                    limit_boxes[cuts-(i+1)] = limit_boxes[cuts]



            if((limit_boxes[cuts,2] <= 1) and  (limit_boxes[cuts,3] <= 1)):
                limit_boxes[cuts] = limit_boxes[cuts - 1]




        if(draw):
            pts = np.array([limit_boxes[cuts,2],limit_boxes[cuts,0], limit_boxes[cuts,3],limit_boxes[cuts,0], limit_boxes[cuts,3],limit_boxes[cuts,1], limit_boxes[cuts,2],limit_boxes[cuts,1] ] )
            pts = pts.reshape((-1,1,2))
            section = cv.polylines(section, np.int32([pts]), True, color=compute_rgb(tissue_color), thickness=1)
            plt.imshow(section)
    return limit_boxes

"""
Function to generate the masks images from the input image (to get the size) and the associated masks points
"""
def generate_mask(image, points):
    # size of the image
    height = image.size[1]
    width = image.size[0]

    # init mask
    img_mask = np.zeros([height,width],dtype=np.uint8)
    img_mask.fill(0)

    cv.fillConvexPoly(img_mask, points, 255)

    return img_mask

"""
Function to generate multiple crop patches from a generated artificial image
It will then save the produced patches following the training structure, with the associated masks
"""
def generate_artificial_patches(wafer, nb_patch, iter_img, DATASET_NUMBER, section_size, temp_path_seg_tissues, temp_path_seg_mag  ):

    index = pd.MultiIndex.from_tuples([('point_1', 'x'), ('point_1', 'y'), ('point_2', 'x'), ('point_2', 'y'),
                                   ('point_3', 'x'), ('point_3', 'y'), ('point_4', 'x'), ('point_4', 'y')])

    seg_tissues = pd.read_csv(temp_path_seg_tissues, sep="\t|,", header=None, names=index, engine='python')
    seg_mag = pd.read_csv(temp_path_seg_mag, sep="\t|,", header=None, names=index, engine='python')

    for i in range(1,nb_patch+1):


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
            image_folder = "../augmented_dataset/stage1/artif"+str(DATASET_NUMBER)+"_"+str(iter_img)+"_crop"+str(i)+"/image/"
            os.makedirs(os.path.dirname(image_folder), exist_ok=True)
            tissue_masks_folder = "../augmented_dataset/stage1/artif"+str(DATASET_NUMBER)+"_"+str(iter_img)+"_crop"+str(i)+"/tissue_masks/"
            os.makedirs(os.path.dirname(tissue_masks_folder), exist_ok=True)
            magnetic_masks_folder = "../augmented_dataset/stage1/artif"+str(DATASET_NUMBER)+"_"+str(iter_img)+"_crop"+str(i)+"/magnetic_masks/"
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
            path_img_tosave = str(image_folder)+"artif"+str(DATASET_NUMBER)+"_"+str(iter_img)+"_crop"+str(i)+".tif"
            cropped_image.save(path_img_tosave)


"""
Function to check if the possible area to put a section is actually free of already inserted section
Takes as input the section and the x/y position to verify in the full image
"""
def check_avail_area(full_image,section,width, width_section, height, height_section, nb_collision_boxes, rgb):

    collision_boxes_section = get_collision_boxes(section, nb_collision_boxes, rgb, False)
    # Find an area in the image free (all pixels = 255)
    timeout = 0
    free_place = False
    while(free_place == False):
        xpos_section = random.randint(0, width - width_section- 1)
        ypos_section = random.randint(0, height - height_section - 1)

        free_place = True
        for i in range(nb_collision_boxes):
            #print(i)
            ymin = ypos_section+collision_boxes_section[i][0]
            ymax = ypos_section+collision_boxes_section[i][1]
            xmin = xpos_section+collision_boxes_section[i][2]
            xmax = xpos_section+collision_boxes_section[i][3]

            # check collision box
            if(rgb==True):
                if(np.mean(np.mean(full_image[ymin:ymax, xmin:xmax],axis=2)) != 0):
                    free_place = False
            else:
                if(np.mean(full_image[ymin:ymax, xmin:xmax]) != 0):
                    free_place = False

        if(timeout>10000):
            # end loop no more space
            break
        else:
            timeout= timeout+1

    return xpos_section, ypos_section, free_place

"""
Function to produce an artificial data, by randomly loading a section, randomly selecting a position in the image,
and check if the space is free. It will then iterate until the algorithm can't find any free spaces anymore
"""
def create_artificial_images(iter_img, DATASET_NUMBER, backgnd, grid_background, seg_coord_tissues, seg_coord_mag, rgb, max_num_section, temp_path_seg_tissues, temp_path_seg_mag ):

    if(rgb==True):
        path_rgb = "rgb"
    else:
        path_rgb="intensity"

    AUGMENTED_DATASET_PATH = "../augmented_dataset"
    WAFER_CROPPED_PATH = str(AUGMENTED_DATASET_PATH)+"/wafer_"+str(path_rgb)+"_cropped_"+str(DATASET_NUMBER)

    # Initialize the artificial image as zeros with a size corresponding to a grid of loaded background_
    # Ex : If grid_background is 4 and the background shape is 100x100 then the resulting image will be 400x400
    height, width = backgnd.shape[0]*grid_background, backgnd.shape[1]*grid_background
    if(rgb==True):
        full_image = np.zeros([height, width,3],dtype=np.uint8)
    else:
        full_image = np.zeros([height, width],dtype=np.uint8)
    full_image.fill(0)
    nb_sections = 150

    f_seg_tissues_artif = open(temp_path_seg_tissues,"w+")
    f_seg_mag_artif= open(temp_path_seg_mag,"w+")

    for index_section in range(nb_sections):

        # Section selection
        # section_num will be use to load the data image, and to load the corrects segmentation boxes

        free_place = False
        try_section = 0
        while((free_place ==False) & (try_section < 5)):
            section_num = random.randint(0, max_num_section)
            path_img= str(WAFER_CROPPED_PATH)+"/extract/"+str(section_num)+".tif"
            if(rgb==True):
                section = cv.imread(path_img)
            else:
                section = cv.imread(path_img,0)

            height_section = section.shape[0]
            width_section = section.shape[1]

            # Collision boxes to not interfere with other sections
            nb_collision_boxes = 40

            xpos_section, ypos_section, free_place = check_avail_area(full_image,section, width, width_section, height, height_section, nb_collision_boxes, rgb)
            try_section += 1

        if(free_place == False):
            # No more space
            break

        # Integrate the section into the full image
        for i in range(height_section):
            for j in range(width_section):
                if(rgb==True):
                    if(np.mean(section[i,j]) != 0):
                        full_image[ypos_section+i,xpos_section+j] = section[i,j]
                else:
                    if(section[i,j] != 0):
                        full_image[ypos_section+i,xpos_section+j] = section[i,j]

        #Store the new segmentation position Brain tissues in txt file
        temp_tissue = seg_coord_tissues.iloc[section_num]
        write_coordinates_file(temp_tissue,xpos_section, ypos_section, f_seg_tissues_artif)

        #Store the new segmentation position Mag in txt file
        temp_mag = seg_coord_mag.iloc[section_num]
        write_coordinates_file(temp_mag,xpos_section, ypos_section, f_seg_mag_artif)


    f_seg_tissues_artif.close()
    f_seg_mag_artif.close()

    # Add background
    width_subbckgnd = backgnd.shape[1]
    height_subbckgnd = backgnd.shape[0]

    for ind_y in range(grid_background):
        for ind_x in range(grid_background):

            for i in range(backgnd.shape[0]):
                for j in range(backgnd.shape[1]):
                    if(np.mean(full_image[height_subbckgnd*ind_y + i , width_subbckgnd*ind_x + j]) == 0):
                        full_image[height_subbckgnd*ind_y + i, width_subbckgnd*ind_x + j] = backgnd[i,j]
    if(rgb==True):
        full_image = cv.cvtColor(full_image, cv.COLOR_BGR2RGB)

    return full_image

"""
Function called by the create_artificial_images to write the coordinates of each section in a file
Each artificial image will have a temporary file, which will be used by the patch_generator to produce the cropped images
with the associated masks
"""
def write_coordinates_file(coordinate_section,xpos_section, ypos_section, file_towrite):

        seg_p1_x = coordinate_section['point_1']['x'] + xpos_section
        seg_p1_y = coordinate_section['point_1']['y'] + ypos_section

        seg_p2_x = coordinate_section['point_2']['x'] + xpos_section
        seg_p2_y = coordinate_section['point_2']['y'] + ypos_section

        seg_p3_x = coordinate_section['point_3']['x'] + xpos_section
        seg_p3_y = coordinate_section['point_3']['y'] + ypos_section

        seg_p4_x = coordinate_section['point_4']['x'] + xpos_section
        seg_p4_y = coordinate_section['point_4']['y'] + ypos_section

        file_towrite.write("%d,%d\t%d,%d\t%d,%d\t%d,%d\r\n" % (seg_p1_x,seg_p1_y,  seg_p2_x,seg_p2_y,  seg_p3_x,seg_p3_y,  seg_p4_x,seg_p4_y))


# ----


def main():


    # -------------------- Arguments input

    if len(sys.argv) < 5:
        if len(sys.argv) == 2 and sys.argv[1] == "help":
            print("""format: python3 gen_artificial_img.py <dataset number> <'rgb' / 'grayscale'> <nb artificial images> <nb patches per images> <section size> """)
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
    number_patches = int(sys.argv[4])

    if(len(sys.argv) == 6):
        section_size = int(sys.argv[5])
    else:
        section_size = 512

    # --------------------- End of args input

    if(rgb==True):
        path_rgb = "rgb"
    else:
        path_rgb="intensity"

    AUGMENTED_DATASET_PATH = "../augmented_dataset"
    WAFER_CROPPED_PATH = str(AUGMENTED_DATASET_PATH)+"/wafer_"+str(path_rgb)+"_cropped_"+str(DATASET_NUMBER)

    index = pd.MultiIndex.from_tuples([('point_1', 'x'), ('point_1', 'y'), ('point_2', 'x'), ('point_2', 'y'),
                                       ('point_3', 'x'), ('point_3', 'y'), ('point_4', 'x'), ('point_4', 'y')])

    #Read the coordinates masks for all the sections to the associated dataset
    seg_coord_tissues = pd.read_csv(str(WAFER_CROPPED_PATH)+"/boxes_tissues.txt", sep="\t|,", header=None, names=index, engine='python')
    seg_coord_mag = pd.read_csv(str(WAFER_CROPPED_PATH)+"/boxes_mag.txt", sep="\t|,", header=None, names=index, engine='python')

    path_background= str(AUGMENTED_DATASET_PATH)+"/background_"+str(path_rgb)+"_wafer"+str(DATASET_NUMBER)+".tif"
    if(rgb==True):
        backgnd = cv.imread(path_background)
    else:
        backgnd = cv.imread(path_background,0)


    # Based on the currently extracted/segmented sections images available
    if(DATASET_NUMBER == 1):
        max_num_section = 100
    elif(DATASET_NUMBER == 2):
        max_num_section = 35
    else:
        max_num_section = 69


    # Size of artificial image (grid_background*size of background) for height and width
    grid_background = 4

    # Temporary file which will be written by create_artificial_images and read by generate_artificial_patches
    temp_path_seg_tissues = str(AUGMENTED_DATASET_PATH)+"/artificial_images/seg_tissues_artif_"+str(DATASET_NUMBER)+".txt"
    temp_path_seg_mag = str(AUGMENTED_DATASET_PATH)+"/artificial_images/seg_mag_artif_"+str(DATASET_NUMBER)+".txt"

    random.seed(22) # Initialize random for the next methods in artifial images generation
    for index_artificial_image in range(nb_artificial_images):
        print("Generating Artificial Image "+str(index_artificial_image))
        res = create_artificial_images(index_artificial_image, DATASET_NUMBER, backgnd, grid_background, seg_coord_tissues, seg_coord_mag, rgb, max_num_section, temp_path_seg_tissues, temp_path_seg_mag )

        # Generate artificial batches
        if(rgb==True):
            wafer = Image.fromarray(np.uint8(res),'RGB')
        else:
            wafer = Image.fromarray(np.uint8(res))
        print("Create  "+str(number_patches)+ " batches from Artificial Image "+str(index_artificial_image))
        generate_artificial_patches(wafer, number_patches, index_artificial_image, DATASET_NUMBER, section_size, temp_path_seg_tissues, temp_path_seg_mag )


if __name__ == "__main__":
    main()
