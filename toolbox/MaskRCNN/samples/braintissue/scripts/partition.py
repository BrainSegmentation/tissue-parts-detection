import sys, os
import glob
from PIL import Image

# Generates boxes which, together, partition the image into ~k**2 parts
def getPartitionBoxes(img, k):
    img_width, img_height = img.size
    step_width, step_height = img_width // k, img_height // k

    bounding_boxes = []
    for i in range(0, img_width, step_width):
        for j in range(0, img_height, step_height):
            box = (i, j, i + step_width, j + step_height)
            bounding_boxes.append(box)
    return bounding_boxes

# Partitions an image
def crop(in_path, k):
    img = Image.open(in_path)
    for box in getPartitionBoxes(img, k):
        yield img.crop(box)


def cropToBox(in_path, box):
    img = Image.open(in_path)
    return img.crop(box)


def mkdir(dir):
    try:
        os.makedirs(dir)
    except FileExistsError:
        pass


# Takes partition (Image), path, section number, and file name
def save_partition(img, out_path, name):
    path = os.path.join(out_path, name)
    mkdir(out_path)
    img.save(path)
    

# Takes in_path, out_path, k (num of partitions is k**2)
def main():
    usage = 'Usage: partition.py <in_path> <mask_path> <out_path> <k>, out and k optional'
    if len(sys.argv) > 4 or len(sys.argv) < 2:
        print(usage)
        sys.exit()
    # Defaults 
    in_path = './' + sys.argv[1]    # wafer file location
    mask_path = './' + sys.argv[2]  # mask files location
    out_path = './partitions/'      # output directory path
    k = 1                           # sqrt of # of partitions
    # User Input
    if len(sys.argv) == 4:
        out_path = sys.argv[3]
    if len(sys.argv) == 5:
        out_path = sys.argv[3]
        k = int(sys.argv[4])

    # Get locations of all mask files
    mask_paths = glob.glob(mask_path + '*.png')
    mask_paths.extend(glob.glob(mask_path + '*.bmp'))
    print('Found {} masks'.format(len(mask_paths)))

    # Generate cropping bounding boxes
    img = Image.open(in_path)
    bounding_boxes = getPartitionBoxes(img, k)

    # Crop all images according to each bounding box
    counter = 0
    print('Partitioning Image(s) into {} parts'.format(len(bounding_boxes)))
    for i, box in enumerate(bounding_boxes):
        sub_counter = 0;
        # Make paths for current partition
        section_str = 'section-{}'.format(i)
        section_path = os.path.join(out_path, section_str)
        images_path = os.path.join(section_path, 'images')
        masks_path = os.path.join(section_path, 'masks')

        name = section_str + '.png'
        wafer_sub = cropToBox(in_path, box)
        save_partition(wafer_sub, images_path, name) # saves to images/
        counter += 1        
        sub_counter += 1
        for j, mask in enumerate(mask_paths):
            name = section_str + '-mask-{}.png'.format(j)
            mask_sub = cropToBox(mask, box)
            save_partition(mask_sub, masks_path, name)  # saves to masks/
            counter += 1
            sub_counter += 1
             
        print('Wrote {0} images to {1}'.format(sub_counter, section_path))
        sub_counter = 0

    print('Saved {0} images to {1}'.format(counter, out_path))


if __name__ == '__main__':
    main()
