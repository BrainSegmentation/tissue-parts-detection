### Github Link :
https://github.com/BrainSegmentation/tissue-parts-detection

# Tissue Part Detection

Detecting brain tissue and magnet parts using [Mask R-CNN](https://github.com/matterport/Mask_RCNN). 

## Getting Started

### Clone
When cloning the repo, use recursive method in order to download data from submodule as well:  
`$ git clone https://github.com/BrainSegmentation/tissue-parts-detection.git --recursive`

### Cloud
Cloud used for Machine Learning pipline is [Paperspace](https://www.paperspace.com/).  
The machine used had following characteristics:
- RAM: **30 GB**
- CPUs: **8**
- HD: **100 GB**
- GPU: **8 GB**

## Run

### Docker
Docker image used is [nvidia-docker](https://github.com/NVIDIA/nvidia-docker), and sometimes [waleedka/modern-deep-learning
](https://hub.docker.com/r/waleedka/modern-deep-learning).

Following command will mount the files from local machine location (left `~/Documents`) to the docker one (right `~/Documents`):
`sudo nvidia-docker run -it -v ~/Documents:/Documents -p 8888:8888 brainsegmentation bash`

### Train

Train a new model using train dataset starting from specific weights file:

`$python brain.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5`

Train a new model starting from specific weights file using the full `stage1_train` dataset:

`$python brain.py train --dataset=/path/to/dataset --subset=stage1_train --weights=/path/to/weights.h5`

Resume training a model that you had trained earlier:

`$python brain.py train --dataset=/path/to/dataset --subset=train --weights=last`

### Jupyter Notebooks

Inside docker container, run Jupyter Notebook server with following command:
`$jupyter notebook --allow-root`

Open the link in your local computer:
`$ipaddress:8888`

## Links

### Training - Test Dataset 
https://mega.nz/#!amo2hYTQ!p0QQQCUAaBEAAhcQ7S6VGyHXEL_66J32FL-vKzF5zKA

### Weights of Braintissue Model (40 epochs)
https://mega.nz/#!3mRTzKbB!rEpygnbG0WGdEysMNa8ULzcuu_AsfuM8PI2SHZs9F0w

### Weights of Resnet50 backbone
https://mega.nz/#!KzBXGC6Q!Sae8SI-7kjzGY3L5IdF7A9KQrcSxSByj8-bCKMjzm4M

### Base Images for Augmented Artificial Images
https://mega.nz/#!n3oSyCiD!yJ4rbm5hgNGH-MgoRQTPs2cn8q3yY6PbiliOWON32kc

### Report
[Read-Only link to Report](https://www.overleaf.com/read/cqhkjdbxtmbr)

### Project
GitHub Organization project [BrainSegmentation/tissue-parts-detection](https://github.com/BrainSegmentation/tissue-parts-detection)

### Initial Dataset
Github Organization initial dataset [BrainSegmentation/section-segmentation-dataset](https://github.com/BrainSegmentation/section-segmentation-dataset)

### Augmented Dataset

Training set : 
https://mega.nz/#!JCpw3IAb!2j91l1G2n5EbvPd3XkEZZNqA1R2VytwXhUrUYTGlm7k

Test set : 
https://mega.nz/#!obgCUAiQ!ebwNPfEdWKcFBFDvX2KP1gFtAjZH2OW3HSXlwwzppG4

### Jupyter Notebooks
- [Create Artificial Images](https://github.com/BrainSegmentation/tissue-parts-detection/blob/master/notebooks/create_artificial_images-full_batches.ipynb)
- [Create Crop Inference](https://github.com/BrainSegmentation/tissue-parts-detection/blob/master/notebooks/create_crop_inference.ipynb)
- [Create Extract](https://github.com/BrainSegmentation/tissue-parts-detection/blob/master/notebooks/create_extract.ipynb)
- [Create Stage1 Data](https://github.com/BrainSegmentation/tissue-parts-detection/blob/master/notebooks/create_stage1_data.ipynb)
- [Crop Images](https://github.com/BrainSegmentation/tissue-parts-detection/blob/master/notebooks/crop_images.ipynb)
- [Generate 3-channel Images](https://github.com/BrainSegmentation/tissue-parts-detection/blob/master/notebooks/generate-3-channel-images.ipynb)
- [Inspect Data](https://github.com/BrainSegmentation/tissue-parts-detection/blob/master/notebooks/inspect_data.ipynb)


## Milestones

### Milestone 1  
_Deadline:_ 19.11.
- Reading materials related to Tissue Segmentation and Mask R-CNN 
- Diving into the data labeling
- Making training (validation) and testing datasets?

### Milestone 2
_Deadline:_ 26.11. 
- Explore existing projects and getting the overview of the results
- Creation of section (mag + brain parts) coordinates (txt file) 
- Rotation Algo : Different Angles and Center of Rotation
- First trial : Apply segmentation modesl on our data
- Create Docker Image (Tensorflow / MAsk RCNN)

### Milestone 3
_Deadline:_ 3.12. 
- From these sections : Create straight boxes of 1 section per boxes (+margin) --> For Detection Model beginning (Friday)
- Saturation algo (create different images by changing color  : darker/lighter)
- Create/clean full dataset of boxes for Detection Model
- Create Detection Model 
- Binary representation of fluorescent images of magnetic part (white on magnetic part)

### Milestone 4
_Deadline:_ 10.12.
- Another iteration of Machine Learning pipeline
- Implementing GUI for proofreading (webpage)
- Code documentation

### Milestone 5
_Deadline:_ 17.12.
- Report 
- Code documentation (final touches)
