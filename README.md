# tissue-parts-detection
Detecting tissue and magnet parts using Mask R-CNN 

# Note
When cloning the repo, use recursive method in order to download data from submodule as well:  
`$ git clone https://github.com/BrainSegmentation/tissue-parts-detection.git --recursive`

__Augmented Dataset__

https://mega.nz/#!lG5m1SbK!A3HqaLfcyfnqYJNgVBFrRBocQAIngIQi1W9mbtQf8R4

# Report
[Read-Only link to Overleaf](https://www.overleaf.com/read/cqhkjdbxtmbr)

# Milestones

## Milestone 1  
_Deadline:_ 19.11.
- Reading materials related to Tissue Segmentation and Mask R-CNN 
- Diving into the data labeling
- Making training (validation) and testing datasets?

## Milestone 2
_Deadline:_ 26.11. 
- Explore existing projects and getting the overview of the results
- Creation of section (mag + brain parts) coordinates (txt file) 
- Rotation Algo : Different Angles and Center of Rotation
- First trial : Apply segmentation modesl on our data
- Create Docker Image (Tensorflow / MAsk RCNN)

## Milestone 3
_Deadline:_ 3.12. 
- From these sections : Create straight boxes of 1 section per boxes (+margin) --> For Detection Model beginning (Friday)
- Saturation algo (create different images by changing color  : darker/lighter)
- Create/clean full dataset of boxes for Detection Model
- Create Detection Model 
- Binary representation of fluorescent images of magnetic part (white on magnetic part)

## Milestone 4
_Deadline:_ 10.12.
- Another iteration of Machine Learning pipeline
- Implementing GUI for proofreading (webpage)
- Code documentation

## Milestone 5
_Deadline:_ 17.12.
- Report 
- Code documentation (final touches)
