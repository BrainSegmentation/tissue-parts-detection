"""
The user manually labels the following ***in this order*** with the labelme GUI and saves a json file:
  for each section template (ideally they have slightly different looks):
    -tissue polygon (in labelme you can create a single label 'tissue' for example)
    -mag polygon (in labelme you can create a single label 'mag' for example)
    -envelope polygon : the envelope typically involves the nearby dummy. The envelope is used for collision checks when throwing sections
  a single *roughly rectangular* background area free of sections (should be larger than the final patch size)

Data structure required : in one folder you have:
  - the brightfield wafer overview that contains "BF_Test" in its name
  - the fluo wafer overview which has the same name as the brightfield channel but with "BF_Test" replaced by "DAPI"
  - the labelme json file saved from the GUI
  
Example call:
python artificialPatchGenerator.py yourJsonPath.json
python artificialPatchGenerator.py yourJsonPath.json -p 100 -t 2000 -a 30 -x 700 -y 700
"""
import json
import os
import random
import pathlib
import copy
import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import cv2 as cv

###################
# Parsing arguments
parser = argparse.ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
# parser.add_argument('-j', '--json', help='the labelme json file', required=True)
parser.add_argument('json', help='the labelme json file')
parser.add_argument('-p', '--patches', help='number of patches (default = 100)', required=False, default=100)
parser.add_argument('-t', '--throws', help='number of section throwing attempts (default = 2000). A small number gives a low section density', required=False, default=2000)
parser.add_argument('-a', '--angles', help='number of angles used to rotate the user templates (default = 30)', required=False, default=30)
parser.add_argument('-x', '--xpatch', help='size in pixel of the final patch in x (default = 512)', required=False, default=512)
parser.add_argument('-y', '--ypatch', help='size in pixel of the final patch in y (default = 512)', required=False, default=512)
args = vars(parser.parse_args())

jsonPath = os.path.normpath(args['json'])
nPatches = int(args['patches'])
nThrows = int(args['throws'])
nAngles = int(args['angles'])
xPatch = int(args['xpatch'])
yPatch = int(args['ypatch'])
patchSize = np.array([xPatch, yPatch])
###################

######################
# Some initializations
rootFolder = os.path.dirname(jsonPath)
artificialFolder = os.path.join(rootFolder, 'artificialData')
pathlib.Path(artificialFolder).mkdir(exist_ok=True)

basketSize = (1.4 * patchSize).astype(int) # size of the large picture in which sections are thrown (sections are thrown into the basket). Will be cropped to patchSize after throwing.
offset = (basketSize - patchSize)//2

with open(jsonPath, 'r') as f:
	labelmeJson = json.load(f)
sections = zip(*(iter(labelmeJson['shapes'][:-1]),) * 3) # 3 types of manual input: tissue, magnet, envelope

imName = labelmeJson['imagePath']
fluoName = imName.replace('BF_Test', 'DAPI')
imPath = os.path.join(rootFolder, imName)
fluoPath = os.path.join(rootFolder, fluoName)
######################

##################################################################################
# populate templates: points, masks, im, envelope area, etc. with different angles
templates = {}
im = cv.imread(imPath, 0)
imFluo = cv.imread(fluoPath, 0)
w,h = im.shape
angles = np.linspace(start=0, stop=360, num=nAngles)[:-1]
counter = 0

for t,m,e in sections: # tissue, magnet, envelope
	for angle in angles:
		# get a larger bbox around the template to make rotations
		bbox = cv.boundingRect(np.array(e['points'])) # the bounding box of the envelope
		largerBbox = np.array([bbox[0]-bbox[2], bbox[1]-bbox[3], 3*bbox[2], 3*bbox[3]])
		
		if largerBbox[0]>0 and largerBbox[1]>0 and largerBbox[0]+largerBbox[2]<w and largerBbox[1]+largerBbox[3]<h: # check that the larger bbox is still in the image
			templates[counter] = {}
			
			# rotation matrix
			M = cv.getRotationMatrix2D((largerBbox[2]/2, largerBbox[3]/2), angle, 1)
			
			# rotate the envelope points
			ePoints = np.array(e['points']) - np.array([largerBbox[0], largerBbox[1]])
			ePointsRotated = cv.transform(np.array([ePoints]), M)[0]
			
			# get new bounding box after rotation
			bboxAfterRotation = cv.boundingRect(np.array(ePointsRotated))			
			templates[counter]['bbox'] = bboxAfterRotation
			
			# create envelope mask
			eMask = np.zeros((largerBbox[3], largerBbox[2]), dtype=np.uint8) # envelope mask
			cv.fillConvexPoly(eMask, ePointsRotated, 255)
			eMask = eMask[bboxAfterRotation[1]:bboxAfterRotation[1]+bboxAfterRotation[3], bboxAfterRotation[0]:bboxAfterRotation[0]+bboxAfterRotation[2]]
			
			# write envelope
			templates[counter]['e'] = {}
			templates[counter]['e']['mask'] = eMask
			templates[counter]['e']['points'] = np.array(ePointsRotated) - np.array([bboxAfterRotation[0], bboxAfterRotation[1]])
			templates[counter]['eSize'] = cv.countNonZero(eMask)

			# tissue
			tPoints = np.array(t['points']) - np.array([largerBbox[0], largerBbox[1]])
			tPointsRotated = cv.transform(np.array([tPoints]), M)[0]

			tMask = np.zeros((largerBbox[3], largerBbox[2]), dtype=np.uint8) # envelope mask
			cv.fillConvexPoly(tMask, tPointsRotated, 255)
			tMask = tMask[bboxAfterRotation[1]:bboxAfterRotation[1]+bboxAfterRotation[3], bboxAfterRotation[0]:bboxAfterRotation[0]+bboxAfterRotation[2]]
			
			templates[counter]['t'] = {}
			templates[counter]['t']['points'] = np.array(tPointsRotated) - np.array([bboxAfterRotation[0], bboxAfterRotation[1]])
			templates[counter]['t']['mask'] = tMask

			# magnet
			mPoints = np.array(m['points']) - np.array([largerBbox[0], largerBbox[1]])
			mPointsRotated = cv.transform(np.array([mPoints]), M)[0]

			mMask = np.zeros((largerBbox[3], largerBbox[2]), dtype=np.uint8) # envelope mask
			cv.fillConvexPoly(mMask, mPointsRotated, 255)
			mMask = mMask[bboxAfterRotation[1]:bboxAfterRotation[1]+bboxAfterRotation[3], bboxAfterRotation[0]:bboxAfterRotation[0]+bboxAfterRotation[2]]
			
			templates[counter]['m'] = {}
			templates[counter]['m']['points'] = np.array(mPointsRotated) - np.array([bboxAfterRotation[0], bboxAfterRotation[1]])
			templates[counter]['m']['mask'] = mMask

			# process the images
			imBoxed = im[largerBbox[1]:largerBbox[1]+largerBbox[3], largerBbox[0]:largerBbox[0]+largerBbox[2]]
			fluoBoxed = imFluo[largerBbox[1]:largerBbox[1]+largerBbox[3], largerBbox[0]:largerBbox[0]+largerBbox[2]]
			
			imRotated = cv.warpAffine(imBoxed, M, (largerBbox[2],largerBbox[3]))			
			fluoRotated = cv.warpAffine(fluoBoxed, M, (largerBbox[2],largerBbox[3]))
			
			imCropped = imRotated[bboxAfterRotation[1]:bboxAfterRotation[1]+bboxAfterRotation[3], bboxAfterRotation[0]:bboxAfterRotation[0]+bboxAfterRotation[2]]
			imMasked = cv.bitwise_and(imCropped, imCropped, mask = eMask)
			templates[counter]['e']['im'] = imMasked

			fluoCropped = fluoRotated[bboxAfterRotation[1]:bboxAfterRotation[1]+bboxAfterRotation[3], bboxAfterRotation[0]:bboxAfterRotation[0]+bboxAfterRotation[2]]
			fluoMasked = cv.bitwise_and(fluoCropped, fluoCropped, mask = eMask)
			templates[counter]['e']['fluo'] = fluoMasked

			counter = counter+1
			
			# cv.imshow('image',eMask)
			# cv.waitKey(0)
			# cv.imshow('image',tMask)
			# cv.waitKey(0)
			# cv.imshow('image',mMask)
			# cv.waitKey(0)
			# cv.imshow('image',imMasked)
			# cv.waitKey(0)
			# cv.imshow('image',fluoMasked)
			# cv.waitKey(0)
			# cv.destroyAllWindows()
##################################################################################

###########################################################################
# Populate backgrounds and fluoBackgrounds from the labeled background area 
backgrounds = []
backgroundsFluo = []

backgroundPoints = np.array(labelmeJson['shapes'][-1]['points'])
backgroundBbox = cv.boundingRect(backgroundPoints)
bx = backgroundBbox[0] + backgroundBbox[2]
by = backgroundBbox[1] + backgroundBbox[3]
shiftSize = (basketSize * 0.2).astype(int) # how much does the window slide to generate the background (these are subpatches of the manually selected (large) background area)
for x in range(10): # no need to bother with correct indices, just try 10x10 boxes and check whether they are in the image
	for y in range(10):
		x1 = backgroundBbox[0] + x*shiftSize[0]
		y1 = backgroundBbox[1] + y*shiftSize[1]
		x2 = x1 + basketSize[0]
		y2 = y1 + basketSize[1]
		
		if x1<bx and x2<bx and y1<by and y2<by:
			background = im[y1:y2,x1:x2]
			backgrounds.append(background)
			
			backgroundFluo = imFluo[y1:y2,x1:x2]
			backgroundsFluo.append(backgroundFluo)			
del im
del imFluo
###########################################################################

###########################################
# Main loop through the patches to be generated
for idPatch in range(nPatches):
	successfulThrows = []
	basket = np.zeros(basketSize, np.uint8) # container in which sections are thrown
	for idThrow in range(nThrows):
		# pick random template with a random orientation
		templateId = random.choice(list(templates.keys()))
		template = templates[templateId]
		bbox = template['bbox']
		throwSize = basketSize - np.array([bbox[3], bbox[2]]) # the thrown bounding boxes will be contained in the basket
		
		# randomly throw
		x = random.randint(0, throwSize[1])
		y = random.randint(0, throwSize[0])

		# extract the box in basket (the current image keeping track of the thrown envelopes)
		basketBox = basket[y:y+bbox[3], x:x+bbox[2]] # attention with x,y flip
		
		# count the number of white pixels before and after throwing
		currentWhite = cv.countNonZero(basketBox)
		basketBoxAfterThrow = cv.add(basketBox, template['e']['mask'])
		whiteAfterThrow = cv.countNonZero(basketBoxAfterThrow)
		if whiteAfterThrow - currentWhite == template['eSize']: # no collision
			# print('no collision', x, y, idThrow)
			basket[y:y+bbox[3], x:x+bbox[2]] = basketBoxAfterThrow
			successfulThrows.append([templateId, x, y])

	# generate tissueMasks, magMasks, artificialPatch, artificialFluo,
	patchFolder = os.path.join(artificialFolder, 'patch_' + str(idPatch).zfill(4))
	mFolder = os.path.join(patchFolder, 'magnetic_masks')
	tFolder = os.path.join(patchFolder, 'tissue_masks')
	imageFolder = os.path.join(patchFolder, 'image')

	#create folders
	pathlib.Path(patchFolder).mkdir(exist_ok=True)
	pathlib.Path(mFolder).mkdir(exist_ok=True)
	pathlib.Path(tFolder).mkdir(exist_ok=True)
	pathlib.Path(imageFolder).mkdir(exist_ok=True)

	# choose a background randomly
	backgroundId = random.randint(0, len(backgrounds)-1)
	imPatch = copy.deepcopy(backgrounds[backgroundId])
	fluoPatch = copy.deepcopy(backgroundsFluo[backgroundId])
	
	# create the images
	for throwId, [templateId, x, y] in enumerate(successfulThrows):
		template = templates[templateId]
		bbox = template['bbox']
		
		tissueMask = np.zeros(basketSize, np.uint8)
		magMask = np.zeros(basketSize, np.uint8)
		
		magMaskPath = os.path.join(mFolder, str(throwId).zfill(2) + '.tif')
		magMask[y:y+bbox[3], x:x+bbox[2]] = cv.add(magMask[y:y+bbox[3], x:x+bbox[2]], template['m']['mask'])
		magMask = magMask[offset[1]:patchSize[1]+offset[1], offset[0]:patchSize[0]+offset[0]] # crop to patchSize
		cv.imwrite(magMaskPath, magMask)

		tissueMaskPath = os.path.join(tFolder, str(throwId).zfill(2) + '.tif')
		tissueMask[y:y+bbox[3], x:x+bbox[2]] = cv.add(tissueMask[y:y+bbox[3], x:x+bbox[2]], template['t']['mask'])
		tissueMask = tissueMask[offset[1]:patchSize[1]+offset[1], offset[0]:patchSize[0]+offset[0]] # crop to patchSize
		cv.imwrite(tissueMaskPath, tissueMask)

		# to add an image to a background, the background is first masked with the invert of the local envelope before adding the image
		eMask = template['e']['mask']
		eMaskInvert = cv.bitwise_not(eMask)
		
		imPatchBox = imPatch[y:y+bbox[3], x:x+bbox[2]]
		imPatchBox = cv.bitwise_and(imPatchBox, imPatchBox, mask = eMaskInvert)
		imPatch[y:y+bbox[3], x:x+bbox[2]] = cv.add(imPatchBox, template['e']['im'])

		fluoPatchBox = fluoPatch[y:y+bbox[3], x:x+bbox[2]]
		fluoPatchBox = cv.bitwise_and(fluoPatchBox, fluoPatchBox, mask = eMaskInvert)
		fluoPatch[y:y+bbox[3], x:x+bbox[2]] = cv.add(fluoPatchBox, template['e']['fluo'])
		
	fluoPath = os.path.join(imageFolder, 'patch_' + str(idPatch).zfill(4) + '_fluo.tif')
	imPath = os.path.join(imageFolder, 'patch_' + str(idPatch).zfill(4) + '.tif')
	
	fluoPatch = fluoPatch[offset[1]:patchSize[1]+offset[1], offset[0]:patchSize[0]+offset[0]] # crop to patchSize
	cv.imwrite(fluoPath, fluoPatch)	
	
	imPatch = imPatch[offset[1]:patchSize[1]+offset[1], offset[0]:patchSize[0]+offset[0]] # crop to patchSize
	cv.imwrite(imPath, imPatch)
###########################################