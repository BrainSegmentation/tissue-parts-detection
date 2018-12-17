# the current gen_artificial_img.py is working (that's good) but can be optimized/rewritten:
	# - the templates are not randomly rotated
	# - the collision boxes look cumbersome, instead we could throw masks and check with the opencv functions whether there is an increase in the number of white pixels equal to the number of white pixels in the mask (if yes, then no collision). These should only involve calls to the efficient functions of opencv that are in C, instead of doing custom checks in python
	# - no need to generate a large image that then needs to be cropped in subpatches: directly generate a small patch. It will be easier to parallelize

# the user provides
	# for each section template
		# brain
		# mag
		# envelope
			# it would typically involve the nearby dummy. The envelope is used for collision checks
	# a background area free of sections

import json
import os
import random
import pathlib
import copy
import numpy as np
import cv2 as cv

jsonPath = r'C:\Collectome\Students\Docs\RawWafers\BIOPWafers\newWafersBIOP\Wafer_14\stitched_BF_Test_small.json'
rootFolder = os.path.dirname(jsonPath)
artificialFolder = os.path.join(rootFolder, 'artificialData')
pathlib.Path(artificialFolder).mkdir(exist_ok=True)

nPatches = 100
nThrows = 1000

patchSize = np.array([600,600])
basketSize = (1.4 * patchSize).astype(int) # size of the large picture in which sections are thrown (sections are thrown into the basket). Will be cropped to patchSize after throwing.
offset = (basketSize - patchSize)//2

with open(jsonPath, 'r') as f:
	labelmeJson = json.load(f)
sections = zip(*(iter(labelmeJson['shapes'][:-1]),) * 3) # 3 types of manual input: tissue, magnet, envelope

imName = labelmeJson['imagePath']
fluoName = imName.replace('BF_Test', 'DAPI')
imPath = os.path.join(rootFolder, imName)
fluoPath = os.path.join(rootFolder, fluoName)
######################################
# populate templates: points, masks, im, envelope area, 
######################################
templates = {}
im = cv.imread(imPath)
imFluo = cv.imread(fluoPath,0)
for id,[t,m,e] in enumerate(sections): # tissue, magnet, envelope
	templates[id] = {}

	bbox = cv.boundingRect(np.array(e['points'])) # the bounding box of the envelope
	templates[id]['bbox'] = bbox

	templates[id]['t'] = {}
	templates[id]['t']['points'] = np.array(t['points']) - np.array([bbox[0], bbox[1]])
	
	templates[id]['m'] = {}
	templates[id]['m']['points'] = np.array(m['points']) - np.array([bbox[0], bbox[1]])

	templates[id]['e'] = {}
	templates[id]['e']['points'] = np.array(e['points']) - np.array([bbox[0], bbox[1]])

	eMask = np.zeros((bbox[3], bbox[2]), dtype=np.uint8) # envelope mask
	cv.fillConvexPoly(eMask, templates[id]['e']['points'], 255)
	templates[id]['e']['mask'] = eMask
		
	tMask = np.zeros((bbox[3], bbox[2]), dtype=np.uint8) # tissue mask
	cv.fillConvexPoly(tMask, templates[id]['t']['points'], 255)
	templates[id]['t']['mask'] = tMask

	mMask = np.zeros((bbox[3], bbox[2]), dtype=np.uint8) # magnetic mask
	cv.fillConvexPoly(mMask, templates[id]['m']['points'], 255)
	templates[id]['m']['mask'] = mMask
	
	imCropped = im[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
	imMasked = cv.bitwise_and(imCropped, imCropped, mask = eMask)
	templates[id]['e']['im'] = imMasked

	fluoCropped = imFluo[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
	fluoMasked = cv.bitwise_and(fluoCropped, fluoCropped, mask = eMask)
	templates[id]['e']['fluo'] = fluoMasked
	
	templates[id]['eSize'] = cv.countNonZero(eMask)

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

######################################
# populate nOrientations for each template
######################################

###############
# Populate backgrounds and fluoBackgrounds from the labeled background area 
###############
backgrounds = []
backgroundsFluo = []

backgroundPoints = np.array(labelmeJson['shapes'][-1]['points'])
backgroundBbox = cv.boundingRect(backgroundPoints)
bx = backgroundBbox[0] + backgroundBbox[2]
by = backgroundBbox[1] + backgroundBbox[3]
shiftSize = (basketSize * 0.2).astype(int) # how much does the window slide to generate the background (these are subpatches of the manually selected (large) background area)
for x in range(10):
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

#########################
# Main loop through the patches to generate
#########################
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
			print('no collision', x, y, idThrow)
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
	
	imPath = os.path.join(imageFolder, 'patch_' + str(idPatch).zfill(4) + '.tif')
	imPatch = copy.deepcopy(backgrounds[backgroundId])
	
	fluoPath = os.path.join(imageFolder, 'patch_' + str(idPatch).zfill(4) + '_fluo.tif')
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

		# to add an image to a background, first the background is masked with the invert of the local envelope
		eMask = template['e']['mask']
		eMaskInvert = cv.bitwise_not(eMask)
		
		imPatchBox = imPatch[y:y+bbox[3], x:x+bbox[2]]
		imPatchBox = cv.bitwise_and(imPatchBox, imPatchBox, mask = eMaskInvert)
		imPatch[y:y+bbox[3], x:x+bbox[2]] = cv.add(imPatchBox, template['e']['im'])

		fluoPatchBox = fluoPatch[y:y+bbox[3], x:x+bbox[2]]
		fluoPatchBox = cv.bitwise_and(fluoPatchBox, fluoPatchBox, mask = eMaskInvert)
		fluoPatch[y:y+bbox[3], x:x+bbox[2]] = cv.add(fluoPatchBox, template['e']['fluo'])
		
	fluoPatch = fluoPatch[offset[1]:patchSize[1]+offset[1], offset[0]:patchSize[0]+offset[0]] # crop to patchSize
	cv.imwrite(fluoPath, fluoPatch)	
	
	imPatch = imPatch[offset[1]:patchSize[1]+offset[1], offset[0]:patchSize[0]+offset[0]] # crop to patchSize
	cv.imwrite(imPath, imPatch)