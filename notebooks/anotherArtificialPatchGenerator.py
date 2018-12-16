# the current gen_artificial_img.py is working (that's good) but can be optimized/rewritten:
	# - the templates are not randomly rotated
	# - the collision boxes look cumbersome, instead we could throw masks and check with the opencv functions whether there is an increase in the number of white pixels equal to the number of white pixels in the mask (if yes, then no collision). These should only involve calls to the efficient functions of opencv that are in C, instead of doing custom checks in python
	# - no need to generate a large image that then needs to be cropped in subpatches: directly generate a small patch. It will be easier to parallelize

# the user provides for each section template
	# brain
	# mag
	# envelope
		# it would typically involve the nearby dummy. The envelope is used for collision checks

# Workflow
	# for throw in range(throws):
		# pick random template
		# randomly rotate template
		# randomly throw
			# extract the box in envIm (the current image keeping track of the thrown envelopes)
			# locally add the throw
			# check if collision
			# fill current enveloppes# Workflow

import json
import os
import random
import numpy as np
import cv2 as cv

rootFolder = r'C:\Collectome\Students\Docs\RawWafers\BIOPWafers\newWafersBIOP\Wafer_14'		
jsonPath = r'C:\Collectome\Students\Docs\RawWafers\BIOPWafers\newWafersBIOP\Wafer_14\stitched_BF_Test_small.json'


nThrows = 300
patchSize = np.array([600,600])
basketSize = (1.4 * patchSize).astype(int) # size of the large picture in which sections are thrown (sections are thrown into the basket). Will be cropped to patchSize after throwing.
throwSize = (1.2 * patchSize).astype(int) # size of the area in which the sections are thrown

# read templates
templates = {}

with open(jsonPath, 'r') as f:
	labelmeJson = json.load(f)
print(labelmeJson.keys())
imName = labelmeJson['imagePath']
fluoName = imName.replace('BF_Test', 'DAPI')

imPath = os.path.join(os.path.dirname(jsonPath), imName)
fluoPath = os.path.join(os.path.dirname(jsonPath), fluoName)

im = cv.imread(imPath)

# templates['n']['mag/tissue/envelope/bbox/centroid']['points/im/mask']
sections = zip(*(iter(labelmeJson['shapes']),) * 3)
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

	templates[id]['centroid'] = np.mean(templates[id]['e']['points'].T, axis=1).astype(int)

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

	# cv.imshow('image',imMasked)
	# cv.waitKey(0)
	# cv.imshow('image',tMask)
	# cv.waitKey(0)
	# cv.imshow('image',mMask)
	# cv.waitKey(0)
	# cv.imshow('image',eMask)
	# cv.waitKey(0)
	# cv.destroyAllWindows()		
		
# populate nOrientations for each template

# templates['n']['mag/tissue/envelope/bbox/centroid']['points/im/mask']
for idThrow in range(nThrows):
	basket = np.zeros(basketSize, np.uint8)
	
	# pick random template with a random orientation
	template = templates[random.choice(list(templates.keys()))]
	
	# randomly throw
	x = random.randint((patchSize[0]-throwSize[0])/2, patchSize[0] - (patchSize[0]-throwSize[0])/2)
	y = random.randint((patchSize[1]-throwSize[1])/2, patchSize[1] - (patchSize[1]-throwSize[1])/2)
				
	# extract the box in basket (the current image keeping track of the thrown envelopes)
	# locally add the throw
	# check if collision
	# fill current enveloppes