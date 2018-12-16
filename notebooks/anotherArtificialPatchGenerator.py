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
import scipy.spatial.distance
import numpy as np
import cv2 as cv

rootFolder = r'C:\Collectome\Students\Docs\RawWafers\BIOPWafers\newWafersBIOP\Wafer_14'
jsonPath = r'C:\Collectome\Students\Docs\RawWafers\BIOPWafers\newWafersBIOP\Wafer_14\stitched_BF_Test_small.json'

nThrows = 1000
patchSize = np.array([300,800])
basketSize = (1.4 * patchSize).astype(int) # size of the large picture in which sections are thrown (sections are thrown into the basket). Will be cropped to patchSize after throwing.
# throwSize = (1.2 * patchSize).astype(int) # size of the area in which the sections are thrown

# read templates
# templates['n']['mag/tissue/envelope/bbox/centroid']['points/im/mask']
templates = {}

with open(jsonPath, 'r') as f:
	labelmeJson = json.load(f)

imName = labelmeJson['imagePath']
fluoName = imName.replace('BF_Test', 'DAPI')
imPath = os.path.join(os.path.dirname(jsonPath), imName)
fluoPath = os.path.join(os.path.dirname(jsonPath), fluoName)
im = cv.imread(imPath)

sections = zip(*(iter(labelmeJson['shapes']),) * 3) # 3 types of manual input: tissue, magnet, envelope
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
	
	templates[id]['eSize'] = cv.countNonZero(eMask)

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
successfulThrows = []
basket = np.zeros(basketSize, np.uint8)
for idThrow in range(nThrows):
	# centroids = [np.array([0,0]).astype(int)] # centroids of thrown sections (simpler to initialize with 0,0)
	
	# pick random template with a random orientation
	templateId = random.choice(list(templates.keys()))
	template = templates[templateId]
	bbox = template['bbox']
	# throwSize = basketSize - np.array([bbox[2], bbox[3]])
	throwSize = basketSize - np.array([bbox[3], bbox[2]])
	# randomly throw
	x = random.randint(0, throwSize[1])
	y = random.randint(0, throwSize[0])
	
	# # quick brute force check with the existing centroids
	# dists = scipy.spatial.distance.cdist(centroids, np.array([[x,y]]))
	# if min(dists) > template['bbox'][2] + template['bbox'][3]
	
	# extract the box in basket (the current image keeping track of the thrown envelopes)
	basketBox = basket[y:y+bbox[3], x:x+bbox[2]] # attention with x,y flip
	# # cv.imshow('image',basketBox)
	# # cv.waitKey(0)
	currentWhite = cv.countNonZero(basketBox)
	print('basketBox.shape', basketBox.shape, "template['e']['mask'].shape", template['e']['mask'].shape)
	print(x, y, idThrow)
	basketBoxAfterThrow = cv.add(basketBox, template['e']['mask'])
	# # cv.imshow('image',basketBoxAfterThrow)
	# # cv.waitKey(0)
	whiteAfterThrow = cv.countNonZero(basketBoxAfterThrow)
	if whiteAfterThrow - currentWhite == template['eSize']:
		print('no collision', x, y, idThrow)
		basket[y:y+bbox[3], x:x+bbox[2]] = basketBoxAfterThrow
		successfulThrows.append([templateId, x, y])
cv.imshow('image',basket)
cv.waitKey(0)
cv.destroyAllWindows()

# generate tissueMasks, magMasks, artiPatch, artiFluo,

magMask = np.zeros(basketSize, np.uint8)
tissueMask = np.zeros(basketSize, np.uint8)
for templateId, x, y in successfulThrows:
	bbox = templates[templateId]['bbox']
	
	magMask[y:y+bbox[3], x:x+bbox[2]] = cv.add(magMask[y:y+bbox[3], x:x+bbox[2]], templates[templateId]['m']['mask'])
cv.imshow('image', magMask)
cv.waitKey(0)
cv.destroyAllWindows()
	