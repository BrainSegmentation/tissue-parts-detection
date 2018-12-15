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
			# fill current enveloppes