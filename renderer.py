import scipy.io as scio
import sklearn.metrics
import cv2
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(formatter={'float_kind': lambda x: "%.4f" % x})

class Params:
    ACC_CONST = 2e3#5e5 #<--- obsolete
    ksize_acc = 15#15
    ksize_weight = 33#33

def badIndex(project, img):
    bad = np.logical_or(project.min(axis=0) < 1, project[1, :] > img.shape[0])
    bad = np.logical_or(bad, project[0, :] > img.shape[1])
    bad = np.asarray(bad).reshape((-1), order='F')
    return bad

def warpImg(img, t_height, t_width, prj, idx):
    new_img = np.zeros((t_height*t_width, 3))
    ## In case we have some points
    if prj.size != 0:
        pixels = cv2.remap(img, np.squeeze( np.asarray( prj[0,:] ) ).astype('float32'),\
         np.squeeze( np.asarray( prj[1,:] ) ).astype('float32'),  cv2.INTER_CUBIC)
        pixels = pixels[:,0,:]
        new_img[idx,:] = pixels
    else:
        print '> Projected points empty'
    new_img = new_img.reshape(( t_height, t_width, 3), order='F')
    new_img[new_img > 255] = 255
    new_img[new_img < 0] = 0
    new_img = new_img.astype('uint8')
    return new_img

def NormalizePoints(out_proj):
    maxOut = out_proj.max(axis=1)
    minOut = out_proj.min(axis=1)
    lenn =maxOut-minOut
    ## Normalize the points somehow inside the image
    ## In theory here we can do better to avoid puttin in the background pixel of the faces
    den = maxOut-minOut
    den[den==0]=1
    out_proj = (out_proj-minOut)/den
    return out_proj, lenn

def UnnormalizePoints(out_proj, size):
    return np.multiply(out_proj,size.T)

def HandleBackground(out_proj,face_proj_in, img, opts):
    if out_proj.size != 0:
        out_proj,lenn = NormalizePoints(out_proj)
        widthX = lenn[1]
        heightY = lenn[0]
        thWidth = face_proj_in[0,:].min()/img.shape[1]*opts.getfloat('renderer','scaleFaceX')
        idxOveral =  np.nonzero(np.squeeze(np.asarray(out_proj[0,:]))>thWidth)[0]
        if idxOveral.size != 0:
        	out_proj[0,idxOveral] = out_proj[0,idxOveral]/out_proj[0,idxOveral].max()*thWidth
        # In case we want to skip the head and go in the right part of the face
        # diffX = out_proj[0,idxOveral]-thWidth#=thWidth
        # #print diffX
        # rempPts = thWidthMax + diffX[0,:]
        # rempPts, lenn = NormalizePoints(rempPts)
        # rempPts = face_proj_in[0,:].max()*1.1 + UnnormalizePoints(rempPts, img.shape[0]-face_proj_in[0,:].max()*1.1 )
        out_proj = UnnormalizePoints(out_proj, np.matrix([img.shape[1],img.shape[0] ]) )
    return out_proj


def render(img, proj_matrix, ref_U, eyemask, facemask, opts):
	print "> Query image shape:", img.shape
	img = img.astype('float32')

	### Projecting 3D model onto the the image
	threedee = np.reshape(ref_U, (-1, 3), order='F').transpose()
	temp_proj = proj_matrix * np.vstack((threedee, np.ones((1, threedee.shape[1]))))
	project = np.divide(temp_proj[0:2, :], np.tile(temp_proj[2, :], (2,1)))
	## Getting only the face for debug purpose and the background mask as well
	bg_mask = np.setdiff1d( np.arange(0, ref_U.shape[0]*ref_U.shape[1]) ,facemask[:,0] )
	face_proj = project[:, facemask[:,0] ]
	#out_proj = project[:, bg_mask]
	## Getting points that are outside the image
	bad = badIndex(project, img)
	nonbadind = np.nonzero(bad == 0)[0]
	badind = np.nonzero(bad == 1)[0]
	## Check which points lie outside of the image
	out_proj = project[:, badind]
	out_proj_disp = out_proj
	ind_all = np.arange(0, ref_U.shape[0]*ref_U.shape[1])
	ind_outside = ind_all[badind]
	############## OUTSIDE ##################################################
	background_img = None
	badface = badIndex(face_proj, img)
	face_in = np.nonzero( badface == 0 )[0]
	face_proj_in = face_proj[:,face_in]
	## In case we have some points outside, handle the bg
	out_proj = HandleBackground(out_proj,face_proj_in, img, opts)
	############## END OUTSIDE ##################################################
	############## INSIDE ##################################################
	in_proj = project[:, nonbadind]
	# because python arrays are zero indexed
	in_proj -= 1 # matlab indexing
	ind_frontal = ind_all[nonbadind]
	############## END INSIDE ##################################################
	# To do all at once
	prj_jnt = np.hstack( (out_proj, in_proj ) )
	ind_jnt = np.hstack( (ind_outside, ind_frontal) )

	if opts.getboolean('renderer', 'background'):
		frontal_raw = warpImg(img, ref_U.shape[0], ref_U.shape[1], prj_jnt, ind_jnt)
	else:
		frontal_raw = warpImg(img, ref_U.shape[0], ref_U.shape[1], face_proj, facemask[:,0])

	## Apply soft-sym if needed
	frontal_sym, sym_weight = mysoftSymmetry(img, frontal_raw, ref_U, in_proj, ind_frontal, bg_mask, facemask[:,0], eyemask, opts)

	return frontal_raw, frontal_sym, face_proj_in, out_proj_disp, out_proj, sym_weight

def mysoftSymmetry(img, frontal_raw, ref_U, in_proj, \
                 ind_frontal, bg_mask,facemask, eyemask, opts):
    weights = None
    ## Eyemask is activate only for frontal so we do soft-sym only on frontal thus when we have eyemask
    if eyemask is not None and opts.getboolean('renderer', 'symmetry'): # one side is ocluded
        ## Soft Symmetry param
        ksize_acc = Params.ksize_acc
        ################
        ## SOFT SYMMETRY 
        ind = np.ravel_multi_index((np.asarray(in_proj[1, :].round(), dtype='int64'), np.asarray(in_proj[0, :].round(),
                                    dtype='int64')), dims=img.shape[:-1], order='F')
        synth_frontal_acc = np.zeros(ref_U.shape[:-1])
        c, ic = np.unique(ind, return_inverse=True)
        bin_edges = np.r_[-np.Inf, 0.5 * (c[:-1] + c[1:]), np.Inf]
        count, bin_edges = np.histogram(ind, bin_edges)
        synth_frontal_acc = synth_frontal_acc.reshape(-1, order='F')
        synth_frontal_acc[ind_frontal] = count[ic]
        synth_frontal_acc = synth_frontal_acc.reshape((ref_U.shape[0], ref_U.shape[1]), order='F')
        synth_frontal_acc = cv2.GaussianBlur(synth_frontal_acc, (ksize_acc, ksize_acc), 30., borderType=cv2.BORDER_REPLICATE)
        ## Checking which side has more occlusions?
        midcolumn = np.round(ref_U.shape[1]/2)
        # apply soft symmetry to use whatever parts are visible in ocluded side
        synth_frontal_acc = synth_frontal_acc.reshape(-1, order='F')
        minacc=synth_frontal_acc[facemask].min()
        maxacc=synth_frontal_acc[facemask].max()
        ## we may need to do something more smooth like in previous softSym()
        synth_frontal_acc[facemask] = (synth_frontal_acc[facemask] - minacc)/(maxacc-minacc)

        if opts.getboolean('symmetry', 'flipBackground'):
            synth_frontal_acc[bg_mask] = 1. #this control sym on/off on background
        else:
            synth_frontal_acc[bg_mask] = 0. #this control sym on/off on background

        synth_frontal_acc = synth_frontal_acc.reshape((ref_U.shape[0], ref_U.shape[1]), order='F')

        synth_frontal_acc = np.tile(synth_frontal_acc.reshape(ref_U.shape[0], ref_U.shape[1], 1), (1, 1, 3))
        ## Flipping
        frontal_flip = frontal_raw.copy()
        frontal_flip[:,0:midcolumn,:] = np.fliplr(frontal_flip)[:,0:midcolumn,:]
        frontal_sym = np.multiply(frontal_raw, 1.-synth_frontal_acc) + np.multiply(frontal_flip, synth_frontal_acc)


        frontal_sym[frontal_sym > 255] = 255
        frontal_sym[frontal_sym < 0] = 0
        frontal_sym = frontal_sym.astype('uint8')
        weights = synth_frontal_acc[:,:,0]
    else: # both sides are occluded pretty much to the same extent -- do not use symmetry
        print '> skipping sym'
        frontal_sym = frontal_raw
    return frontal_sym, weights

#############################################
##### Old Symmetry code ###################################
##################################################
# def softSymmetry(img, frontal_raw, ref_U, in_proj, \
#                  ind_frontal, bg_mask, eyemask, opts):
#     ## Eyemask is activate only for frontal so we do soft-sym only on frontal thus when we have eyemask
#     if eyemask is not None and opts.getboolean('renderer', 'symmetry'): # one side is ocluded
#         ## Soft Symmetry param
#         ACC_CONST = Params.ACC_CONST
#         ksize_acc = Params.ksize_acc
#         ksize_weight = Params.ksize_weight
#         ################
#         ## SOFT SYMMETRY 
#         ind = np.ravel_multi_index((np.asarray(in_proj[1, :].round(), dtype='int64'), np.asarray(in_proj[0, :].round(),
#                                     dtype='int64')), dims=img.shape[:-1], order='F')
#         synth_frontal_acc = np.zeros(ref_U.shape[:-1])
#         c, ic = np.unique(ind, return_inverse=True)
#         bin_edges = np.r_[-np.Inf, 0.5 * (c[:-1] + c[1:]), np.Inf]
#         count, bin_edges = np.histogram(ind, bin_edges)
#         synth_frontal_acc = synth_frontal_acc.reshape(-1, order='F')
#         synth_frontal_acc[ind_frontal] = count[ic]
#         synth_frontal_acc = synth_frontal_acc.reshape((ref_U.shape[0], ref_U.shape[1]), order='F')
#         synth_frontal_acc = cv2.GaussianBlur(synth_frontal_acc, (ksize_acc, ksize_acc), 30., borderType=cv2.BORDER_REPLICATE)
#         ## Checking which side has more occlusions?
#         midcolumn = np.round(ref_U.shape[1]/2)
#         sumaccs = synth_frontal_acc.sum(axis=0)
#         sum_left = sumaccs[0:midcolumn].sum()
#         sum_right = sumaccs[midcolumn+1:].sum()
#         sum_diff = sum_left - sum_right
#         #print '----------------->np.abs(sum_diff), ',  np.abs(sum_diff)
#         if np.abs(sum_diff) > ACC_CONST:
#             print '> Using Face symmetry'
#             ones = np.ones((ref_U.shape[0], midcolumn))
#             zeros = np.zeros((ref_U.shape[0], ref_U.shape[1]-midcolumn))
#             if sum_diff > ACC_CONST: # left side of face has more occlusions
#                 weights = np.hstack((zeros, ones))
#             else: # right side of face has more occlusions
#                 weights = np.hstack((ones, zeros))
#             weights = cv2.GaussianBlur(weights, (ksize_weight, ksize_weight), 60.5, borderType=cv2.BORDER_REPLICATE)

#             # apply soft symmetry to use whatever parts are visible in ocluded side
#             synth_frontal_acc /= synth_frontal_acc.max()
#             weight_take_from_org = 1. / np.exp( synth_frontal_acc )

#             ### This to avoid symmetry in the background
#             #Symmetry only on the face, on the background we simply copy pase the other part
#             weight_take_from_org = weight_take_from_org.reshape(-1, order='F')
#             if opts.getboolean('symmetry', 'flipBackground'):
#                 weight_take_from_org[bg_mask] = 0. #this control sym on/off on background
#             else:
#                 weight_take_from_org[bg_mask] = 1. #this control sym on/off on background
#             weight_take_from_org = weight_take_from_org.reshape((ref_U.shape[0], ref_U.shape[1]), order='F')
#             ###############
#             weight_take_from_sym = 1 - weight_take_from_org
#             #print 'weight_take_from_org.shape,',  weight_take_from_org.shape
#             #print 'weights,',  np.fliplr(weights).shape
#             weight_take_from_org = np.multiply(weight_take_from_org, np.fliplr(weights))
#             weight_take_from_sym = np.multiply(weight_take_from_sym, np.fliplr(weights))

#             weight_take_from_org = np.tile(weight_take_from_org.reshape(ref_U.shape[0], ref_U.shape[1], 1), (1, 1, 3))
#             weight_take_from_sym = np.tile(weight_take_from_sym.reshape(ref_U.shape[0], ref_U.shape[1], 1), (1, 1, 3))
#             weights = np.tile(weights.reshape(ref_U.shape[0], ref_U.shape[1], 1), (1, 1, 3))

#             denominator = weights + weight_take_from_org + weight_take_from_sym
#             frontal_sym =    np.multiply(frontal_raw, weights) +\
#                              np.multiply(frontal_raw, weight_take_from_org) +\
#                              np.multiply(np.fliplr(frontal_raw), weight_take_from_sym)
#             frontal_sym = np.divide(frontal_sym, denominator)
#             ## Eye-Mask
#             #frontal_sym = np.multiply(frontal_sym, 1-eyemask) + np.multiply(frontal_raw, eyemask)
#             #########################################
#             frontal_sym[frontal_sym > 255] = 255
#             frontal_sym[frontal_sym < 0] = 0
#             frontal_sym = frontal_sym.astype('uint8')
#         else:
#             print '> not occluded, not doing sym'
#             frontal_sym = frontal_raw
#     else: # both sides are occluded pretty much to the same extent -- do not use symmetry
#         print '> skipping sym'
#         frontal_sym = frontal_raw
#     return frontal_sym
