__author__ = 'Iacopo'

import scipy.io as scio
import sklearn.metrics
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import ThreeD_Model
import camera_calibration as calib

## Index to remap landmarks in case we flip an image
repLand = [ 17,16,15,14,13,12,11,10, 9,8,7,6,5,4,3,2,1,27,26,25, \
            24,23,22,21,20,19,18,28,29,30,31,36,35,34,33,32,46,45,44,43, \
            48,47,40,39,38,37,42,41,55,54,53,52,51,50,49,60,59,58,57,56, \
            65,64,63,62,61,68,67,66 ]

def mymkdir(output):
	if not os.path.exists(output):
		os.makedirs(output)

def parse(argv):
	fileList = []
	outputFolder = 'output/'
	## Case in which only an image is provided
	if len(argv) == 2:
	    head, tail = os.path.split(argv[1])
	    fileList = [tail.split('.')[0]+','+str(argv[1])+',None']
	## Ok landmarks are provided as well or we are in batch mode
	elif len(argv) == 3:
	    #print argv[1]
	    ## If we are not in batch mode
	    if "--batch" not in str(argv[1]):
	        head, tail = os.path.split(argv[1])
	        fileList = [tail.split('.')[0]+','+str(argv[1])+','+str(argv[2])]
	    else:
	        print '> Batch mode detected - reading from file: ' + str(argv[2])
	        filep = str(argv[2])
	        fileList = [line.strip() for line in open(filep)]
	else:
		print 'Usage for face rendering. See below'
		print 'Usage: python demo.py <image-path>'
		print 'Usage: python demo.py <image-path> <landmark-path>'
		print 'Usage: python demo.py --batch <file-list-path>'
		print 'where <file-list-path> is a csv file where each line has'
		print 'image_key,<image-path>,<landmark-path> (lines that contain # are skipped)'
		exit(1)
	return fileList, outputFolder

def isFrontal(pose):
	if '_-00_' in pose:
		return True
	return False

def preload(this_path, pose_models,nSub):
    print '> Preloading all the models for efficiency'
    allModels= dict()
    for posee in pose_models:
        ## Looping over the subjects
        for subj in range(1,nSub+1):
            pose =   posee + '_' + str(subj).zfill(2) +'.mat'
            # load detections performed by dlib library on 3D model and Reference Image
            print "> Loading pose model in " + pose
            model3D = ThreeD_Model.FaceModel(this_path + "/models3d/" + pose, 'model3D')
            allModels[pose] = model3D
    return allModels


def cropFunc(pose,frontal_raw,crop_model):
	frontal_raw = crop_face(frontal_raw, crop_model)
	return frontal_raw

def crop_face(img, cropping):
    if cropping is not None:
        img = img[cropping[1]:cropping[3],\
               cropping[0]:cropping[2],:]
        print '> Cropping with: ', cropping
    else:
        print '> No Cropping'
    return img

def flipInCase(img, lmarks, allModels):
	## Check if we need to flip the image
	yaws= []#np.zeros(1,len(allModels))
	## Getting yaw estimate over poses and subjects
	for mmm in allModels.itervalues():
		proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(mmm, lmarks[0])
		yaws.append( calib.get_yaw(rmat) )
	yaws=np.asarray(yaws)
	yaw = yaws.mean()
	print '> Yaw value mean: ',  yaw
	if yaw  < 0:
	    print '> Positive yaw detected, flipping the image'
	    img = cv2.flip(img,1)
	    # Flipping X values for landmarks
	    lmarks[0][:,0] = img.shape[1] - lmarks[0][:,0]
	    # Creating flipped landmarks with new indexing 
	    lmarks3 =  np.zeros((1,68,2))
	    for i in range(len(repLand)):
	        lmarks3[0][i,:] = lmarks[0][repLand[i]-1,:]
	    lmarks = lmarks3
	return img, lmarks, yaw

def show(img_display, img, lmarks, frontal_raw, \
	      face_proj, background_proj, temp_proj2_out_2, sym_weight):
    plt.ion()
    plt.show()
    plt.subplot(221)
    plt.title('Query Image')
    plt.imshow(img_display[:, :, ::-1])
    plt.axis('off')

    plt.subplot(222)
    plt.title('Landmarks Detected')
    plt.imshow(img[:, :, ::-1])
    plt.scatter(lmarks[0][:, 0], lmarks[0][:, 1],c='red', marker='.',s=100,alpha=0.5)
    plt.axis('off')
    plt.subplot(223)
    plt.title('Rendering')

    plt.imshow(frontal_raw[:, :, ::-1])
    plt.axis('off')

    plt.subplot(224)
    if sym_weight is None:
	    plt.title('Face Mesh Projected')
	    plt.imshow(img[:, :, ::-1])
	    plt.axis('off')
	    face_proj = np.transpose(face_proj)
	    plt.plot( face_proj[1:-1:100,0], face_proj[1:-1:100,1] ,'b.')
	    background_proj = np.transpose(background_proj)
	    temp_proj2_out_2 = temp_proj2_out_2.T
	    plt.plot( background_proj[1:-1:100,0], background_proj[1:-1:100,1] ,'r.')
	    plt.plot( temp_proj2_out_2[1:-1:100,0], temp_proj2_out_2[1:-1:100,1] ,'m.')
    else:
	    plt.title('Face Symmetry')
	    plt.imshow(sym_weight)
	    plt.axis('off')
	    plt.colorbar()

    plt.draw()
    plt.pause(0.001)
    enter = raw_input("Press [enter] to continue.")
    plt.clf()

def decidePose(yaw,opts):
	if opts.getboolean('renderer', 'nearView'):
		yaw = abs(yaw)
		# If yaw is near-frontal we render everything
		if yaw < 15:
			return [0,1,2]
		# otherwise we render only 2 profiles (from profile to frontal is noisy)
		else:
			return [1,2]
	else:
		return [0,1,2]