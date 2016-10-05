import scipy.io as scio
import sklearn.metrics
import cv2
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(formatter={'float_kind': lambda x: "%.4f" % x})


class FaceModel:
    def __init__(self, path, name):
        self.load_model(path, name)
        self.eyemask = self.getEyeMask(width=8,plot=False)

    def load_model(self, path, name):
        model = scio.loadmat(path)[name]
        self.out_A = np.asmatrix(model['outA'][0, 0], dtype='float32') #3x3
        self.size_U = model['sizeU'][0, 0][0] #1x2
        self.model_TD = np.asarray(model['threedee'][0,0], dtype='float32') #68x3
        self.indbad = model['indbad'][0, 0]#0x1
        self.ref_U = np.asarray(model['refU'][0,0])
        self.facemask = np.asarray(model['facemask'][0,0])
        self.facemask-=1 #matlab indexing

    def getEyeMask(self,width=1, plot=False):
        X = self.ref_U[:,:,0]
        X = X.reshape( (-1), order='F' )
        Y = self.ref_U[:,:,1]
        Y = Y.reshape( (-1), order='F' ) 
        Z = self.ref_U[:,:,2]
        Z = Z.reshape( (-1), order='F' )
        cloud = np.vstack( (X,Y,Z) ).transpose()
        [idxs, dist] = sklearn.metrics.pairwise_distances_argmin_min(self.model_TD, cloud)
        eyeLeft = idxs[36:42]
        eyeRight = idxs[42:48]
        output1 = self.createMask(eyeLeft, width=width)
        output2 = self.createMask(eyeRight, width=width)
        output = output1 + output2
        if plot:
            plt.figure()
            plt.imshow(output)
            plt.draw()
            plt.pause(0.001)
            enter = raw_input("Press [enter] to continue.")

        output[output==255]=1
        return output

    def createMask(self,eyeLeft,width=1):
        eyeLefPix = np.unravel_index( eyeLeft, dims=self.ref_U.shape[::-1][1:3] ) 
        eyeLefPix = np.asarray(eyeLefPix)
        eyemask = np.zeros((self.ref_U.shape[0]*self.ref_U.shape[1], 3))
        eyemask[eyeLeft,:] = 255
        eyemask = eyemask.reshape((self.ref_U.shape[0], self.ref_U.shape[1], 3), order='F')
        eyemask = eyemask.astype('uint8')
        eyemask = cv2.cvtColor(eyemask,cv2.COLOR_BGR2GRAY)

        for i in range(eyeLefPix.shape[1]):
            cv2.line(eyemask,(eyeLefPix[0,i],eyeLefPix[1,i]),(eyeLefPix[0,(i+1)%eyeLefPix.shape[1]],\
                eyeLefPix[1,(i+1)%eyeLefPix.shape[1]]),(255,255,255),width)

        contours, hierarchy = cv2.findContours(eyemask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        eyemaskfill = np.zeros((self.ref_U.shape[0],self.ref_U.shape[1], 3))

        for r in range(self.ref_U.shape[0]):
            for c in range(self.ref_U.shape[1]):
                if cv2.pointPolygonTest(contours[0], (c,r), False ) > 0:
                    eyemaskfill[r,c,:] = 255

        eyemaskfill = eyemaskfill.astype('uint8')        

        return eyemaskfill