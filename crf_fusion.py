import pydensecrf.densecrf as dcrf
import numpy as np
import sys

from skimage.io import imread, imsave
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax

from os import listdir, makedirs
from os.path import isfile, join
import os

davis_path = '/home/cvlab/FAISAL/DAVIS_Dataset/DAVIS-trainval-480p'
setting = sys.argv[1]
out_folder = sys.argv[2]
for d in listdir(davis_path + '/Results/' + setting):

    vidDir = join(davis_path + '/JPEGImages/480p/', d)
    resDir = join(davis_path + '/Results/' + out_folder, d)

    if not os.path.exists(resDir):
    	makedirs(resDir)
    
    for f in listdir(vidDir):       

        img = imread(join(vidDir, f))
        segDir = join(davis_path + '/Results/' + setting, d)
        frameName = str.split(f, '.')[0]
        if not isfile(segDir + '/raw_' + frameName + '.png'):
            print('Not found')
            continue
        anno_rgb = imread(segDir + '/raw_' + frameName + '.png').astype(np.uint32)
        
        min_val = np.min(anno_rgb.ravel())
        max_val = np.max(anno_rgb.ravel())
        out = (anno_rgb.astype('float') - min_val) / (max_val - min_val)
        labels = np.zeros((2, img.shape[0], img.shape[1]))
        labels[1, :, :] = out
        labels[0, :, :] = 1 - out

        colors = [0, 255]
        colorize = np.empty((len(colors), 1), np.uint8)
        colorize[:,0] = colors

        n_labels = 2

        crf = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

        U = unary_from_softmax(labels)
        crf.setUnaryEnergy(U)

        feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
        crf.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This creates the color-dependent features and then add them to the CRF
        feats = create_pairwise_bilateral(sdims=(50, 50), schan=(10, 10, 10),
                                      img=img, chdim=2)
        crf.addPairwiseEnergy(feats, compat=5,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

        Q = crf.inference(5)

        MAP = np.argmax(Q, axis=0)
        MAP = colorize[MAP]
        
        imsave(resDir + '/' + frameName + '.png', MAP.reshape(anno_rgb.shape))
