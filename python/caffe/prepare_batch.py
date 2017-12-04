# ------------------------------------------------------------------------
# function images = prepare_batch(image_files,IMAGE_MEAN,batch_size)
import math
import numpy as np
import os
import cv2
import caffe

#from numpy.matlib import rand,zeros,ones,empty,eye
def prepare_batch(*args): #image_files, IMAGE_MEAN, batch_size):
# ------------------------------------------------------------------------
    image_files = args[0]
    batch_size = args[1]
    dir_path = os.path.dirname(os.path.realpath(__file__))
    nargin = len(args)

    num_images = len(image_files);
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path += '/../..'
    IMAGE_DIM = 256;
    CROPPED_DIM = 227;
    indices = (IMAGE_DIM - CROPPED_DIM) + 1
    center = math.floor(indices / 2) + 1

    num_images = len(image_files)
    a = np.matlib.zeros((CROPPED_DIM, CROPPED_DIM), float)
    images = np.matlib.zeros((CROPPED_DIM, CROPPED_DIM), float)

    for i in range(0, num_images):
    # read file
        print('%c Preparing %s\n', 13, image_files[i])
        try:
            #im = cv2.imread(dir_path + '/' + image_files[i])
            im = caffe.io.load_image(dir_path + '/' + image_files[i])
        # resize to fixed input size
            #im = single(im)
            im = im.astype('float32')
            #im = cv2.resize(im, (IMAGE_DIM, IMAGE_DIM))
            #im = numpy.matlib.resize(im, IMAGE_DIM, IMAGE_DIM)
        # Transform GRAY to RGB
            if im.shape == 1:
                im = np.array(im, im, im, im)
        
        # permute from RGB to BGR (IMAGE_MEAN is already BGR)
        #im = im[...,  [3, 2, 1]]
        # Crop the center of the image
            images[:, :, :, i] = np.transpose(im[center:center + CROPPED_DIM - 1, center:center + CROPPED_DIM - 1, :], [2, 1, 3])
        
        except:
            print('Problems with file', image_files[i])

    return images