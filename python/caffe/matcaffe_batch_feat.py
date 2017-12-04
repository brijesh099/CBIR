# from _overlapped import NULL
import numpy as np

from matcaffe_init_feat import matcaffe_init_feat
from read_cell import read_cell
import math
from prepare_batch import prepare_batch
from ctypes import cdll
import caffe
import os


# Definationf of the function.
def matcaffe_batch_feat(list_im, use_gpu, feat_len, model_def_file, model_file):
    # lib = cdll.LoadLibrary('/home/aditya/Downloads/caffe-cvprw15-master/build/lib/libcaffe.so')
        
    if list_im != None:  # isinstance(list_im, str[None]):
    # if list_im is np.string:#ischar(list_im) :
    # Assume it is a file contaning the list of images
        filename = list_im
        print(list_im)
        list_im = read_cell(filename)

# Adjust the batch size and dim to match with models/bvlc_reference_caffenet/deploy.prototxt
    batch_size = 10
    dim = feat_len
    print(list_im)
    if (len(list_im) % batch_size) :
        print(['Assuming batches of ' + str(batch_size) + ' images rest will be filled with zeros'])
    

# init caffe network (spews logging info)
    matcaffe_init_feat(use_gpu, model_def_file, model_file, list_im)

    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # d = open(dir_path + '/imagenet/ilsvrc_2012_mean.npy');
    # IMAGE_MEAN = d.image_mean;
    # net.set_mean('data', np.load('../python/caffe/imagenet/ilsvrc_2012_mean.npy'))



    """
    num_images = len(list_im)
    scores = 0  # zeros(dim,num_images,'single');
    num_batches = math.ceil(len(list_im) / batch_size)
    # initic=tic;
    print('Num batches are: ' + str(num_batches))
    print(list_im)

    for bb in range(1, int(num_batches)+1):  # = 1 : num_batches
        # batchtic = tic;
        range1 = 1 + batch_size * (bb - 1), getMinimum([num_images, (batch_size * bb)])
        # tic
        print("The range is " + str(range1))
        print(list_im[0])
        input_data = prepare_batch(list_im[range1[0]: range1[1]], batch_size)
        # toc, tic
        # fprintf('Batch %d out of %d %.2f%% Complete ETA %.2f seconds\n',...
        #       bb,num_batches,bb/num_batches*100,toc(initic)/bb*(num_batches-bb));
        print("Processing.." + str(bb))
        output_data = caffe('forward', {input_data})
        # toc
        output_data = np.squeeze(output_data[1], None)
        scores[:, range1] = output_data[:, ((range1 - 1) % batch_size) + 1]
        # toc(batchtic)
    
    # toc(initic);
"""
def getMinimum(list):
    minimum = None       
    for num in list:
        if minimum is None or num < minimum:
            minimum = num

    print minimum
    return minimum

