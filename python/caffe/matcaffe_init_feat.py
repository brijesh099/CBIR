import os
from ctypes import cdll
# from python.caffe.pycaffe import pycaffe #import _caffe
import sys

sys.path.append('/usr/aditya/caffe-cvprw15-master/python')
import caffe
import numpy as np
from prepare_batch import prepare_batch
import math
import cv2


def matcaffe_init_feat(*args):  # use_gpu, model_def_file, model_file):

    nargin = len(args)
    model_def_file = args[1]
    use_gpu = args[0]
    model_file = args[2]
    list_im = args[3]

    if nargin < 1:
        # By default use CPU
        use_gpu = 0

    if nargin < 2 or model_def_file == "":  # isempty(model_def_file):
        # By default use imagenet_deploy
        model_def_file = str('./examples/cvprw15-cifar10/CIFAR10_48_deploy.prototxt')

    if nargin < 3 or model_file == "":  # isempty(model_file):
        model_file = str('./examples/cvprw15-cifar10/CIFAR10_48.caffemodel')

    # if caffe('is_initialized') == 0
    if not os.path.exists(model_file):  # exist(model_file, mstring('file')) == 0:
        # NOTE: you'll have to get the pre-trained ILSVRC network
        print('ERROR: You need a network model file')

    if not os.path.exists(model_def_file):
        # NOTE: you'll have to get network definition
        print('ERROR: You need the network prototxt definition')

    # lib = os.popen('/home/aditya/Downloads/caffe-cvprw15-master/build/tools/caffe.o')
    print('Init caffe')
    print(model_def_file)
    print(model_file)
    net = caffe.Net(model_def_file, model_file)
    # end
    print(str('Done with init\\n'))

    # set to use GPU or CPU
    if use_gpu:
        print(str('Using GPU Mode\\n'))
        net.set_device(1)
        net.set_mode_gpu()
        # caffe(str('set_mode_gpu'))
    else:
        print(str('Using CPU Mode\\n'))
        net.set_mode_cpu()
        # caffe(str('set_mode_cpu'))

    print(str('Done with set_mode\\n'))
    batch_size = 10


    # put into test mode
    dir_path = os.path.dirname(os.path.realpath(__file__))
    net.set_mean('data', np.load(dir_path + '/imagenet/ilsvrc_2012_mean.npy'))
    net.set_phase_test()

    num_images = len(list_im)
    scores = 0  # zeros(dim,num_images,'single');
    num_batches = math.ceil(len(list_im) / batch_size)
    # initic=tic;
    print('Num batches are: ' + str(num_batches))
    print(list_im)
    ####################################################################################
    for bb in range(1, int(num_batches) + 1):  # = 1 : num_batches
        # batchtic = tic;
        range1 = 1 + batch_size * (bb - 1), getMinimum([num_images, (batch_size * bb)])
        # tic
        print("The range is " + str(range1))
        print(list_im[0])
        ####################################################################################
        # input_data = prepare_batch(list_im[range1[0]: range1[1]], batch_size)
        image_files = list_im[range1[0]: range1[1]]
        num_images = len(image_files)
        images = None
        im = None
        for i in range(0, num_images):
            # read file
            print('%c Preparing %s\n', 13, image_files[i])
            #try:

            img = cv2.imread(image_files[i], 0)
            img_blobinp = img[np.newaxis, np.newaxis, :, :]
            #net.blobs['data'].reshape(*img_blobinp.shape)
            #net.blobs['data'].data[...] = img_blobinp

            #transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
            ##transformer.set_mean('data', (np.load(dir_path + '/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)))
            #transformer.set_transpose('data', (2, 0, 1))
            #transformer.set_channel_swap('data', (2, 1, 0))
            #transformer.set_raw_scale('data', 255.0)
            #im = caffe.io.load_image(image_files[i])
            ##im = im[np.newaxis, np.newaxis, :, :]
            #im = im[:, :, 0]
            #transformer.preprocess('data', img_blobinp)
            #net.blobs['data'].reshape(1, 1, 32, 32)#im.shape)
            #except:
            print('Problems with file', image_files[i])

            if images is None:
                images = im
            else:
                images += im
        ########################################################################################
        # toc, tic
        # fprintf('Batch %d out of %d %.2f%% Complete ETA %.2f seconds\n',...
        #       bb,num_batches,bb()/num_batches*100,toc(initic)/bb*(num_batches-bb));
        print("Processing.." + str(bb))
        #net.blobs['data'].reshape(1, 32, 32, 3)
        #net.blobs['data'].data[...] = im
        #output_data = net.forward()  # caffe('forward', {input_data})
        #img_blobinp = img_blobinp.transpose(2, 1, 0)

        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_mean('data', (np.load(dir_path + '/imagenet/ilsvrc_2012_mean.npy')))
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_channel_swap('data', (2, 1, 0))
        transformer.set_raw_scale('data', 255)
        out = net.forward()
        #out = net.forward_all(data=np.asarray([transformer.preprocess('data', img_blobinp)]))
        # toc
        # output_data = np.squeeze(output_data[1], None)
        #print output_data['prob'].argmax()
        #scores[:, range1] = output_data[:, ((range1 - 1) % batch_size) + 1]
        # toc(batchtic)

        # toc(initic);
        print('Done with forward')

###############################################################################################

def getMinimum(list):
    minimum = None
    for num in list:
        if minimum is None or num < minimum:
            minimum = num

    print minimum
    return minimum


