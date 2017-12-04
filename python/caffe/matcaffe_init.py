
import numpy as np
import scipy
#import matcompat
from code import args
from ctypes import cdll
import os

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def matcaffe_init(*args):#use_gpu, model_def_file, model_file):
    lib = cdll.LoadLibrary('./libfoo.so')
    nargin = len(args)
    use_gpu = args[0]
    model_def_file = args[1]
    model_file = args[2]
    # Local Variables: use_gpu, model_file, model_def_file
    # Function calls: matcaffe_init, fprintf, nargin, exist, error, caffe, isempty
    #% matcaffe_init(model_def_file, model_file, use_gpu)
    #% Initilize matcaffe wrapper
    if nargin<1:
        #% By default use CPU
        use_gpu = 0.
    
    if nargin<2 or model_def_file == "": #isempty(model_def_file):
        #% By default use imagenet_deploy
        model_def_file = '../../models/bvlc_reference_caffenet/deploy.prototxt'
    
    if nargin<3 or model_file == "": #isempty(model_file):
        #% By default use caffe reference model
        model_file = '../../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    
    if lib.caffe('is_initialized') == 0.:
        if os.path.isfile(model_file) == 0.:
            #% NOTE: you'll have to get the pre-trained ILSVRC network
            #matcompat.error('You need a network model file')
            print('ERROR: You need a network model file')
        
        if not os.path.isfile(model_def_file, 'file'):
            #% NOTE: you'll have to get network definition
            print('ERROR: You need the network prototxt definition')
        
        lib.caffe('init', model_def_file, model_file)
    
    
    print('Done with init\n')
    #% set to use GPU or CPU
    if use_gpu:
        print('Using GPU Mode\n')
        lib.caffe('set_mode_gpu')
    else:
        print('Using CPU Mode\n')
        lib.caffe('set_mode_cpu')
        
    
    print('Done with set_mode\n')
    #% put into test mode
    lib.caffe('set_phase_test')
    print('Done with set_phase_test\n')
    return 