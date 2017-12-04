import threading

import numpy as np
import sys, os
import matplotlib

matplotlib.use('Agg')
import caffe
import time
import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2

MODEL_DIR = '/home/aditya/Downloads/Temp/caffe-cvprw15-master/examples/cvprw15-cifar10/'
MODEL = "/home/aditya/Downloads/Temp/caffe-cvprw15-master/examples/cvprw15-cifar10/KevinNet_CIFAR10_48.caffemodel"
PROTO = "/home/aditya/Downloads/Temp/caffe-cvprw15-master/examples/cvprw15-cifar10/KevinNet_CIFAR10_48_deploy.prototxt"
MEAN = "/home/aditya/Downloads/Temp/caffe-cvprw15-master/python/caffe/imagenet/ilsvrc_2012_mean.npy"
TRAIN_BINARY = '/home/aditya/Downloads/Temp/caffe-cvprw15-master/analysis/binary-train.pycode'
BINARY_PATH = '/home/aditya/Downloads/Temp/caffe-cvprw15-master/analysis/'
TRAIN_DATASET = '/home/aditya/Downloads/Temp/caffe-cvprw15-master/examples/cvprw15-cifar10/dataset/train-file-list.txt'
TEST_DATASET = '/home/aditya/Downloads/Temp/caffe-cvprw15-master/examples/cvprw15-cifar10/dataset/test-file-list-1.txt'
TRAIN_LABEL = '/home/aditya/Downloads/Temp/caffe-cvprw15-master/examples/cvprw15-cifar10/dataset/train-label.txt'
TEST_LABEL = '/home/aditya/Downloads/Temp/caffe-cvprw15-master/examples/cvprw15-cifar10/dataset/test-label.txt'


class binaryCodesGenerator(object):
    def __init__(self, gpuid, modelDir):
        self.gpuid = gpuid
        self.model = os.path.join(modelDir, MODEL)
        self.proto = os.path.join(modelDir, PROTO)
        self.mean = os.path.join(modelDir, MEAN)
        self.initcaffe()

    def initcaffe(self):
        # caffe.set_device(self.gpuid)
        # caffe.set_mode_cpu()
        self.net = caffe.Net(self.proto, self.model)  # , caffe.TEST)
        self.net.set_phase_test()
        self.net.set_mode_cpu()
        self.net.forward()
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.load(self.mean).mean(1).mean(1))
        self.transformer.set_raw_scale('data', 255)
        self.transformer.set_channel_swap('data', (2, 1, 0))

    def hashing(self, image):
        array = np.fromstring(image, dtype='uint8')
        im = cv2.imdecode(array, 1)
        im = im / 255.
        im = im[:, :, (2, 1, 0)]
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', im)
        self.net.forward()
        # obtain the output probabilities
        feat = self.net.blobs['fc8_kevin_encode'].data[0]
        # generate binary codes
        binary_codes = feat > 0.5
        binary_codes = binary_codes.astype(int)
        return binary_codes

    def traindataset(self):
        print('Training Data Sets')
        binFile = open(TRAIN_BINARY, "w")
        with open(TRAIN_DATASET) as f:
            lines = f.read().splitlines()

        i = 1
        prev = 0
        allThreads = []

        self.traindatasetthread(lines[0:100], 'batch')
        return

    def traindatasetthread(self, lines, filename):
        print('Training Data Sets')
        binFile = open(BINARY_PATH + filename, "w")
        i = 0
        for str1 in lines:
            with open(str1, 'rb') as infile:
                print(filename + ' : image: ' + str(i) + 'Generating HashCode for' + str1)
                buf = infile.read()
                binary_codes = bCG.hashing(buf)
                binFile.write("".join(map(str, binary_codes)).replace('[', '').replace(']', ''))
                binFile.write("\n")
                i += 1
        return

    def testdataset(self):
        print('Test Dataset')
        # Open train dataset
        with open(TRAIN_DATASET) as f1:
            traindata = f1.read().splitlines()
        # Open test Dataset
        with open(TEST_DATASET) as f2:
            testdata = f2.read().splitlines()
            # codes = bCG.hashing(lines[0])
            # print("Binary code for ", lines[0], "is ".join(map(str, codes)).replace('[', '').replace(']', ''))

        i = 1
        # Open train Binary file
        with open(TRAIN_BINARY) as f:
            binary = f.read().splitlines()
        # guessed = 0
        for line in testdata:
            print('Query: ', i, ' for ', line)
            i = i + 1
            guessed = 0
            with open(line, 'rb') as infile:
                buf = infile.read()
                binary_codes = bCG.hashing(buf)
                binary_codes = "".join(map(str, binary_codes)).replace('[', '').replace(']', '')
                print(
                    "Hamming code for the test image: ",
                    "".join(map(str, binary_codes)).replace('[', '').replace(']', ''))
                filecount = 0
                for bin in binary:
                    # buf1 =
                    hamming_dis = self.hamming2(binary_codes, bin)  # np.count_nonzero(binary_codes != bin)
                    if hamming_dis == 0:
                        with open(TRAIN_LABEL) as f3:
                            trlabel = f3.read().splitlines()
                        with open(TEST_LABEL) as f4:
                            tslabel = f4.read().splitlines()

                        if (tslabel[i] == trlabel[filecount]):
                            # print "hamming distance: %d" % hamming_dis
                            print "Match found with" + traindata[filecount]
                            guessed += 1
                        filecount += 1

                print ('Iterated over: ', filecount, ' matches found: ', guessed)
        print "Acurracy: ", (guessed / i)
        return

    def hamming2(self, s1, s2):
        """Calculate the Hamming distance between two bit strings"""
        assert len(s1) == len(s2)
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))


    def printmatch(self, list):
        with open(TRAIN_DATASET) as f:
            match = f.read().splitlines()
        print
        return


if __name__ == "__main__":
    gpuID = 0
    bCG = binaryCodesGenerator(gpuID, MODEL_DIR)
    img_path = '/home/aditya/Downloads/caffe-cvprw15-master/examples/cvprw15-cifar10/dataset/test/1314.jpg'
    with open(img_path, 'rb') as infile:
        buf = infile.read()
    binary_codes_1 = bCG.hashing(buf)
    # img2
    img_path = '/home/aditya/Downloads/caffe-cvprw15-master/examples/cvprw15-cifar10/dataset/batch1/3094.jpg'
    with open(img_path, 'rb') as infile:
        buf = infile.read()
    binary_codes_2 = bCG.hashing(buf)
    # compute hamming distance
    hamming_dis = np.count_nonzero(binary_codes_1 != binary_codes_2)
    print "hamming distance: %d" % hamming_dis
