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

dir_path = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = dir_path + '/examples/cbir-cifar10/'
MODEL = dir_path + "/examples/cbir-cifar10/CBIR_CPU.caffemodel"
PROTO = dir_path + "/examples/cbir-cifar10/CIFAR10_CBIR_48_deploy.prototxt"
MEAN = dir_path + "/python/caffe/imagenet/ilsvrc_2012_mean.npy"
TRAIN_BINARY = dir_path + '/analysis/binary-train.pycode'
BINARY_PATH = dir_path + '/analysis/'
TRAIN_DATASET = dir_path + '/examples/cbir-cifar10/dataset/train-file-list.txt'
TEST_DATASET = dir_path + '/examples/cbir-cifar10/dataset/test-file-list-1.txt'
TRAIN_LABEL = dir_path + '/examples/cbir-cifar10/dataset/train-label.txt'
TEST_LABEL = dir_path + '/examples/cbir-cifar10/dataset/test-label.txt'


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
        feat = self.net.blobs['fc8_cbir_encode'].data[0]
        # generate binary codes
        binary_codes = feat > 0.5
        binary_codes = binary_codes.astype(int)
        return binary_codes

    def traindataset(self):
        print('Training Data Sets')
        binFile = open(TRAIN_BINARY, "w")
        with open(TRAIN_DATASET) as f:
            lines = f.read().splitlines()
        start = time.time()
        i = 1
        prev = 0
        allThreads = []

        for batch in range(5000, 55000, 5000):
            print('Starting Thread for ', prev, ' to ', batch)
            t = threading.Thread(target=self.traindatasetthread, args=(lines[prev: batch], 'batch' + str(i)))
            i = i + 1
            prev = batch + 1
            t.start()
            allThreads.append(t)


        for th in allThreads:
            th.join()

        for i in range(1, 10, 1):
            with open(BINARY_PATH + 'batch' + str(i)) as f:
                lines = f.read().splitlines()

            for line in lines:
                binFile.write(line + '\n')

                os.remove(BINARY_PATH + 'batch' + str(i))

        end = time.time()
        print("Total time to train: ", end - start, " seconds")
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
        start = time.time()
        with open(TRAIN_DATASET) as f1:
            traindata = f1.read().splitlines()
        # Open test Dataset
        with open(TEST_DATASET) as f2:
            testdata = f2.read().splitlines()
            # codes = bCG.hashing(lines[0])
            # print("Binary code for ", lines[0], "is ".join(map(str, codes)).replace('[', '').replace(']', ''))

        i = 0
        # Open train Binary file
        with open(TRAIN_BINARY) as f:
            binary = f.read().splitlines()
        # guessed = 0
        accuracy = 0
        for line in testdata:
            print('Query: ', i, ' for ', line)
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
                if guessed > 0:
                    accuracy += 1
                    # print accuracy
                print ('Iterated over: ', filecount, ' matches found: ', guessed)
            i = i + 1
        print("Accuracy: ", (accuracy / (i * 1.0)))
        end = time.time()
        print("Total time to test 8 images: ", (end - start), " seconds.")
        return

    def hamming2(self, s1, s2):
        """Calculate the Hamming distance between two bit strings"""
        assert len(s1) == len(s2)
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    def testDataSet2(self):
        timeStart = time.time()
        with open(TEST_DATASET) as f2:
            testdata = f2.read().splitlines()
        with open(TEST_LABEL) as f4:
            tslabel = f4.read().splitlines()
        i = 0
        AP = [0 for x in range(len(testdata))]
        for t in testdata:
            predict = time.time()
            with open(t, 'rb') as infile:
                print('Query: ', i+1, ' for ', t)
                buf = infile.read()
                binary_codes = bCG.hashing(buf)
                binary_codes = "".join(map(str, binary_codes)).replace('[', '').replace(']', '')
                AP[i] = self.getTopMatches(1000, binary_codes, tslabel[i])
                i = i + 1
                print("Estimated time to complete: ", (time.time() - predict) * (len(testdata) - i))

        timeEnd = time.time()
        print("Total time reqired to complete Test: ",timeEnd - timeStart," seconds.")
        print("The precision is: ",self.calMean(AP))

    def calMean(self, mat):
        length = len(mat) * 1.0
        sum = 0
        for m in mat:
            sum = sum +  m
        return sum/length

    def getTopMatches(self, top, testCode, testLabel):
        var = np.mat
        a = np.zeros((2, 1), dtype = str)

        with open(TRAIN_LABEL) as f1:
            trlabel = f1.read().splitlines()

        with open(TRAIN_BINARY) as f2:
            binary = f2.read().splitlines()

        with open(TRAIN_DATASET) as f3:
            trainDat = f3.read().splitlines()

        matrix = [[0 for x in range(3)] for y in range(len(binary))]

        i = 0
        for bin in binary:
            hamming_dis = self.hamming2(bin, testCode)
            matrix[i] = [float(hamming_dis), trainDat[i], trlabel[i]]
            i = i + 1

        matrix.sort()
        topMatrix = [[0 for x in range(3)] for y in range(top)]

        for i in range(0, top):
            topMatrix[i] = matrix[i]

        buffer_yes = [0 for x in range(top)]

        for i in range(0, top):
            if(testLabel == topMatrix[i][2]):
                buffer_yes[i] = 1.0
                print("Matched with: ", topMatrix[i][1])

        cumsum = self.calCumSum(buffer_yes)
        P = self.elementDivision(cumsum)
        result = self.calculation2(cumsum, P)
        return result

    def calCumSum(self, Matrix):
        first = 0
        for mat in range(0, len(Matrix)):
            Matrix[mat] = Matrix[mat] + first
            first = Matrix[mat]

        return Matrix

    def elementDivision(self, Matrix):
        for mat in range(0, len(Matrix)):
            Matrix[mat] = Matrix[mat] / (mat+1)

        return Matrix

    def calculation2(self, buffer_sum, p):
        sum1 = 0
        result = 0
        for m in buffer_sum:
            sum1 = sum1 + m

        if sum1 == 0:
            return 0

        for m in range(0, len(buffer_sum)):
            buffer_sum[m] = p[m] * buffer_sum[m]

        sum2 = 0
        for m in buffer_sum:
            sum2 = sum2 + (m * 1.0)

        return sum2/sum1

if __name__ == "__main__":
    gpuID = 0
    bCG = binaryCodesGenerator(gpuID, MODEL_DIR)

    bCG = binaryCodesGenerator(gpuID, MODEL_DIR)
    if not os.path.exists(TRAIN_BINARY):
        bCG.traindataset()

    bCG.testDataSet2()
