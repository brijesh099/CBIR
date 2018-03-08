1. Download the project from the git repository.

2. Open the terminal

3. Install the dependencies of the caffe found on following link.
	http://caffe.berkeleyvision.org/installation.html

4. Install all required dependencies for executing Python 2.7:
	i. Open terminal at CBIR/python and execute following command
		for req in $(cat requirements.txt); do pip install $req; done

5. Execute following commands: 
   i. make all -j8
   ii. make test -j8
   iii. make runtest
   iv. make pycaffe
   v. ./prepare.sh
   vi. cd /examples/cbir-cifar10/
	   chmod 777 train_CPU_48 train_GPU_48
	   ./train_CPU_48  -- for training with CPU
	   ./train_GPU_48  -- for training with GPU

6. Add the CBIR/python to your python path.

7. Modify the model with newly created model in step 5.vi. and execute the RunPythonCIfarGpu/Cpu.py - This will train and test the code. 

8. If you want to manually download the dataset, we have kept it at : https://www.dropbox.com/l/scl/AAASnbNHfs9b9BF5kPMTplWVJc45X9e3dIU
   i. Extract the dataset and keep it at CBIR/examples/cbir-cifar10/dataset
   
9. The trained model for GPU is kept at this location: https://www.dropbox.com/s/dlak33m5icvdg1o/CBIR_GPU.caffemodel?dl=0

10. The trained model for CPU is kept at this location: https://www.dropbox.com/s/iixuym610fntld6/CBIR_CPU.caffemodel?dl=0


