echo "Downloading ImageNet pre-trained model and weight"
wget -O ./models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel "https://www.dropbox.com/s/nlggnj47xxdmwkb/bvlc_reference_caffenet.caffemodel?dl=1"

echo "Downloading CIFAR10 Dataset"
wget -O cifar10-dataset.zip "https://www.dropbox.com/s/h8swscihlmjvznm/cifar10-dataset.zip?dl=0" 
unzip cifar10-dataset.zip -d ./examples/cbir-cifar10/dataset

echo "Convert CIFAR10 to leveldb"
./examples/cbir-cifar10/create_imagenet.sh 
