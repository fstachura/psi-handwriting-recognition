# Handwriting recognition 

This project aims to create a model that recognizes handwritten digits and letters. Various models were evaluated on the classic MNIST dataset. A convolutional neural network was also evaluated on the EMNIST MNSIT combined with EMNIST letters dataset.

## Setup

To install required dependencies, run `pip install -r requirements.txt`. 
This project uses Jupyter Notebook. After installing dependencies, run `python3 -m jupyter notebook` in project directory to start the Jupyter server.

### Datasets

To train models, datasets need to be downloaded first. 

* MNIST is downloaded using functions built into used libraries (keras and sklearn)
* EMNSIT can be downloaded from https://www.nist.gov/itl/products-and-services/emnist-dataset 
    * Files emnist-mnist-test-images-idx3-ubyte.gz, emnist-mnist-test-labels-idx1-ubyte.gz, emnist-mnist-train-images-idx3-ubyte.gz and emnist-mnist-train-labels-idx1-ubyte.gz should be unpacked and decompressed using gzip into `emnist_mnist` directory
    * Files emnist-letters-test-images-idx3-ubyte.gz, emnist-letters-test-labels-idx1-ubyte.gz, emnist-letters-train-images-idx3-ubyte.gz and emnist-letters-train-labels-idx1-ubyte.gz should be unpacked and decompressed with gzip into emnist_letters directory

Detailed information about datasets and data examples can be found in datasets.ipynb

Final structure of dataset directories:

```
% ls emnist_letters emnist_mnist 
emnist_letters:
emnist-letters-test-images-idx3-ubyte  emnist-letters-test-labels-idx1-ubyte  emnist-letters-train-images-idx3-ubyte  emnist-letters-train-labels-idx1-ubyte

emnist_mnist:
emnist-mnist-test-images-idx3-ubyte  emnist-mnist-test-labels-idx1-ubyte  emnist-mnist-train-images-idx3-ubyte  emnist-mnist-train-labels-idx1-ubyte
```

## Usage examples

Trained models can be downloaded from Github Releases. Model files should be put in the main directory of this project.

All notebooks with actual models contain a section called "Manual test". Code in this section loads a model and displays a widget that allows user to draw a digit and see how the model classifies it. While this of course is not a good way to measure accuracy of any model, it allows users to quickly test models with natural input. It's of course recommended to use a graphic tablet (if possible) while testing the model.

### Notebooks

#### Info

* info.ipynb - sources and miscellaneous information about the project

#### Models

##### Digits and letters

* cnn.ipynb - Convolutional Neural Network

##### Digits

* svm.ipynb - Support Vector Machines with deskewing
* ova.ipynb - Simple one-vs-all logistic regression classifier
* neural_network.ipynb - Dense neural network 
* neural_network_2.ipynb - Dense neural network with dropout and batch normalization

