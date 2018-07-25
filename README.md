# Semantic Segmentation
### Introduction
In this project, a fully convolutional neural network (FCN) was implemented to perform semantic segmentation of the pixels of road in the images from Kitti Road Dataset. 

### Approach:

#### Network Architecture:
For this semantic segmentation task I used a pre-trained VGG-16 network (trained on 'imagenet') and adding additional 1x1 layers and skip conections to build a FCN that was used for training.
The total number of classes for this task is two as we are performing a binary classification task to segment road vs non-road pixels from the images.
Final convolution layer from VGG-16 i.e. layer 7 was input to 1x1 convolution with depth being equal to the desired clases. The other intermediate convolution layers from VGG-16 i.e. layer3 and layer4
are used as skip connection along with 1x1 convolutions and upsampled later to finally give the network output.

HYPERPARAMETERS:

Batch size = 5
Training Epochs = 50
L2 regularizer = 1e-3
learning rate = 0.0009
keep probability = 0.5

#### RESULTS:
The final cross entropy loss after 50 epochs was reduced to 0.026 !

The results on the test set using the save_inference_samples from the helper function:

![image1](./runs/1532462630.7094371/um_000000.png)

![image2](./runs/1532462630.7094371/um_000010.png)

![iamge3](./runs/1532462630.7094371/um_000030.png) 

### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  
Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.
