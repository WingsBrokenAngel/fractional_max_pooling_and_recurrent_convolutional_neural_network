# Fractional Max Pooling and Recurrent Convolutional Neural Network

## Requirements
- Python3.6
- TensorFlow-gpu 1.13
- NumPy
- Keras

## Fractional Max Pooling
This repository is implemented based on [Fractional Max-Pooling](https://arxiv.org/abs/1412.6071) which is written by Benjamin Graham in 2014. It is the top-1 model in [CIFAR-10 result list](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130).

![The effect of fractional max-pooling](./imgs/fractional_max_pooling.png)

#### Usage
`python train_dev_model.py [--name model_name] [--gpu gpu_idx] [--lr initial_learning_rate] [--drop drop_rate] [--filters filters_number] [--wdecay weight_decay]`

#### MNIST
The input images are resized to (38, 38) which is slightly different with the size in that paper. The model is trained without dropout and data argumentation. The overall structure of the model can be expressed as follows: 
```
(32nC2 - FMP 2^{1/2})_6 - C2 - C1 - output.
```
Here is the comparison table on error rates.  

| The number of repeated tests 	| pseudorandom overlapping 	| random overlapping 	|
|---------------------------	|----------------------		|	--------------		|
| 1 test (mine)				   	|   0.53%                  	|	0.45%				|
| 12 test (mine)                |   0.34%                  	|	0.32%				|
| 1 test (paper)				|	0.44%					|	0.50%				|
| 12 test (paper)				|   0.34%					|	0.32%				|



##### Model Checkpoint
- pseudorandom overlapping
	- [meta](https://cloud.tsinghua.edu.cn/f/4773c8f9ca694b9dbdc4/?dl=1) 
	SHA256: e1593968648bb6665e2cede56b793945ef6369e89e8fe24bf1ab8bafb8d73c07
	- [index](https://cloud.tsinghua.edu.cn/f/781d47b47ee549d9831e/?dl=1) 
	SHA256: 14b991f3d0a4baedb8da761181130b48484eda9ec86d08683f9034141f1f8e5a
	- [data](https://cloud.tsinghua.edu.cn/f/fcc97c71d2c74c38b527/?dl=1) 
	SHA256: f71c969600dfac18b6a0c5af2702319dc97139b38e79a4c19046abf0b593157b
- random overlapping
	- [meta](https://cloud.tsinghua.edu.cn/f/df7e7aef99324e34ae2d/?dl=1) 
	SHA256: 483c51833be1078a172108ae8b743d1c888264ad18558f3f2fdd710e2e23c893
	- [index](https://cloud.tsinghua.edu.cn/f/b2d4eb813b43440fb358/?dl=1) 
	SHA256: 83bea54b67784e89064b7fb003e512a7ed9544b332ab2ce843c46651b01ad3cd
	- [data](https://cloud.tsinghua.edu.cn/f/a9724b756c164574aab3/?dl=1) 
	SHA256: 7677d1138e39191c6c7da8029e611b23f775e966ac4b07a6a582cbb10b5f5c17

#### CIFAR10
The model structure can be written as follows: `(96nC2 - FMP 2^{1/3})_7 - C1 - output` which is a simplified network. 
The size of input image is 32. 
Fraction max pooling layer is too slow on Tensorflow and it is not implemented on PyTorch. 

| The number of repeated tests 	| pseudorandom overlapping 	|
|---------------------------	|----------------------		|
| 1 test (mine)				   	|   10.76%                  |
| 12 test (mine)                |   0.34%                  	|

#### CIFAR100
The overall structure of the model can be represented as follows:
```
(128nC2 - FMP 2^{1/3})_6 - C2 - C1 - output
```
The number of layers is reduced since it is too time consuming to train the network described in the paper and its performance is not very good on the train set and validation set. The input images are resized to (46, 46). The model is trained without dropout and data argumentation.

| The number of repeated tests 	| pseudorandom overlapping 	| 
|---------------------------	|----------------------		|
| 1 test (mine)				   	|   ---                  	|	
| 12 test (mine)                |   ---                 	|	


## Recurrent Convolutional Neural Network
The code of this part is implemented according to [Recurrent Convolutional Neural Network for Object Recognition](https://ieeexplore.ieee.org/document/7298958/). 
Local response normalization is replaced by batch normalization. Only last hidden layer is followed by a dropout layer. Adam optimizer is used to optimize the model. Learning rate decays exponentially. Image shift and horizental flip are utilized for data augmentation.

![Recurrent Convolutional Neural Network](./imgs/recurrent_convolutional_neural_network.jpg)

### Usage
`python train_dev_model.py [--name model_name] [--gpu gpu_idx] [--lr initial_learning_rate] [--drop drop_rate] [--filters filters_number] [--wdecay weight_decay]`

### MNIST

| model                      	| Error Rate            | 
|---------------------------	|----------------------	|
| filters 96 (one crop)			|   0.93%               |	
| filters 128 (one crop)        |   0.87%               |	
| filters 160 (one crop)		|	0.86%				|
| filters 32 (paper)			| 	0.42%				|
| filters 64 (paper)			| 	0.32%				|
| filters 96 (paper)			|	0.31%				|

### CIFAR10

| model                      	| Error Rate            | 
|---------------------------	|----------------------	|
| 96 filters (one crop)			|   10.53%              |
| 128 filters (one crop)		|	9.21%				|
| 160 filters (one crop)		|	8.89%				|
| 96 filters (paper, nine crop) |   7.37%               |
| 128 filters (paper, nine crop)| 	7.24%				|
| 160 filters (paper, nine crop)|	7.09%				|

### CIFAR100
The model reported by the paper is trained without data augmentation. Our model adopt the same hyper-parameters from the model for CIFAR10.

| model                      	| Error Rate            | 
|---------------------------	|----------------------	|
| 96 filters (one crop)			|   39.58%              |
| 128 filters (one crop)		|	36.26%				|
| 160 filters (one crop)		|	34.75%				|
| 96 filters (paper, nine crop) |   34.18%              |
| 128 filters (paper, nine crop)|	32.59%				|
| 160 filters (paper, nine crop)|	31.75%				|	
