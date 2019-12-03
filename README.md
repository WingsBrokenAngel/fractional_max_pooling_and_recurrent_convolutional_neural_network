# fractional_max_pooling
This repository is implemented based on [Fractional Max-Pooling](https://arxiv.org/abs/1412.6071) which is written by Benjamin Graham in 2014. It is the top-1 model in [CIFAR-10 result list](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130).
## Requirements
- Python3.6
- TensorFlow-gpu 1.13
- NumPy
- PIL

## Model Description
![The effect of fractional max-pooling](./imgs/fractional_max_pooling.png)
## MNIST
The input images are resized to (38, 38) which is slightly different with the size in that paper. The model is trained without dropout and data argumentation. The overall structure of the model can be expressed as follows: `(32nC2 - FMP \sqrt{2})_6-C2-C1-output`.

| The number of repeated tests 	| pseudorandom overlapping 	|
|---------------------------|----------------------|
| 1 test (ours)				   	|   0.53%                  	|
| 12 test (ours)                |   0.34%                  	|
| 1 test (paper)				|	0.44%					|
| 12 test (paper)				|   0.34%					|

### Usage
- train: `python train_dev_test_FMP.py WEIGHT_DECAY_RATE GPU_DEVICE_NUM MODEL_NAME`
- test: `python testFMP.py ./PATH/TO/MODEL GPU_DEVICE_NUM`

### Model Checkpoint
- [meta](https://cloud.tsinghua.edu.cn/f/4773c8f9ca694b9dbdc4/?dl=1)
    - SHA256: e1593968648bb6665e2cede56b793945ef6369e89e8fe24bf1ab8bafb8d73c07
- [index](https://cloud.tsinghua.edu.cn/f/781d47b47ee549d9831e/?dl=1)
    - SHA256: 14b991f3d0a4baedb8da761181130b48484eda9ec86d08683f9034141f1f8e5a
- [data](https://cloud.tsinghua.edu.cn/f/fcc97c71d2c74c38b527/?dl=1)
    - SHA256: f71c969600dfac18b6a0c5af2702319dc97139b38e79a4c19046abf0b593157b