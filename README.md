# Image segmentation of ischemic stroke regions (dead and salvable tissues)

## Release v1.0
It contains the code described in the paper "Segmentation of damaged brain in acute ischemic stroke patients using early fusion multi-input CNN"

### 1 - Code
Code for this repository will be uploaded upon paper acceptance

### 2 - Abstract
Time is a fundamental factor during stroke treatments. A fast automatic approach that segments the ischemic regions helps the treatment decisions. In clinical use today, a set of color-coded parametric maps generated from computed tomography perfusion (CTP) images are investigated manually to decide a treatment plan.
We propose an automatic method based on a neural network using color-coded parametric maps to segment the two ischemic regions: dead tissue (core) and tissue at risk (penumbra) in patients affected by an acute ischemic stroke. Our model is based on a convolution-deconvolution bottleneck structure with a multi-input and early data-fusion. To address the data imbalance issue, we use a loss function based on the focal Tversky index. Our architecture demonstrates effective performances and results comparable to the ground truth annotated by neuroradiologists.

![alt text](images/intro-pipeline.png?raw=true)

### 2.1 - Link to paper
```
TBA
```


### 3 - Dependecies:
```
pip install -r requirements.txt
```

### 4 - Usage
Assuming that you already have a dataset to work with, you can use a json file to define the setting of your model.

Refer to  [Setting_explained.json](Setting/Setting_explained.json) for explanations of the various settings.


### 4.1 Train/Test

```
Usage: python main.py gpu sname
                [-h] [-v] [-d] [-o] [-s SETTING_FILENAME] [-t TILE] [-dim DIMENSION] [-c {2,3,4}]

    positional arguments:
      gpu                   Give the id of gpu (or a list of the gpus) to use
      sname                 Select the setting filename

    optional arguments:
      -h, --help            show this help message and exit
      -v, --verbose         Increase output verbosity
      -d, --debug           DEBUG mode
      -o, --original        Set the shape of the testing dataset to be compatible with the original shape
                            (T,M,N) [time in front]
      -pm, --pm             Set the flag to train the parametric maps as input
      -t TILE, --tile TILE  Set the tile pixels dimension (MxM) (default = 16)
      -dim DIMENSION, --dimension DIMENSION
                            Set the dimension of the input images (width X height) (default = 512)
      -c {2,3,4}, --classes {2,3,4}
                            Set the # of classes involved (default = 4)
      -w, --weights         Set the weights for the categorical losses

```


### 5 - How to cite our work
The code is released free of charge as open-source software under the GPL-3.0 License. Please cite our paper when you have used it in your study.
```
TBA
```

### Got Questions?
Email me at luca.tomasetti@uis.no
