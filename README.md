# Multi-input segmentation of damaged brain in acute ischemic stroke patients using slow fusion with skip connection

## Release v1.0
It contains the code described in the paper "Multi-input segmentation of damaged brain in acute ischemic stroke patients using slow fusion with skip connection".


### 1 - Abstract
Time is a fundamental factor during stroke treatments. A fast, automatic approach that segments the ischemic regions helps treatment decisions. In clinical use today, a set of color-coded parametric maps generated from computed tomography perfusion (CTP) images are investigated manually to decide a treatment plan.
We propose an automatic method based on a neural network using a set of parametric maps to segment the two ischemic regions (core and penumbra) in patients affected by acute ischemic stroke.
Our model is based on a convolution-deconvolution bottleneck structure with multi-input and slow fusion.
A loss function based on the focal Tversky index addresses the data imbalance issue.
The proposed architecture demonstrates effective performance and results comparable to the ground truth annotated by neuroradiologists.
A Dice coefficient of 0.81 for penumbra and 0.52 for core over the large vessel occlusion test set is achieved.

![alt text](images/intro-pipeline.png?raw=true)

### 2 - Code
Code for this repository will be uploaded upon paper acceptance

### 2.1 - Table Validation Results 

<table class="tg">
<thead>
  <tr>
    <th class="tg-9wq8" rowspan="3">Model</th>
    <th class="tg-9wq8" colspan="3" rowspan="2">Input</th>
    <th class="tg-9wq8" rowspan="3">Layer<br>Weights</th>
    <th class="tg-9wq8" colspan="4">Dice coeff. (Avg)&plusmn;SD </th>
  </tr>
  <tr>
    <td class="tg-9wq8" colspan="2">LVO</td>
    <td class="tg-9wq8" colspan="2">Non-LVO</td>
  </tr>
  <tr>
    <td class="tg-9wq8">PMs</td>
    <td class="tg-9wq8">MIP</td>
    <td class="tg-9wq8">NIHSS</td>
    <td class="tg-9wq8">Penumbra</td>
    <td class="tg-9wq8">Core</td>
    <td class="tg-9wq8">Penumbra</td>
    <td class="tg-9wq8">Core</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8"><a href="Settings/Model_F(PMs).json">Model_F(PMs)</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8" rowspan="4">Frozen</td>
    <td class="tg-9wq8">0.71&plusmn;0.1</td>
    <td class="tg-9wq8"><b>0.37&plusmn;0.3</b></td>
    <td class="tg-9wq8">0.27&plusmn;0.3</td>
    <td class="tg-9wq8">0.22&plusmn;0.3</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Settings/Model_F(PMs,M).json">Model_F(PMs,M)</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8">0.69&plusmn;0.2</td>
    <td class="tg-9wq8">0.36&plusmn;0.3</td>
    <td class="tg-9wq8">0.29&plusmn;0.3</td>
    <td class="tg-9wq8">0.20&plusmn;0.3</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Settings/Model_F(PMs,N).json">Model_F(PMs,N)</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">0.70&plusmn;0.2</td>
    <td class="tg-9wq8">0.36&plusmn;0.3</td>
    <td class="tg-9wq8">0.29&plusmn;0.3</td>
    <td class="tg-9wq8">0.16&plusmn;0.2</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Settings/Model_F(PMs,M,N).json">Model_F(PMs,M,N)</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">0.68&plusmn;0.2</td>
    <td class="tg-9wq8">0.34&plusmn;0.3</td>
    <td class="tg-9wq8">0.30&plusmn;0.3</td>
    <td class="tg-9wq8">0.18&plusmn;0.3</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Settings/Model_U(PMs).json">Model_U(PMU)</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8" rowspan="4">Unfrozen</td>
    <td class="tg-9wq8">0.70&plusmn;0.2</td>
    <td class="tg-9wq8">0.34&plusmn;0.3</td>
    <td class="tg-9wq8">0.29&plusmn;0.4</td>
    <td class="tg-9wq8">0.24&plusmn;0.3</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Settings/Model_F(PMs,M).json">Model_F(PMs,M)</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8">0.70&plusmn;0.2</td>
    <td class="tg-9wq8">0.36&plusmn;0.3</td>
    <td class="tg-9wq8">0.34&plusmn;0.3</td>
    <td class="tg-9wq8"><b>0.24&plusmn;0.3</b></td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Settings/Model_U(PMs,N).json">Model_U(PMs,N)</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8"><b>0.72&plusmn;0.2</b></td>
    <td class="tg-9wq8">0.36&plusmn;0.3</td>
    <td class="tg-9wq8">0.29&plusmn;0.3</td>
    <td class="tg-9wq8">0.23&plusmn;0.3</td>    
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Settings/Model_U(PMs,M,N).json">Model_U(PMs,M,N)</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">X</td>
    <td class="tg-c3ow">0.71&plusmn;0.2</td>
    <td class="tg-c3ow">0.36&plusmn;0.3</td>
    <td class="tg-c3ow">0.32&plusmn;0.3</td>
    <td class="tg-c3ow">0.22&plusmn;0.3</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Settings/Model_G(PMs).json">Model_G(PMs)</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8" rowspan="4">Gradual<br>Fine-tuning</td>
    <td class="tg-9wq8">0.71&plusmn;0.2</td>
    <td class="tg-9wq8">0.35&plusmn;0.3</td>
    <td class="tg-9wq8">0.30&plusmn;0.3</td>
    <td class="tg-9wq8">0.19&plusmn;0.3</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Settings/Model_G(PMs,M).json">Model_G(PMs,M)</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8">0.69&plusmn;0.2</td>
    <td class="tg-9wq8">0.35&plusmn;0.3</td>
    <td class="tg-9wq8"><b>0.34&plusmn;0.6</b></td>
    <td class="tg-9wq8">0.22&plusmn;0.4</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Settings/Model_G(PMs,N).json">Model_G(PMs,N)</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8"><b>0.72&plusmn;0.2</b></td>
    <td class="tg-9wq8"><b>0.37&plusmn;0.3</b></td>
    <td class="tg-9wq8">0.31&plusmn;0.3</td>
    <td class="tg-9wq8">0.21&plusmn;0.3</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Settings/Model_G(PMs,M,N).json">Model_G(PMs,M,N)</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">0.68&plusmn;0.2</td>
    <td class="tg-9wq8">0.34&plusmn;0.3</td>
    <td class="tg-9wq8">0.31&plusmn;0.3</td>
    <td class="tg-9wq8">0.18&plusmn;0.3</td>
  </tr>
</tbody>
</table>

### 2.2 - Link to paper
https://doi.org/10.7557/18.6223

https://arxiv.org/abs/2203.10039


### 3 - Dependecies:
```
pip install -r requirements.txt
```

### 4 - Usage
Assuming that you already have a dataset to work with, you can use a json file to define the setting of your model.

Refer to  [Setting_explained.json](Settings/Setting_explained.json) for explanations of the various settings.


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
