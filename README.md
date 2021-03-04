# Segmentation of damaged brain in acute ischemic stroke patients using early fusion multi-input CNN

## Release v1.0
It contains the code described in the paper "Segmentation of damaged brain in acute ischemic stroke patients using early fusion multi-input CNN"


### 1 - Abstract
Time is a fundamental factor during stroke treatments. A fast automatic approach that segments the ischemic regions helps the treatment decisions. In clinical use today, a set of color-coded parametric maps generated from computed tomography perfusion (CTP) images are investigated manually to decide a treatment plan.
We propose an automatic method based on a neural network using color-coded parametric maps to segment the two ischemic regions: dead tissue (core) and tissue at risk (penumbra) in patients affected by an acute ischemic stroke. Our model is based on a convolution-deconvolution bottleneck structure with a multi-input and early data-fusion. To address the data imbalance issue, we use a loss function based on the focal Tversky index. Our architecture demonstrates effective performances and results comparable to the ground truth annotated by neuroradiologists.

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
    <th class="tg-9wq8" colspan="4">F1-score (AIS)</th>
    <th class="tg-9wq8" colspan="2" rowspan="2">F1-score (AIS+WIS)</th>
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
    <td class="tg-9wq8">Penumbra</td>
    <td class="tg-9wq8">Core</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8"><a href="Setting/Model_F(PMs).json">Model_F(PMs)</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8" rowspan="4">Frozen</td>
    <td class="tg-9wq8">0.739</td>
    <td class="tg-9wq8">0.698</td>
    <td class="tg-9wq8">0.559</td>
    <td class="tg-9wq8">0.571</td>
    <td class="tg-9wq8">0.72</td>
    <td class="tg-9wq8">0.692</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Setting/Model_F(PMs,M).json">Model_F(PMs,M)</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8">0.74</td>
    <td class="tg-9wq8">0.706</td>
    <td class="tg-9wq8">0.601</td>
    <td class="tg-9wq8">0.537</td>
    <td class="tg-9wq8">0.725</td>
    <td class="tg-9wq8">0.698</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Setting/Model_F(PMs,N).json">Model_F(PMs,N)</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">0.753</td>
    <td class="tg-9wq8">0.74</td>
    <td class="tg-9wq8">0.674</td>
    <td class="tg-9wq8">0.492</td>
    <td class="tg-9wq8">0.746</td>
    <td class="tg-9wq8">0.732</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Setting/Model_F(PMs,M,N).json">Model_F(PMs,M,N)</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">0.741</td>
    <td class="tg-9wq8">0.728</td>
    <td class="tg-9wq8">0.665</td>
    <td class="tg-9wq8">0.524</td>
    <td class="tg-9wq8">0.733</td>
    <td class="tg-9wq8">0.722</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Setting/Model_U(PMs).json">Model_U(PMs)</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8" rowspan="4">Unfrozen</td>
    <td class="tg-9wq8">0.752</td>
    <td class="tg-9wq8">0.74</td>
    <td class="tg-9wq8">0.668</td>
    <td class="tg-9wq8">0.621</td>
    <td class="tg-9wq8">0.746</td>
    <td class="tg-9wq8">0.736</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Setting/Model_U(PMs,M).json">Model_U(PMs,M)</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8">0.749</td>
    <td class="tg-9wq8">0.751</td>
    <td class="tg-9wq8">0.659</td>
    <td class="tg-9wq8">0.641</td>
    <td class="tg-9wq8">0.742</td>
    <td class="tg-9wq8">0.748</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Setting/Model_U(PMs,N).json">Model_U(PMs,N)</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">0.764</td>
    <td class="tg-9wq8">0.76</td>
    <td class="tg-9wq8">0.636</td>
    <td class="tg-9wq8">0.535</td>    
    <td class="tg-9wq8">0.752</td>
    <td class="tg-9wq8">0.751</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Setting/Model_U(PMs,M,N).json">Model_U(PMs,M,N)</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">X</td>
    <td class="tg-c3ow">0.763</td>
    <td class="tg-c3ow">0.734</td>
    <td class="tg-c3ow">0.63</td>
    <td class="tg-c3ow">0.555</td>
    <td class="tg-9wq8">0.749</td>
    <td class="tg-9wq8">0.727</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Setting/Model_G(PMs).json">Model_G(PMs)</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8" rowspan="4">Gradual<br>Fine-tuning</td>
    <td class="tg-9wq8">0.769</td>
    <td class="tg-9wq8">0.752</td>
    <td class="tg-9wq8">0.59</td>
    <td class="tg-9wq8">0.562</td>
    <td class="tg-9wq8">0.75</td>
    <td class="tg-9wq8">0.746</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Setting/Model_G(PMs,M).json">Model_G(PMs,M)</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8">0.756</td>
    <td class="tg-9wq8">0.757</td>
    <td class="tg-9wq8">0.73</td>
    <td class="tg-9wq8">0.598</td>
    <td class="tg-9wq8">0.753</td>
    <td class="tg-9wq8">0.753</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Setting/Model_G(PMs,N).json">Model_G(PMs,N)</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">0.767</td>
    <td class="tg-9wq8">0.759</td>
    <td class="tg-9wq8">0.644</td>
    <td class="tg-9wq8">0.595</td>
    <td class="tg-9wq8"><b>0.757</b></td>
    <td class="tg-9wq8"><b>0.754</b></td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Setting/Model_G(PMs,M,N).json">Model_G(PMs,M,N)</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">0.761</td>
    <td class="tg-9wq8">0.746</td>
    <td class="tg-9wq8">0.677</td>
    <td class="tg-9wq8">0.557</td>
    <td class="tg-9wq8">0.754</td>
    <td class="tg-9wq8">0.74</td>
  </tr>
</tbody>
</table>

### 2.2 - Link to paper
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
