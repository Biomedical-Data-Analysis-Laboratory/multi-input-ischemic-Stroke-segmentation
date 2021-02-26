# Image segmentation of ischemic stroke regions (dead and salvable tissues)

## Release v1.0
It contains the code described in the paper "Segmentation of damaged brain in acute ischemic stroke patients using early fusion multi-input CNN"


### 1 - Abstract
Time is a fundamental factor during stroke treatments. A fast automatic approach that segments the ischemic regions helps the treatment decisions. In clinical use today, a set of color-coded parametric maps generated from computed tomography perfusion (CTP) images are investigated manually to decide a treatment plan.
We propose an automatic method based on a neural network using color-coded parametric maps to segment the two ischemic regions: dead tissue (core) and tissue at risk (penumbra) in patients affected by an acute ischemic stroke. Our model is based on a convolution-deconvolution bottleneck structure with a multi-input and early data-fusion. To address the data imbalance issue, we use a loss function based on the focal Tversky index. Our architecture demonstrates effective performances and results comparable to the ground truth annotated by neuroradiologists.

![alt text](images/intro-pipeline.png?raw=true)

### 2 - Code
Code for this repository will be uploaded upon paper acceptance

### 2.1 - Table Results 

<table class="tg">
<thead>
  <tr>
    <th class="tg-9wq8" rowspan="3">Model</th>
    <th class="tg-9wq8" colspan="3" rowspan="2">Input</th>
    <th class="tg-9wq8" rowspan="3">Layer<br>Weights</th>
    <th class="tg-9wq8" colspan="4">F1-score</th>
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
    <td class="tg-9wq8"><a href="Setting/Baseline.json">Baseline</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8" rowspan="4">Frozen</td>
    <td class="tg-9wq8">0.739</td>
    <td class="tg-9wq8">0.698</td>
    <td class="tg-9wq8">0.559</td>
    <td class="tg-9wq8">0.571</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Setting/Mdl_F_PmM.json">Mdl_F_PmM</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8"></td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Setting/Mdl_F_PmN.json">Mdl_F_PmN</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">0.753</td>
    <td class="tg-9wq8">0.74</td>
    <td class="tg-9wq8">0.674</td>
    <td class="tg-9wq8">0.492</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Setting/Mdl_F_PmMN.json">Mdl_F_PmMN</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">0.741</td>
    <td class="tg-9wq8">0.728</td>
    <td class="tg-9wq8">0.665</td>
    <td class="tg-9wq8">0.524</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Setting/Mdl_U_Pm.json">Mdl_U_Pm</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8" rowspan="4">Unfrozen</td>
    <td class="tg-9wq8">0.752</td>
    <td class="tg-9wq8">0.74</td>
    <td class="tg-9wq8">0.668</td>
    <td class="tg-9wq8">0.621</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Setting/Mdl_U_PmM.json">Mdl_U_PmM</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8">0.749</td>
    <td class="tg-9wq8">0.751</td>
    <td class="tg-9wq8">0.659</td>
    <td class="tg-9wq8">0.641</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Setting/Mdl_U_PmN.json">Mdl_U_PmN</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8"></td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Setting/Mdl_U_PmMN.json">Mdl_U_PmMN</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">X</td>
    <td class="tg-c3ow">0.764</td>
    <td class="tg-c3ow">0.755</td>
    <td class="tg-c3ow">0.609</td>
    <td class="tg-c3ow">0.624</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Setting/Mdl_G_Pm.json">Mdl_G_Pm</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8" rowspan="4">Gradual<br>Fine-tuning</td>
    <td class="tg-9wq8">0.769</td>
    <td class="tg-9wq8">0.752</td>
    <td class="tg-9wq8">0.59</td>
    <td class="tg-9wq8">0.562</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Setting/Mdl_G_PmM.json">Mdl_G_PmM</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8"></td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Setting/Mdl_G_PmN.json">Mdl_G_PmN</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8"></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">0.767</td>
    <td class="tg-9wq8">0.759</td>
    <td class="tg-9wq8">0.644</td>
    <td class="tg-9wq8">0.595</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="Setting/Mdl_G_PmMN.json">Mdl_G_PmMN</a></td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">X</td>
    <td class="tg-9wq8">0.761</td>
    <td class="tg-9wq8">0.746</td>
    <td class="tg-9wq8">0.677</td>
    <td class="tg-9wq8">0.557</td>
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
