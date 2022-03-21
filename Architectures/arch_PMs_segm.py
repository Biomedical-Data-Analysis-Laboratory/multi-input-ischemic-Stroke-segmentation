from Model import constants
from Utils import general_utils, spatial_pyramid

from tensorflow.keras import layers, models, regularizers, initializers
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, Dropout, Concatenate
import tensorflow.keras.backend as K
from tensorflow.keras.applications import VGG16


################################################################################
# Class that define a PM object
class PM_obj(object):
    def __init__(self, name, params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch):
        self.name = ("_" + name)
        self.input_shape = (constants.getM(), constants.getN(), 3)

        # Create base model
        self.base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
        self.base_model._name += self.name
        for layer in self.base_model.layers: layer._name += self.name
        # Freeze base model
        self.base_model.trainable = False if params["trainable"]==0 else True
        self.input = self.base_model.input

        # Creating dictionary that maps layer names to the layers
        self.layer_dict = dict([(layer.name, layer) for layer in self.base_model.layers])

        # Conv layers after the VGG16
        self.conv_1 = layers.Conv2D(128, kernel_size=(3, 3), padding='same',activation=activ_func,
                                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(self.base_model.output)
        self.conv_2 = layers.Conv2D(128, kernel_size=(3, 3), padding='same',activation=activ_func,
                                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(self.conv_1)
        if batch: self.conv_2 = layers.BatchNormalization()(self.conv_2)
        self.conv_2 = Dropout(params["dropout"][name+".1"])(self.conv_2)


################################################################################
# mJ-Net model version for the parametric maps as input
def PMs_segmentation(params, to_categ, moreinfo, batch=True):

    activ_func = 'relu'
    l1_l2_reg = regularizers.l1_l2(l1=1e-6, l2=1e-5)
    kernel_init = "glorot_uniform"  # Xavier uniform initializer.
    kernel_constraint, bias_constraint = max_norm(2.), max_norm(2.)

    cbf = PM_obj("cbf", params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch)
    cbv = PM_obj("cbv", params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch)
    ttp = PM_obj("ttp", params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch)
    tmax = PM_obj("tmax", params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch)
    PMS = [cbf, cbv, ttp, tmax]

    if "mip" in moreinfo.keys() and moreinfo["mip"]==1:
        mip = PM_obj("mip", params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch)
        PMS.append(mip)

    layersAfterTransferLearning, inputs, block5_conv3, block4_conv3, block3_conv3, block2_conv2, block1_conv2 = [], [], [], [], [], [], []

    for pm in PMS:
        layersAfterTransferLearning.append(pm.conv_2)
        inputs.append(pm.input)
        block5_conv3.append(pm.layer_dict["block5_conv3" + pm.name].output)
        block4_conv3.append(pm.layer_dict["block4_conv3" + pm.name].output)
        block3_conv3.append(pm.layer_dict["block3_conv3" + pm.name].output)
        block2_conv2.append(pm.layer_dict["block2_conv2" + pm.name].output)
        block1_conv2.append(pm.layer_dict["block1_conv2" + pm.name].output)

    # MORE INFO as input = NIHSS score, age, gender
    input_dim = 0
    concat_input = []
    flag_dense = 0

    if "nihss" in moreinfo.keys() and moreinfo["nihss"]==1:
        NIHSS_input = layers.Input(shape=(1,))
        input_dim += 1
        concat_input.append(NIHSS_input)
        flag_dense = 1
    if "age" in moreinfo.keys() and moreinfo["age"]==1:
        age_input = layers.Input(shape=(1,))
        input_dim += 1
        concat_input.append(age_input)
        flag_dense = 1
    if "gender" in moreinfo.keys() and moreinfo["gender"]==1:
        gender_input = layers.Input(shape=(1,))
        input_dim += 1
        concat_input.append(gender_input)
        flag_dense = 1

    if flag_dense:
        if input_dim==1: conc = concat_input[0]
        else: conc = Concatenate(1)(concat_input)
        dense_1 = layers.Dense(100, input_dim=input_dim, activation="relu")(conc)
        dense_2 = layers.Dense(layersAfterTransferLearning[0].shape[1]*layersAfterTransferLearning[0].shape[2], activation="relu")(dense_1)
        out = layers.Reshape((layersAfterTransferLearning[0].shape[1], layersAfterTransferLearning[0].shape[2], 1))(dense_2)
        moreinfo_mdl = models.Model(inputs=concat_input, outputs=[out])
        inputs = [inputs, moreinfo_mdl.input]
        layersAfterTransferLearning.append(moreinfo_mdl.output)

    conc_layer = Concatenate(-1)(layersAfterTransferLearning)

    transp_1 = layers.Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), padding='same',activation=activ_func,
                                      kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                                      kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conc_layer)

    block5_conv3_conc = Concatenate(-1)(block5_conv3)
    up_1 = Concatenate(-1)([transp_1,block5_conv3_conc])

    # going up with the layers
    up_2 = upLayer(up_1, 128, block4_conv3, len(PMS), activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint)
    up_3 = upLayer(up_2, 64, block3_conv3, len(PMS), activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint)
    up_4 = upLayer(up_3, 32, block2_conv2, len(PMS), activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint)
    up_5 = upLayer(up_4, 16, block1_conv2, len(PMS), activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint)

    final_conv_1 = layers.Conv2D(16, kernel_size=(3, 3), padding='same',activation=activ_func,
                                 kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(up_5)
    if batch: final_conv_1 = layers.BatchNormalization()(final_conv_1)
    # general_utils.print_int_shape(final_conv_1)
    final_conv_2 = layers.Conv2D(16, kernel_size=(3, 3), padding='same',activation=activ_func,
                                 kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(final_conv_1)
    if batch: final_conv_2 = layers.BatchNormalization()(final_conv_2)
    # general_utils.print_int_shape(final_conv_2)

    act_name = "sigmoid"
    n_chann = 1
    shape_output = (constants.getM(), constants.getN())

    # set the softmax activation function if the flag is set
    if to_categ:
        act_name = "softmax"
        n_chann = len(constants.LABELS)
        shape_output = (constants.getM(), constants.getN(), n_chann)

    final_conv_3 = layers.Conv2D(n_chann, kernel_size=(1, 1), activation=act_name, padding='same',
                                 kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(final_conv_2)
    # general_utils.print_int_shape(final_conv_3)

    y = layers.Reshape(shape_output)(final_conv_3)
    # general_utils.print_int_shape(y)

    model = models.Model(inputs=inputs, outputs=[y])
    return model


################################################################################
# Helpful function to define up-layers based on the previous layer
def upLayer(prev_up, filters, block, howmanypms, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint):
    conv = layers.Conv2D(filters * howmanypms, kernel_size=(3, 3), padding='same',activation=activ_func,
                         kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                         kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(prev_up)
    conv = layers.Conv2D(filters * howmanypms, kernel_size=(3, 3), padding='same',activation=activ_func,
                         kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                         kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv)
    transp = layers.Conv2DTranspose(filters * howmanypms, kernel_size=(2, 2), strides=(2, 2), padding='same',
                                    activation=activ_func,kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv)

    block_conc = Concatenate(-1)(block)
    up = Concatenate(-1)([transp, block_conc])

    return up
