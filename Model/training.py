import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import cv2, matplotlib, glob

from Model import constants
from Utils import callback, general_utils

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K


################################################################################
# Return the optimizer based on the setting
def getOptimizer(optInfo):
    optimizer = None
    if optInfo["name"].lower() == "adam":
        optimizer = optimizers.Adam(
            lr=optInfo["lr"],
            beta_1=optInfo["beta_1"],
            beta_2=optInfo["beta_2"],
            epsilon=None if optInfo["epsilon"] == "None" else optInfo["epsilon"],
            decay=optInfo["decay"],
            amsgrad=True if "amsgrad" in optInfo.keys() and optInfo["amsgrad"] == "True" else False,
            clipvalue=0.5
        )
    elif optInfo["name"].lower() == "sgd":
        optimizer = optimizers.SGD(
            learning_rate=optInfo["learning_rate"],
            decay=optInfo["decay"],
            momentum=optInfo["momentum"],
            nesterov=True if optInfo["nesterov"] == "True" else False,
            clipvalue=0.5
        )
    elif optInfo["name"].lower() == "rmsprop":
        optimizer = optimizers.RMSprop(
            learning_rate=optInfo["learning_rate"],
            rho=0.9,
            momentum=0.0,
            epsilon=1e-07,
            centered=False
        )
    elif optInfo["name"].lower() == "adadelta":
        optimizer = optimizers.Adadelta(
            learning_rate=optInfo["learning_rate"],
            rho=0.95,
            epsilon=1e-07,
            clipvalue=0.5
        )

    return optimizer


################################################################################
# Return the callbacks defined in the setting
def getCallbacks(info, root_path, filename, textFolderPath, dataset, sample_weights, nn_id, add_for_finetuning):
    # add by default the TerminateOnNaN callback
    cbs = [callback.TerminateOnNaN()]

    for key in info.keys():
        # save the weights
        if key == "ModelCheckpoint":
            cbs.append(callback.modelCheckpoint(filename, info[key]["monitor"], info[key]["mode"], info[key]["period"]))
        # stop if the monitor is not improving
        elif key == "EarlyStopping":
            cbs.append(callback.earlyStopping(info[key]["monitor"], info[key]["min_delta"], info[key]["patience"]))
        # reduce the learning rate if the monitor is not improving
        elif key == "ReduceLROnPlateau":
            cbs.append(callback.reduceLROnPlateau(info[key]["monitor"], info[key]["factor"], info[key]["patience"],
                                                  info[key]["min_delta"], info[key]["cooldown"], info[key]["min_lr"]))
        # reduce learning_rate every fix number of epochs
        elif key == "LearningRateScheduler":
            cbs.append(callback.LearningRateScheduler(info[key]["decay_step"], info[key]["decay_rate"]))
        # collect info
        elif key == "CollectBatchStats":
            cbs.append(callback.CollectBatchStats(root_path, filename, textFolderPath, info[key]["acc"]))
        # save the epoch results in a csv file
        elif key == "CSVLogger":
            cbs.append(callback.CSVLogger(textFolderPath, nn_id, add_for_finetuning+info[key]["filename"], info[key]["separator"]))
        elif key == "RocCallback":
            training_data = (dataset["train"]["data"], dataset["train"]["labels"])
            validation_data = (dataset["val"]["data"], dataset["val"]["labels"])
            # # TODO: no model passed!
            # # TODO: filename is different (is the TMP_MODELS not MODELS folder)
            cbs.append(
                callback.RocCallback(training_data, validation_data, model, sample_weights, filename, textFolderPath))
        # elif key=="TensorBoard":
        #     cbs.append(callback.TensorBoard(log_dir=textFolderPath, update_freq=info[key]["update_freq"], histogram_freq=info[key]["histogram_freq"]))

    return cbs


################################################################################
# Fit the model
def fitModel(model, dataset, batch_size, epochs, listOfCallbacks, sample_weights, initial_epoch, save_activation_filter,
             intermediate_activation_path, use_multiprocessing):
    validation_data = None
    if dataset["val"]["data"] is not None and dataset["val"]["labels"] is not None:
        validation_data = (dataset["val"]["data"], dataset["val"]["labels"])

    training = model.fit(dataset["train"]["data"],
                         dataset["train"]["labels"],
                         batch_size=batch_size,
                         epochs=epochs,
                         callbacks=listOfCallbacks,
                         shuffle=True,
                         validation_data=validation_data,
                         sample_weight=sample_weights,
                         initial_epoch=initial_epoch,
                         verbose=1,
                         use_multiprocessing=use_multiprocessing)

    if save_activation_filter: saveIntermediateLayers(model, intermediate_activation_path=intermediate_activation_path)

    return training


################################################################################
# Function that call a fit_generator to load the training dataset on the fly
def fit_generator(model, train_sequence, val_sequence, steps_per_epoch, validation_steps, epochs, listOfCallbacks,
                  initial_epoch, save_activation_filter, intermediate_activation_path, use_multiprocessing, clear):
    multiplier = 16
    # steps_per_epoch is given by the len(train_sequence)*steps_per_epoch_ratio rounded to the nearest integer
    training = model.fit_generator(
        generator=train_sequence,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_sequence,
        validation_steps=validation_steps,
        callbacks=listOfCallbacks,
        initial_epoch=initial_epoch,
        verbose=1,
        max_queue_size=10*multiplier,
        workers=1*multiplier,
        shuffle=True,
        use_multiprocessing=use_multiprocessing)

    if save_activation_filter: saveIntermediateLayers(model, intermediate_activation_path=intermediate_activation_path)

    if clear: K.clear_session()

    return training


################################################################################
# Save the intermediate layers
def saveIntermediateLayers(model, intermediate_activation_path):
    count = 0
    pixels = np.zeros(shape=(constants.getM(), constants.getN(), constants.NUMBER_OF_IMAGE_PER_SECTION))
    path = "/home/prosjekt/PerfusionCT/StrokeSUS/FINAL_TIFF_HU_v1/CTP_01_010/10/*."
    for imagename in np.sort(glob.glob(path + constants.SUFFIX_IMG)):
        img = cv2.imread(imagename, cv2.IMREAD_GRAYSCALE)
        pixels[:, :, count] = general_utils.getSlicingWindow(img, 320, 320)
        count += 1

    pixels = pixels.reshape(1, constants.getM(), constants.getN(), constants.NUMBER_OF_IMAGE_PER_SECTION, 1)

    for layer in model.layers:
        if layer.name != "input_1" and layer.name != "reshape":
            visualizeLayer(model, pixels, layer.name, intermediate_activation_path)


################################################################################
# Function to visualize (save) a single layer given the name
def visualizeLayer(model, pixels, layer_name, intermediate_activation_path, save=True):
    layer_output = model.get_layer(layer_name).output
    intermediate_model = tf.keras.models.Model(inputs=model.input, outputs=layer_output)

    intermediate_prediction = intermediate_model.predict(pixels)
    if save:
        for img_index in range(0, intermediate_prediction.shape[3]):
            for c in range(0, intermediate_prediction.shape[4]):
                cv2.imwrite(intermediate_activation_path + layer_name + str(img_index) + str(c) + ".png",
                            intermediate_prediction[0, :, :, img_index, c])
    else:
        row_size = intermediate_prediction.shape[4]
        col_size = intermediate_prediction.shape[3]

        fig, ax = plt.subplots(row_size, col_size, figsize=(10, 8))

        for row in range(0, row_size):
            for col in range(0, col_size):
                ax[row][col].imshow(intermediate_prediction[0, :, :, col, row], cmap='gray', vmin=0, vmax=255)


################################################################################
# For plotting the loss and accuracy of the trained model
def plotLossAndAccuracy(nn, p_id):
    for key in nn.train.history.keys():
        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        ax.plot(nn.train.history[key], 'r', linewidth=3.0)
        ax.legend([key], fontsize=10)
        ax.set_xlabel('Epochs ', fontsize=16)
        ax.set_ylabel(key, fontsize=16)
        ax.set_title(key + 'Curves', fontsize=16)

        fig.savefig(nn.savePlotFolder + nn.getNNID(p_id) + "_" + key + "_" + str(constants.SLICING_PIXELS) + "_" + str(
            constants.getM()) + "x" + str(constants.getN()) + ".png")
        plt.close(fig)


################################################################################
# For plotting the loss and accuracy of the trained model
def plotMetrics(nn, p_id, list_metrics):
    for metric in list_metrics:
        key = metric["name"]
        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        ax.plot(metric["val"], 'r', linewidth=3.0)
        ax.legend([key], fontsize=10)
        ax.set_xlabel('Batch ', fontsize=16)
        ax.set_ylabel(key, fontsize=16)
        ax.set_title(key + 'Curves', fontsize=16)

        fig.savefig(nn.savePlotFolder + nn.getNNID(p_id) + "_" + key + "_" + str(constants.SLICING_PIXELS) + "_" + str(
            constants.getM()) + "x" + str(constants.getN()) + ".png")
        plt.close(fig)
