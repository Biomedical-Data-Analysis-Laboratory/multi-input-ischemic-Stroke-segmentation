import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import math, cv2, glob, time, os
from typing import Set, Dict, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from Model import constants
from Utils import general_utils, dataset_utils


################################################################################
# https://faroit.com/keras-docs/2.1.3/models/sequential/#fit_generator
class datasetSequence(Sequence):
    def __init__(self, dataframe, indices, sample_weights, x_label, y_label, moreinfo,
                 to_categ, batch_size, back_perc, flagtype="train", loss=None):
        self.indices = indices
        self.dataframe = dataframe.iloc[self.indices]

        self.sample_weights = sample_weights
        self.x_label = x_label
        self.y_label = y_label
        self.moreinfo = moreinfo
        self.batch_size = batch_size
        self.to_categ = to_categ
        self.back_perc = back_perc
        self.flagtype = flagtype
        self.loss = loss

        if self.flagtype != "test":
            # get ALL the rows with label != from background
            self.dataframenoback = self.dataframe.loc[self.dataframe.label != constants.LABELS[0]]
            # also, get a back_perc of rows with label == background
            self.dataframeback = self.dataframe.loc[self.dataframe.label == constants.LABELS[0]]
            if self.back_perc < 100: self.dataframeback = self.dataframeback[:int((len(self.dataframeback)/100)*self.back_perc)]
            # combine the two dataframes
            self.dataframe = pd.concat([self.dataframenoback, self.dataframeback], sort=False)

        self.index_pd_DA: Dict[str, Set[Any]] = {"0": set(),"1": set(),"2": set(),"3": set(),"4": set(),"5": set()}
        self.index_batch = None

    def on_epoch_end(self):
        self.dataframe = self.dataframe.sample(frac=1)  # shuffle the dataframe rows at the end of a epoch

    # Every Sequence must implement the __getitem__ and the __len__ methods
    def __len__(self):
        return math.ceil(len(self.dataframe) / self.batch_size)

    def __getitem__(self, idx):
        start = idx*self.batch_size
        end = (idx+1)*self.batch_size

        batch_index_list = list(range(start,end))
        current_batch = self.dataframe[start:end]
        self.index_batch = current_batch.index

        # empty initialization
        X = np.empty((len(current_batch), constants.getM(), constants.getN(), constants.NUMBER_OF_IMAGE_PER_SECTION, 1))
        Y = np.empty((len(current_batch), constants.getM(), constants.getN()))
        weights = np.empty((len(current_batch),))

        if constants.getUSE_PM(): X = np.empty((len(current_batch), constants.getM(), constants.getN()))

        # reset the index for the data augmentation
        self.index_pd_DA: Dict[str, Set[Any]] = {"0": set(),"1": set(),"2": set(),"3": set(),"4": set(),"5": set()}

        # path to the folder containing the NUMBER_OF_IMAGE_PER_SECTION time point images
        if self.x_label=="pixels":  X, weights = self.getX_pixels(X,current_batch,weights,batch_index_list)
        elif self.x_label== constants.getList_PMS():X, weights = self.getX_PM(weights, current_batch, batch_index_list)

        # path to the ground truth image
        if self.y_label=="ground_truth": Y, weights = self.getY(current_batch,weights)

        # Check if any value is NaN
        if np.isnan(np.unique(Y)).any(): print([gt for gt in current_batch.ground_truth])

        return X, Y, weights

    ################################################################################
    # return the X set and the relative weights based on the pixels column
    def getX_pixels(self, X, current_batch, weights, batch_index_list):
        for index, (_, row) in enumerate(current_batch.iterrows()):
            # weights[index] = self.sample_weights[batch_index_list[index]]
            folder = row[self.x_label]
            coord = row["x_y"]
            data_aug_idx = row["data_aug_idx"]
            # add the index into the correct set
            self.index_pd_DA[str(data_aug_idx)].add(index)

            for timeIndex, filename in enumerate(np.sort(glob.glob(folder + "*" + constants.SUFFIX_IMG))):
                # TODO: for ISLES2018 (to change in the future) --> if the number of time-points per slice
                #  is > constants.NUMBER_OF_IMAGE_PER_SECTION
                if timeIndex >= constants.NUMBER_OF_IMAGE_PER_SECTION: break

                sliceW = general_utils.getSlicingWindow(cv2.imread(filename,cv2.IMREAD_GRAYSCALE),coord[0],coord[1])
                sliceW = general_utils.performDataAugmentationOnTheImage(sliceW, data_aug_idx)

                # reshape it for the correct input in the model
                X[index, :, :, timeIndex, :] = sliceW.reshape(sliceW.shape + (1,))

        return X, weights

    ################################################################################
    # Return the X set and relative weights based on the parametric maps columns
    def getX_PM(self, weights, current_batch, batch_index_list):
        pms = dict()
        nihss, age, gender = [], [], []

        for index, (_, row) in enumerate(current_batch.iterrows()):
            # weights[index] = self.sample_weights[batch_index_list[index]]
            coord = row["x_y"]
            data_aug_idx = row["data_aug_idx"]
            if "nihss" in self.moreinfo.keys() and self.moreinfo["nihss"]==1: nihss.append(int(row["NIHSS"]) if row["NIHSS"]!="-" else 0)
            if "age" in self.moreinfo.keys() and self.moreinfo["age"]==1: age.append(int(row["age"]))
            if "gender" in self.moreinfo.keys() and self.moreinfo["gender"]==1: gender.append(int(row["gender"]))
            # add the index into the correct set
            self.index_pd_DA[str(data_aug_idx)].add(index)

            for pm in self.x_label:
                if pm not in pms.keys(): pms[pm] = []
                totimg = cv2.imread(row[pm])

                if totimg is not None:
                    img = general_utils.getSlicingWindow(totimg, coord[0], coord[1], removeColorBar=True)
                    img = general_utils.performDataAugmentationOnTheImage(img, data_aug_idx)
                    pms[pm].append(img)

        X = [np.array(pms["CBF"]), np.array(pms["CBV"]), np.array(pms["TTP"]), np.array(pms["TMAX"])]
        if "mip" in self.moreinfo.keys() and self.moreinfo["mip"]==1: X.append(np.array(pms["MIP"]))
        if "nihss" in self.moreinfo.keys() and self.moreinfo["nihss"]==1: X.append(np.array(nihss))
        if "age" in self.moreinfo.keys() and self.moreinfo["age"]==1: X.append(np.array(age))
        if "gender" in self.moreinfo.keys() and self.moreinfo["gender"]==1: X.append(np.array(gender))

        return X, weights

    ################################################################################
    # Return the Y set and the weights
    def getY(self, current_batch, weights):
        Y = []
        for aug_idx in self.index_pd_DA.keys():
            for index in self.index_pd_DA[aug_idx]:
                row_index = self.index_batch[index]
                filename = current_batch.loc[row_index][self.y_label]
                coord = current_batch.loc[row_index]["x_y"]  # coordinates of the slice window
                img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
                img = general_utils.getSlicingWindow(img, coord[0], coord[1], isgt=True)

                # remove the brain from the image ==> it becomes background
                if constants.N_CLASSES<=3: img[img == 85] = constants.PIXELVALUES[0]
                # remove the penumbra ==> it becomes core
                if constants.N_CLASSES==2: img[img == 170] = constants.PIXELVALUES[1]

                # Override the weights based on the pixel values
                if constants.N_CLASSES>2:
                    core_idx, penumbra_idx = 3, 2
                    if constants.N_CLASSES == 3: core_idx, penumbra_idx = 2, 1
                    core_value, core_weight = constants.PIXELVALUES[core_idx], constants.HOT_ONE_WEIGHTS[0][core_idx]
                    penumbra_value, penumbra_weight = constants.PIXELVALUES[penumbra_idx], constants.HOT_ONE_WEIGHTS[0][penumbra_idx]

                    # focus on the SVO core only during training!
                    if self.flagtype=="train" and current_batch.loc[row_index]["severity"]=="02":
                        core_weight *= 6
                        penumbra_weight *= 6

                    f = lambda x: np.sum(np.where(np.array(x) == core_value, core_weight,
                                                  np.where(np.array(x) == penumbra_value, penumbra_weight, constants.HOT_ONE_WEIGHTS[0][0])))
                    weights[index] = f(img) / (constants.getM() * constants.getN())

                # convert the label in [0, 1] values,
                # for to_categ the division happens inside dataset_utils.getSingleLabelFromIndexCateg
                if not self.to_categ: img = np.divide(img,255)

                if aug_idx=="0": img = img if not self.to_categ or self.loss=="sparse_categorical_crossentropy" else dataset_utils.getSingleLabelFromIndexCateg(img)
                elif aug_idx=="1": img = np.rot90(img) if not self.to_categ or self.loss=="sparse_categorical_crossentropy" else dataset_utils.getSingleLabelFromIndexCateg(np.rot90(img))
                elif aug_idx=="2": img = np.rot90(img, 2) if not self.to_categ or self.loss=="sparse_categorical_crossentropy" else dataset_utils.getSingleLabelFromIndexCateg(np.rot90(img, 2))
                elif aug_idx=="3": img = np.rot90(img, 3) if not self.to_categ or self.loss=="sparse_categorical_crossentropy" else dataset_utils.getSingleLabelFromIndexCateg(np.rot90(img, 3))
                elif aug_idx=="4": img = np.flipud(img) if not self.to_categ or self.loss=="sparse_categorical_crossentropy" else dataset_utils.getSingleLabelFromIndexCateg(np.flipud(img))
                elif aug_idx=="5": img = np.fliplr(img) if not self.to_categ or self.loss=="sparse_categorical_crossentropy" else dataset_utils.getSingleLabelFromIndexCateg(np.fliplr(img))

                Y.append(img)

        return np.array(Y), weights
