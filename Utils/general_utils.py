# DO NOT import dataset_utils here!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from Model import constants
from Utils import metrics, losses

import sys, argparse, os, json, time, pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K

################################################################################
######################## UTILS FUNCTIONS #######################################
# The file should only contains functions!
################################################################################


################################################################################
# get the arguments from the command line
def getCommandLineArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")
    parser.add_argument("-d", "--debug", help="DEBUG mode", action="store_true")
    parser.add_argument("-o", "--original", help="ORIGINAL_SHAPE flag", action="store_true")
    parser.add_argument("-pm", "--pm", help="Use parametric maps", action="store_true")
    parser.add_argument("-t", "--tile", help="Set the tile pixels dimension (MxM)", type=int)
    parser.add_argument("-dim", "--dimension", help="Set the dimension of the input images (widthXheight)", type=int)
    parser.add_argument("-c", "--classes", help="Set the # of classes involved (default = 4)", default=4, type=int, choices=[2,3,4])
    parser.add_argument("-w", "--weights", help="Set the weights for the categorical losses", type=float, nargs='+')
    parser.add_argument("-e", "--exp", help="Set the number of the experiment", type=float)
    parser.add_argument("-j", "--jump", help="Jump the training and go directly on the gradual fine-tuning function", action="store_true")
    parser.add_argument("gpu", help="Give the id of gpu (or a list of the gpus) to use")
    parser.add_argument("sname", help="Select the setting filename")
    args = parser.parse_args()

    constants.setVerbose(args.verbose)
    constants.setDEBUG(args.debug)
    constants.setOriginalShape(args.original)
    constants.setUSE_PM(args.pm)
    constants.setTileDimension(args.tile)
    constants.setImageDimension(args.dimension)
    constants.setNumberOfClasses(args.classes)
    constants.setWeights(args.weights)

    return args


################################################################################
# get the setting file
def getSettingFile(filename):
    setting = dict()

    # the path of the setting file start from the main.py
    # (= current working directory)
    with open(os.path.join(os.getcwd(), filename)) as f:
        setting = json.load(f)

    if constants.getVerbose():
        printSeparation("-",50)
        print("Load setting file: {}".format(filename))

    return setting


################################################################################
# setup the global environment
def setupEnvironment(args, setting):
    # important: set up the root path for later uses
    constants.setRootPath(setting["root_path"])

    if "NUMBER_OF_IMAGE_PER_SECTION" in setting["init"].keys(): constants.setImagePerSection(setting["init"]["NUMBER_OF_IMAGE_PER_SECTION"])
    if "3D" in setting["init"].keys() and setting["init"]["3D"]: constants.set3DFlag()
    if "ONE_TIME_POINT" in setting["init"].keys() and setting["init"]["ONE_TIME_POINT"]: constants.setONETIMEPOINT(getStringFromIndex(setting["init"]["ONE_TIME_POINT"]))

    experimentFolder = "EXP"+convertExperimentNumberToString(setting["EXPERIMENT"])+"/"
    N_GPU = setupEnvironmentForGPUs(args, setting)

    for key, rel_path in setting["relative_paths"].items():
        if isinstance(rel_path, dict):
            prefix = key.upper()+"/"
            createDir(prefix)
            createDir(prefix+experimentFolder)
            for sub_path in setting["relative_paths"][key].values():
                createDir(prefix+experimentFolder+sub_path)
        else:
            if rel_path!="": createDir(rel_path)

    return N_GPU


################################################################################
# setup the environment for the GPUs
def setupEnvironmentForGPUs(args, setting):
    GPU = args.gpu
    N_GPU = len(GPU.split(","))

    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = setting["init"]["TF_CPP_MIN_LOG_LEVEL"]

    config = tf.compat.v1.ConfigProto()
    if setting["init"]["allow_growth"]:
        config.gpu_options.allow_growth = True
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        for physical_device in physical_devices: tf.config.experimental.set_memory_growth(physical_device, True)

    config.gpu_options.per_process_gpu_memory_fraction = setting["init"]["per_process_gpu_memory_fraction"] * N_GPU
    session = tf.compat.v1.Session(config=config)

    if constants.getVerbose():
        printSeparation("-",50)
        print("Use {0} GPU(s): {1}".format(N_GPU, GPU))

    return N_GPU


################################################################################
# return the selected window for an image
def getSlicingWindow(img, startX, startY, isgt=False, removeColorBar=False):
    M, N = constants.getM(), constants.getN()
    sliceWindow = img[startX:startX+M,startY:startY+N]

    # check if there are any NaN elements
    if np.isnan(sliceWindow).any():
        where = list(map(list, np.argwhere(np.isnan(sliceWindow))))
        for w in where: sliceWindow[w] = constants.PIXELVALUES[0]

    if isgt:
        for pxval in constants.PIXELVALUES:
            sliceWindow = np.where(np.logical_and(
                sliceWindow>=np.rint(pxval-(256/6)), sliceWindow<=np.rint(pxval+(256/6))
            ), pxval, sliceWindow)

    # Remove the colorbar! starting coordinate: (129,435)
    if removeColorBar:
        if M== constants.IMAGE_WIDTH and N== constants.IMAGE_HEIGHT:sliceWindow[:, constants.colorbar_coord[1]:] = 0
        # if the tile is smaller than the entire image
        elif startY+N>= constants.colorbar_coord[1]:sliceWindow[:, constants.colorbar_coord[1] - startY:] = 0

    sliceWindow = np.cast["float32"](sliceWindow)  # cast the window into a float

    return sliceWindow


################################################################################
# Perform a data augmentation based on the index and return the image
def performDataAugmentationOnTheImage(img, data_aug_idx):
    if data_aug_idx == 1: img = np.rot90(img)  # rotate 90 degree counterclockwise
    elif data_aug_idx == 2: img = np.rot90(img, 2)  # rotate 180 degree counterclockwise
    elif data_aug_idx == 3: img = np.rot90(img, 3)  # rotate 270 degree counterclockwise
    elif data_aug_idx == 4: img = np.flipud(img)  # rotate 270 degree counterclockwise
    elif data_aug_idx == 5: img = np.fliplr(img)  # flip the matrix left/right

    return img


################################################################################
# Get the epoch number from the partial weight filename
def getEpochFromPartialWeightFilename(partialWeightsPath):
    return int(partialWeightsPath[partialWeightsPath.index(constants.suffix_partial_weights) +
                                  len(constants.suffix_partial_weights):partialWeightsPath.index(".h5")])


################################################################################
# Get the loss defined in the settings
def getLoss(modelInfo):
    name = modelInfo["loss"]
    hyperparameters = modelInfo[name] if name in modelInfo.keys() else {}
    if name=="focal_tversky_loss": constants.setFocal_Tversky(hyperparameters)

    general_losses = [
        "binary_crossentropy",
        "categorical_crossentropy",
        "sparse_categorical_crossentropy",
        "mean_squared_error"
    ]
    loss = {}

    if name in general_losses: loss["loss"] = name
    else: loss["loss"] = getattr(losses, name)

    loss["name"] = name

    if constants.getVerbose(): print("[INFO] - Use {} Loss".format(name))

    return loss


################################################################################
# Get the statistic functions (& metrics) defined in the settings
def getMetricFunctions(listStats):
    general_metrics = [
        "binary_crossentropy",
        "categorical_crossentropy",
        "sparse_categorical_crossentropy",
        "mean_squared_error",
        "accuracy"
    ]

    statisticFuncs = []
    for m in listStats:
        if m in general_metrics: statisticFuncs.append(m)
        else: statisticFuncs.append(getattr(metrics, m))

    if constants.getVerbose(): print("[INFO] - Getting {} functions".format(listStats))

    if len(statisticFuncs)==0: statisticFuncs = None

    return statisticFuncs


################################################################################
# Return a flag to check if the filename (partial) is inside the list of patients
def isFilenameInListOfPatient(filename, patients):
    ret = False
    start_index = filename.rfind("/") + len(constants.DATASET_PREFIX) + 1
    patient_id = filename[start_index:start_index+len(str(patients[-1]))]
    # don't load the dataframe if patient_id NOT in the list of patients
    if constants.PREFIX_IMAGES== "PA": patient_id = int(patient_id)
    if patient_id in patients: ret = True

    return ret

################################################################################
################################################################################
################################################################################
########################### GENERAL UTILS ######################################
################################################################################
################################################################################
################################################################################
################################################################################


################################################################################
# get the string of the patient id given the integer
def getStringFromIndex(index):
    p_id = str(index)
    if len(p_id)==1: p_id = "0"+p_id
    return p_id


################################################################################
# return the suffix for the model and the patient dataset
def getSuffix():
    return "_" + str(constants.SLICING_PIXELS) +\
           "_" + str(constants.getM()) + "x" + str(constants.getN()) + \
           constants.get3DFlag() + constants.getONETIMEPOINT()


################################################################################
# get the full directory path, given a relative path
def getFullDirectoryPath(path):
    return constants.getRootPath() + path


################################################################################
# Generate a directory in dir_path
def createDir(dir_path):
    if not os.path.isdir(dir_path):
        if constants.getVerbose(): print("[INFO] - Creating folder: " + dir_path)
        os.makedirs(dir_path)


################################################################################
# print a separation for verbose purpose
def printSeparation(what, howmuch):
    print(what*howmuch)


################################################################################
# Convert the experiment number to a string of 3 letters
def convertExperimentNumberToString(expnum):
    exp = str(expnum)
    while len(exp.split(".")[0])<3: exp = "0"+exp
    return exp


################################################################################
# Print the shape of the layer if we are in debug mode
def print_int_shape(layer):
    if constants.getVerbose(): print(K.int_shape(layer))


################################################################################
def addPIDToWatchdog():
    # Add PID to watchdog list
    if constants.ENABLE_WATCHDOG is True:
        if os.path.isfile(constants.PID_WATCHDOG_PICKLE_PATH):
            PID_list_for_watchdog = pickle_load(constants.PID_WATCHDOG_PICKLE_PATH)
            PID_list_for_watchdog.append(dict(pid=os.getpid()))
        else:
            PID_list_for_watchdog = [dict(pid=os.getpid())]

        # Save list
        pickle_save(PID_list_for_watchdog, constants.PID_WATCHDOG_PICKLE_PATH)

        # Create a empty list for saving to when the model finishes
        if not os.path.isfile(constants.PID_WATCHDOG_FINISHED_PICKLE_PATH):
            PID_list_finished_for_watchdog = []
            pickle_save(PID_list_finished_for_watchdog, constants.PID_WATCHDOG_FINISHED_PICKLE_PATH)
    else:
        print('Warning: WATCHDOG IS DISABLED!')


################################################################################
def stopPIDToWatchdog():
    if constants.ENABLE_WATCHDOG is True:
        # Add PID to finished-watchdog-list
        if os.path.isfile(constants.PID_WATCHDOG_FINISHED_PICKLE_PATH):
            PID_list_finished_for_watchdog = pickle_load(constants.PID_WATCHDOG_FINISHED_PICKLE_PATH)
            PID_list_finished_for_watchdog.append(dict(pid=os.getpid()))
            pickle_save(PID_list_finished_for_watchdog, constants.PID_WATCHDOG_FINISHED_PICKLE_PATH)

        # Remove PID from watchdog list
        if os.path.isfile(constants.PID_WATCHDOG_PICKLE_PATH):
            PID_list_for_watchdog = pickle_load(constants.PID_WATCHDOG_PICKLE_PATH)
            PID_list_for_watchdog.remove(dict(pid=os.getpid()))
            pickle_save(PID_list_for_watchdog, constants.PID_WATCHDOG_PICKLE_PATH)

################################################################################
def pickle_save(variable_to_save, path):
    with open(path, 'wb') as handle:
        pickle.dump(variable_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)


################################################################################
def pickle_load(path):
    with open(path, 'rb') as handle:
        output = pickle.load(handle)
    return output

