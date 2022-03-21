import socket

verbose = False
USE_PM = False
DEBUG = False
ORIGINAL_SHAPE = False

root_path = ""

M, N = 16, 16
SLICING_PIXELS = int(M/4)
IMAGE_WIDTH, IMAGE_HEIGHT = 512, 512
NUMBER_OF_IMAGE_PER_SECTION = 30  # number of image (divided by time) for each section of the brain
N_CLASSES = 4
LABELS = ["background", "brain", "penumbra", "core"]  # background:0, brain:85, penumbra:170, core:255
PIXELVALUES = [0, 85, 170, 255]
# weights for the various weighted losses: 1) position: background, 2) brain, 3) penumbra, 4) core
HOT_ONE_WEIGHTS = [[0, 0.1, 50, 440]]  # [[0.1, 0.1, 2, 15]]

# hyperparameters for the multi focal loss
ALPHA = [[0.25,0.25,0.25,0.25]]
GAMMA = [[2.,2.,2.,2.]]
focal_tversky_loss = {
    "alpha": 0.7,
    "gamma": 1.33
}

PREFIX_IMAGES = "PA"
DATASET_PREFIX = "patient"
SUFFIX_IMG = ".tiff"  # ".png"
colorbar_coord = (129, 435)

suffix_partial_weights = "__"
threeD_flag, ONE_TIME_POINT = "", ""

list_PMS = list()
dataFrameColumns = ['patient_id', 'label', 'pixels', 'CBF', 'CBV', 'TTP', 'TMAX', "MIP", "NIHSS", 'ground_truth', 'x_y',
                    'data_aug_idx','timeIndex', 'sliceIndex', 'severity', "age", "gender", 'label_code']

ENABLE_WATCHDOG = True
PID_WATCHDOG_PICKLE_PATH = '../PID_list_{}.obj'.format(socket.gethostname())
PID_WATCHDOG_FINISHED_PICKLE_PATH = '../PID_finished_list_{}.obj'.format(socket.gethostname())


################################################################################
def getVerbose():
    return verbose


def getDEBUG():
    return DEBUG


def getM():
    return M


def getN():
    return N


def get3DFlag():
    return threeD_flag


def getONETIMEPOINT():
    return ONE_TIME_POINT


def getRootPath():
    return root_path


def getPrefixImages():
    return PREFIX_IMAGES


def getUSE_PM():
    return USE_PM


def getList_PMS():
    return list_PMS


################################################################################
################################################################################
# Functions used to set the various GLOBAl variables
def setVerbose(v):
    global verbose
    verbose = v


def setDEBUG(d):
    global DEBUG
    DEBUG = d


def setOriginalShape(o):
    global ORIGINAL_SHAPE, PIXELVALUES
    ORIGINAL_SHAPE = o
    if ORIGINAL_SHAPE: PIXELVALUES = [255, 1, 76, 150]


def setTileDimension(t):
    global M, N, SLICING_PIXELS
    if t is not None:
        M = int(t)
        N = int(t)
        SLICING_PIXELS = int(M / 4)


def setImageDimension(d):
    global IMAGE_WIDTH, IMAGE_HEIGHT
    if d is not None:
        IMAGE_WIDTH = int(d)
        IMAGE_HEIGHT = int(d)


def setRootPath(path):
    global root_path
    root_path = path


def setImagePerSection(num):
    global NUMBER_OF_IMAGE_PER_SECTION
    if num is not None: NUMBER_OF_IMAGE_PER_SECTION = num


def setNumberOfClasses(c):
    global N_CLASSES, LABELS, PIXELVALUES, HOT_ONE_WEIGHTS, GAMMA, ALPHA

    if c == 2:
        N_CLASSES = c
        LABELS = ["background", "core"]
        PIXELVALUES = [0, 255]
        HOT_ONE_WEIGHTS = [[0.1, 1]]
        GAMMA = [[2., 2.]]
        ALPHA = [[.25,.25]]
    elif c == 3:
        N_CLASSES = c
        LABELS = ["background", "penumbra", "core"]
        PIXELVALUES = [0, 170, 255]
        HOT_ONE_WEIGHTS = [[0.1, 50, 440]]
        GAMMA = [[2., 2., 2.]]
        ALPHA = [[0.25,0.25,0.25]]


def set3DFlag():
    global threeD_flag
    threeD_flag = "_3D"


def setONETIMEPOINT(timepoint):
    global ONE_TIME_POINT
    ONE_TIME_POINT = "_" + timepoint


def setPrefixImagesSUS2020_v2():
    global PREFIX_IMAGES
    PREFIX_IMAGES = "CTP_"


def setUSE_PM(pm):
    global USE_PM, list_PMS
    USE_PM = pm
    if USE_PM: list_PMS = ["CBF", "CBV", "TTP", "TMAX", "MIP"]


def setFocal_Tversky(hyperparameters):
    global focal_tversky_loss
    for key in hyperparameters.keys(): focal_tversky_loss[key] = hyperparameters[key]


def setWeights(weights):
    global HOT_ONE_WEIGHTS
    if weights is not None: HOT_ONE_WEIGHTS = [weights]

