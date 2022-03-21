# Run the testing function, save the images ..
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from Utils import general_utils, dataset_utils, sequence_utils, metrics
from Model import constants, training

import os, time, cv2, glob
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing
from scipy import ndimage
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import pickle as pkl


################################################################################
# Predict the model based on the input
def predictFromModel(nn, x_input):
    return nn.model.predict(x=x_input, steps=nn.test_steps, use_multiprocessing=nn.mp)


################################################################################
# Generate the images for the patient and save the images
def predictAndSaveImages(nn, p_id):
    start = time.time()
    suffix = general_utils.getSuffix()  # es == "_4_16x16"

    suffix_filename = ".pkl"
    if nn.use_hickle: suffix_filename = ".hkl"
    filename_test = nn.datasetFolder + constants.DATASET_PREFIX + str(p_id) + suffix + suffix_filename

    relativePatientFolder = constants.getPrefixImages() + p_id + "/"
    relativePatientFolderHeatMap = relativePatientFolder + "HEATMAP/"
    relativePatientFolderGT = relativePatientFolder + "GT/"
    relativePatientFolderTMP = relativePatientFolder + "TMP/"
    patientFolder = nn.patientsFolder+relativePatientFolder

    filename_saveImageFolder = nn.saveImagesFolder+nn.experimentID+"__"+nn.getNNID(p_id)+suffix
    # create the related folders
    general_utils.createDir(filename_saveImageFolder)
    for subpath in [relativePatientFolder,relativePatientFolderHeatMap,relativePatientFolderGT,relativePatientFolderTMP]:
        general_utils.createDir(filename_saveImageFolder+"/"+subpath)

    if constants.getVerbose(): general_utils.printSeparation("-", 100)

    # for all the slice folders in patientFolder
    for subfolder in glob.glob(patientFolder+"*/"):
        prefix = nn.experimentID + constants.suffix_partial_weights + nn.getNNID(p_id) + suffix + "/"
        subpatientFolder = prefix+relativePatientFolder
        patientFolderHeatMap = prefix+relativePatientFolderHeatMap
        patientFolderGT = prefix+relativePatientFolderGT
        patientFolderTMP = prefix+relativePatientFolderTMP

        # Predict the images
        if constants.getUSE_PM(): predictImagesFromParametricMaps(nn, subfolder, p_id, subpatientFolder, patientFolderHeatMap, patientFolderGT, patientFolderTMP, filename_test)
        else: predictImage(nn, subfolder, p_id, patientFolder, subpatientFolder, patientFolderHeatMap, patientFolderGT, patientFolderTMP, filename_test)

    end = time.time()


################################################################################
def predictImage(nn, subfolder, p_id, patientFolder, relativePatientFolder, relativePatientFolderHeatMap, relativepatientFolderGT, relativePatientFolderTMP, filename_test):
    """
    Generate a SINGLE image for the patient and save it.

    Input parameters:
    - nn                            : NeuralNetwork class
    - subfolder                     : Name of the slice subfolder
    - p_id                          : Patient ID
    - patientFolder                 : folder of the patient
    - relativePatientFolder         : relative name of the patient folder
    - relativePatientFolderHeatMap  : relative name of the patient heatmap folder
    - relativepatientFolderGT      : relative name of the patient gt folder
    - relativePatientFolderTMP      : relative name of the patient tmp folder
    - filename_test                 : Name of the test pandas dataframe
    """

    start = time.time()
    imagesDict = {}  # faster access to the images
    startingX, startingY = 0, 0
    imagePredicted = np.zeros(shape=(constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT))
    categoricalImage = np.zeros(shape=(constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT, constants.N_CLASSES))
    checkImageProcessed = np.zeros(shape=(constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT))

    idx = general_utils.getStringFromIndex(subfolder.replace(patientFolder, '').replace("/", ""))  # image index

    # remove the old logs.
    logsName = nn.saveImagesFolder+relativePatientFolder+idx+"_logs.txt"
    if os.path.isfile(logsName): os.remove(logsName)

    if constants.getVerbose(): print("[INFO] - Analyzing Patient {0}, image {1}.".format(p_id, idx))
    s1 = time.time()

    # get the label image only if the path is set
    if nn.labeledImagesFolder!="":
        filename = nn.labeledImagesFolder + constants.PREFIX_IMAGES + p_id + "/" + idx + constants.SUFFIX_IMG
        if not os.path.exists(filename):
            print("[WARNING] - {0} does NOT exists, try another...".format(filename))
            filename = nn.labeledImagesFolder + constants.PREFIX_IMAGES + p_id + "/" + p_id + idx + constants.SUFFIX_IMG
            if not os.path.exists(filename):
                raise Exception("[ERROR] - {0} does NOT exist".format(filename))

        checkImageProcessed = cv2.imread(filename, cv2.COLOR_BGR2RGB)

    # get the images in a dictionary
    for imagename in np.sort(glob.glob(subfolder +"*" + constants.SUFFIX_IMG)):  # sort the images !
        filename = imagename.replace(subfolder, '')
        if not nn.supervised or nn.patientsFolder!="OLDPREPROC_PATIENTS/": imagesDict[filename] = cv2.imread(imagename, cv2.IMREAD_GRAYSCALE)
        else:  # don't take the first image (the manually annotated one)
            if filename != "01"+ constants.SUFFIX_IMG: imagesDict[filename] = cv2.imread(imagename, cv2.IMREAD_GRAYSCALE)

    # Portion for the prediction of the image
    if constants.get3DFlag()!= "":
        if not os.path.exists(filename_test):
            if constants.getVerbose(): print("[WARNING] - File {} does NOT exist".format(filename_test))
            return

        test_df = dataset_utils.readFromPickleOrHickle(filename_test, nn.use_hickle)
        # get only the rows with data_aug_idx==0 (no rotation or any data augmentation)
        test_df = test_df[test_df.data_aug_idx==0]
        test_df = test_df[test_df.timeIndex==idx]  # todo: why timeindex?
        imagePredicted = generateTimeImagesAndConsensus(nn, test_df, relativepatientFolderGT, relativePatientFolderTMP, idx)
    else:  # usual behaviour
        while True:
            if constants.ORIGINAL_SHAPE: pixels_shape = (
            constants.NUMBER_OF_IMAGE_PER_SECTION, constants.getM(), constants.getN())
            else: pixels_shape = (constants.getM(), constants.getN(), constants.NUMBER_OF_IMAGE_PER_SECTION)

            pixels = np.zeros(shape=pixels_shape)
            binary_mask = np.zeros(shape=(constants.getM(), constants.getN()))
            count = 0

            # for each image get the array for prediction
            for imagename in np.sort(glob.glob(subfolder +"*" + constants.SUFFIX_IMG)):
                filename = imagename.replace(subfolder, '')
                if not nn.supervised or nn.patientsFolder!="OLDPREPROC_PATIENTS/":
                    image = imagesDict[filename]
                    if constants.ORIGINAL_SHAPE:pixels[count, :, :] = general_utils.getSlicingWindow(image, startingX, startingY)
                    else: pixels[:,:,count] = general_utils.getSlicingWindow(image, startingX, startingY)
                    binary_mask += (pixels[:,:,count]>0)  # add the mask of the pixels that are > 0
                    count+=1
                else:
                    if filename != "01"+ constants.SUFFIX_IMG:
                        image = imagesDict[filename]
                        if constants.ORIGINAL_SHAPE:pixels[count, :, :] = general_utils.getSlicingWindow(image, startingX, startingY)
                        else: pixels[:,:,count] = general_utils.getSlicingWindow(image, startingX, startingY)
                        binary_mask += (pixels[:, :, count] > 0)  # add the mask of the pixels that are > 0
                        count+=1

            # the final binary mask is a consensus among the all timepoints
            binary_mask = (binary_mask/(count+1) >= 0.5)

            if constants.get3DFlag()== "": pixels = pixels.reshape(1, pixels.shape[0], pixels.shape[1], pixels.shape[2], 1)
            else: pixels = pixels.reshape(1, pixels.shape[0], pixels.shape[1], pixels.shape[2])
            imagePredicted, categoricalImage = generate2DImage(nn, pixels, (startingX,startingY), imagePredicted, categoricalImage, binary_mask)

            # if we reach the end of the image, break the while loop.
            if startingX>= constants.IMAGE_WIDTH- constants.getM() and startingY>= constants.IMAGE_HEIGHT- constants.getN(): break

            # going to the next slicingWindow
            if startingY< constants.IMAGE_HEIGHT- constants.getN(): startingY+= constants.getN()
            else:
                if startingX< constants.IMAGE_WIDTH:
                    startingY=0
                    startingX+= constants.getM()

    s2 = time.time()
    # save the image
    saveImage(nn, relativePatientFolder, idx, imagePredicted, categoricalImage,
                      relativePatientFolderHeatMap, relativepatientFolderGT, relativePatientFolderTMP,
                      checkImageProcessed)

    end = time.time()


################################################################################
#
def predictImagesFromParametricMaps(nn, subfolder, p_id, relativePatientFolder, relativePatientFolderHeatMap, relativepatientFolderGT, relativePatientFolderTMP, filename_test):
    """
    Generate ALL the images for the patient using the PMs and save it.

    Input parameters:
    - nn                            : NeuralNetwork class
    - subfolder                     : Name of the slice subfolder
    - p_id                          : Patient ID
    - patientFolder                 : folder of the patient
    - relativePatientFolder         : relative name of the patient folder
    - relativePatientFolderHeatMap  : relative name of the patient heatmap folder
    - relativepatientFolderGT      : relative name of the patient gt folder
    - relativePatientFolderTMP      : relative name of the patient tmp folder
    - filename_test                 : Name of the test pandas dataframe
    """

    # if the patient folder contains the correct number of subfolders
    # ATTENTION: careful here...
    if len(glob.glob(subfolder+"*/"))>=7:
        for idx in glob.glob(subfolder+"/TTP/*"):
            idx = general_utils.getStringFromIndex(idx.replace(subfolder, '').replace("/TTP/", ""))  # image index
            idx = idx.replace(".png","")
            start = time.time()
            checkImageProcessed = np.zeros(shape=(constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT))

            # remove the old logs.
            logsName = nn.saveImagesFolder+relativePatientFolder+idx+"_logs.txt"
            if os.path.isfile(logsName): os.remove(logsName)

            if constants.getVerbose(): print("[INFO] - Analyzing Patient {0}, image {1}.".format(p_id, idx))
            s1 = time.time()

            # get the label image only if the path is set
            if nn.labeledImagesFolder!="":
                filename = nn.labeledImagesFolder + constants.PREFIX_IMAGES + p_id + "/" + idx + constants.SUFFIX_IMG
                if not os.path.exists(filename):
                    print("[WARNING] - {0} does NOT exists, try another...".format(filename))
                    filename = nn.labeledImagesFolder + constants.PREFIX_IMAGES + p_id + "/" + p_id + idx + constants.SUFFIX_IMG
                    if not os.path.exists(filename):
                        raise Exception("[ERROR] - {0} does NOT exist".format(filename))

                checkImageProcessed = cv2.imread(filename, cv2.COLOR_BGR2RGB)

            if not os.path.exists(filename_test):
                if constants.getVerbose(): print("[WARNING] - File {} does NOT exist".format(filename_test))
                return

            # get the pandas dataframe
            test_df = dataset_utils.readFromPickleOrHickle(filename_test, nn.use_hickle)
            # get only the rows with data_aug_idx==0 (no rotation or any data augmentation)
            test_df = test_df[test_df.data_aug_idx == 0]
            test_df = test_df[test_df.sliceIndex == idx]

            imagePredicted, categoricalImage = generateImageFromParametricMaps(nn, test_df)
            s2 = time.time()

            # save the image
            saveImage(nn, relativePatientFolder, idx, imagePredicted, categoricalImage, relativePatientFolderHeatMap,
                      relativepatientFolderGT, relativePatientFolderTMP, checkImageProcessed)

            end = time.time()


################################################################################
# Util function to save image
def saveImage(nn, relativePatientFolder, idx, imagePredicted, categoricalImage, relativePatientFolderHeatMap,
              relativepatientFolderGT, relativePatientFolderTMP, checkImageProcessed):

    if nn.save_images:
        # s1 = time.time()
        # TODO: rotate the predictions for the ISLES2018 dataset
        # if "ISLES2018" in nn.datasetFolder: imagePredicted = np.rot90(imagePredicted,-1)
        # save the image predicted in the specific folder
        cv2.imwrite(nn.saveImagesFolder+relativePatientFolder+idx+".png", imagePredicted)
        # create and save the HEATMAP only if we are using softmax activation
        if nn.to_categ:
            p_idx, c_idx = 2,3
            if constants.N_CLASSES==3: p_idx, c_idx = 1, 2
            heatmap_img_p = cv2.convertScaleAbs(categoricalImage[:, :, p_idx] * 255)
            heatmap_img_c = cv2.convertScaleAbs(categoricalImage[:, :, c_idx] * 255)
            heatmap_img_p = cv2.applyColorMap(heatmap_img_p, cv2.COLORMAP_JET)
            checkImageProcessed_rgb = cv2.cvtColor(checkImageProcessed, cv2.COLOR_GRAY2RGB)
            blend_p = cv2.addWeighted(checkImageProcessed_rgb, 0.5, heatmap_img_p, 0.5, 0.0)
            heatmap_img_c = cv2.applyColorMap(heatmap_img_c, cv2.COLORMAP_JET)
            blend_c = cv2.addWeighted(checkImageProcessed_rgb, 0.5, heatmap_img_c, 0.5, 0.0)

            cv2.imwrite(nn.saveImagesFolder + relativePatientFolderHeatMap + idx + "_heatmap_penumbra.png", blend_p)
            cv2.imwrite(nn.saveImagesFolder + relativePatientFolderHeatMap + idx + "_heatmap_core.png", blend_c)
            # sns.heatmap(categoricalImage[:,:,p_idx], cmap="jet", yticklabels=False, xticklabels=False, square=True, cbar=False)
            # plt.savefig(nn.saveImagesFolder+relativePatientFolderHeatMap+idx+"_heatmap_penumbra.png", transparent=True, bbox_inches='tight')
            # sns.heatmap(categoricalImage[:,:,c_idx], cmap="jet", yticklabels=False, xticklabels=False, square=True, cbar=False)
            # plt.savefig(nn.saveImagesFolder+relativePatientFolderHeatMap+idx+"_heatmap_core.png", transparent=True, bbox_inches='tight')

            # f = open(nn.saveImagesFolder+relativePatientFolderHeatMap+idx+".pkl", 'wb')
            # pkl.dump(categoricalImage, f)

        # Save the ground truth and the contours
        if constants.get3DFlag()== "":
            # save the GT
            cv2.imwrite(nn.saveImagesFolder + relativepatientFolderGT + idx + constants.SUFFIX_IMG, checkImageProcessed)

            imagePredicted = cv2.cvtColor(np.uint8(imagePredicted),cv2.COLOR_GRAY2RGB)  # back to rgb
            _, penumbra_mask = cv2.threshold(checkImageProcessed, 85, constants.PIXELVALUES[-2], cv2.THRESH_BINARY)
            penumbra_cnt, _ = cv2.findContours(penumbra_mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            imagePredicted = cv2.drawContours(imagePredicted, penumbra_cnt, -1, (255,0,0), 2)
            _, core_mask = cv2.threshold(checkImageProcessed, constants.PIXELVALUES[-2], constants.PIXELVALUES[-1], cv2.THRESH_BINARY)
            core_cnt, _ = cv2.findContours(core_mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            imagePredicted = cv2.drawContours(imagePredicted, core_cnt, -1, (0,0,255), 2)
            # save the GT image with predicted contours
            # checkImageProcessed = cv2.addWeighted(checkImageProcessed, 1, penumbra_area, 0.5, 0.0)
            # checkImageProcessed = cv2.addWeighted(checkImageProcessed, 1, core_area, 0.5, 0.0)
            cv2.imwrite(nn.saveImagesFolder+relativePatientFolderTMP+idx+".png",imagePredicted)

        # s2 = time.time()
        # if constants.getVerbose(): print("save time: {}".format(round(s2-s1, 3)))


################################################################################
# Helpful function that return the 2D image from the pixel and the starting coordinates
def generate2DImage(nn, pixels, startingXY, imagePredicted, categoricalImage, binary_mask,
                    slicingWindowPredicted=None):
    """
    Generate a 2D image from the test_df

    Input parameters:
    - nn                    : NeuralNetwork class
    - pixels                : pixel in a numpy array
    - startingXY            : (x,y) coordinates
    - imagePredicted        : the predicted image
    - checkImageProcessed   : the ground truth image

    Return:
    - imagePredicted        : the predicted image
    """
    startingX, startingY = startingXY
    # slicingWindowPredicted_orig contain only the prediction for the last step
    slicingWindowPredicted_orig = predictFromModel(nn, pixels)[nn.test_steps-1]
    if nn.save_images and nn.to_categ: categoricalImage[startingX:startingX + constants.getM(),
                                       startingY:startingY + constants.getN()] = slicingWindowPredicted_orig

    # convert the categorical into a single array using a threshold (0.6) for removing some uncertain predictions
    if nn.to_categ: slicingWindowPredicted = K.eval((K.argmax(slicingWindowPredicted_orig)*255)/(constants.N_CLASSES-1))
    else: slicingWindowPredicted *= 255

    # save the predicted images
    if nn.save_images:
        # Remove the parts already classified by the model
        binary_mask = np.array(binary_mask, dtype=np.float)
        overlapping_pred = np.array(slicingWindowPredicted>0,dtype=np.float)
        overlapping_pred *= 85.
        binary_mask *= 85.  # multiply the binary mask for the brain pixel value
        slicingWindowPredicted += (binary_mask-overlapping_pred)  # add the brain to the prediction window
        imagePredicted[startingX:startingX + constants.getM(), startingY:startingY + constants.getN()] = slicingWindowPredicted
    return imagePredicted, categoricalImage


################################################################################
def generateTimeImagesAndConsensus(nn, test_df, relativePatientFolderTMP, idx):
    """
    Generate the image from the 3D sequence of time index (create also these images) with a consensus

    Input:
    - nn                        : NeuralNetwork class
    - test_df                   : pandas dataframe for testing
    - relativePatientFolderTMP  : tmp folder for the patient
    - idx                       : image index (slice)

    Return:
    - imagePredicted        : the predicted image
    """

    imagePredicted = np.zeros(shape=(constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT))
    categoricalImage = np.zeros(shape=(constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT, constants.N_CLASSES))
    checkImageProcessed = np.zeros(shape=(constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT))
    arrayTimeIndexImages = dict()

    for test_row in test_df.itertuples():  # for every rows of the same image
        if str(test_row.timeIndex) not in arrayTimeIndexImages.keys(): arrayTimeIndexImages[str(test_row.timeIndex)] = np.zeros(shape=(
        constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT), dtype=np.uint8)
        if constants.get3DFlag() == "": test_row.pixels = test_row.pixels.reshape(1, test_row.pixels.shape[0], test_row.pixels.shape[1], test_row.pixels.shape[2], 1)
        else: test_row.pixels = test_row.pixels.reshape(1, test_row.pixels.shape[0], test_row.pixels.shape[1], test_row.pixels.shape[2])
        arrayTimeIndexImages[str(test_row.timeIndex)], categoricalImage = generate2DImage(nn, test_row.pixels, test_row.x_y, arrayTimeIndexImages[str(test_row.timeIndex)], categoricalImage, checkImageProcessed)

    if nn.save_images:              # remove one class from the ground truth
        if constants.N_CLASSES==3: checkImageProcessed[checkImageProcessed == 85] = constants.PIXELVALUES[0]
        cv2.imwrite(nn.saveImagesFolder + relativePatientFolderTMP +"orig_" + idx + constants.SUFFIX_IMG, checkImageProcessed)

        for tidx in arrayTimeIndexImages.keys():
            curr_image = arrayTimeIndexImages[tidx]
            # save the images
            cv2.imwrite(nn.saveImagesFolder + relativePatientFolderTMP + idx +"_" + general_utils.getStringFromIndex(tidx) + constants.SUFFIX_IMG, curr_image)
            # add the predicted image in the imagePredicted for consensus
            imagePredicted += curr_image

        imagePredicted /= len(arrayTimeIndexImages.keys())

    return imagePredicted, categoricalImage


################################################################################
# Function to predict an image starting from the parametric maps
def generateImageFromParametricMaps(nn, test_df):
    """
    Generate a 2D image from the test_df using the parametric maps

    Input:
    - nn                        : NeuralNetwork class
    - test_df                   : pandas dataframe for testing
    - checkImageProcessed       : the labeled image (Ground truth img)
    - relativePatientFolderTMP  : tmp folder for the patient
    - idx                       : image index (slice)

    Return:
    - imagePredicted        : the predicted image
    """

    imagePredicted = np.zeros(shape=(constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT))
    categoricalImage = np.zeros(shape=(constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT, constants.N_CLASSES))
    startX, startY = 0, 0

    while True:
        row_to_analyze = test_df[test_df.x_y == (startX, startY)]
        binary_mask = np.zeros(shape=(constants.getM(), constants.getN()))
        if len(row_to_analyze)>0:
            pms = dict()
            for pm_name in constants.getList_PMS():
                filename = row_to_analyze[pm_name].iloc[0]
                # pm = img_to_array(load_img(filename))
                pm = cv2.imread(filename, cv2.COLOR_BGR2RGB)

                pms[pm_name] = general_utils.getSlicingWindow(pm, startX, startY, removeColorBar=True)
                # add the mask of the pixels that are > 0 only if it's the MIP image
                if pm_name=="MIP": binary_mask += (cv2.cvtColor(pms[pm_name], cv2.COLOR_RGB2GRAY) > 0)
                pms[pm_name] = np.array(pms[pm_name])
                pms[pm_name] = pms[pm_name].reshape((1,) + pms[pm_name].shape)

            # the final binary mask is a consensus among the all parametric maps
            # binary_mask = (binary_mask >= len(constants.getList_PMS())/2)
            # binary_mask = ndimage.binary_fill_holes(binary_mask).astype(int)

            X = [pms["CBF"], pms["CBV"], pms["TTP"], pms["TMAX"]]

            if "mip" in nn.moreinfo.keys() and nn.moreinfo["mip"] == 1: X.append(pms["MIP"])
            if "nihss" in nn.moreinfo.keys() and nn.moreinfo["nihss"] == 1: X.append(np.array([int(row_to_analyze["NIHSS"].iloc[0])]) if row_to_analyze["NIHSS"].iloc[0]!="-" else np.array([0]))
            if "age" in nn.moreinfo.keys() and nn.moreinfo["age"] == 1: X.append(np.array([int(row_to_analyze["age"].iloc[0])]))
            if "gender" in nn.moreinfo.keys() and nn.moreinfo["gender"] == 1: X.append(np.array([int(row_to_analyze["gender"].iloc[0])]))

            # slicingWindowPredicted contain only the prediction for the last step
            imagePredicted, categoricalImage = generate2DImage(nn, X, (startX,startY), imagePredicted, categoricalImage, binary_mask)

        # if we reach the end of the image, break the while loop.
        if startX>= constants.IMAGE_WIDTH- constants.getM() and startY>= constants.IMAGE_HEIGHT- constants.getN(): break

        # check for M == WIDTH & N == HEIGHT
        if constants.getM()== constants.IMAGE_WIDTH and constants.getN()== constants.IMAGE_HEIGHT: break

        # going to the next slicingWindow
        if startY<=(constants.IMAGE_HEIGHT - constants.getN()): startY+= constants.getN()
        else:
            if startX < constants.IMAGE_WIDTH:
                startY = 0
                startX += constants.getM()

    return imagePredicted, categoricalImage


################################################################################
# Test the model with the selected patient
def evaluateModel(nn, p_id, isAlreadySaved):
    suffix = general_utils.getSuffix()

    if isAlreadySaved:
        suffix_filename = ".pkl"
        if nn.use_hickle: suffix_filename = ".hkl"
        filename_train = nn.datasetFolder + constants.DATASET_PREFIX + str(p_id) + suffix + suffix_filename

        if not os.path.exists(filename_train): return

        nn.train_df = dataset_utils.readFromPickleOrHickle(filename_train, nn.use_hickle)

        nn.dataset = dataset_utils.getTestDataset(nn.dataset, nn.train_df, p_id, nn.use_sequence, nn.mp)
        if not nn.use_sequence: nn.dataset["test"]["labels"] = dataset_utils.getLabelsFromIndex(train_df=nn.train_df, dataset=nn.dataset["test"], modelname=nn.name, to_categ=nn.to_categ, flag="test")
        nn.compileModel()  # compile the model and then evaluate it

    sample_weights = nn.getSampleWeights("test")
    if nn.use_sequence:
        multiplier = 16

        nn.test_sequence = sequence_utils.datasetSequence(
            dataframe=nn.train_df,
            indices=nn.dataset["test"]["indices"],
            sample_weights=sample_weights,
            x_label="pixels" if not constants.getUSE_PM() else constants.getList_PMS(),
            y_label="ground_truth",
            moreinfo=nn.moreinfo,
            to_categ=nn.to_categ,
            batch_size=nn.batch_size,
            flagtype="test",
            back_perc=100,
            loss=nn.loss["name"]
        )

        testing = nn.model.evaluate_generator(
            generator=nn.test_sequence,
            max_queue_size=10*multiplier,
            workers=1*multiplier,
            use_multiprocessing=nn.mp
        )

    else:
        testing = nn.model.evaluate(
            x=nn.dataset["test"]["data"],
            y=nn.dataset["test"]["labels"],
            callbacks=nn.callbacks,
            sample_weight=sample_weights,
            verbose=constants.getVerbose(),
            batch_size=nn.batch_size,
            use_multiprocessing=nn.mp
        )

    general_utils.printSeparation("-",50)
    if not isAlreadySaved:
        for metric_name in nn.train.history:
            print("TRAIN %s: %.2f%%" % (metric_name, round(float(nn.train.history[metric_name][-1]), 6)*100))
    for index, val in enumerate(testing):
        print("TEST %s: %.2f%%" % (nn.model.metrics_names[index], round(val,6)*100))
    general_utils.printSeparation("-",50)

    with open(general_utils.getFullDirectoryPath(nn.saveTextFolder)+nn.getNNID(p_id)+suffix+".txt", "a+") as text_file:
        if not isAlreadySaved:
            for metric_name in nn.train.history:
                text_file.write("TRAIN %s: %.2f%% \n" % (metric_name, round(float(nn.train.history[metric_name][-1]), 6)*100))
        for index, val in enumerate(testing):
            text_file.write("TEST %s: %.2f%% \n" % (nn.model.metrics_names[index], round(val,6)*100))
        text_file.write("----------------------------------------------------- \n")
