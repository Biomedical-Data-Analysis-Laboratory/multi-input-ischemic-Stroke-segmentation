import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from Utils import general_utils, dataset_utils, sequence_utils, architectures, losses
from Model import training, testing, constants

import os, glob, math
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import multi_gpu_model, plot_model


################################################################################
# Class that defines a NeuralNetwork
################################################################################
def getVerbose():
    return constants.getVerbose()


class NeuralNetwork(object):
    """docstring for NeuralNetwork."""

    def __init__(self, modelInfo, setting):
        super(NeuralNetwork, self).__init__()
        self.summaryFlag = 0

        # Used to override the path for the saved model in order to test patients with a specific model
        self.OVERRIDE_MODELS_ID_PATH = setting["OVERRIDE_MODELS_ID_PATH"] if setting["OVERRIDE_MODELS_ID_PATH"]!="" else False
        self.use_sequence = setting["USE_SEQUENCE_TRAIN"] if "USE_SEQUENCE_TRAIN" in setting.keys() else 0

        self.name = modelInfo["name"]
        self.epochs = modelInfo["epochs"]
        self.batch_size = modelInfo["batch_size"] if "batch_size" in modelInfo.keys() else 32

        # use only for the fit_generator
        self.steps_per_epoch_ratio = modelInfo["steps_per_epoch_ratio"] if "steps_per_epoch_ratio" in modelInfo.keys() else 1
        self.validation_steps_ratio = modelInfo["validation_steps_ratio"] if "validation_steps_ratio" in modelInfo.keys() else 1

        self.val = {
            "validation_perc": modelInfo["val"]["validation_perc"],
            "random_validation_selection": modelInfo["val"]["random_validation_selection"],
            "number_patients_for_validation": modelInfo["val"]["number_patients_for_validation"] if "number_patients_for_validation" in modelInfo["val"].keys() else 0,
            "number_patients_for_testing": modelInfo["val"]["number_patients_for_testing"] if "number_patients_for_testing" in modelInfo["val"].keys() else 0
        }
        self.test_steps = modelInfo["test_steps"]

        self.dataset = {"train": {},"val": {},"test": {}}

        # get parameter for the model
        self.optimizerInfo = modelInfo["optimizer"]
        self.params = modelInfo["params"]
        self.loss = general_utils.getLoss(modelInfo)
        self.metricFuncs = general_utils.getMetricFunctions(modelInfo["metrics"])

        # FLAGS for the model
        self.moreinfo = modelInfo["moreinfo"] if "moreinfo" in modelInfo.keys() else dict()
        self.to_categ = True if modelInfo["to_categ"]==1 else False
        self.save_images = True if modelInfo["save_images"]==1 else False
        self.da = True if modelInfo["data_augmentation"]==1 else False
        self.train_again = True if modelInfo["train_again"]==1 else False
        self.cross_validation = True if modelInfo["cross_validation"]==1 else False
        self.supervised = True if modelInfo["supervised"]==1 else False
        self.save_activation_filter = True if modelInfo["save_activation_filter"]==1 else False
        self.use_hickle = True if "use_hickle" in modelInfo.keys() and modelInfo["use_hickle"]==1 else False

        # paths
        self.rootPath = setting["root_path"]
        self.datasetFolder = setting["dataset_path"]
        self.labeledImagesFolder = setting["relative_paths"]["labeled_images"]
        self.patientsFolder = setting["relative_paths"]["patients"]
        self.experimentID = "EXP"+general_utils.convertExperimentNumberToString(setting["EXPERIMENT"])
        self.experimentFolder = "SAVE/" + self.experimentID + "/"
        self.savedModelFolder = self.experimentFolder+setting["relative_paths"]["save"]["model"]
        self.savePartialModelFolder = self.experimentFolder+setting["relative_paths"]["save"]["partial_model"]
        self.saveImagesFolder = self.experimentFolder+setting["relative_paths"]["save"]["images"]
        self.savePlotFolder = self.experimentFolder+setting["relative_paths"]["save"]["plot"]
        self.saveTextFolder = self.experimentFolder+setting["relative_paths"]["save"]["text"]
        self.intermediateActivationFolder = self.experimentFolder+setting["relative_paths"]["save"]["intermediate_activation"] if "intermediate_activation" in setting["relative_paths"]["save"].keys() else None

        self.infoCallbacks = modelInfo["callbacks"]

        # empty variables initialization
        self.model = None
        self.testing_score = []
        self.partialWeightsPath = ""
        self.callbacks = None
        self.initial_epoch = 0
        self.train_df, self.val_list, self.test_list = None, None, None
        self.N_BACKGROUND, self.N_BRAIN, self.N_PENUMBRA, self.N_CORE, self.N_TOT = 0, 0, 0, 0, 0
        self.optimizer = None
        self.sample_weights = None
        self.train = None
        self.train_sequence, self.val_sequence = None, None
        self.mp = False

        # change the prefix if SUS2020_v2 is in the dataset name
        if "SUS2020" in self.datasetFolder: constants.setPrefixImagesSUS2020_v2()

    ################################################################################
    # Initialize the callbacks
    def setCallbacks(self, p_id, sample_weights=None, add_for_finetuning=""):
        if getVerbose():
            general_utils.printSeparation("-", 50)
            print("[INFO] - Setting callbacks...")

        self.callbacks = training.getCallbacks(
            root_path=self.rootPath,
            info=self.infoCallbacks,
            filename=self.getSavedInformation(p_id, path=self.savePartialModelFolder),
            textFolderPath=self.saveTextFolder,
            dataset=self.dataset,
            sample_weights=sample_weights, # only for ROC callback (NOT working)
            nn_id=self.getNNID(p_id),
            add_for_finetuning=add_for_finetuning
        )

    ################################################################################
    # return a Boolean to control if the model was already saved
    def isModelSaved(self, p_id):
        saved_modelname = self.getSavedModel(p_id)
        saved_weightname = self.getSavedWeight(p_id)

        return os.path.isfile(saved_modelname) and os.path.isfile(saved_weightname)

    ################################################################################
    # load json and create model
    def loadSavedModel(self, p_id):
        saved_modelname = self.getSavedModel(p_id)
        saved_weightname = self.getSavedWeight(p_id)
        json_file = open(saved_modelname, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(saved_weightname)

        if getVerbose():
            general_utils.printSeparation("+",100)
            print("[INFO - Loading] - --- MODEL {} LOADED FROM DISK! --- ".format(saved_modelname))
            print("[INFO - Loading] - --- WEIGHTS {} LOADED FROM DISK! --- ".format(saved_weightname))

    ################################################################################
    # Check if there are saved partial weights
    def arePartialWeightsSaved(self, p_id):
        # path ==> weight name plus a suffix ":" <-- constants.suffix_partial_weights
        path = self.getSavedInformation(p_id, path=self.savePartialModelFolder) + constants.suffix_partial_weights
        for file in glob.glob(self.savePartialModelFolder+"*.h5"):
            if path in self.rootPath+file:  # we have a match
                self.partialWeightsPath = file
                return True

        return False

    ################################################################################
    # Load the partial weights and set the initial epoch where the weights were saved
    def loadModelFromPartialWeights(self):
        if self.partialWeightsPath!="":
            self.model.load_weights(self.partialWeightsPath)
            self.initial_epoch = general_utils.getEpochFromPartialWeightFilename(self.partialWeightsPath)

            if getVerbose():
                general_utils.printSeparation("+",100)
                print("[INFO - Loading] - --- WEIGHTS {} LOADED FROM DISK! --- ".format(self.partialWeightsPath))
                print("[INFO] - --- Start training from epoch {} --- ".format(str(self.initial_epoch)))

    ################################################################################
    # Function to divide the dataframe in train and test based on the patient id;
    def splitDataset(self, train_df, p_id, listOfPatientsToTrainVal, listOfPatientsToTest):
        # set the dataset inside the class
        self.train_df = train_df
        # split the dataset (return in dataset just the indices and labels)
        self.dataset, self.val_list, self.test_list = dataset_utils.splitDataset(self, p_id, listOfPatientsToTrainVal, listOfPatientsToTest)
        # get the number of element per class in the dataset
        self.N_BACKGROUND, self.N_BRAIN, self.N_PENUMBRA, self.N_CORE, self.N_TOT = dataset_utils.getNumberOfElements(self.train_df)

        return self.val_list

    ################################################################################
    # Function to reshape the pixel array and initialize the model.
    def prepareDataset(self, p_id):
        # split the dataset (set the data key inside dataset [NOT for the sequence generator])
        self.dataset = dataset_utils.prepareDataset(self, p_id)

    ################################################################################
    # compile the model, callable also from outside
    def compileModel(self):
        # set the optimizer (or reset)
        self.optimizer = training.getOptimizer(optInfo=self.optimizerInfo)

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss["loss"],
            metrics=self.metricFuncs
        )

    ################################################################################
    # Function that initialize the training, print the model summary and set the weights
    def initializeTraining(self, p_id, n_gpu):
        if getVerbose():
            general_utils.printSeparation("*", 50)
            print("[INFO] - Start runTraining function.")
            print("[INFO] - Getting model {0} with {1} optimizer...".format(self.name, self.optimizerInfo["name"]))

        # Based on the number of GPUs available, call the function called self.name in architectures.py
        if n_gpu==1:
            self.model = getattr(architectures, self.name)(params=self.params, to_categ=self.to_categ, moreinfo=self.moreinfo)
        else:
            # TODO: problems during the load of the model with multiple GPUs...
            with tf.device('/cpu:0'):
                self.model = getattr(architectures, self.name)(params=self.params, to_categ=self.to_categ, moreinfo=self.moreinfo)
            self.model = multi_gpu_model(self.model, gpus=n_gpu)

        if self.summaryFlag==0:
            if getVerbose(): print(self.model.summary())
            for rankdir in ["LR", "TB"]:
                plot_model(
                    self.model,
                    to_file=general_utils.getFullDirectoryPath(self.savedModelFolder)+self.getNNID("model")+"_"+rankdir+".png",
                    show_shapes=True,
                    rankdir=rankdir
                )
            self.summaryFlag+=1

        # Check if the model has some saved weights to load...
        if self.arePartialWeightsSaved(p_id):  self.loadModelFromPartialWeights()

        # Compile the model with optimizer, loss function and metrics
        self.compileModel()
        # Get the sample weights
        self.sample_weights = self.getSampleWeights("train")
        # Set the callbacks
        self.setCallbacks(p_id, self.sample_weights)

    ################################################################################
    # Run the training over the dataset based on the model
    def runTraining(self, p_id, n_gpu):
        self.initializeTraining(p_id, n_gpu)

        self.dataset["train"]["labels"] = dataset_utils.getLabelsFromIndex(train_df=self.train_df, dataset=self.dataset["train"], modelname=self.name, to_categ=self.to_categ, flag="train")
        self.dataset["val"]["labels"] = None if self.val["validation_perc"]==0 else dataset_utils.getLabelsFromIndex(train_df=self.train_df, dataset=self.dataset["val"], modelname=self.name, to_categ=self.to_categ, flag="val")
        if self.supervised: self.dataset["test"]["labels"] = dataset_utils.getLabelsFromIndex(train_df=self.train_df, dataset=self.dataset["test"], modelname=self.name, to_categ=self.to_categ, flag="test")

        # fit and train the model
        self.train = training.fitModel(
            model=self.model,
            dataset=self.dataset,
            batch_size=self.batch_size,
            epochs=self.epochs,
            listOfCallbacks=self.callbacks,
            sample_weights=self.sample_weights,
            initial_epoch=self.initial_epoch,
            save_activation_filter=self.save_activation_filter,
            intermediate_activation_path=self.intermediateActivationFolder,
            use_multiprocessing=self.mp)

        # plot the loss and accuracy of the training
        training.plotLossAndAccuracy(self, p_id)

        # deallocate memory
        for flag in ["train", "val", "test"]:
            for t in ["labels", "data"]:
                if t in self.dataset[flag]: del self.dataset[flag][t]

    ################################################################################
    # Function to prepare the train and validation sequence using the datasetSequence class
    def prepareSequenceClass(self):
        # train data sequence
        self.train_sequence = sequence_utils.datasetSequence(
            dataframe=self.train_df,
            indices=self.dataset["train"]["indices"],
            sample_weights=self.getSampleWeights("train"),
            x_label="pixels" if not constants.getUSE_PM() else constants.getList_PMS(),
            y_label="ground_truth",
            moreinfo=self.moreinfo,
            to_categ=self.to_categ,
            batch_size=self.batch_size,
            back_perc=2 if not constants.getUSE_PM() or (constants.getM() != constants.IMAGE_WIDTH
                                                         and constants.getN() != constants.IMAGE_HEIGHT) else 100,
            loss=self.loss["name"]
        )

        # validation data sequence
        self.val_sequence = sequence_utils.datasetSequence(
            dataframe=self.train_df,
            indices=self.dataset["val"]["indices"],
            sample_weights=self.getSampleWeights("val"),
            x_label="pixels" if not constants.getUSE_PM() else constants.getList_PMS(),
            y_label="ground_truth",
            moreinfo=self.moreinfo,
            to_categ=self.to_categ,
            batch_size=self.batch_size,
            back_perc=2 if not constants.getUSE_PM() or (constants.getM() != constants.IMAGE_WIDTH
                                                         and constants.getN() != constants.IMAGE_HEIGHT) else 100,
            flagtype="val",
            loss=self.loss["name"]
        )

    ################################################################################
    # Function to start the train using the sequence as input and the fit_generator function
    def runTrainSequence(self, clear=True):
        if "gradual_finetuning_solution" not in self.params.keys(): clear=False
        self.train = training.fit_generator(
            model=self.model,
            train_sequence=self.train_sequence,
            val_sequence=self.val_sequence,
            steps_per_epoch=math.ceil((self.train_sequence.__len__()*self.steps_per_epoch_ratio)),
            validation_steps=math.ceil((self.val_sequence.__len__()*self.validation_steps_ratio)),
            epochs=self.epochs,
            listOfCallbacks=self.callbacks,
            initial_epoch=self.initial_epoch,
            save_activation_filter=self.save_activation_filter,
            intermediate_activation_path=self.intermediateActivationFolder,
            use_multiprocessing=self.mp,
            clear=clear
        )

    ################################################################################
    # Check if we need to perform the hybrid solution or not
    def gradualFineTuningSolution(self, p_id):
        # Hybrid solution to fine-tuning the model unfreezing the layers in the VGG-16 architectures
        if "gradual_finetuning_solution" in self.params.keys() and self.params["trainable"] == 0:
            finished_first_half = False
            layer_indexes = []
            for pm in constants.getList_PMS(): layer_indexes.extend([i for i, l in enumerate(self.model.layers) if pm.lower() in l.name])
            layer_indexes = np.sort(layer_indexes)

            # The optimizer (==ADAM) should have a low learning rate
            if self.optimizerInfo["name"].lower() != "adam":
                print("The optimizer is not Adam!")
                return

            self.optimizerInfo["lr"] = 1e-5
            self.infoCallbacks["ModelCheckpoint"]["period"] = 1
            previousEarlyStoppingPatience = self.infoCallbacks["EarlyStopping"]["patience"]
            self.infoCallbacks["EarlyStopping"]["patience"] = 25

            if self.params["gradual_finetuning_solution"]["type"] == "half":
                # Perform fine tuning twice: first on the bottom half, then on the totality
                # Make the bottom half of the VGG-16 layers trainable
                for ind in layer_indexes[len(layer_indexes) // 2:]: self.model.layers[ind].trainable = True
                if getVerbose(): print("Fine-tuning setting: {} layers trainable".format(layer_indexes[len(layer_indexes) // 2:]))
                if self.arePartialWeightsSaved(p_id):
                    self.model.load_weights(self.partialWeightsPath)
                    self.initial_epoch = general_utils.getEpochFromPartialWeightFilename(self.partialWeightsPath) + previousEarlyStoppingPatience
                # Compile the model again
                self.compileModel()
                # Get the sample weights
                self.sample_weights = self.getSampleWeights("train")
                # Set the callbacks
                self.setCallbacks(p_id, self.sample_weights, "_half")
                # Train the model again
                self.runTrainSequence()
                finished_first_half = True

            if self.params["gradual_finetuning_solution"]["type"] == "full" or finished_first_half:
                # Make ALL the VGG-16 layers trainable
                for ind in layer_indexes: self.model.layers[ind].trainable = True
                if getVerbose(): print("Fine-tuning setting: {} layers trainable".format(layer_indexes))
                if self.arePartialWeightsSaved(p_id):
                    self.model.load_weights(self.partialWeightsPath)
                    self.initial_epoch = general_utils.getEpochFromPartialWeightFilename(self.partialWeightsPath) + previousEarlyStoppingPatience
                # Compile the model again
                self.compileModel()
                # Get the sample weights
                self.sample_weights = self.getSampleWeights("train")
                # Set the callbacks
                self.setCallbacks(p_id, self.sample_weights, "_full")
                # Train the model again
                self.runTrainSequence(clear=False)

    ################################################################################
    # Get the sample weight from the dataset
    def getSampleWeights(self, flagDataset):
        sample_weights = None
        self.N_BACKGROUND, self.N_BRAIN, self.N_PENUMBRA, self.N_CORE, self.N_TOT = dataset_utils.getNumberOfElements(self.train_df)

        if constants.N_CLASSES==4:
            # and the (M,N) == image dimension
            if constants.getM()== constants.IMAGE_WIDTH and constants.getN()== constants.IMAGE_HEIGHT:
                if self.use_sequence:  # set everything == 1
                    sample_weights = self.train_df.assign(ground_truth=1)
                    sample_weights = sample_weights.ground_truth
                else:
                    # function that map each PIXELVALUES[2] with 150, PIXELVALUES[3] with 20
                    # and the rest with 0.1 and sum them
                    f = lambda x: np.sum(np.where(np.array(x) == constants.PIXELVALUES[2],
                                                  constants.HOT_ONE_WEIGHTS[0][3],
                                                  np.where(np.array(x) == constants.PIXELVALUES[3],
                                                           constants.HOT_ONE_WEIGHTS[0][2],
                                                           constants.HOT_ONE_WEIGHTS[0][0])))

                    sample_weights = self.train_df.ground_truth.map(f)
                    sample_weights = sample_weights/(constants.getM() * constants.getN())
            else:
                # see: "ISBI 2019 C-NMC Challenge: Classification in Cancer Cell Imaging" section 4.1 pag 68
                sample_weights = self.train_df.label.map({
                    constants.LABELS[0]: self.N_TOT / (constants.N_CLASSES * self.N_BACKGROUND) if self.N_BACKGROUND > 0 else 0,
                    constants.LABELS[1]: self.N_TOT / (constants.N_CLASSES * self.N_BRAIN) if self.N_BRAIN > 0 else 0,
                    constants.LABELS[2]: self.N_TOT / (constants.N_CLASSES * self.N_PENUMBRA) if self.N_PENUMBRA > 0 else 0,
                    constants.LABELS[3]: self.N_TOT / (constants.N_CLASSES * self.N_CORE) if self.N_CORE > 0 else 0,
                })
        elif constants.N_CLASSES==3:
            # and the (M,N) == image dimension
            if constants.getM()== constants.IMAGE_WIDTH and constants.getN()== constants.IMAGE_HEIGHT:
                if self.use_sequence:  # set everything == 1
                    sample_weights = self.train_df.assign(ground_truth=1)
                    sample_weights = sample_weights.ground_truth
                else:
                    # function that map each PIXELVALUES[2] with 150, PIXELVALUES[3] with 20
                    # and the rest with 0.1 and sum them
                    f = lambda x: np.sum(np.where(np.array(x)==150,150,np.where(np.array(x)==76,20,0.1)))

                    sample_weights = self.train_df.ground_truth.map(f)
                    sample_weights = sample_weights/(constants.getM() * constants.getN())
            else:
                # see: "ISBI 2019 C-NMC Challenge: Classification in Cancer Cell Imaging" section 4.1 pag 68
                sample_weights = self.train_df.label.map({
                    constants.LABELS[0]: self.N_TOT / (constants.N_CLASSES * (self.N_BACKGROUND + self.N_BRAIN)) if self.N_BACKGROUND + self.N_BRAIN > 0 else 0,
                    constants.LABELS[1]: self.N_TOT / (constants.N_CLASSES * self.N_PENUMBRA) if self.N_PENUMBRA > 0 else 0,
                    constants.LABELS[2]: self.N_TOT / (constants.N_CLASSES * self.N_CORE) if self.N_CORE > 0 else 0,
                })
        elif constants.N_CLASSES==2:  # we are in a binary class problem
            # f = lambda x : np.sum(np.array(x))
            # sample_weights = self.train_df.ground_truth.map(f)
            sample_weights = self.train_df.label.map({
                constants.LABELS[0]: self.N_TOT / (constants.N_CLASSES * self.N_BACKGROUND) if self.N_BACKGROUND > 0 else 0,
                constants.LABELS[1]: self.N_TOT / (constants.N_CLASSES * self.N_CORE) if self.N_CORE > 0 else 0,
            })

        return np.array(sample_weights.values[self.dataset[flagDataset]["indices"]])

    ################################################################################
    # Save the trained model and its relative weights
    def saveModelAndWeight(self, p_id):
        saved_modelname = self.getSavedModel(p_id)
        saved_weightname = self.getSavedWeight(p_id)

        p_id = general_utils.getStringFromIndex(p_id)
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(saved_modelname, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(saved_weightname)

        if getVerbose():
            general_utils.printSeparation("-", 50)
            print("[INFO - Saving] - Saved model and weights to disk!")

    ################################################################################
    # Call the function located in testing for predicting and saved the images
    def predictAndSaveImages(self, listPatients, isAlreadySaved):
        stats = {}
        for p_id in listPatients:
            # evaluate the model with the testing patient (not necessary)
            # if self.supervised: self.evaluateModelWithCategorics(p_id, isAlreadySaved)

            if getVerbose():
                general_utils.printSeparation("+", 50)
                print("[INFO] - Executing function: predictAndSaveImages for patient {}".format(p_id))

            testing.predictAndSaveImages(self, p_id)

        return stats


    ################################################################################
    # Test the model with the selected patient (if the number of patient to test is > 0)
    def evaluateModelWithCategorics(self, p_id, isAlreadySaved):
        if getVerbose():
            general_utils.printSeparation("+", 50)
            print("[INFO] - Evaluating the model for patient {}".format(p_id))

        self.testing_score.append(testing.evaluateModel(self, p_id, isAlreadySaved))

    ################################################################################
    # set the flag for single/multi PROCESSING
    def setProcessingEnv(self, mp):
        self.mp = mp

    ################################################################################
    # return the saved model or weight (based on the suffix)
    def getSavedInformation(self, p_id, path, other_info="", suffix=""):
        # mJ-Net_DA_ADAM_4_16x16.json <-- example weights name
        # mJ-Net_DA_ADAM_4_16x16.h5 <-- example model name
        path = general_utils.getFullDirectoryPath(path)+self.getNNID(p_id)+other_info+general_utils.getSuffix()
        return path+suffix

    ################################################################################
    # return the saved model
    def getSavedModel(self, p_id):
        return self.getSavedInformation(p_id, path=self.savedModelFolder, suffix=".json")

    ################################################################################
    # return the saved weight
    def getSavedWeight(self, p_id):
        return self.getSavedInformation(p_id, path=self.savedModelFolder, suffix=".h5")

    ################################################################################
    # return NeuralNetwork ID
    def getNNID(self, p_id):
        # CAREFUL WITH THIS
        # needs to override the model id to use a different model to test various patients
        if self.OVERRIDE_MODELS_ID_PATH: ret_id = self.OVERRIDE_MODELS_ID_PATH
        else:
            ret_id = self.name
            if self.da: ret_id += "_DA"
            ret_id += ("_" + self.optimizerInfo["name"].upper())

            ret_id += ("_VAL" + str(self.val["validation_perc"]))
            if self.val["random_validation_selection"]: ret_id += "_RANDOM"

            if self.to_categ: ret_id += "_SOFTMAX"  # differentiate between softmax and sigmoid last activation layer

            # if there is cross validation, add the PATIENT_ID to differentiate the models
            if self.cross_validation: ret_id += ("_" + p_id)

        return ret_id
