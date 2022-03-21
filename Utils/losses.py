import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from Utils import metrics
from Model import constants
import tensorflow.keras.backend as K


################################################################################
# Function that calculates the modified DICE coefficient loss. Util for the LOSS
# function during the training of the model (for image in input and output)!
def squared_dice_coef_loss(y_true, y_pred):
    return 1 - metrics.squared_dice_coef(y_true, y_pred)


################################################################################
# Calculate the real value for the Dice coefficient, but it returns lower values than
# the other dice_coef + lower specificity and precision
def dice_coef_loss(y_true, y_pred):
    return 1-metrics.dice_coef(y_true, y_pred)


################################################################################
# Tversky loss.
# Based on this paper: https://arxiv.org/abs/1706.05721
def tversky_loss(y_true, y_pred):
    tv = metrics.tversky_coef(y_true, y_pred)
    return 1 - tv


################################################################################
# Focal Tversky loss: a generalisation of the tversky loss.
# From this paper: https://arxiv.org/abs/1810.07842
def focal_tversky_loss(y_true, y_pred):
    gamma = constants.focal_tversky_loss["gamma"]
    tv = metrics.tversky_coef(y_true, y_pred)
    return K.pow((1 - tv), (1/gamma))


################################################################################
# Function that calculates the JACCARD index loss. Util for the LOSS function during
# the training of the model (for image in input and output)!
def jaccard_index_loss(y_true, y_pred, smooth=100):
    return (1-metrics.jaccard_distance(y_true, y_pred, smooth)) * smooth


################################################################################
# Function that calculate the weighted categorical crossentropy based on the
# article: https://doi.org/10.1109/ACCESS.2019.2910348
def weighted_categorical_cross_entropy_loss(y_true, y_pred):
    return metrics.weighted_categorical_cross_entropy(y_true, y_pred)


################################################################################
#  Focal loss: https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf
def focal_loss(y_true, y_pred):
    return metrics.focal_loss(y_true, y_pred)


################################################################################
# Tanimoto loss. https://arxiv.org/pdf/1904.00592.pdf
def tanimoto_loss(y_true, y_pred):
    return 1-metrics.tanimoto(y_true, y_pred)


################################################################################
# Tanimoto loss with its complement. https://arxiv.org/pdf/1904.00592.pdf
def tanimoto_with_dual_loss(y_true, y_pred):
    return 1-((metrics.tanimoto(y_true, y_pred)+metrics.tanimoto(1-y_true, 1-y_pred))/2)
