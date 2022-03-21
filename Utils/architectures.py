from Model import constants
from Architectures import arch_mJNet, arch_PMs_segm, arch_autoencoder, arch_van_De_Leemput, arch_UNet, arch_VNet
from Utils import general_utils, spatial_pyramid


################################################################################
# mJ-Net model
def mJNet(params, to_categ, moreinfo):
    return arch_mJNet.mJNet(params, to_categ)


################################################################################
# Function to call the mJ-net with dropout
def mJNet_NOBatch(params, to_categ, moreinfo):
    return arch_mJNet.mJNet(params, to_categ, batch=False)


################################################################################
# Function to call the mJ-net with dropout
def mJNet_Drop(params, to_categ, moreinfo):
    return arch_mJNet.mJNet(params, to_categ, drop=True)


################################################################################
# Function to call the mJ-net with dropout and a long "J"
def mJNet_LongJ_Drop(params, to_categ, moreinfo):
    return arch_mJNet.mJNet(params, to_categ, drop=True, longJ=True)


################################################################################
# Function to call the mJ-net with a long "J"
def mJNet_LongJ(params, to_categ, moreinfo):
    return arch_mJNet.mJNet(params, to_categ, longJ=True)


################################################################################
# mJ-Net model version 2
def mJNet_v2(params, to_categ, moreinfo):
    return arch_mJNet.mJNet(params, to_categ, drop=True, longJ=True, v2=True)


################################################################################
# mJ-Net model version 3D
def mJNet_v3D(params, to_categ, moreinfo):
    return arch_mJNet.mJNet_v3D(params, to_categ)


################################################################################
def PMs_segmentation(params, to_categ, moreinfo):
    return arch_PMs_segm.PMs_segmentation(params, to_categ, moreinfo)


################################################################################
def PMs_segmentation_NOBatch(params, to_categ, moreinfo):
    return arch_PMs_segm.PMs_segmentation(params, to_categ, moreinfo, batch=False)


################################################################################
def PMs_segmentation_MoreInfo(params, to_categ, moreinfo):
    return arch_PMs_segm.PMs_segmentation(params, to_categ, moreinfo)


################################################################################
def PMs_segmentation_MoreInfo_NOBatch(params, to_categ, moreinfo):
    return arch_PMs_segm.PMs_segmentation(params, to_categ, moreinfo, batch=False)


################################################################################
def simple_autoencoder(params, to_categ, moreinfo):
    return arch_autoencoder.simple_autoencoder(params, to_categ)


################################################################################
# Model from Van De Leemput
def van_De_Leemput(params, to_categ, moreinfo):
    return arch_van_De_Leemput.van_De_Leemput(params, to_categ)


################################################################################
# Model from Ronneberger (original paper of U-Net) (https://doi.org/10.1007/978-3-319-24574-4_28)
def Ronneberger_UNET(params, to_categ, moreinfo):
    return arch_UNet.Ronneberger_UNET(params, to_categ)


################################################################################
# Model from Milletari V-Net (https://arxiv.org/pdf/1606.04797.pdf)
def VNet_Milletari(params, to_categ, moreinfo):
    return arch_VNet.VNet_Milletari(params, to_categ)


################################################################################
# Model for the V-Net Light https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9098643
def VNet_Light(params, to_categ, moreinfo):
    return arch_VNet.VNet_Light(params, to_categ)
