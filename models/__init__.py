
from models.generator import *

from models.slr_i3d.i3d import InceptionI3d
from models.slr_i3d.slr_i3d import SLR_I3D
from models.DNF.dnf import DNFmodel,GoogLeNet_TConvsAtt
from models.stmc.stmc_model import STMC
from models.stmc.vgg_pose import VGGPose_Conv1D_RNN, VGGPose_Conv1D
from models.subunet.subunet import SubUNet


def SLR_video_encoder(config,args, N_classes):

    if args.model == 'STMC':
        return STMC(args, N_classes, num_of_visual_cues=2, K=7, mode=args.mode)
    elif args.model == 'VGGPose_Conv1D':
        return VGGPose_Conv1D(N_classes)
    elif args.model == 'GoogLeNet_TConvs':
        return DNFmodel(config, args, N_classes=N_classes)
    elif args.model == 'SubUNet':
        return SubUNet(config,args, N_classes=N_classes)
    elif args.model == 'I3D':
        return SLR_I3D(config,args, num_classes=N_classes,  temporal_resolution=25)
