
from models.generator import *

from models.slr_i3d.i3d import InceptionI3d
from models.slr_i3d.slr_i3d import SLR_I3D
from models.googlenet_tconvs.googlenet_tconvs import GoogLeNet_TConvs,GoogLeNet_TConvsAtt
from models.stmc.stmc_model import STMC
from models.stmc.vgg_pose import VGGPose_Conv1D_RNN, VGGPose_Conv1D


def select_model(args, N_classes):
    if (args.discriminator == 'CLM'):
        d = CritisizingLM(N_classes, args.hidden_size)
    elif (args.discriminator == 'rnn'):
        d = SLRDiscriminator(args, N_classes, args.filter_size, clm='rnn')
    elif (args.discriminator == 'tcn'):
        d = SLRDiscriminator(args, N_classes, args.filter_size, clm='tcn')
    elif (args.discriminator == 'cnn'):
        d = SLRDiscriminator(args, N_classes, args.filter_size, clm='cnn')
    elif (args.discriminator == 'gloss'):
        d = GlossDiscriminator(args, N_classes, args.filter_size, clm='lstm')
    elif (args.discriminator == 'hybrid'):
        d = Hybridiscriminator(args, N_classes)
    elif (args.discriminator == 'context'):
        d = Context_discriminator(args, N_classes, args.filter_size, clm='lstm', fusion='concat')
    elif (args.discriminator == 'pair'):
        d = HybridConcatDiscriminator(args, N_classes, args.filter_size, clm='rnn')

    if (args.modality == 'feats'):
        g = SLR_Feature_Generator(args, N_classes,
                                  mode='continuous')
    else:
        g = SLR_Generator(args, N_classes,
                          mode='continuous')

    return g, d


def SLR_video_encoder(config,args, N_classes):

    if args.model == 'STMC':
        return STMC(args, N_classes, num_of_visual_cues=2, K=7, mode=args.mode)
    elif args.model == 'VGGPose_Conv1D':
        return VGGPose_Conv1D(N_classes)
    elif args.model == 'GoogLeNet_TConvs':
        return GoogLeNet_TConvs(config,args, N_classes=N_classes)
    elif args.model == 'GoogLeNet_TConvsAtt':
        return GoogLeNet_TConvsAtt(args, N_classes=N_classes)
    elif args.model == 'I3D':
        return SLR_I3D(config,args, num_classes=N_classes,  temporal_resolution=25)
