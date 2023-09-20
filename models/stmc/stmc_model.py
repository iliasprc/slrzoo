import torch
import torch.nn as nn
import torchvision.models as models


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def soft_argmax(voxels):
    """
    Arguments: voxel patch in shape (batch_size, channel, H, W, depth)
    Return: 3D coordinates in shape (batch_size, channel, 3)
    """
    assert voxels.dim() == 4
    # alpha is here to make the largest element really big, so it
    # would become very close to 1 after softmax
    alpha = 1000.0
    N, C, H, W = voxels.shape
    soft_max = nn.functional.softmax(voxels.view(N, C, -1) * alpha, dim=2)
    soft_max = soft_max.view(voxels.shape)
    # print(voxels.device)
    indices_kernel = torch.arange(start=0, end=H * W).unsqueeze(0).float().to(voxels.device)
    indices_kernel = indices_kernel.view((H, W)).to(voxels.device)
    conv = soft_max * indices_kernel
    indices = conv.sum(2).sum(2)

    y = (indices).floor() % W
    x = (((indices).floor()) / W).floor() % H
    coords = torch.stack([x, y], dim=2)
    return coords


class SMC(nn.Module):
    def __init__(self, args, num_of_visual_cues=2, K=7):
        super(SMC, self).__init__()
        if (args.cnn == 'vgg11'):
            self.cnn1 = models.vgg11(True).features[0:10]
            self.cnn2 = models.vgg11(True).features[10:18]
            self.cnn3 = models.vgg11(True).features[18:]
            self.num_of_feats = 512
            self.transpose_input_feats = 512
        elif (args.cnn == 'googlenet'):
            self.cnn1 = nn.Sequential(*list(models.googlenet(pretrained=True, aux_logits=False).children())[:-10])

            self.cnn2 = nn.Sequential(*list(models.googlenet(pretrained=True, aux_logits=False).children())[-10:-6])
            self.cnn3 = nn.Sequential(*list(models.googlenet(pretrained=True, aux_logits=False).children())[-6:-3])
            self.num_of_feats = 1024
            self.transpose_input_feats = 832
        elif (args.cnn == 'resnet18'):
            self.cnn1 = nn.Sequential(*list(models.resnet18(True).children())[:-5])

            self.cnn2 = nn.Sequential(*list(models.resnet18(True).children())[-5:-3])
            self.cnn3 = nn.Sequential(*list(models.resnet18(True).children())[-3:-2])
            self.num_of_feats = 512
            self.transpose_input_feats = 256
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.deconv = nn.Sequential(nn.ConvTranspose2d(self.transpose_input_feats, 512, 2, stride=2), nn.ReLU(),
                                    nn.ConvTranspose2d(512, 512, 2, stride=2), nn.ReLU(), nn.Conv2d(512, K, 1, 1))
        self.pose_fc = nn.Sequential(nn.Linear(K * 2, 256), nn.ReLU())
        # self.cnn = nn.Sequential(*list(models.vgg11(True).children())[:-2])
        # self.cnn = nn.Sequential(*list(models.vgg11(True).children()))

    def forward(self, x):
        batch_size_x_Timesteps, C, H, W = x.size()
        out1 = self.cnn1(x)
        # print('cnn1 ',out1.shape)
        out2 = self.cnn2(out1)
        # print('cnn2 ',out2.shape)
        out3 = self.cnn3(out2)
        full_frame_features = self.global_pool(out3).squeeze(-1).squeeze(-1)
        deco = self.deconv(out2)
        # print(deco.shape)
        pose_coords = soft_argmax(deco)
        b, _, _ = pose_coords.shape
        pose_coords = pose_coords.view(batch_size_x_Timesteps, -1)
        # print(pose_coords.shape)
        pose_feats = self.pose_fc(pose_coords)
        # print(pose_feats.shape,full_frame_features.shape)
        out = torch.cat((full_frame_features, pose_feats), dim=-1)

        return out, full_frame_features, pose_feats


class TMC(nn.Module):
    def __init__(self, out_channels, features=[512, 256], names=['intra_cue_1', 'intra_cue_2'], kernel_size=5,
                 pool_size=2, padding=2):
        super(TMC, self).__init__()
        self.tmc_block1 = TMC_Block(out_channels=out_channels, features=features, names=names, kernel_size=kernel_size,
                                    pool_size=pool_size, padding=padding)
        f = [out_channels // len(features) for i in range(len(features))]
        # print('f ', f)
        self.tmc_block2 = TMC_Block(out_channels=out_channels, features=f,
                                    names=names, kernel_size=kernel_size, pool_size=pool_size, padding=padding)

    def forward(self, x1, x2):
        o1, f1 = self.tmc_block1(x1, x2)
        o2, f2 = self.tmc_block2(o1, f1)
        return o2, f2


class TMC_Block(nn.Module):
    def __init__(self, out_channels=512, features=[256, 256], names=['intra_cue_1', 'intra_cue_2'],
                 kernel_size=5, pool_size=2, padding=2):
        super(TMC_Block, self).__init__()
        self.names = names
        self.end_points = {}
        for i in range(len(names)):
            self.end_points[names[i]] = nn.Sequential(
                nn.Conv1d(features[i], out_channels // len(names), kernel_size=kernel_size, padding=padding),
                nn.ReLU())
        self.end_points['pw_conv'] = nn.Conv1d(out_channels, out_channels // 2, kernel_size=1)
        self.end_points['inter_cue'] = nn.Conv1d(sum(features), out_channels // 2, kernel_size=kernel_size,
                                                 padding=padding)
        self.end_points['relu'] = nn.ReLU()
        self.end_points['pool'] = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        self.build()

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, inter_cue_input, intra_cue_input):
        outputs = []

        # print(intra_cue_input.shape)
        for i, name in enumerate(self.names):
            # print(intra_cue_input[:,i,...].shape)
            out = self._modules[name](intra_cue_input[i])
            outputs.append(out)
            # print(out.shape)

        fl = torch.cat(outputs, dim=1)
        # print(fl.shape)
        pw_out = self._modules['pw_conv'](fl)
        # print(pw_out.shape)
        inter_cue_conv = self._modules['inter_cue'](inter_cue_input)
        # print(inter_cue_conv.shape)
        ol = self._modules['relu'](torch.cat((pw_out, inter_cue_conv), dim=1))

        out_inter = self._modules['pool'](ol)

        out_intra = []
        for i in outputs:
            out_intra.append(self._modules['pool'](i))
            # print(self._modules['pool'](i).shape)

        return out_inter, out_intra


class STMC(nn.Module):
    def __init__(self, args, num_classes, num_of_visual_cues=2, K=7, mode='continuous'):
        super(STMC, self).__init__()
        self.end_points = {}
        self.end_points['spatial'] = SMC(args, num_of_visual_cues, K)
        features = [self.end_points['spatial'].num_of_feats, 256]
        if mode == 'continuous':
            padding = 2
            self._forward_ = self.continous_forward
        else:
            padding = 0
            self._forward_ = self.isolated_forward
        self.n_cl = num_classes
        self.end_points['temporal'] = TMC(out_channels=args.out_channels, features=features,
                                          kernel_size=args.kernel1d,
                                          pool_size=args.pool1d, padding=padding)
        self.end_points['seq_model'] = nn.LSTM(
            input_size=args.out_channels,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional)
        for i in range(num_of_visual_cues):
            self.end_points['seq' + str(i)] = nn.LSTM(
                input_size=args.out_channels // num_of_visual_cues,
                hidden_size=256,
                num_layers=1,
                dropout=args.dropout,
                bidirectional=args.bidirectional)

        if (args.bidirectional):
            self.end_points['fc'] = nn.Linear(args.hidden_size, self.n_cl)
        else:
            self.end_points['fc'] = nn.Linear(args.hidden_size, self.n_cl)
        for i in range(num_of_visual_cues):
            if (args.bidirectional):
                self.end_points['fc_' + str(i)] = nn.Linear(256, self.n_cl)
            else:
                self.end_points['fc_' + str(i)] = nn.Linear(256, self.n_cl)

        self.end_points['iso_fc'] = torch.nn.Linear(args.hidden_size, self.n_cl)
        self.build()

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x):
        return self._forward_(x)

    def continous_forward(self, x):

        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        cnn_out, full_frame_features, pose_feats = self._modules['spatial'](c_in)
        cnn_out = cnn_out.contiguous().view(batch_size, -1, timesteps)
        full_frame_features = full_frame_features.contiguous().view(batch_size, -1, timesteps)
        pose_feats = pose_feats.contiguous().view(batch_size, -1, timesteps)
        # print(cnn_out.shape)

        o, f = self._modules['temporal'](cnn_out, [full_frame_features, pose_feats])
        # print(o.permute(2, 0, 1).shape, f[0].shape, f[1].shape)
        full_feats_blstm, _ = self._modules['seq_model'](o.permute(2, 0, 1))
        feats_blstm_0, _ = self._modules['seq0'](f[0].permute(2, 0, 1))
        feats_blstm_1, _ = self._modules['seq1'](f[1].permute(2, 0, 1))

        full_logits = self._modules['fc'](full_feats_blstm)
        logits0 = self._modules['fc_0'](feats_blstm_0)
        logits1 = self._modules['fc_1'](feats_blstm_1)
        return full_logits, logits0, logits1

    def isolated_forward(self, x):

        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        cnn_out, full_frame_features, pose_feats = self._modules['spatial'](c_in)
        cnn_out = cnn_out.contiguous().view(batch_size, -1, timesteps)
        full_frame_features = full_frame_features.contiguous().view(batch_size, -1, timesteps)
        pose_feats = pose_feats.contiguous().view(batch_size, -1, timesteps)
        # print(cnn_out.shape)

        o, f = self._modules['temporal'](cnn_out, [full_frame_features, pose_feats])

        full_logits = self._modules['fc'](o.permute(2, 0, 1)).squeeze(0)
        logits_frame = self._modules['fc_0'](f[0].permute(2, 0, 1)).squeeze(0)
        logits_pose = self._modules['fc_1'](f[1].permute(2, 0, 1)).squeeze(0)
        # print(full_logits.shape)
        return full_logits, logits_frame, logits_pose


# a = TMC(1024, features=[512, 256], names=['t1', 't2'])
# print(a)
# a(torch.randn(3, 512 + 256, 5), [torch.randn(3, 512, 5), torch.randn(3, 256, 5)])
# torchsummary.summary(a.cuda(),(2,256,5),batch_size=4)

# a = SMC()
# print(a)
# torchsummary.summary(a.cuda(), (3, 224, 224))
def arguments():
    import argparse
    parser = argparse.ArgumentParser(description='SLR-GAN weakly supervised training')
    parser.add_argument('--modality', type=str, default='full', metavar='rc',
                        help='hands or full image')
    parser.add_argument('--dataset', type=str, default='greek_SI', metavar='rc',
                        help='slr dataset   phoenixI5 phoenix2014feats  phoenix2014  csl  signum_continuous ')

    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')

    parser.add_argument('--epochs', type=int, default=25, metavar='N',
                        help='number of epochs to train (default: 10)')

    parser.add_argument('--cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1234)')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='image normalization mean std')
    parser.add_argument('--run_full', action='store_true', default=True
                        )
    parser.add_argument('--padding', action='store_true', default=False,
                        help='video padding')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='For Saving the current Model')

    ## optimizer and scheduler

    parser.add_argument('--adversarial_loss', type=str, default='bce')
    parser.add_argument('--optimD', type=str, default='adam', metavar='optim number', help='optimizer sgd or adam')
    parser.add_argument('--optimG', type=str, default='adam', metavar='optim number', help='optimizer sgd or adam')
    parser.add_argument('--lr_Discriminator', type=float, default=0.000051, metavar='LR',

                        help='learning rate (default: 0.0001)')
    parser.add_argument('--lr_Generator', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--scheduler', type=str, default='ReduceLRonPlateau', metavar='scheduler lr ',
                        help='scheduler lr ')

    parser.add_argument('--gradient_clipping', action='store_true', default=False)
    parser.add_argument('--gradient_penalty', action='store_true', default=False)

    parser.add_argument('--scheduler_factor', type=float, default=0.5)
    parser.add_argument('--scheduler_patience', type=float, default=1)
    parser.add_argument('--scheduler_min_lr', type=float, default=5e-6)
    parser.add_argument('--scheduler_verbose', type=float, default=5e-6)
    ## DISCRIMINATOR ARGUMENTS

    parser.add_argument('--discriminator', type=str, default='pair', help='DISCRIMINATOR model')
    parser.add_argument('--filter_size', type=int, default=128, metavar='num', help='num of discriminator filters')
    parser.add_argument('--D_layers', type=int, default=1, metavar='num', help='num of layers')

    parser.add_argument('--D_bidir', action='store_true', default=False, help='bidirectional for rnn')

    ## GENERATOR ARGUMENTS
    parser.add_argument('--generator', type=str, default='features', help='features or full')

    parser.add_argument('--seq-length', type=int, default=250, metavar='num', help='frame sequence length')
    parser.add_argument('--hidden_size', type=int, default=512, metavar='num', help='GENERATOR lstm units')
    parser.add_argument('--out_channels', type=int, default=512, metavar='num', help='GENERATOR lstm units')
    parser.add_argument('--num_layers', type=int, default=2, metavar='num', help=' GENERATOR rnn layers')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='num', help='GENERATOR hidden size')
    parser.add_argument('--bidirectional', action='store_true', default=True, help='GENERATOR bidirectional for rnn')
    parser.add_argument('--cnn', type=str, default='resnet18', help='GENERATOR cnn backbone')

    parser.add_argument('--rnn', type=str, default='rnn',
                        help='GENERATOR rnn backbone , tcn for dilated 1dconvs, ti for timecption')
    parser.add_argument('--trainG', action='store_true', default=False)
    parser.add_argument('--ctc', type=str, default='normal',
                        help='normal for vanilla-CTC or focal or ent_ctc or custom or weighted or aggregation or stim_ctc')
    parser.add_argument('--kernel1d', type=int, default=5, metavar='num', help='GENERATOR TConv kernel size ')
    parser.add_argument('--pool1d', type=int, default=2, metavar='num', help='GENERATOR TConv pooling size')

    args = parser.parse_args()

    return args

#
# args = arguments()
# m = STMC(args, 1000, mode='isolated')
# print(m)
# a = m(torch.randn(1, 16, 3, 224, 224))
