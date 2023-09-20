from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

# from main_weaklys import SEED
SEED = 1234
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# random.seed(SEED)
MAX_LENGTH = 30
# torch.cuda.manual_seed(SEED)

from models.slr_i3d.i3d import InceptionI3d


class SLR_I3D(nn.Module):
    def __init__(self, config, args, num_classes=400, temporal_resolution=24):
        super(SLR_I3D, self).__init__()

        self.n_classes = num_classes
        self.mode = config.model.mode

        temporal_resolution = config.dataset.train.seq_length

        self.cnn = InceptionI3d(num_classes=self.n_classes, temporal_resolution=temporal_resolution, mode=self.mode,
                                in_channels=3)

        if config.model.pretrained:
            print("load imagenet  weights")
            self.cnn.load_state_dict(torch.load('./checkpoints/rgb_imagenet.pt', map_location='cpu'))
            self.cnn.replace_logits(self.n_classes)

        self.rnn = nn.LSTM(
            input_size=1024,
            hidden_size=config.model.rnn.hidden_size,
            num_layers=config.model.rnn.num_layers,
            dropout=config.model.rnn.dropout,
            bidirectional=config.model.rnn.bidirectional)
        if self.mode == 'continuous':
            self.freeze_param()

    def forward(self, x):

        x = self.cnn(x)

        if (self.mode == 'isolated'):

            return torch.mean(x, dim=-1)
        elif (self.mode == 'continuous'):

            r_out, (h_n, h_c) = self.rnn(x)

            x = self.cnn.logits(r_out.permute(1, 2, 0).unsqueeze(-1).unsqueeze(-1))

            return x.squeeze(-1).squeeze(-1).permute(2, 0, 1)

    def freeze_param(self):
        count = 0
        for param in self.cnn.parameters():
            count += 1

            param.requires_grad = False
        for param in self.cnn.logits.parameters():
            count += 1

            param.requires_grad = True
