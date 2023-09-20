import torch
import torch.nn as nn

from base.base_model import BaseModel
from utils.model_utils import select_backbone


class SubUNet(BaseModel):
    def __init__(self, config, args, N_classes=1232):
        super(SubUNet, self).__init__()

        self.n_classes = N_classes

        model_config = config['model']

        self.mode = config.model.mode
        self.sequence_module = model_config.rnn.type  # ['rnn']['type']
        self.num_layers = model_config.rnn.num_layers  # ['rnn']['num_layers']
        self.hidden_size = model_config.rnn.hidden_size  # ['rnn']['hidden_size']
        self.rnn_dropout = model_config.rnn.dropout  # ['rnn']['dropout']
        self.bidirectional = model_config.rnn.bidirectional  # ['rnn']['bidirectional']
        self.cnn, self.dim_feats = select_backbone(config.model.backbone.cnn)  # ['backbone'])

        self.dropout = True
        self.return_forward = False
        self.dp = nn.Dropout(p=0.2)

        self.rnn = nn.LSTM(
            input_size=self.dim_feats,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.rnn_dropout,
            bidirectional=self.bidirectional)

        if self.bidirectional:
            self.last_linear = nn.Linear(2 * self.hidden_size, self.n_classes)
            self.isolated_fc = nn.Linear(2 * self.hidden_size, self.n_classes)
        else:
            self.last_linear = nn.Linear(self.hidden_size, self.n_classes)
            self.isolated_fc = nn.Linear(self.hidden_size, self.n_classes)

        if (self.mode == 'continuous'):
            print('Run end-2-end freeze cnn params ')
            self.init_param()

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()

        if self.mode == 'continuous':
            with torch.no_grad():
                c_in = x.view(batch_size * timesteps, C, H, W)
                c_out = self.cnn(c_in)
            rnn_out, _ = self.rnn(c_out.view(timesteps, batch_size, -1))
            logits = self.last_linear(rnn_out)

            return logits, rnn_out

        elif self.mode == 'isolated':
            c_in = x.view(batch_size * timesteps, C, H, W)

            c_out = self.cnn(c_in)
            r_in = c_out.view(timesteps, batch_size, -1)
            r_out, _ = self.rnn(r_in)
            r_out2 = self.last_linear(r_out[-1, :, :])

            return r_out2

    def init_param(self):
        count = 0
        for param in self.cnn.parameters():
            count += 1

            param.requires_grad = False
