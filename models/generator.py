import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel
from utils import select_backbone, select_sequence_module
from utils.model_utils import init_weights_rnn





class SLR_Context_from_D_Generator(BaseModel):
    def __init__(self, args, N_classes=1232, mode='continuous'):
        super(SLR_Context_from_D_Generator, self).__init__()

        self.hidden_size = args.hidden_size
        self.num_layers = args.n_layers
        self.n_classes = N_classes
        self.mode = mode
        self.sequence_module = args.rnn
        self.rnn_dropout = args.dropout
        self.bidirectional = args.bidirectional
        print('bi ', self.bidirectional)
        if (self.mode == 'continuous'):
            print('RUN FULL MODEL END-TO-END')
            # for end-to-end

            self.padding = args.padding1d

        else:
            # for feature extractor
            self.padding = 0
        self.use_context = args.return_context
        self.temp_channels = 1024
        self.tc_kernel_size = args.kernel1d
        self.tc_pool_size = args.pool1d
        self.dropout = True
        self.return_forward = False
        self.dp = nn.Dropout(p=0.2)
        self.dim_feats = 1024
        self.cnn = None
        self.temporal = torch.nn.Sequential(
            nn.Conv1d(self.dim_feats, self.temp_channels, kernel_size=self.tc_kernel_size, stride=1,
                      padding=self.padding),
            nn.ReLU(),
            nn.MaxPool1d(self.tc_pool_size, self.tc_pool_size),
            nn.Conv1d(self.temp_channels, self.temp_channels, kernel_size=self.tc_kernel_size, stride=1,
                      padding=self.padding),
            nn.ReLU(),
            nn.MaxPool1d(self.tc_pool_size, self.tc_pool_size))

        self.last_linear, self.rnn, self.hidden, self.cell = select_sequence_module(self.sequence_module,
                                                                                    self.temp_channels,
                                                                                    self.hidden_size, self.num_layers,
                                                                                    self.rnn_dropout,
                                                                                    self.bidirectional,
                                                                                    self.n_classes)

        self.mapper = nn.LSTM(input_size=self.n_classes, hidden_size=self.hidden_size, num_layers=self.num_layers,
                              bidirectional=True, batch_first=True)
        init_weights_rnn(self.mapper)
        # self.mapper = nn.Linear(args.filter_size, self.hidden_size)
        # torch.nn.init.xavier_uniform_(self.mapper.weight)
        # self.mapper.bias.data.fill_(0.01)
        self.isolated_fc = nn.Linear(self.temp_channels, self.n_classes)
        if (self.mode == 'continuous'):
            print('Run end-2-end continuous')

    def forward(self, x, h=None, c=None):
        ## select continous or isolated
        if (self.mode == 'continuous'):
            return self.continuous_forward(x, h)

        return None

    def continuous_forward(self, x, prev_out=None):

        batch_size, timesteps, C = x.size()
        temp_input = x.permute(0, 2, 1)

        temp = self.temporal(temp_input)

        # temporal layers output size batch_size x dim_feats x timesteps
        # rnn input must be timesteps x batch_size x dim_feats

        if (self.sequence_module == 'rnn'):
            rnn_input = temp.permute(2, 0, 1)
            if self.use_context:
                if prev_out is not None:
                    # print(prev_out.shape)
                    _, (h1, c1) = self.mapper(prev_out.unsqueeze(0).detach())
                    # c1 = self.mapper(c)
                    # print(f'mapper h {h} c {c}')
                    rnn_out1, (hidden, cell) = self.rnn(rnn_input, (h1, c1))
                    self.hidden = hidden.detach()
                    self.cell = cell.detach()
                else:
                    rnn_out1, (hidden, cell) = self.rnn(rnn_input)  # , (self.hidden, self.cell))
                    self.hidden = hidden.detach()
                    self.cell = cell.detach()
            else:
                rnn_out1, (hidden, cell) = self.rnn(rnn_input)  # , (self.hidden, self.cell))
                self.hidden = hidden.detach()
                self.cell = cell.detach()

            r_out = self.last_linear(rnn_out1)

            return r_out
        elif (self.sequence_module == 'tcn' or self.sequence_module == 'edtcn'):
            tcn_out = self.tcn(temp).permute(2, 0, 1)
            r_out = self.last_linear(tcn_out)
            return r_out, None
        elif (self.sequence_module == 'ti'):
            ti_out = self.ti(temp.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1).permute(2, 0, 1)
            r_out = self.last_linear(ti_out)

            return r_out, None


class SLR_Feature_Generator_context(BaseModel):
    def __init__(self, args, N_classes=1232, mode='continuous'):
        super(SLR_Feature_Generator_context, self).__init__()

        self.hidden_size = args.hidden_size
        self.num_layers = args.n_layers
        self.n_classes = N_classes
        self.mode = mode
        self.sequence_module = args.rnn
        self.rnn_dropout = args.dropout
        self.bidirectional = args.bidirectional
        self.cnn, self.dim_feats = select_backbone(args.cnn)
        print('bi ', self.bidirectional)
        if (self.mode == 'continuous'):
            print('RUN FULL MODEL END-TO-END')
            # for end-to-end

            self.padding = args.padding1d

        else:
            # for feature extractor
            self.padding = 0
        self.use_context = args.return_context
        self.temp_channels = 1024
        self.tc_kernel_size = args.kernel1d
        self.tc_pool_size = args.pool1d
        self.dropout = True
        self.return_forward = False
        self.dp = nn.Dropout(p=0.2)
        self.dim_feats = 1024

        self.temporal = torch.nn.Sequential(
            nn.Conv1d(self.dim_feats, self.temp_channels, kernel_size=self.tc_kernel_size, stride=1,
                      padding=self.padding),
            nn.ReLU(),
            nn.MaxPool1d(self.tc_pool_size, self.tc_pool_size),
            nn.Conv1d(self.temp_channels, self.temp_channels, kernel_size=self.tc_kernel_size, stride=1,
                      padding=self.padding),
            nn.ReLU(),
            nn.MaxPool1d(self.tc_pool_size, self.tc_pool_size))

        self.last_linear, self.rnn, self.hidden, self.cell = select_sequence_module(self.sequence_module,
                                                                                    self.temp_channels,
                                                                                    self.hidden_size, self.num_layers,
                                                                                    self.rnn_dropout,
                                                                                    self.bidirectional,
                                                                                    self.n_classes)
        # self.mapper = NonLinearLayer(args.filter_size, self.hidden_size)
        # self.mapper = nn.LSTM(input_size=self.n_classes, hidden_size=self.hidden_size, num_layers=self.num_layers,
        #                       bidirectional=True, batch_first=True)
        # init_weights_rnn(self.mapper)
        self.mapper = nn.Linear(args.filter_size, self.hidden_size)
        # torch.nn.init.xavier_uniform_(self.mapper.weight)
        # self.mapper.bias.data.fill_(0.01)
        self.isolated_fc = nn.Linear(self.temp_channels, self.n_classes)
        if (self.mode == 'continuous'):
            print('Run end-2-end continuous')

    def forward(self, x, h=None, c=None):
        ## select continous or isolated
        if (self.mode == 'continuous'):
            return self.continuous_forward(x, h,c)

        return None

    def continuous_forward(self, x, h=None,c=None):

        batch_size, timesteps, C = x.size()
        temp_input = x.permute(0, 2, 1)

        temp = self.temporal(temp_input)

        # temporal layers output size batch_size x dim_feats x timesteps
        # rnn input must be timesteps x batch_size x dim_feats

        if (self.sequence_module == 'rnn'):
            rnn_input = temp.permute(2, 0, 1)
            if self.use_context:
                if h is not None:
                    # print(prev_out.shape)

                    c1 = self.mapper(c.detach())
                    h1 = self.mapper(h.detach())
                    # print(f'mapper h {h} c {c}')
                    rnn_out1, (hidden, cell) = self.rnn(rnn_input, (h1, c1))
                    self.hidden = hidden.detach()
                    self.cell = cell.detach()
                else:
                    rnn_out1, (hidden, cell) = self.rnn(rnn_input)  # , (self.hidden, self.cell))
                    self.hidden = hidden.detach()
                    self.cell = cell.detach()
            else:
                rnn_out1, (hidden, cell) = self.rnn(rnn_input)  # , (self.hidden, self.cell))
                self.hidden = hidden.detach()
                self.cell = cell.detach()

            r_out = self.last_linear(rnn_out1)

            return r_out




