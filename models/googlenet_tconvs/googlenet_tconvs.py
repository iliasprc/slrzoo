from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from base.base_model import BaseModel
from utils import Loss
from utils.model_utils import select_backbone, select_sequence_module


class GoogLeNet_TConvs(BaseModel):
    def __init__(self, config, args, N_classes=1232, ):
        super(GoogLeNet_TConvs, self).__init__()

        print(config)

        self.n_classes = N_classes

        model_config = config['model']

        self.mode = config.model.mode
        self.sequence_module = model_config.rnn.type  # ['rnn']['type']
        self.num_layers = model_config.rnn.num_layers  # ['rnn']['num_layers']
        self.hidden_size = model_config.rnn.hidden_size  # ['rnn']['hidden_size']
        self.rnn_dropout = model_config.rnn.dropout  # ['rnn']['dropout']
        self.bidirectional = model_config.rnn.bidirectional  # ['rnn']['bidirectional']
        self.cnn, self.dim_feats = select_backbone(config.model.backbone.cnn)  # ['backbone'])

        temp_channels = model_config.backbone.temporal.filters  # ['temporal']['filters']
        tc_kernel_size = model_config.backbone.temporal.kernel_size  # ['temporal']['kernel_size']
        tc_stride = model_config.backbone.temporal.stride  # ['temporal']['stride']
        tc_pool_size = model_config.backbone.temporal.pool_size  # ['temporal']['pool_size']
        tc_padding = model_config.backbone.temporal.padding  # ['temporal']['padding']
        self.dropout = True
        self.return_forward = False
        self.dp = nn.Dropout(p=0.2)

        # self.use_context = args.return_context
        self.temporal = torch.nn.Sequential(
            nn.Conv1d(self.dim_feats, temp_channels, kernel_size=tc_kernel_size, stride=tc_stride,
                      padding=tc_padding),
            nn.ReLU(),
            nn.MaxPool1d(tc_pool_size, tc_pool_size),
            nn.Conv1d(temp_channels, temp_channels, kernel_size=tc_kernel_size, stride=tc_stride,
                      padding=tc_padding),
            nn.ReLU(),
            nn.MaxPool1d(tc_pool_size, tc_pool_size))

        self.last_linear, self.rnn, self.hidden, self.cell = select_sequence_module(self.sequence_module,
                                                                                    temp_channels,
                                                                                    self.hidden_size, self.num_layers,
                                                                                    self.rnn_dropout,
                                                                                    self.bidirectional,
                                                                                    self.n_classes)
        self.isolated_fc = nn.Linear(temp_channels, self.n_classes)
        if (self.mode == 'continuous'):
            print('Run end-2-end ')
            self.init_param()
            self.criterion = Loss(args.ctc, average=True)

    def forward(self, x):
        ## select continous or isolated
        if (self.mode == 'continuous'):
            return self.continuous_forward(x)
        elif (self.mode == 'isolated'):

            return self.isolated_forward(x)

        return None

    def training_step(self, train_batch, batch_idx=0):
        x, y = train_batch
        if (self.mode == 'continuous'):
            y_hat = self.continuous_forward(x)
            loss = self.criterion(y_hat, y)
            return y_hat, loss
        elif (self.mode == 'isolated'):

            y_hat = self.isolated_forward(x)
            loss = F.cross_entropy(y_hat, y)
            return y_hat, loss

    def validation_step(self, train_batch, batch_idx=0):
        x, y = train_batch
        if (self.mode == 'continuous'):
            y_hat = self.continuous_forward(x)
            loss = self.criterion(y_hat, y)
            return y_hat, loss
        elif (self.mode == 'isolated'):

            y_hat = self.isolated_forward(x)
            loss = F.cross_entropy(y_hat, y)
            return y_hat, loss

    def continuous_forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_outputs = self.cnn(c_in)

        c_out = c_outputs.contiguous().view(batch_size, timesteps, -1)

        # c_out has size timesteps x dim feats
        # temporal layers gets input size batch_size x dim_feats x timesteps

        temp_input = c_out.permute(0, 2, 1)
        temp = self.temporal(temp_input)

        # temporal layers output size batch_size x dim_feats x timesteps
        # rnn input must be timesteps x batch_size x dim_feats
        rnn_input = temp.permute(2, 0, 1)

        # rnn_out = self.transformer(rnn_input)
        rnn_out1, hidden = self.rnn(rnn_input)

        r_out = self.last_linear(rnn_out1)

        return r_out
        # if (self.sequence_module == 'rnn' or self.sequence_module == 'indrnn'):

    def isolated_forward(self, x):

        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_outputs = self.cnn(c_in)

        c_out = c_outputs.contiguous().view(batch_size, timesteps, -1)

        # train only feature extractor
        # c_out has size batch_size x timesteps x dim feats
        # temporal layers gets input size batch_size x dim_feats x timesteps
        temp_input = c_out.permute(0, 2, 1)
        temp = self.dp(self.temporal(temp_input))
        # print('temp out ', temp.size())
        # last linear input must be timesteps x batch_size x dim_feats

        fc_input = temp.permute(2, 0, 1)

        r_out = self.isolated_fc(fc_input)
        # print(" out ,",r_out.size())
        if (r_out.size(0) > 1):
            ## MS ASL
            r_out = torch.mean(r_out, dim=0)
            # print(fc_input.size())
        else:
            r_out = r_out.squeeze(0)
        return r_out
        # print(r_out.size())

        return F.log_softmax(r_out, dim=-1)

    def init_param(self):
        count = 0
        for param in self.cnn.parameters():
            count += 1

            param.requires_grad = False


class GoogLeNet_TConvsAtt(BaseModel):
    def __init__(self, args, N_classes=1232, mode='isolated'):
        super(GoogLeNet_TConvsAtt, self).__init__()
        config = OmegaConf.load(os.path.join(args.cwd, 'models/googlenet_tconvs/model.yml'))['model']
        print(config)
        print(config.model)

        self.n_classes = N_classes

        dataset_config = config['datasets'][args.dataset]
        print(dataset_config)
        self.mode = dataset_config['mode']
        self.sequence_module = dataset_config.rnn.type  # ['rnn']['type']
        self.num_layers = dataset_config.rnn.num_layers  # ['rnn']['num_layers']
        self.hidden_size = dataset_config.rnn.hidden_size  # ['rnn']['hidden_size']
        self.rnn_dropout = dataset_config.rnn.dropout  # ['rnn']['dropout']
        self.bidirectional = dataset_config.rnn.bidirectional  # ['rnn']['bidirectional']
        self.cnn, self.dim_feats = select_backbone(config.backbone)  # ['backbone'])
        self.cnn_att = nn.MultiheadAttention(self.dim_feats, 8)
        self.temp_channels = dataset_config.temporal.filters  # ['temporal']['filters']
        self.tc_kernel_size = dataset_config.temporal.kernel_size  # ['temporal']['kernel_size']
        self.tc_stride = dataset_config.temporal.stride  # ['temporal']['stride']
        self.tc_pool_size = dataset_config.temporal.pool_size  # ['temporal']['pool_size']
        self.tc_padding = dataset_config.temporal.padding  # ['temporal']['padding']
        self.dropout = True
        self.return_forward = False
        self.dp = nn.Dropout(p=0.2)
        self.dim_feats = 1024
        # self.use_context = args.return_context
        self.temporal = torch.nn.Sequential(
            nn.Conv1d(self.dim_feats, self.temp_channels, kernel_size=self.tc_kernel_size, stride=self.tc_stride,
                      padding=self.tc_padding),
            nn.ReLU(),
            nn.MaxPool1d(self.tc_pool_size, self.tc_pool_size),
            nn.Conv1d(self.temp_channels, self.temp_channels, kernel_size=self.tc_kernel_size, stride=self.tc_stride,
                      padding=self.tc_padding),
            nn.ReLU(),
            nn.MaxPool1d(self.tc_pool_size, self.tc_pool_size))
        self.tcl_att = nn.MultiheadAttention(self.dim_feats, 8)
        self.last_linear, self.rnn, self.hidden, self.cell = select_sequence_module(self.sequence_module,
                                                                                    self.temp_channels,
                                                                                    self.hidden_size, self.num_layers,
                                                                                    self.rnn_dropout,
                                                                                    self.bidirectional,
                                                                                    self.n_classes)
        self.isolated_fc = nn.Linear(self.temp_channels, self.n_classes)
        self.init_param()
        if (self.mode == 'continuous'):
            print('Run end-2-end ')
            self.init_param()
            self.criterion = Loss(args.ctc, average=True)

    def forward(self, x):
        ## select continous or isolated
        if (self.mode == 'continuous'):
            return self.continuous_forward(x)
        elif (self.mode == 'isolated'):

            return self.isolated_forward(x)

        return None

    def training_step(self, train_batch, batch_idx=0):
        x, y = train_batch
        if (self.mode == 'continuous'):
            y_hat = self.continuous_forward(x)
            loss = self.criterion(y_hat, y)
            return y_hat, loss
        elif (self.mode == 'isolated'):

            y_hat = self.isolated_forward(x)
            loss = F.cross_entropy(y_hat, y)
            return y_hat, loss

    def validation_step(self, train_batch, batch_idx=0):
        x, y = train_batch
        if (self.mode == 'continuous'):
            y_hat = self.continuous_forward(x)
            loss = self.criterion(y_hat, y)
            return y_hat, loss
        elif (self.mode == 'isolated'):

            y_hat = self.isolated_forward(x)
            loss = F.cross_entropy(y_hat, y)
            return y_hat, loss

    def continuous_forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_outputs = self.cnn(c_in)

        c_out = c_outputs.contiguous().view(batch_size, timesteps, -1)

        # c_out has size timesteps x dim feats
        # temporal layers gets input size batch_size x dim_feats x timesteps

        temp_input = c_out.permute(0, 2, 1)
        temp = self.temporal(temp_input)

        # temporal layers output size batch_size x dim_feats x timesteps
        # rnn input must be timesteps x batch_size x dim_feats
        rnn_input = temp.permute(2, 0, 1)

        # rnn_out = self.transformer(rnn_input)
        rnn_out1, hidden = self.rnn(rnn_input)

        r_out = self.last_linear(rnn_out1)

        return r_out
        # if (self.sequence_module == 'rnn' or self.sequence_module == 'indrnn'):

    def isolated_forward(self, x):

        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_outputs = self.cnn(c_in)

        c_out = c_outputs.contiguous().view(timesteps, batch_size, -1)
        c_att, _ = self.cnn_att(c_out, c_out, c_out)
        c_out = c_out + c_att
        # train only feature extractor
        # c_out has size batch_size x timesteps x dim feats
        # temporal layers gets input size batch_size x dim_feats x timesteps
        temp_input = c_out.permute(1, 2, 0)

        temp = self.dp(self.temporal(temp_input))
        # print('temp out ', temp.size())
        # last linear input must be timesteps x batch_size x dim_feats

        fc_input = temp.permute(2, 0, 1)

        tmp_att, _ = self.tcl_att(fc_input, fc_input, fc_input)
        # print(fc_input.shape)
        fc_input = fc_input + tmp_att
        r_out = self.isolated_fc(fc_input)
        # print(" out ,",r_out.size())
        if (r_out.size(0) > 1):
            ## MS ASL
            r_out = torch.mean(r_out, dim=0)
            # print(fc_input.size())
        else:
            r_out = r_out.squeeze(0)

        # print(r_out.size())

        return F.log_softmax(r_out, dim=-1)

    def init_param(self):
        count = 0
        for param in self.cnn.parameters():
            count += 1

            param.requires_grad = False


class GoogLeNet_TConvsAtt1(BaseModel):
    def __init__(self, args, N_classes=1232, mode='isolated'):
        super(GoogLeNet_TConvsAtt1, self).__init__()
        config = OmegaConf.load(os.path.join(args.cwd, 'models/googlenet_tconvs/model.yml'))['model']
        print(config)
        print(config.model)

        self.n_classes = N_classes

        dataset_config = config['datasets'][args.dataset]
        print(dataset_config)
        self.mode = dataset_config['mode']
        self.sequence_module = dataset_config.rnn.type  # ['rnn']['type']
        self.num_layers = dataset_config.rnn.num_layers  # ['rnn']['num_layers']
        self.hidden_size = dataset_config.rnn.hidden_size  # ['rnn']['hidden_size']
        self.rnn_dropout = dataset_config.rnn.dropout  # ['rnn']['dropout']
        self.bidirectional = dataset_config.rnn.bidirectional  # ['rnn']['bidirectional']
        self.cnn, self.dim_feats = select_backbone(config.backbone)  # ['backbone'])
        self.cnn_att = nn.MultiheadAttention(self.dim_feats, 8)
        self.temp_channels = dataset_config.temporal.filters  # ['temporal']['filters']
        self.tc_kernel_size = dataset_config.temporal.kernel_size  # ['temporal']['kernel_size']
        self.tc_stride = dataset_config.temporal.stride  # ['temporal']['stride']
        self.tc_pool_size = dataset_config.temporal.pool_size  # ['temporal']['pool_size']
        self.tc_padding = dataset_config.temporal.padding  # ['temporal']['padding']
        self.dropout = True
        self.return_forward = False
        self.dp = nn.Dropout(p=0.2)
        self.dim_feats = 1024
        # self.use_context = args.return_context
        self.temporal = torch.nn.Sequential(
            nn.Conv1d(self.dim_feats, self.temp_channels, kernel_size=self.tc_kernel_size, stride=self.tc_stride,
                      padding=self.tc_padding),
            nn.ReLU(),
            nn.MaxPool1d(self.tc_pool_size, self.tc_pool_size),
            nn.Conv1d(self.temp_channels, self.temp_channels, kernel_size=self.tc_kernel_size, stride=self.tc_stride,
                      padding=self.tc_padding),
            nn.ReLU(),
            nn.MaxPool1d(self.tc_pool_size, self.tc_pool_size))
        self.tcl_att = nn.MultiheadAttention(self.dim_feats, 8)
        self.last_linear, self.rnn, self.hidden, self.cell = select_sequence_module(self.sequence_module,
                                                                                    self.temp_channels,
                                                                                    self.hidden_size, self.num_layers,
                                                                                    self.rnn_dropout,
                                                                                    self.bidirectional,
                                                                                    self.n_classes)
        self.isolated_fc = nn.Linear(self.temp_channels, self.n_classes)
        # self.init_param()
        if (self.mode == 'continuous'):
            print('Run end-2-end ')
            self.init_param()
            self.criterion = Loss(args.ctc, average=True)

    def forward(self, x):
        ## select continous or isolated
        if (self.mode == 'continuous'):
            return self.continuous_forward(x)
        elif (self.mode == 'isolated'):

            return self.isolated_forward(x)

        return None

    def training_step(self, train_batch, batch_idx=0):
        x, y = train_batch
        if (self.mode == 'continuous'):
            y_hat = self.continuous_forward(x)
            loss = self.criterion(y_hat, y)
            return y_hat, loss
        elif (self.mode == 'isolated'):

            y_hat = self.isolated_forward(x)
            loss = F.cross_entropy(y_hat, y)
            return y_hat, loss

    def validation_step(self, train_batch, batch_idx=0):
        x, y = train_batch
        if (self.mode == 'continuous'):
            y_hat = self.continuous_forward(x)
            loss = self.criterion(y_hat, y)
            return y_hat, loss
        elif (self.mode == 'isolated'):

            y_hat = self.isolated_forward(x)
            loss = F.cross_entropy(y_hat, y)
            return y_hat, loss

    def continuous_forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_outputs = self.cnn(c_in)

        c_out = c_outputs.contiguous().view(batch_size, timesteps, -1)

        # c_out has size timesteps x dim feats
        # temporal layers gets input size batch_size x dim_feats x timesteps

        temp_input = c_out.permute(0, 2, 1)
        temp = self.temporal(temp_input)

        # temporal layers output size batch_size x dim_feats x timesteps
        # rnn input must be timesteps x batch_size x dim_feats
        rnn_input = temp.permute(2, 0, 1)

        # rnn_out = self.transformer(rnn_input)
        rnn_out1, hidden = self.rnn(rnn_input)

        r_out = self.last_linear(rnn_out1)

        return r_out
        # if (self.sequence_module == 'rnn' or self.sequence_module == 'indrnn'):

    def isolated_forward(self, x):

        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_outputs = self.cnn(c_in)

        c_out = c_outputs.contiguous().view(timesteps, batch_size, -1)
        c_out, _ = self.cnn_att(c_out, c_out, c_out)
        # train only feature extractor
        # c_out has size batch_size x timesteps x dim feats
        # temporal layers gets input size batch_size x dim_feats x timesteps
        temp_input = c_out.permute(1, 2, 0)

        temp = self.dp(self.temporal(temp_input))
        # print('temp out ', temp.size())
        # last linear input must be timesteps x batch_size x dim_feats

        fc_input = temp.permute(2, 0, 1)

        # fc_input,_ = self.tcl_att(fc_input,fc_input,fc_input)
        # print(fc_input.shape)
        r_out = self.isolated_fc(fc_input)
        # print(" out ,",r_out.size())
        if (r_out.size(0) > 1):
            ## MS ASL
            r_out = torch.mean(r_out, dim=0)
            # print(fc_input.size())
        else:
            r_out = r_out.squeeze(0)

        # print(r_out.size())

        return r_out

    def init_param(self):
        count = 0
        for param in self.cnn.parameters():
            count += 1

            param.requires_grad = False


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)

        # attention= [batch size, src len]

        return F.softmax(attention, dim=1)
