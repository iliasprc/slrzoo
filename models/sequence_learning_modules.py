import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class NRelU(nn.Module):
    def __init__(self, epsilon=0.00001):
        super(NRelU, self).__init__()
        self.epsilon = epsilon
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(x)
        out = out / (out.max() + self.epsilon)
        return out

        # layer_name = 'bn_b3_g%d_tc%d' % (group_num, layer_num)
        # layer._name = layer_name
        # setattr(self, layer_name, layer)


class DTRM(nn.Module):
    def __init__(self, in_channels, pool_kernel=2):
        super(DTRM, self).__init__()

        self.deformable_conv = nn.Conv1d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=5,
                                         dilation=1,
                                         padding=2)
        self.nrelu = NRelU()
        self.residual_pool = nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_kernel, return_indices=True)
        self.conv1x1 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1)
        # self.nrelu = nn.Re
        self.temporal_unpool = nn.MaxUnpool1d(kernel_size=pool_kernel, stride=pool_kernel, padding=0)
        # self.unpool = nn.Upsample(scale_factor=pool_kernel)

    def forward(self, res_stream, pool_stream):
        # print(len(stream))
        # #print(stream.size())
        # res_stream = stream[0]
        # pool_stream = stream[1]
        ## pool residual stream and concat with pooling stream
        # print(pool_stream.size())
        # print(res_stream.type())
        res_pooled, indices = self.residual_pool(res_stream)
        # print("pooled residual stream size {} poolstream {} indices{} ".format(res_pooled.size(),pool_stream.size(),indices.size()))

        df_input = torch.cat([res_pooled, pool_stream], dim=1)

        ## deformable convolution
        # print("df_input size {} ".format(df_input.size()))
        f1_T_2n = self.nrelu(self.deformable_conv(df_input))
        # print("f1_T_2n size {} ".format(f1_T_2n.size()))
        c11 = self.conv1x1(f1_T_2n)
        # print("conv1x1 size {} ".format(c11.size()))
        for_sum_fusion = self.temporal_unpool(c11, indices)
        # print("upsample size {} ".format(for_sum_fusion.size()))

        if (res_stream.size() != for_sum_fusion.size()):
            print(res_stream[:, :, 0:-1].size(), for_sum_fusion.size())
            f1_T = torch.cat([res_stream[:, :, 0:-1] + for_sum_fusion, res_stream[:, :, -1].unsqueeze(-1)], dim=-1)
        else:
            f1_T = res_stream + for_sum_fusion

        # print(f1_T.size())

        return f1_T, f1_T_2n





class TRN(nn.Module):
    def __init__(self, in_channels):
        super(TRN, self).__init__()
        self.bottom_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.temporal_pool1 = nn.MaxPool1d(kernel_size=2, stride=2, return_indices=True)
        self.bottom_dtrm = DTRM(in_channels)
        self.temporal_pool2 = nn.MaxPool1d(kernel_size=2, stride=2, return_indices=True)
        self.middle_dtrm = DTRM(in_channels, pool_kernel=4)
        self.unpool1 = nn.MaxUnpool1d(kernel_size=2, stride=2)
        self.top_dtrm = DTRM(in_channels)
        self.unpool2 = nn.MaxUnpool1d(kernel_size=2, stride=2)
        self.top_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        out_conv1 = self.bottom_conv(x)
        ## conv1 1 residual connection and one after pool as temporal connection
        pooled_temporal1, indices1 = self.temporal_pool1(out_conv1)
        # print(out_conv1.size())
        f1_T, f1_T_2n = self.bottom_dtrm(out_conv1, pooled_temporal1)
        print("DTRM1 out {} {}".format(f1_T.size(), out_conv1.size()))
        # 1st residual
        residual_stream = out_conv1 + f1_T
        pooled_temporal2, indices2 = self.temporal_pool2(f1_T_2n)
        print("DTRM2 inputs {} {}".format(residual_stream.size(), pooled_temporal2.size()))
        f2_T, f2_T_2n = self.middle_dtrm(residual_stream, pooled_temporal2)
        # print("DTRM2 out {} {}".format(f2_T.size(), f2_T_2n.size()))
        # print(pooled_temporal2.size())
        residual_stream = residual_stream + f2_T

        ### 1st unpool
        # print("sizes before 1st unpool {} {}".format(residual_stream.size(),f2_T_2n.size()))
        unpooled_stream1 = self.unpool1(f2_T_2n, indices2)

        f3_T, f3_T_2n = self.top_dtrm(residual_stream, unpooled_stream1)
        # print("DTRM2 out {} {}".format(f3_T.size(), f3_T_2n.size()))
        residual_stream = residual_stream + f3_T
        unpooled_stream2 = self.unpool1(f3_T_2n, indices1)
        # print("sizes after last unpool {} {}".format(residual_stream.size(), unpooled_stream2.size()))

        last_out = self.top_conv(unpooled_stream2 + residual_stream)

        return last_out


class ED_TCN(nn.Module):
    def __init__(self, input_size, num_filters):
        super(ED_TCN, self).__init__()
        self.encoder = torch.nn.Sequential(
            nn.Conv1d(input_size, num_filters // 2, kernel_size=5, stride=1, padding=2),
            NRelU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(num_filters // 2, num_filters, kernel_size=5, stride=1, padding=2),
            NRelU(),
            nn.MaxPool1d(2, 2))

        self.decoder = torch.nn.Sequential(
            nn.Conv1d(num_filters, num_filters, kernel_size=5, stride=1, padding=2),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(num_filters, num_filters, kernel_size=5, stride=1, padding=2),
            nn.Upsample(scale_factor=2))

    def forward(self, x):
        encoder_out = self.encoder(x)
        # print("ENCODER out {}".format(encoder_out.size()))
        decoder_out = self.decoder(encoder_out)

        return decoder_out


from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.PReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CausalConv1d(nn.Module):
    """
    A causal 1D convolution.
    """

    def __init__(self, kernel_size, in_channels, out_channels, dilation):
        super(CausalConv1d, self).__init__()

        # attributes:
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.dilation = dilation

        # modules:
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels,
                                      kernel_size, stride=1,
                                      padding=(kernel_size - 1),
                                      dilation=dilation)

    def forward(self, seq):
        """
        Note that Conv1d expects (batch, in_channels, in_length).
        We assume that seq ~ (len(seq), batch, in_channels), so we'll reshape it first.
        """

        conv1d_out = self.conv1d(seq).permute(2, 0, 1)
        print(conv1d_out.shape, conv1d_out[0:-(self.kernel_size - 1)].shape)
        # remove k-1 values from the end:
        return conv1d_out[0:-(self.kernel_size - 1)]


class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList(
            [copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)) for s in
             range(num_stages - 1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]

# m = MultiStageModel(num_stages=4,num_layers=2,num_f_maps=512,dim=512,num_classes=1000)
# m = CausalConv1d(3,512,512,2)
# print(m)
# from torchsummary import summary
#
#
# summary(m,(512,10),device='cpu')
