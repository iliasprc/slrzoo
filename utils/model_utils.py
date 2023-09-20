import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau


def showgradients(model):
    for name, param in model.named_parameters():
        print(name, ' ', type(param.data), param.size())
        print("GRADS= \n", param.grad)


def save_checkpoint_cslr(model, optimizer, epoch, wer, checkpoint, name, save_seperate_layers=False, is_best=False):
    state = {}
    if (save_seperate_layers):
        for name1, module in model.named_children():
            # print(name1)
            state[name1 + '_dict'] = module.state_dict()

    state['model_dict'] = model.state_dict()
    state['optimizer_dict'] = optimizer.state_dict()
    state['wer'] = str(wer)
    state['epoch'] = str(epoch)
    filepath = os.path.join(checkpoint, name + '.pth')

    if is_best:
        filepath = os.path.join(checkpoint, 'best' + name + f'.pth')

    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)

    torch.save(state, filepath)
    # os.rename(filepath,os.path.join(checkpoint,'best'+ name + f'_WER_{str(wer)}.pth'))


def save_checkpoint_islr(model, optimizer, epoch, acc1, checkpoint, name, save_seperate_layers=False, is_best=False):
    state = {}
    if (save_seperate_layers):
        for name1, module in model.named_children():
            # print(name1)
            state[name1 + '_dict'] = module.state_dict()

    state['model_dict'] = model.state_dict()
    state['optimizer_dict'] = optimizer.state_dict()
    state['acc1'] = str(acc1)
    state['epoch'] = str(epoch)
    filepath = os.path.join(checkpoint, name + '.pth')

    if is_best:
        filepath = os.path.join(checkpoint, 'best' + name + f'.pth')

    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)

    torch.save(state, filepath)


def load_checkpoint(checkpoint, model, strict=True, optimizer=None, load_seperate_layers=False):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint, map_location='cpu')

    if (not load_seperate_layers):

        model.load_state_dict(checkpoint['model_dict'], strict=strict)
    else:
        print("load cnn dict")

        model.cnn.load_state_dict(checkpoint['cnn_dict'], strict=strict)

        print("load rnn dict")
        model.rnn.load_state_dict(checkpoint['rnn_dict'], strict=strict)
        # print("load temporal dict")
        model.temporal.load_state_dict(checkpoint['temporal_dict'], strict=strict)
        print("load fc dict")
        # model.last_linear.load_state_dict(checkpoint['last_linear_dict'], strict=strict)

    epoch = 0
    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer_dict'])

    return checkpoint, epoch


def load_checkpoint_modules(checkpoint, model, strict=True, optimizer=None, load_seperate_layers=False):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint, map_location='cpu')
    print(checkpoint.keys())
    if (not load_seperate_layers):

        model.load_state_dict(checkpoint['model_dict'], strict=strict)
    else:
        for name1, module in model.named_children():
            # print(name1)
            module.load_state_dict(checkpoint[name1 + '_dict'], strict=strict)

    epoch = 0
    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer_dict'])

    return checkpoint, epoch


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        print(classname)
        m.eval()
    return m


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def init_weights_linear(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    return m


def init_weights_rnn(model):
    for m in model.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
    return model


def weights_init_uniform(net):
    for name, param in net.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            nn.init.uniform_(param, a=-0.1, b=0.1)
    return net


def select_optimizer(model, config, checkpoint=None):
    opt = config['optimizer']['type']
    lr = config['optimizer']['lr']
    if (opt == 'Adam'):
        print(" use optimizer Adam lr ", lr)
        optimizer = optim.Adam(model.parameters(), lr=float(config['optimizer']['lr']), weight_decay=0.00001)
    elif (opt == 'SGD'):
        print(" use optimizer SGD lr ", lr)
        optimizer = optim.SGD(model.parameters(), lr=float(config['optimizer']['lr']), momentum=0.9)
    elif (opt == 'RMSprop'):
        print(" use RMS  lr", lr)
        optimizer = optim.RMSprop(model.parameters(), lr=float(config['optimizer']['lr']))
    if (checkpoint):
        optimizer.load_state_dict(checkpoint['optimizer_dict'])
    if config['scheduler']['type'] == 'ReduceLRonPlateau':
        scheduler = ReduceLROnPlateau(optimizer, factor=config['scheduler']['scheduler_factor'],
                                      patience=config['scheduler']['scheduler_patience'],
                                      min_lr=config['scheduler']['scheduler_min_lr'],
                                      verbose=config['scheduler']['scheduler_verbose'])
        return optimizer, scheduler

    return optimizer, None


def select_backbone(backbone):
    if (backbone == 'alexnet'):
        aux_logits = False
        cnn = torchvision.models.alexnet(pretrained=False)

        dim_feats = 4096

        cnn.classifier[-1] = torch.nn.Identity()

    elif (backbone == 'googlenet'):

        aux_logits = False
        from torchvision.models import googlenet

        cnn = googlenet(pretrained=True, transform_input=False, aux_logits=aux_logits)
        cnn.fc = torch.nn.Identity()
        dim_feats = 1024
        count = 0
        return cnn, dim_feats
    elif (backbone == 'resnet18'):
        aux_logits = False
        from torchvision.models import resnet18

        cnn = resnet18(pretrained=True)
        cnn.fc = torch.nn.Identity()
        dim_feats = 512
        count = 0
    elif (backbone == 'seres'):
        aux_logits = False
        from torchvision.models import resnext50_32x4d

        cnn = resnext50_32x4d(pretrained=True)
        cnn.fc = torch.nn.Identity()
        cnn.classifier = torch.nn.Identity()
        dim_feats = 2048
        count = 0
    elif (backbone == 'mobilenet'):
        aux_logits = False
        from torchvision.models import mobilenet_v2

        cnn = mobilenet_v2(pretrained=True)
        cnn.fc = torch.nn.Identity()
        cnn.classifier = torch.nn.Identity()
        dim_feats = 1280
        count = 0
    elif (backbone == 'shufflenet'):
        aux_logits = False
        from torchvision.models import shufflenet_v2_x1_0

        cnn = shufflenet_v2_x1_0(True)
        cnn.fc = torch.nn.Identity()
        cnn.classifier = torch.nn.Identity()
        dim_feats = 1024
        count = 0
    return cnn, dim_feats


def select_sequence_module(sequence_module, temp_channels, hidden_size, num_layers, rnn_dropout, bidirectional,
                           n_classes):
    if (sequence_module == 'lstm'):
        rnn = nn.LSTM(
            input_size=temp_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=rnn_dropout,
            bidirectional=bidirectional)

        if (bidirectional):
            last_linear = nn.Linear(2 * hidden_size, n_classes)
            hidden = torch.zeros(2 * num_layers, 1, hidden_size).cuda()
            cell = torch.zeros(2 * num_layers, 1, hidden_size).cuda()
        else:
            last_linear = nn.Linear(hidden_size, n_classes)
            hidden = torch.zeros(num_layers, 1, hidden_size).cuda()
            cell = torch.zeros(num_layers, 1, hidden_size).cuda()
            rnn = init_weights_rnn(rnn)
            last_linear = init_weights_linear(last_linear)
    elif (sequence_module == 'gru'):
        rnn = nn.GRU(
            input_size=temp_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=rnn_dropout,
            bidirectional=bidirectional)

        if (bidirectional):
            last_linear = nn.Linear(2 * hidden_size, n_classes)
            hidden = torch.zeros(2 * num_layers, 1, hidden_size).cuda()
            cell = torch.zeros(2 * num_layers, 1, hidden_size).cuda()
        else:
            hidden = torch.zeros(num_layers, 1, hidden_size).cuda()
            cell = torch.zeros(num_layers, 1, hidden_size).cuda()
            last_linear = nn.Linear(hidden_size, n_classes)
            rnn = init_weights_rnn(rnn)
            last_linear = init_weights_linear(last_linear)
    else:
        encoder_layer = nn.TransformerEncoderLayer(1024, nhead=8)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        return None, transformer_encoder, None, None
    return last_linear, rnn, hidden, cell
