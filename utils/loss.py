import numpy as np
import torch
import torch.nn as nn

from utils.ctc_losses import ctc_ent_cost


def calc_gradient_penalty(args, netD, real_data, fake_data, LAMBDA=1.0):
    use_cuda = torch.cuda.is_available()
    if (False):
        interpolates = real_data
    else:
        interpolates = fake_data

    if use_cuda:
        interpolates = interpolates.cuda()

    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    # TODO: Make ConvBackward diffentiable
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(
                                        disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                        disc_interpolates.size()),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


class Dist(nn.Module):
    def __init__(self, crit):
        super(Dist, self).__init__()
        self.distance = torch.nn.CosineSimilarity(dim=-1)
        self.crit = crit
        if (crit == 'mse'):
            # self.crit = nn.MSELoss(reduction='sum')
            self.distance = torch.nn.PairwiseDistance()

    def forward(self, x1, x2):
        if (self.crit == 'mse'):
            return self.distance(x1, x2)
        else:
            return 1.0 - self.distance(x1, x2)


class Loss(nn.Module):
    def __init__(self, crit, average=True, alpha=0.99, gamma=2.0, beta=0.1, return_ctc_cost=False):
        super(Loss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.crit = crit
        self.average = average
        self.return_ctc_cost = return_ctc_cost

        if (crit == 'normal'):
            self.loss = self.normal_ctc_loss

        elif (crit == 'ent_ctc'):

            self.loss = self.ent_ctc_loss

        elif (crit == 'focal'):
            self.loss = self.focal_ctc_loss

    def forward(self, output, target):

        cost = self.loss(output, target)

        return cost

    def normal_ctc_loss(self, log_probs, target):

        if (self.average):
            criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        else:
            criterion = nn.CTCLoss(blank=0, reduction='sum', zero_infinity=True)
        input_len = torch.tensor([log_probs.size(0)], dtype=torch.int)
        target_len = torch.tensor([target.size(1)], dtype=torch.int)
        loss = criterion(nn.functional.log_softmax(log_probs, dim=2), target, input_len, target_len)
        return loss

    def focal_ctc_loss(self, log_probs, target):

        if (self.average):
            criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        else:
            criterion = nn.CTCLoss(blank=0, reduction='sum', zero_infinity=True)
        input_len = torch.tensor([log_probs.size(0)], dtype=torch.int)
        target_len = torch.tensor([target.size(1)], dtype=torch.int)
        loss = criterion(nn.functional.log_softmax(log_probs, dim=2), target, input_len, target_len)
        p = torch.exp((-1) * loss)
        focal_loss = self.alpha * ((1 - p) ** self.gamma) * loss
        return focal_loss

    def ent_ctc_loss(self, log_probs, target):

        input_len = torch.tensor([log_probs.size(0)], dtype=torch.int)
        target_len = torch.tensor([target.size(1)], dtype=torch.int)

        H, cost = ctc_ent_cost(nn.functional.log_softmax(log_probs, dim=-1).cpu(), target.cpu(), input_len.cpu(),
                               target_len.cpu())
        check = (cost - self.beta * H)
        if (check.item() > 2000):
            cost = cost * 0.0
            H = H * 0.0

        if (self.average):
            if (self.return_ctc_cost):
                return (cost - self.beta * H) / target.size(1), cost / target.size(1)

            return (cost - self.beta * H) / target.size(1)
        else:
            return (cost - self.beta * H)


def dynamic_label_decode(probs, labels):
    probs = torch.log_softmax(probs, dim=-1).squeeze(1)
    labels = labels.squeeze(0)

    probs1, labels1 = probs.detach(), labels.detach()

    labelling, betas, q = dpd_decode.dpd(probs1.cpu().numpy().astype(np.float64),
                                         labels1.cpu().numpy().astype(np.int32), blank=0)

    return torch.tensor(labelling), torch.tensor(betas), torch.tensor(q)


class FocalLoss(nn.Module):

    def __init__(self, weight=None,
                 gamma=0., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.criterion = nn.NLLLoss(reduction=reduction)

    def forward(self, input_tensor, target_tensor):
        # log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(input_tensor)
        return self.criterion(
            ((1 - prob) ** self.gamma) * input_tensor,
            target_tensor,

        )
