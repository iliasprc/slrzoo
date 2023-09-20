# This Python file uses the following encoding: utf-8
# All rights reserved for the GIPAS Lab in USTC.
# It's shared by Hao Zhou for academic exchange and non-commercial usage.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.stmc.temporal_model import EncoderRNN


class VGGPose(nn.Module):
    def __init__(self, num_classes=-1, input_channels=3):
        super(VGGPose, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        # base model 
        # 64, M (Max-pool)
        self.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 128, M == 112, 112
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.poo2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 256, 256, M == 56, 56
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 512, 512, M == 28, 28
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 512, 512, M == 14, 14
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu8 = nn.ReLU(inplace=True)
        self.pool8 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 512 == 7, 7
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu9 = nn.ReLU(inplace=True)
        self.pool9 = nn.AvgPool2d(kernel_size=7, stride=7)

        # hand  224/4=56, 24, 24
        self.handnet = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 12, 12
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 6, 6
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=6, stride=6)
        )

        # face 224/4=56, 16, 16
        self.facenet = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 8, 8
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8, stride=8),
        )

        # pose 14, 14
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2,
                               padding=1, output_padding=0, bias=True),
            nn.ReLU(inplace=True),
            # 28, 28
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2,
                               padding=1, output_padding=0, bias=True),
            nn.ReLU(inplace=True),
            # 56, 56
            nn.Conv2d(256, 7, kernel_size=1, stride=1, padding=0)
        )

        expect_matrix = torch.linspace(0, 1, steps=56).squeeze(0).repeat(56, 1)
        self.h_expectation = nn.Parameter(expect_matrix.t(), requires_grad=False)
        self.w_expectation = nn.Parameter(expect_matrix, requires_grad=False)

        if self.num_classes != -1:
            self.fc_full = nn.Linear(512, self.num_classes)
            self.fc_hand = nn.Linear(512, self.num_classes)
            self.fc_face = nn.Linear(256, self.num_classes)
            self.fc_concat = nn.Linear(1024, self.num_classes)

    def forward(self, x, teacher_ratio=0.0, pose_given=None):
        # full 224
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        # 128, M  = 112
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.poo2(x)
        # 256, 256 = 56
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        middle_map = x
        x = self.pool4(x)
        # 512, 512 = 28
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.pool6(x)
        # 512, 512 = 14
        x = self.conv7(x)
        x = self.relu7(x)
        pose_map = x
        x = self.conv8(x)
        x = self.relu8(x)
        x = self.pool8(x)
        # 512 = 7
        x = self.conv9(x)
        x = self.relu9(x)
        x = self.pool9(x)

        # pose
        pose_map = self.deconv(pose_map)
        # pose_map = torch.zeros(, 7, 56, 56).cuda()
        # pose_map[0, 0, 10, 15] = 30
        # pose_map[0, 5, 2, 46] = 30
        # pose_map[0, 6, 7, 53] = 30
        # N, 7, 56, 56
        pose_map = F.softmax(pose_map.reshape(-1, 7, 56 * 56), dim=-1).reshape(-1, 7, 56, 56)
        pose_h = torch.mul(pose_map, self.h_expectation).sum(-1).sum(-1)
        pose_w = torch.mul(pose_map, self.w_expectation).sum(-1).sum(-1)
        pose_predict = torch.stack((pose_h, pose_w), dim=-1)
        if 1 < teacher_ratio:
            pose = pose_given
        else:
            pose = pose_predict
        # N, 7, 2
        pose_detach = (pose.detach().cpu() * 55).round().int()
        pose_detach[:, 0, :] = torch.clamp(pose_detach[:, 0, :], min=7, max=47)  # 7, 55-8
        pose_detach[:, 5, :] = torch.clamp(pose_detach[:, 5, :], min=11, max=43)  # 11, 55-12
        pose_detach[:, 6, :] = torch.clamp(pose_detach[:, 6, :], min=11, max=43)  # 11, 55-12
        # print(pose*55)
        # print(pose_detach.shape)

        # 56, 56  24, 24  16, 16
        hand_l = []
        hand_r = []
        face = []
        for i, mapi in enumerate(middle_map):
            # 256, 56, 56
            # face
            h, w = pose_detach[i, 0].tolist()
            face.append(mapi[:, h - 7:h + 9, w - 7:w + 9])
            # hand
            h, w = pose_detach[i, 5].tolist()
            hand_l.append(mapi[:, h - 11:h + 13, w - 11:w + 13])
            h, w = pose_detach[i, 6].tolist()
            hand_r.append(mapi[:, h - 11:h + 13, w - 11:w + 13])
        face = torch.stack(face, dim=0)
        hand_l = torch.stack(hand_l, dim=0)
        hand_r = torch.stack(hand_r, dim=0)

        # ! # codes used to visualize the result of pose #
        # from matplotlib import pyplot as plt
        # for i in range(10):
        #     a = middle_map[i].permute(1,2,0).mean(-1).detach().cpu().numpy()
        #     b = hand_l[i].permute(1,2,0).mean(-1).detach().cpu().numpy()
        #     c = hand_r[i].permute(1,2,0).mean(-1).detach().cpu().numpy()
        #     d = face[i].permute(1,2,0).mean(-1).detach().cpu().numpy()
        #     plt.figure(i)
        #     plt.subplot(1,4,1)
        #     plt.imshow(a)
        #     plt.subplot(1,4,2)
        #     plt.imshow(b)
        #     plt.subplot(1,4,3)
        #     plt.imshow(c)
        #     plt.subplot(1,4,4)
        #     plt.imshow(d)

        #     plt.colorbar()
        #     plt.axis('on')
        #     plt.savefig('demo/'+str(i)+'.png')
        #     plt.close(i)
        # exit()
        # !

        face = self.facenet(face)
        hand_l = self.handnet(hand_l)
        hand_r = self.handnet(hand_r)
        # print('face', face.shape)
        # print('hand_l', hand_l.shape)
        # print('hand_r', hand_r.shape)

        return pose_predict, x, hand_l, hand_r, face


class TMC_Conv1D(nn.Module):
    def __init__(self, in_cat_size, in_part_size, out_cat_size, out_part_size):
        super(TMC_Conv1D, self).__init__()
        self.cat_size = out_cat_size
        self.part_size = out_part_size
        self.part_index = []
        start = 0
        for size in in_part_size:
            self.part_index.append((start, start + size))
            start = start + size
        self.conv_cat = nn.Conv1d(in_cat_size, self.cat_size // 2, kernel_size=5, stride=1, padding=0)
        self.conv_parts = nn.ModuleList()
        for i, size in enumerate(self.part_size):
            self.conv_parts.append(nn.Conv1d(in_part_size[i], size, kernel_size=5, stride=1, padding=0))

        self.conv_merge_1 = nn.Sequential(
            nn.Conv1d(sum(self.part_size), self.cat_size // 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        # self.conv_merge_2 = nn.Sequential(
        #     nn.Conv1d(self.cat_size, self.cat_size//8, kernel_size=1, stride=1, padding=0),
        #     nn.Conv1d(self.cat_size//8, self.cat_size, kernel_size=1, stride=1, padding=0),
        # )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, cat, parts):
        '''
            input: (B, C, T)
        '''
        cat = self.conv_cat(cat)
        cat = self.relu(cat)
        parts_tmp = []
        for i, index in enumerate(self.part_index):
            parts_tmp.append(self.relu(self.conv_parts[i](parts[:, index[0]:index[1], :])))
        parts = torch.cat(parts_tmp, dim=1)
        cat = torch.cat((cat, self.conv_merge_1(parts)), dim=1)

        return cat, parts


class VGGPose_Conv1D(nn.Module):
    def __init__(self, num_classes=-1, input_channels=3, frame_level=False):
        super(VGGPose_Conv1D, self).__init__()
        self.vggpose = VGGPose(input_channels=input_channels)
        self.posenet = nn.Sequential(
            nn.Linear(14, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True)
        )
        self.part_index = []
        self.part_size = [256, 256, 256, 256]  # full, hand, face, pose
        start = 0
        for size in self.part_size:
            self.part_index.append((start, start + size))
            start = start + size
        self.conv1d_1 = TMC_Conv1D(1536, [512, 512, 256, 256], 1024, [256, 256, 256, 256])
        self.pool1 = nn.MaxPool1d(kernel_size=2, ceil_mode=False)
        self.conv1d_2 = TMC_Conv1D(1024, [256, 256, 256, 256], 1024, [256, 256, 256, 256])
        self.pool2 = nn.MaxPool1d(kernel_size=2, ceil_mode=False)
        if num_classes != -1:
            self.fc = nn.ModuleList([
                nn.Linear(self.part_size[0], num_classes),
                nn.Linear(self.part_size[1], num_classes),
                nn.Linear(self.part_size[2], num_classes),
                nn.Linear(self.part_size[3], num_classes),
                nn.Linear(sum(self.part_size), num_classes),
            ])
        self.relu = nn.ReLU(inplace=True)
        self.frame_level = frame_level
        self.dropout = nn.Dropout(0.4, inplace=True)
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, x, targets, pose_given=None, teacher_ratio=0.0):
        B, C, T, H, W = x.shape
        if pose_given is not None:
            pose_given = pose_given.reshape(B * T, 7, 2)
        # (B, C, T, H, W) 
        x = x.transpose(1, 2).reshape(B * T, C, H, W)
        # (B*T, C, H, W)
        pose, full, hand_l, hand_r, face = self.vggpose(x, pose_given=pose_given, teacher_ratio=teacher_ratio)
        pose = pose.reshape(B, T, 7, 2)
        # (B*T, C)
        full = full.reshape(B, T, -1).transpose(1, 2)
        hand = torch.cat((hand_l, hand_r), dim=-1).reshape(B, T, -1).transpose(1, 2)
        face = face.reshape(B, T, -1).transpose(1, 2)
        pose_feat = self.posenet(pose.detach().reshape(B, T, 14)).transpose(1, 2)

        full = self.dropout(full)
        cat = torch.cat((full, hand, face, pose_feat), dim=1)
        # (B, C, T)
        if not self.frame_level:
            parts = torch.cat((full, hand, face, pose_feat), dim=1)
            cat, parts = self.conv1d_1(cat, parts)
            cat = self.pool1(cat)
            parts = self.pool1(parts)
            cat, parts = self.conv1d_2(cat, parts)
            cat = self.pool2(cat)
            # cat = self.pool2(cat) ### shorter
            # cat = self.pool2(cat) ###
            parts = self.pool2(parts)
            full = parts[:, self.part_index[0][0]:self.part_index[0][1], :]
            hand = parts[:, self.part_index[1][0]:self.part_index[1][1], :]
            face = parts[:, self.part_index[2][0]:self.part_index[2][1], :]
            pose_feat = parts[:, self.part_index[3][0]:self.part_index[3][1], :]
            if self.num_classes != -1:
                full = self.fc[0](full.transpose(1, 2)).transpose(1, 2)
                hand = self.fc[1](hand.transpose(1, 2)).transpose(1, 2)
                face = self.fc[2](face.transpose(1, 2)).transpose(1, 2)
                pose_feat = self.fc[3](pose_feat.transpose(1, 2)).transpose(1, 2)
                cat = self.fc[4](cat.transpose(1, 2)).transpose(1, 2)

            self.loss = self.ISLR_loss([full, hand, face, pose_feat, cat], targets)
            # self.optimize()
            return full, self.loss
        return [full, hand, face, pose_feat, cat], pose

    def ISLR_loss(self, x, y):
        loss = 0.0

        for out in range(x):
            loss += self.criterion(out, y)  # , input_len, target_len)
        loss = loss / len(x)
        self.loss = loss
        return loss

    def optimize(self):
        self.loss.backward()
        self.optimizer.step()  # Now we can do an optimizer step
        self.optimizer.zero_grad()  # Reset gradients tensors


class VGGPose_Conv1D_RNN(nn.Module):
    def __init__(self, num_classes, mode, rnn_type='LSTM', input_channels=3):
        super(VGGPose_Conv1D_RNN, self).__init__()
        self.vggpose = VGGPose(input_channels=input_channels)
        self.posenet = nn.Sequential(
            nn.Linear(14, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True)
        )
        self.part_index = []
        self.part_size = [256, 256, 256, 256]  # full, hand, face, pose
        start = 0
        for size in self.part_size:
            self.part_index.append((start, start + size))
            start = start + size
        self.conv1d_1 = TMC_Conv1D(1536, [512, 512, 256, 256], 1024, [256, 256, 256, 256])
        self.pool1 = nn.MaxPool1d(kernel_size=2, ceil_mode=False)
        self.conv1d_2 = TMC_Conv1D(1024, [256, 256, 256, 256], 1024, [256, 256, 256, 256])
        self.pool2 = nn.MaxPool1d(kernel_size=2, ceil_mode=False)
        if mode == 'continuous':
            self.rnn = nn.ModuleList([
                EncoderRNN(rnn_type=rnn_type, input_size=self.part_size[0], hidden_size=256,
                           num_classes=num_classes, num_layers=2, bidirectional=True),
                EncoderRNN(rnn_type=rnn_type, input_size=self.part_size[1], hidden_size=256,
                           num_classes=num_classes, num_layers=2, bidirectional=True),
                EncoderRNN(rnn_type=rnn_type, input_size=self.part_size[2], hidden_size=256,
                           num_classes=num_classes, num_layers=2, bidirectional=True),
                EncoderRNN(rnn_type=rnn_type, input_size=self.part_size[3], hidden_size=256,
                           num_classes=num_classes, num_layers=2, bidirectional=True),
                EncoderRNN(rnn_type=rnn_type, input_size=sum(self.part_size), hidden_size=1024,
                           num_classes=num_classes, num_layers=2, bidirectional=True),
            ])
        else:
            self.fc = nn.ModuleList([nn.Linear(in_features=self.part_size[0], out_features=num_classes),
                                     nn.Linear(in_features=self.part_size[1], out_features=num_classes),
                                     nn.Linear(in_features=self.part_size[2], out_features=num_classes),
                                     nn.Linear(in_features=self.part_size[3], out_features=num_classes),
                                     nn.Linear(in_features=sum(self.part_size), out_features=num_classes)])
        self.dropout = nn.Dropout(0.4, inplace=True)

    def forward(self, x, targets, len_x=2, pose_given=None, teacher_ratio=0):
        B, C, T, H, W = x.shape
        len_x = torch.tensor([25])
        if pose_given is not None:
            pose_given = pose_given.reshape(B * T, 7, 2)
        # (B, C, T, H, W) 
        x = x.transpose(1, 2).reshape(B * T, C, H, W)
        # (B*T, C, H, W)
        pose, full, hand_l, hand_r, face = self.vggpose(x, pose_given=pose_given, teacher_ratio=teacher_ratio)
        pose = pose.reshape(B, T, 7, 2)
        # (B*T, C)
        full = full.reshape(B, T, -1).transpose(1, 2)
        hand = torch.cat((hand_l, hand_r), dim=-1).reshape(B, T, -1).transpose(1, 2)
        face = face.reshape(B, T, -1).transpose(1, 2)
        pose_feat = self.posenet(pose.detach().reshape(B, T, 14)).transpose(1, 2)

        full = self.dropout(full)
        cat = torch.cat((full, hand, face, pose_feat), dim=1)
        parts = torch.cat((full, hand, face, pose_feat), dim=1)
        # (B, C, T)
        cat, parts = self.conv1d_1(cat, parts)
        cat = self.pool2(cat)
        parts = self.pool2(parts)
        cat, parts = self.conv1d_2(cat, parts)
        cat = self.pool2(cat)
        parts = self.pool2(parts)
        cat = self.dropout(cat)

        # (B, C, (T-12/4) => (T', B, C)
        full = parts[:, self.part_index[0][0]:self.part_index[0][1], :].permute(2, 0, 1)
        hand = parts[:, self.part_index[1][0]:self.part_index[1][1], :].permute(2, 0, 1)
        face = parts[:, self.part_index[2][0]:self.part_index[2][1], :].permute(2, 0, 1)
        pose_feat = parts[:, self.part_index[3][0]:self.part_index[3][1], :].permute(2, 0, 1)
        cat = cat.permute(2, 0, 1)

        len_x = (len_x - 12) / 4  # len_x must be a long or int tensor
        # print(
        #     f' feats {full.shape} hand {hand.shape} face {face.shape} pose_feat {pose_feat.shape}  cat {cat.shape} {len_x}')
        full, _ = self.rnn[0](full, len_x)
        hand, _ = self.rnn[1](hand, len_x)
        face, _ = self.rnn[2](face, len_x)
        pose_feat, _ = self.rnn[3](pose_feat, len_x)
        cat, _ = self.rnn[4](cat, len_x)

        pose = pose.reshape(B, T, 7, 2)
        self.loss = self.CSLR_loss([full, hand, face, pose_feat, cat], targets)
        self.optimize()
        return full

    def CSLR_loss(self, x, y):
        loss = 0.0
        criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

        for out in range(x):
            input_len = torch.tensor([out.size(0)], dtype=torch.int)
            target_len = torch.tensor([y.size(1)], dtype=torch.int)
            loss += criterion(nn.functional.log_softmax(out, dim=2), y, input_len, target_len)
        loss = loss / len(x)
        self.loss = loss
        return loss

    def optimize(self):
        self.loss.backward()
        self.optimizer.step()  # Now we can do an optimizer step
        self.optimizer.zero_grad()  # Reset gradients tensors

    def iso_forward(self, x, len_x=2, pose_given=None, teacher_ratio=0):
        B, C, T, H, W = x.shape
        len_x = torch.tensor([25])
        if pose_given is not None:
            pose_given = pose_given.reshape(B * T, 7, 2)
        # (B, C, T, H, W)
        x = x.transpose(1, 2).reshape(B * T, C, H, W)
        # (B*T, C, H, W)
        pose, full, hand_l, hand_r, face = self.vggpose(x, pose_given=pose_given, teacher_ratio=teacher_ratio)
        pose = pose.reshape(B, T, 7, 2)
        # (B*T, C)
        full = full.reshape(B, T, -1).transpose(1, 2)
        hand = torch.cat((hand_l, hand_r), dim=-1).reshape(B, T, -1).transpose(1, 2)
        face = face.reshape(B, T, -1).transpose(1, 2)
        pose_feat = self.posenet(pose.detach().reshape(B, T, 14)).transpose(1, 2)

        full = self.dropout(full)
        cat = torch.cat((full, hand, face, pose_feat), dim=1)
        parts = torch.cat((full, hand, face, pose_feat), dim=1)
        # (B, C, T)
        cat, parts = self.conv1d_1(cat, parts)
        cat = self.pool2(cat)
        parts = self.pool2(parts)
        cat, parts = self.conv1d_2(cat, parts)
        cat = self.pool2(cat)
        parts = self.pool2(parts)
        cat = self.dropout(cat)

        # (B, C, (T-12/4) => (T', B, C)
        full = parts[:, self.part_index[0][0]:self.part_index[0][1], :].permute(2, 0, 1)
        hand = parts[:, self.part_index[1][0]:self.part_index[1][1], :].permute(2, 0, 1)
        face = parts[:, self.part_index[2][0]:self.part_index[2][1], :].permute(2, 0, 1)
        pose_feat = parts[:, self.part_index[3][0]:self.part_index[3][1], :].permute(2, 0, 1)
        cat = cat.permute(2, 0, 1)

        len_x = (len_x - 12) / 4  # len_x must be a long or int tensor
        #
        full = self.fc[0](full, len_x)
        hand = self.fc[1](hand, len_x)
        face = self.fc[2](face, len_x)
        pose_feat = self.fc[3](pose_feat, len_x)
        cat = self.fc[4](cat, len_x)

        pose = pose.reshape(B, T, 7, 2)

        return [full, hand, face, pose_feat, cat], len_x, pose


