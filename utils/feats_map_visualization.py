import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import matplotlib.pyplot as plt
import torch

from models.slr_models import SLR_I3D,GoogLeNet_TConvs
from dataloaders.loader_functions import load_video_sequence

class Hook():
    def __init__(self, module, backward=False):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


select_model = 4

visualisation = {}


def hook_fn(m, i, o):
    visualisation[m] = o

def select_model_for_vizulization():
    if (select_model == 1):
        model = I3D_RNN(pretrained=False, mode='continuous', temp_resolution=8, hidden_size=512)
        model.load_state_dict(torch.load(
            '/home/papastrat/SLR/checkpoints/pretrained/i3d_phoenix2014_ws_loss_ent_ctc_test_dayFri Sep 27 02:04:19 2019/best_wer.pth',
            map_location='cpu')[
                                  'model_dict'])
        print(model.named_modules())
        # model.cuda()
        model.cnn.Conv3d_1a_7x7.register_forward_hook(get_activation('conv1'))
    elif (select_model == 2):
        model = SubUnet(hidden_size=1024, n_layers=2, bi=True, dropt=0.5, N_classes=1232)
        print(model)
        model.load_state_dict(
            torch.load(
                '/home/papastrat/SLR/checkpoints/model_subunet/model_subunetweakly_supervised_loss_ent_ctctest_dayWed Sep 25 21:34:43 2019/best_wer.pth',
                map_location='cpu')[
                'model_dict'])

        model.cnn.features[0].register_forward_hook(get_activation('conv1'))

    elif (select_model == 3):
        model = Cui2019(hidden_size=512, n_layers=2, bi=True, dropt=0.5, N_classes=1232,backbone='googlenet')
        model.load_state_dict(
            torch.load(
                '/home/papastrat/SLR/checkpoints/model_cui/model_cuiweakly_supervised_loss_ent_ctctest_dayTue Sep 24 17:55:12 2019/best_wer.pth',
        map_location = 'cpu')[
            'model_dict'])
        print(model)
        model.cnn.conv1.register_forward_hook(get_activation('conv1'))

    elif ( select_model == 4):
        model = Resnet3d(pretrained=False, mode='continuous', temp_resolution=8, hidden_size=512)
        print(model)
        model.load_state_dict(
            torch.load(
                '/home/papastrat/SLR/checkpoints/model_resnet/dataset_phoenix2014/ws_loss_normal_test_dayWed Oct 16 12:44:26 2019/best_wer.pth',
        map_location = 'cpu')[
            'model_dict'])

        model.module.conv1.register_forward_hook(get_activation('conv1'))
    paths = {
        'phoenix': 'data/phoenix_version1/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/test/28September_2009_Monday_tagesschau_default-7/1'}

x = load_video_sequence(path=paths['phoenix'], time_steps=250, dim=(224, 224),
                        augmentation='test', padding=False, normalize=True, img_type='png').unsqueeze(0)  # .cuda()

output = model(x)  # .cpu()
if select_model == 1:
    print(activation['conv1'].squeeze().shape)
    act = activation['conv1'].squeeze()[0, :, :, :]
elif select_model == 2:
    print(activation['conv1'].squeeze().shape)
    ## select one conv layer fileter
    act = activation['conv1'].squeeze()[:, 0, :, :]
elif select_model == 3:
    print(activation['conv1'].squeeze().shape)
    act = activation['conv1'].squeeze()[0, :, :, :]

elif select_model == 4:
    print(activation['conv1'].squeeze().shape)
    ## select one conv layer fileter
    act = activation['conv1'].squeeze()[:, 0, :, :]
# act = visualisation
# # print(act)
# for key, value in act.items():
#      print("{} {}".format(key, 0))
#
# act = visualisation[
#     'Conv3d_1a_7x7']
print("shaep", act.shape)
num_plot = 4

# for i in range(act.size(0)):
#     plt.imshow(act[i, :, :])
#     #plt.show()
fig, axarr = plt.subplots(1, num_plot)
print("len ", axarr.shape)
for idx in range(min(act.size(0), num_plot)):
    axarr[idx].imshow(act[idx+5])
    axarr[idx].axis('off')
plt.show()

# print(x.shape)
#
# for idx in range(min(act.size(0), num_plot)):
#     axarr[idx].imshow(x[0,idx + 3])
#     axarr[idx].axis('off')
# plt.show()

#
# class LayerActivations():
#     features = []
#
#     def __init__(self, model):
#         self.features = []
#         self.hook = model.register_forward_hook(self.hook_fn)
#
#         def hook_fn(self, module, input, output):
#             self.features.extend(output.view(output.size(0), -1).cpu().data)
#
#         def remove(self):
#             self.hook.remove()
