import glob
import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import os

def get_instance_segmentation_model(detector_path='flask_server/detector.pth', num_classes=2):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    # detector_path = os.getcwd() + '/flask_server/' + detector_path
    detector_path = detector_path
    checkpoint = torch.load(detector_path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=True)
    print('detector weights loaded')

    return model



def predict_masks(data,detector):
    detector.eval()
    with torch.no_grad():
        masks = []
        for index in range(len(data)):

            frame = data[index]
            pred = detector([frame])
            if pred[0]['scores'].nelement() == 0:
                continue
            C, H, W = data[index].shape
            mask_img = pred[0]['masks'][0, ...].cpu().squeeze()
            kernel = np.ones((3,3), np.uint8)
            mask_img = mask_img.cpu().numpy()

            mask_img_blurred = cv2.dilate(mask_img, kernel, iterations=5)
            _, mask_img_blurred = cv2.threshold(mask_img_blurred * 255, 127, 255, cv2.THRESH_BINARY)
            mask_img_blurred = np.expand_dims(mask_img_blurred / 255.0, axis=-1)
            mask_mul = np.concatenate((mask_img_blurred, mask_img_blurred, mask_img_blurred), axis=-1)
            # print(frame.max(),mask_mul.max())
            #b#ackground = cv2.cvtColor(cv2.imread('/home/iliask/Desktop/tsipras.jpg'), cv2.COLOR_BGR2RGB)
            #background = (1.0 - mask_mul) * (cv2.resize(background, (W, H)))

            framecv2 = cv2.cvtColor((frame.cpu().permute(1, 2, 0).numpy()) * mask_mul,cv2.COLOR_BGR2RGB)
            masks.append(torch.tensor(framecv2).permute(2, 0, 1))
            # cv2.imshow('blur', mask_img_blurred)
            # #mask_img = mask_img.numpy()
            # cv2.waitKey(10)
            # cv2.imshow('F, framecv2)
            # #cv2.imshow('BCK', background/255.0)
            # cv2.waitKey(100)



    return framecv2


def img_to_tensor(img):
    img1 = torch.tensor(img / 255.0, dtype=torch.float32).permute(2, 0, 1)

    return img1

rgb_path = '/home/iliask/Desktop/ilias/datasets/GSL_continuous/'
depth_path = '/home/iliask/Desktop/ilias/datasets/GSL_MASKS/'
detector_path = '/home/iliask/Desktop/ilias/pretrained_checkpoints/bbox_detector/detector.pth'

def run_on_GSL():
    detector = get_instance_segmentation_model(detector_path)
    detector = detector.cuda()
    folders = sorted(glob.glob(f'{rgb_path}*/*/*'))
    print(len(folders))
    fs = []
    for i in folders:
        if 'health1' not in i :
            fs.append(i)
    print(len(folders),len(fs))

    parent_dir = os.getcwd()
    for fold in fs:
        images = sorted(glob.glob(f'{fold}/*.jpg'))
        print(len(images))
        print(images)
        for img_path in images:
            cv2_image = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
            scale = 2
            #print(cv2_image.shape)
            H,W,_ = cv2_image.shape
            cv2_image = cv2.resize(cv2_image, (W // scale, H // scale))
            img = img_to_tensor(cv2_image).unsqueeze(0).cuda()
            masked_img = predict_masks(img,detector)
            masked_img = masked_img
            #print(img_path)
            new_path = img_path.replace('GSL_continuous','GSL_MASKS')
            new_folder = new_path.rsplit('/',1)[0]
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
            cv2.imwrite(new_path,masked_img*255)

            #print(new_folder)


run_on_GSL()