# !/usr/bin/python
# -*- coding: utf-8 -*-

import os
import time
import argparse
from PIL import Image
import PIL.ImageOps
import torch
from torchvision import transforms
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

from torch.autograd import Variable
from PIL import Image
import numpy as np
import copy

import albumentations as album
from albumentations.pytorch import ToTensorV2



def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)

def deepfool(img ='/content/gdrive/MyDrive/EQ2445/Pytorch_Classification_Intergration/my_data/Master project data/test_image/6tjqTU40.png', 
             num_classes=3, overshoot=0.02, max_iter=50):

    #load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load("/content/gdrive/MyDrive/EQ2445/Pytorch_Classification_Intergration/checkpoints/b0300/weights/best.pt", map_location=device)
    #print('Load Model Done!!!')
    net = model.eval()

    #transforms
    in_transform = transforms.Compose([
                                      transforms.Resize(size=(400, 400), interpolation=Image.LANCZOS),
                                      transforms.ToTensor(),
                                      #transforms.Normalize([0.7790739, 0.58031166, 0.75693846], [0.17999493, 0.27398717, 0.1621663]),
                                      ])

    out_transform = transforms.Compose([
                                       transforms.ToPILImage(),
                                       ])

    in_transform2 = album.Compose([
        album.Resize(400, 400),
        album.Normalize(mean=[0.7790739, 0.58031166, 0.75693846], std=[0.17999493, 0.27398717, 0.1621663]),
        ToTensorV2(),
    ])

    out_transform2 = album.Compose([
        album.Resize(400, 400),
        album.Normalize(mean=[0.7790739, 0.58031166, 0.75693846], std=[0.17999493, 0.27398717, 0.1621663]),
        ToTensorV2(),
    ])

    #load image
    image = Image.open(img)
    image = in_transform(image)
    #image = in_transform2(image = np.array(image))
    img_rt = out_transform(image) #a copy of unattacked image
    #img_rt = out_transform2(image)

    #fool
    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.numpy().flatten())

        loop_i += 1

    r_tot = (1+overshoot)*r_tot

    pert_image_rt = out_transform(pert_image[0])
    r_rt = np.array(pert_image_rt) - np.array(img_rt)
    return loop_i, label, k_i, pert_image_rt, img_rt, r_rt
  
if __name__ == '__main__':
    files = os.listdir('/content/gdrive/MyDrive/EQ2445/Pytorch_Classification_Intergration/my_data/Master project data/test_image/')
    fool_count = 0
    count = 0
    for file in files:
        print('###')
        file_name = '/content/gdrive/MyDrive/EQ2445/Pytorch_Classification_Intergration/my_data/Master project data/test_image/' + file
        loop_i, label, k_i, pert_image, ori_image, r_image = deepfool(img = file_name, num_classes=3, overshoot=0.02, max_iter=10)
        print(file)
        print(label, k_i)

        pert_file = '/content/gdrive/MyDrive/EQ2445/Pytorch_Classification_Intergration/my_data/Master project data/pert_image/pert_' + file
        #print((np.array(pert_image)).shape)
        Image.fromarray((np.array(pert_image)), 'RGB').save(pert_file)

        ori_file = '/content/gdrive/MyDrive/EQ2445/Pytorch_Classification_Intergration/my_data/Master project data/pert_image/ori_' + file
        #print((np.array(ori_image)).shape)
        Image.fromarray((np.array(ori_image)), 'RGB').save(ori_file)

        r_file = '/content/gdrive/MyDrive/EQ2445/Pytorch_Classification_Intergration/my_data/Master project data/pert_image/r_' + file
        #print((np.array(r_image)).shape)
        Image.fromarray((np.array(r_image)), 'RGB').save(r_file)

        if label != k_i: fool_count = fool_count + 1
        count = count + 1
        #break
    print('###')
    print('total number of attacked images: ', count)
    print("total number of fooled classifications: ", fool_count)
    print('fool ratio: ', fool_count/count)






  