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
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
import copy
# ssim
from skimage.metrics import structural_similarity
# load data
import albumentations as album
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import PIL
# save data
import pickle


def default_image_loader(image_path):
    """
    Helper function to open image file as PIL image in RBG format.
    """
    return PIL.Image.open(image_path).convert('RGB')

class LoadDatasetFromCSV(Dataset):
    """
    Class to import dataset from .csv files. Can perform transformations defined
    by agumentations using albumentations.
    The class inhereits from from torch.utils.data's Dataset.
    """
    def __init__(self, image_root, csv_path, transforms=None, loader=default_image_loader):
        # Root directory containing images
        self.image_root = image_root
        # Read data using pandas
        self.data = pd.read_csv(csv_path, header=None)
        # Initialize list for images
        imgs = []
        # Get file names in first column as numpy array
        files_names = np.array(self.data.iloc[:,0])
        # Iterate over all files names and join the image name with the root path
        for img in files_names:
            imgs.append(os.path.join(self.image_root, str(img)))
        # Get labels in second column as numpy array
        self.labels = np.asarray(self.data.iloc[:,1])
        # Define images, now with full path, as attribute
        self.images = imgs
        # Define provided transform as attribute
        self.transforms = transforms
        # Use the provided loader to load the images
        self.loader = loader

    def __getitem__(self, index):
        # Get image and label
        img = self.images[index]
        lbl = self.labels[index]
        # Load image using the loader
        img = np.array(self.loader(img))
        # Perform transormation if it was provided
        if self.transforms is not None:
            img = self.transforms(image=img)
        return img, lbl

    def __len__(self):
        # Return number of samples in dataset
        return len(self.data.index)

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)

def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class DeepFoolAttack:
    def __init__(self, img_path, mean, std, model_path, num_classes, overshoot, max_iter):
        self.img_path = img_path
        self.mean = mean
        self.std = std
        self.model_path = model_path
        self.num_classes = num_classes
        self.overshoot = overshoot
        self.max_iter = max_iter
        self.dataloader_attack = None

    def load(self, label_path):
        # resize images and rescale values
        data_mean = torch.tensor(self.mean)
        data_std = torch.tensor(self.std)
        album_compose = album.Compose([
            album.Resize(400, 400),                                                        # Resize to IMAGE_SIZE x IMAGE_SIZE
            album.Normalize(mean=[0.0,0.0,0.0], std=[1.0,1.0,1.0], max_pixel_value=255.0), # Rescale values from [0,255] to [0,1]
            album.Normalize(mean=data_mean, std=data_std, max_pixel_value=1.0),            # Rescale values according to above
            ToTensorV2(),
        ])
        # load data
        dataset_attack = LoadDatasetFromCSV(image_root = img_path, csv_path = label_path,transforms=album_compose)
        dataloader_attack = DataLoader(dataset=dataset_attack, batch_size = 1, shuffle = False)
        self.dataloader_attack = dataloader_attack

    def dfattack(self, save_path, min_ssim = 0.995, restrict_iter = False, restrict_ssim = True):
        #intialize
        label_arr = []
        k_i_arr = []
        ssim_arr = []
        i_arr = []
        count = 0
        fool_count = 0
        attacked_arr = []
        ori_arr = []
        #load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load(self.model_path, map_location=device)
        net = model.eval()
        # output transform
        in_transform = transforms.Compose([
                        transforms.Scale(400),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = self.mean, std = self.std)
                        ])
        clip = lambda x: clip_tensor(x, 0, 255)
        out_transform = transforms.Compose([
                        transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, self.std))),
                        transforms.Normalize(mean=list(map(lambda x: -x, self.mean)), std=[1, 1, 1]),
                        transforms.Lambda(clip),
                        transforms.ToPILImage(),
                        ])
        #handle restrictions
        if restrict_iter == False: self.max_iter = np.inf
        if restrict_ssim == False: min_ssim = 0
        # attack
        for batch, (images, labels) in enumerate(self.dataloader_attack):
            # batch: int
            # labels: tensor([int])
            image = images['image'] # image: tensor([[[...]]]) shape: 1, 3, 400, 400
            image = torch.squeeze(image) # squeeze to 3*400*400
            ori_arr.append(image)
            img_rt = out_transform(image) #a copy of unattacked image
            f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.numpy().flatten()
            I = (np.array(f_image)).flatten().argsort()[::-1]
            I = I[0:num_classes]
            label = I[0]
            input_shape = image.numpy()[0].shape
            pert_image = copy.deepcopy(image)
            w = np.zeros(input_shape)
            r_tot = np.zeros(input_shape)
            loop_i = 0
            x = Variable(pert_image[None, :], requires_grad=True)
            fs = net.forward(x)
            fs_list = [fs[0,I[k]] for k in range(num_classes)]
            k_i = label
            ssim = 1
            while k_i == label and loop_i <self.max_iter:
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
                r_tot_tmp = np.float32(r_tot + r_i)
                # compute ssim
                pert_image_tmp = image + (1+overshoot)*torch.from_numpy(r_tot_tmp)
                pert_image_tmp = out_transform(pert_image_tmp[0])
                pert_image_tmp = Image.fromarray((np.array(pert_image_tmp)), 'RGB')
                pert_image_tmp.save('tmp.png')
                pert_image_quantized = in_transform(Image.open('tmp.png'))
                #pert_image_quantized_rt = out_transform(pert_image_quantized)
                pert_image_quantized_rt = pert_image_tmp
                ssim_tmp = structural_similarity(np.array(img_rt), np.array(pert_image_quantized_rt), data_range = np.array(img_rt).max() - np.array(img_rt).min(), multichannel=True)
                if ssim_tmp > min_ssim: 
                    ssim = ssim_tmp
                    r_tot = np.float32(r_tot + r_i)
                    pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)
                    x = Variable(pert_image_quantized[None, :, :, :], requires_grad=True)
                    #x = Variable(pert_image, requires_grad=True)
                    fs = net.forward(x)
                    k_i = np.argmax(fs.data.numpy().flatten())
                    loop_i += 1
                else: 
                    #print(ssim_tmp)
                    r_tot = np.float32(r_tot + r_i * 0)
                    #pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)
                    pert_image_tmp = image + (1+overshoot)*torch.from_numpy(r_tot)
                    pert_image_tmp = out_transform(pert_image_tmp[0])
                    pert_image_tmp = Image.fromarray((np.array(pert_image_tmp)), 'RGB')
                    pert_image_tmp.save('tmp.png')
                    pert_image_quantized = in_transform(Image.open('tmp.png'))
                    #pert_image_quantized_rt = out_transform(pert_image_quantized)
                    pert_image_quantized_rt = pert_image_tmp
                    break
            r_tot = (1+overshoot)*r_tot
            # process output
            #pert_image_rt = np.array(out_transform(pert_image[0]))
            attacked_arr.append(pert_image_quantized)#[0])
            r_rt = np.array(pert_image_quantized_rt) - np.array(img_rt)
            # return loop_i, label, k_i, pert_image_rt, img_rt, r_rt, ssim
            # save outputs 
            label_arr.append(label)
            k_i_arr.append(k_i)
            # print labels
            print('###')
            print(batch)
            print('original label: ', label)
            print('new label: ', k_i)
            print('ssim: ', ssim)
            print('loop num: ', loop_i)
            ###
            #x = Variable(pert_image_quantized[None, :, :, :], requires_grad=True)
            #fs = net.forward(x)
            #k_i = np.argmax(fs.data.numpy().flatten())
            #print('test: ', k_i)
            #img = pert_image_quantized.unsqueeze(0)
            #net.to(device)
            #img = img.to(device)
            #out = net(img)
            #preds = F.softmax(out, dim=1)
            #prod, index = torch.max(preds, 1)
            #print('test: ', index)

            # save images
            pert_file = save_path + 'pert_' + str(batch) + '.png'
            Image.fromarray((np.array(pert_image_quantized_rt)), 'RGB').save(pert_file)
            ori_file = save_path + 'ori_' + str(batch) + '.png'
            Image.fromarray((np.array(img_rt)), 'RGB').save(ori_file)
            r_file = save_path + 'r_' + str(batch) + '.png'
            Image.fromarray((np.array(r_rt)), 'RGB').save(r_file)
            if label != k_i: fool_count = fool_count + 1
            count = count + 1
            ssim_arr.append(ssim)
            i_arr.append(loop_i)
            #break
        print('###')
        print('total number of attacked images: ', count)
        print("total number of fooled classifications: ", fool_count)
        print('fool ratio: ', fool_count/count)
        print('mean ssim: ', np.mean(np.array(ssim_arr)))
        print('std ssim: ', np.std(np.array(ssim_arr)))
        print('avg loop num: ', np.mean(np.array(i_arr)))
        return ori_arr, attacked_arr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restrict_iter', type=str2bool, default=False, help='restrict_iter') 
    parser.add_argument('--restrict_ssim', type=str2bool, default=True, help='restrict_ssim') 
    parser.add_argument('--overshoot', type=int, default=0.02, help='overshoot')
    parser.add_argument('--max_iter', type=int, default=10, help='max_iter')
    args = parser.parse_args()
    print('restrict_iter :', args.restrict_iter)
    print('restrict_ssim :', args.restrict_ssim)
    restrict_iter = args.restrict_iter
    restrict_ssim = args.restrict_ssim

    img_path = '/content/gdrive/MyDrive/EQ2445/Pytorch_Classification_Intergration/my_data/Master project data/test_image/'
    mean = [0.7750, 0.5888, 0.7629]
    std = [0.2129, 0.2971, 0.1774]
    model_path = "/content/gdrive/MyDrive/EQ2445/Pytorch_Classification_Intergration/checkpoints/new/best.pt"
    num_classes = 3
    overshoot = args.overshoot
    max_iter = args.max_iter
    df = DeepFoolAttack(img_path, mean, std, model_path, num_classes, overshoot, max_iter)
    save_path = '/content/gdrive/MyDrive/EQ2445/Pytorch_Classification_Intergration/my_data/Master project data/pert_image/'
    df.load('/content/gdrive/MyDrive/EQ2445/Pytorch_Classification_Intergration/my_data/Master project data/data_labels_test.csv')
    ori_arr, attacked_arr = df.dfattack(save_path,  restrict_iter = restrict_iter, restrict_ssim = restrict_ssim)
    f = open('/content/gdrive/MyDrive/EQ2445/Pytorch_Classification_Intergration/my_data/Master project data/pert_image/ori_arr.pckl', 'wb')
    pickle.dump(ori_arr, f)
    f.close()
    f = open('/content/gdrive/MyDrive/EQ2445/Pytorch_Classification_Intergration/my_data/Master project data/pert_image/attacked_arr.pckl', 'wb')
    pickle.dump(attacked_arr, f)
    f.close()


  