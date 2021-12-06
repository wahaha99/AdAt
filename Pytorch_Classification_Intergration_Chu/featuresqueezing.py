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
# entropy
#from scipy.stats import entropy
# median filter
from scipy import ndimage

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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
 
def color_bit_squeezer(img, i):
    #img: tensor 3*400*400 [0, 1]
    #i: number of target color bit depth (1 leq i leq 7)
    #return: tensor 3*400*400 [0, 1]
    ret = torch.round(img * (2 ** i - 1))
    ret = img / (2 ** i - 1)
    return ret

def median_filter(img, size):
    #img: tensor 3*400*400 [0, 1]
    #size: filter size (odd)
    #return: tensor 3*400*400 [0, 1]   
    image = img.numpy()
    ret = np.zeros((3,400,400))
    ret[0] = ndimage.median_filter(image[0], size=size)
    ret[1] = ndimage.median_filter(image[1], size=size)
    ret[2] = ndimage.median_filter(image[2], size=size)
    ret = torch.from_numpy(ret).float()
    return ret



class FeatureSqueezing:
    def __init__(self, img_path, mean, std, model_path, num_classes, ori_arr, attacked_arr, learn_ratio, metric):
        self.img_path = img_path
        self.mean = mean
        self.std = std
        self.model_path = model_path
        self.num_classes = num_classes
        self.ori_arr = ori_arr
        self.attacked_arr = attacked_arr
        self.learn_ratio = learn_ratio
        self.metric = metric
        self.threshold_c = 0
        self.threshold_m = 0

    def calculate_e(self, a, b):
        if self.metric == 0: return entropy(a, b) # entropy
        if self.metric == 1: return np.linalg.norm((np.array(a) - np.array(b)), ord=1) # l1 norm

    def learn_threshold_c(self, net, bit_depth, ori_learn, attacked_learn):
        print("### learning threshold c ###")
        e_ori_arr = []
        e_attacked_arr = []
        for i in range(len(ori_learn)):
            image = ori_learn[i]        
            image_sq = color_bit_squeezer(image, bit_depth)
            logit = net.forward(Variable(image[None, :, :, :], requires_grad=True))
            p = torch.nn.functional.softmax(logit, dim=1).data.numpy().flatten()
            logit_sq = net.forward(Variable(image_sq[None, :, :, :], requires_grad=True))
            p_sq = torch.nn.functional.softmax(logit_sq, dim=1).data.numpy().flatten()
            e = self.calculate_e(p, p_sq)
            #print(e)
            e_ori_arr.append(e)
        for i in range(len(attacked_learn)):
            image = attacked_learn[i]      
            #print(image.numpy().shape)     
            image_sq = color_bit_squeezer(image, bit_depth)
            logit = net.forward(Variable(image[None, :, :, :], requires_grad=True))
            p = torch.nn.functional.softmax(logit, dim=1).data.numpy().flatten()
            logit_sq = net.forward(Variable(image_sq[None, :, :, :], requires_grad=True))
            p_sq = torch.nn.functional.softmax(logit_sq, dim=1).data.numpy().flatten()
            e = self.calculate_e(p, p_sq)
            #print(e)
            e_attacked_arr.append(e)    
        center_ori = np.mean(e_ori_arr)
        center_attacked = np.mean(e_attacked_arr)
        #th = (center_ori + center_attacked) / 2
        selected_distance_idx = int(np.ceil(len(ori_learn) * 0.9))
        th = sorted(e_ori_arr)[selected_distance_idx-1]
        print("threshold: ", th)
        self.threshold_c = th

    def color_bit_detection(self, bit_depth = 5):
        # intialize
        ad_count_ori = 0
        ad_count_attacked = 0
        # load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load(self.model_path, map_location=device)
        net = model.eval()
        # learn threshold
        n_ori_learn = int(len(self.ori_arr) * self.learn_ratio)
        n_attacked_learn = int(len(self.attacked_arr) * self.learn_ratio)
        ori_learn = self.ori_arr[0:n_ori_learn - 1]
        attacked_learn = self.attacked_arr[0:n_attacked_learn - 1]
        ori_detection = self.ori_arr[n_ori_learn:]
        attacked_detection = self.attacked_arr[n_attacked_learn:]
        self.learn_threshold_c(net, bit_depth, ori_learn, attacked_learn)
        # detection
        print('### testing threshold ###')
        for i in range(len(ori_detection)):
            image = ori_detection[i]        
            image_sq = color_bit_squeezer(image, bit_depth)
            logit = net.forward(Variable(image[None, :, :, :], requires_grad=True))
            p = torch.nn.functional.softmax(logit, dim=1).data.numpy().flatten()
            logit_sq = net.forward(Variable(image_sq[None, :, :, :], requires_grad=True))
            p_sq = torch.nn.functional.softmax(logit_sq, dim=1).data.numpy().flatten()
            e = self.calculate_e(p, p_sq)
            if e > self.threshold_c: ad_count_ori = ad_count_ori + 1
        print('###')
        print('total number of unttacked images: ', len(ori_detection))
        print("total number of detected adversarial samples: ", ad_count_ori)
        print('false positive ratio: ', ad_count_ori/len(ori_detection))
        for i in range(len(attacked_detection)):
            image = attacked_detection[i]        
            image_sq = color_bit_squeezer(image, bit_depth)
            logit = net.forward(Variable(image[None, :, :, :], requires_grad=True))
            p = torch.nn.functional.softmax(logit, dim=1).data.numpy().flatten()
            logit_sq = net.forward(Variable(image_sq[None, :, :, :], requires_grad=True))
            p_sq = torch.nn.functional.softmax(logit_sq, dim=1).data.numpy().flatten()
            e = self.calculate_e(p, p_sq)
            if e > self.threshold_c: ad_count_attacked = ad_count_attacked + 1
        print('###')
        print('total number of attacked images: ', len(attacked_detection))
        print("total number of detected adversarial samples: ", ad_count_attacked)
        print('detection ratio: ', ad_count_attacked/len(attacked_detection))

    def learn_threshold_m(self, net, size, ori_learn, attacked_learn):
        print("### learning threshold m ###")
        e_ori_arr = []
        e_attacked_arr = []
        for i in range(len(ori_learn)):
            image = ori_learn[i]        
            image_sq = median_filter(image, size)
            logit = net.forward(Variable(image[None, :, :, :], requires_grad=True))
            p = torch.nn.functional.softmax(logit, dim=1).data.numpy().flatten()
            logit_sq = net.forward(Variable(image_sq[None, :, :, :], requires_grad=True))
            p_sq = torch.nn.functional.softmax(logit_sq, dim=1).data.numpy().flatten()
            e = self.calculate_e(p, p_sq)
            #print(e)
            e_ori_arr.append(e)
        for i in range(len(attacked_learn)):
            image = attacked_learn[i]     
            image_sq = median_filter(image, size)
            logit = net.forward(Variable(image[None, :, :, :], requires_grad=True))
            p = torch.nn.functional.softmax(logit, dim=1).data.numpy().flatten()
            logit_sq = net.forward(Variable(image_sq[None, :, :, :], requires_grad=True))
            p_sq = torch.nn.functional.softmax(logit_sq, dim=1).data.numpy().flatten()
            e = self.calculate_e(p, p_sq)
            #print(e)
            e_attacked_arr.append(e)    
        center_ori = np.mean(e_ori_arr)
        center_attacked = np.mean(e_attacked_arr)
        #th = (center_ori + center_attacked) / 2
        selected_distance_idx = int(np.ceil(len(ori_learn) * 0.9))
        th = sorted(e_ori_arr)[selected_distance_idx-1]
        print("threshold: ", th)
        self.threshold_m = th

    def median_smooth_detection(self, size = 3):
        # intialize
        ad_count_ori = 0
        ad_count_attacked = 0
        # load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load(self.model_path, map_location=device)
        net = model.eval()
        # learn threshold
        n_ori_learn = int(len(self.ori_arr) * self.learn_ratio)
        n_attacked_learn = int(len(self.attacked_arr) * self.learn_ratio)
        ori_learn = self.ori_arr[0:n_ori_learn - 1]
        attacked_learn = self.attacked_arr[0:n_attacked_learn - 1]
        ori_detection = self.ori_arr[n_ori_learn:]
        attacked_detection = self.attacked_arr[n_attacked_learn:]
        self.learn_threshold_m(net, size, ori_learn, attacked_learn)
        # detection
        print('### testing threshold ###')
        for i in range(len(ori_detection)):
            image = ori_detection[i]        
            image_sq = median_filter(image, size)
            logit = net.forward(Variable(image[None, :, :, :], requires_grad=True))
            p = torch.nn.functional.softmax(logit, dim=1).data.numpy().flatten()
            logit_sq = net.forward(Variable(image_sq[None, :, :, :], requires_grad=True))
            p_sq = torch.nn.functional.softmax(logit_sq, dim=1).data.numpy().flatten()
            e = self.calculate_e(p, p_sq)
            if e > self.threshold_m: ad_count_ori = ad_count_ori + 1
        print('###')
        print('total number of unttacked images: ', len(ori_detection))
        print("total number of detected adversarial samples: ", ad_count_ori)
        print('false positive ratio: ', ad_count_ori/len(ori_detection))
        for i in range(len(attacked_detection)):
            image = attacked_detection[i]        
            image_sq = median_filter(image, size)
            logit = net.forward(Variable(image[None, :, :, :], requires_grad=True))
            p = torch.nn.functional.softmax(logit, dim=1).data.numpy().flatten()
            logit_sq = net.forward(Variable(image_sq[None, :, :, :], requires_grad=True))
            p_sq = torch.nn.functional.softmax(logit_sq, dim=1).data.numpy().flatten()
            e = self.calculate_e(p, p_sq)
            if e > self.threshold_m: ad_count_attacked = ad_count_attacked + 1
        print('###')
        print('total number of attacked attacked images: ', len(attacked_detection))
        print("total number of detected adversarial samples: ", ad_count_attacked)
        print('detection ratio: ', ad_count_attacked/len(attacked_detection))

    def joint_detection(self, bit_depth = 5, size = 3):
        # intialize
        ad_count_ori = 0
        ad_count_attacked = 0
        # load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load(self.model_path, map_location=device)
        net = model.eval()
        # learn threshold
        n_ori_learn = int(len(self.ori_arr) * self.learn_ratio)
        n_attacked_learn = int(len(self.attacked_arr) * self.learn_ratio)
        ori_learn = self.ori_arr[0:n_ori_learn - 1]
        attacked_learn = self.attacked_arr[0:n_attacked_learn - 1]
        ori_detection = self.ori_arr[n_ori_learn:]
        attacked_detection = self.attacked_arr[n_attacked_learn:]
        self.learn_threshold_c(net, bit_depth, ori_learn, attacked_learn)
        self.learn_threshold_m(net, size, ori_learn, attacked_learn)
        # detection
        print('### testing threshold ###')
        for i in range(len(ori_detection)):
            image = ori_detection[i]        
            image_sq = median_filter(image, size)
            logit = net.forward(Variable(image[None, :, :, :], requires_grad=True))
            p = torch.nn.functional.softmax(logit, dim=1).data.numpy().flatten()
            logit_sq = net.forward(Variable(image_sq[None, :, :, :], requires_grad=True))
            p_sq = torch.nn.functional.softmax(logit_sq, dim=1).data.numpy().flatten()
            e_c = self.calculate_e(p, p_sq)
            image_sq = median_filter(image, size)
            logit = net.forward(Variable(image[None, :, :, :], requires_grad=True))
            p = torch.nn.functional.softmax(logit, dim=1).data.numpy().flatten()
            logit_sq = net.forward(Variable(image_sq[None, :, :, :], requires_grad=True))
            p_sq = torch.nn.functional.softmax(logit_sq, dim=1).data.numpy().flatten()
            e_m = self.calculate_e(p, p_sq)
            if e_c > e_m:
                if e_c > self.threshold_c: ad_count_ori = ad_count_ori + 1
            else:
                if e_m > self.threshold_m: ad_count_ori = ad_count_ori + 1
        print('###')
        print('total number of unttacked images: ', len(ori_detection))
        print("total number of detected adversarial samples: ", ad_count_ori)
        print('false positive ratio: ', ad_count_ori/len(ori_detection))
        for i in range(len(attacked_detection)):
            image = attacked_detection[i]        
            image_sq = median_filter(image, size)
            logit = net.forward(Variable(image[None, :, :, :], requires_grad=True))
            p = torch.nn.functional.softmax(logit, dim=1).data.numpy().flatten()
            logit_sq = net.forward(Variable(image_sq[None, :, :, :], requires_grad=True))
            p_sq = torch.nn.functional.softmax(logit_sq, dim=1).data.numpy().flatten()
            e_c = self.calculate_e(p, p_sq)
            image_sq = median_filter(image, size)
            logit = net.forward(Variable(image[None, :, :, :], requires_grad=True))
            p = torch.nn.functional.softmax(logit, dim=1).data.numpy().flatten()
            logit_sq = net.forward(Variable(image_sq[None, :, :, :], requires_grad=True))
            p_sq = torch.nn.functional.softmax(logit_sq, dim=1).data.numpy().flatten()
            e_m = self.calculate_e(p, p_sq)
            if e_c > e_m:
                if e_c > self.threshold_c: ad_count_attacked = ad_count_attacked + 1
            else:
                if e_m > self.threshold_m: ad_count_attacked = ad_count_attacked + 1
        print('###')
        print('total number of attacked attacked images: ', len(attacked_detection))
        print("total number of detected adversarial samples: ", ad_count_attacked)
        print('detection ratio: ', ad_count_attacked/len(attacked_detection))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--color_depth', type=str2bool, default=True, help='color_depth') 
    parser.add_argument('--bit', type=int, default=5, help='bit') 
    parser.add_argument('--median_smoothing', type=str2bool, default=True, help='median_smoothing')
    parser.add_argument('--size', type=int, default=2, help='size') 
    parser.add_argument('--joint', type=str2bool, default=False, help='joint')
    parser.add_argument('--lr', type=float, default=0.7, help='joint')

    args = parser.parse_args()
    print('color_depth: ', args.color_depth)
    if args.color_depth == True:
        print('reduce to: ', args.bit, ' bits')
    print('median_smoothing: ', args.median_smoothing)
    if args.median_smoothing == True:
        print('filter size: ', args.size, 'x', args.size)

    img_path = '/content/gdrive/MyDrive/EQ2445/Pytorch_Classification_Intergration/my_data/Master project data/test_image/'
    mean = [0.7750, 0.5888, 0.7629]
    std = [0.2129, 0.2971, 0.1774]
    model_path = "/content/gdrive/MyDrive/EQ2445/Pytorch_Classification_Intergration/checkpoints/new/best.pt"
    num_classes = 3

    f = open('/content/gdrive/MyDrive/EQ2445/Pytorch_Classification_Intergration/my_data/Master project data/pert_image/ori_arr.pckl', 'rb')
    ori_arr = pickle.load(f)
    f.close()
    f = open('/content/gdrive/MyDrive/EQ2445/Pytorch_Classification_Intergration/my_data/Master project data/pert_image/attacked_arr.pckl', 'rb')
    attacked_arr = pickle.load(f)
    f.close() 
    fq = FeatureSqueezing(img_path, mean, std, model_path, num_classes, ori_arr, attacked_arr, args.lr, 1)
    if args.color_depth == True:
        print('########## bit depth ##########')
        fq.color_bit_detection(args.bit)
    if args.median_smoothing == True:
        print('########## median smoothing ##########')
        fq.median_smooth_detection(args.size)
    if args.joint == True:
        print('########## joint detection ##########')
        fq.joint_detection(args.bit, args.size)

  