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

def exif_transpose(img):
    if not img:
        return img

    exif_orientation_tag = 274

    # Check for EXIF data (only present on some files)
    if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
        exif_data = img._getexif()
        orientation = exif_data[exif_orientation_tag]

        # Handle EXIF Orientation
        if orientation == 1:
            # Normal image - nothing to do!
            pass
        elif orientation == 2:
            # Mirrored left to right
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # Rotated 180 degrees
            img = img.rotate(180)
        elif orientation == 4:
            # Mirrored top to bottom
            img = img.rotate(180).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            # Mirrored along top-left diagonal
            img = img.rotate(-90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            # Rotated 90 degrees
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            # Mirrored along top-right diagonal
            img = img.rotate(90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            # Rotated 270 degrees
            img = img.rotate(90, expand=True)

    return img


def load_image_file(file, mode='RGB'):
    # Load the image with PIL
    img = PIL.Image.open(file)

    if hasattr(PIL.ImageOps, 'exif_transpose'):
        # Very recent versions of PIL can do exit transpose internally
        img = PIL.ImageOps.exif_transpose(img)
    else:
        # Otherwise, do the exif transpose ourselves
        img = exif_transpose(img)

    img = img.convert(mode)

    return img


def default_loader(path):
    return Image.open(path).convert('RGB')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='/content/gdrive/MyDrive/EQ2445/Pytorch_Classification_Intergration/my_data/Master project data/test_image/',
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--model', type=str,
                        default="/content/gdrive/MyDrive/EQ2445/Pytorch_Classification_Intergration/checkpoints/b0300/weights/best.pt")
    parser.add_argument('--image_size', type=int, default=400, help='image_size')
    args = parser.parse_args()
    print(args)

    # Test image folder
    source = args.source + os.sep
    images = os.listdir(source)

    # Result list
    result_list_1 = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(args.model, map_location=device)
    print('Load Model Done!!!')
    outfile = "/content/gdrive/MyDrive/EQ2445/Pytorch_Classification_Intergration/train_label_by_model.csv"
    fileout = open(outfile, "w")
    if not fileout:
        print("cannot open the file %s for writing" % outfile)
    fileout.write("image_name" + "," +  "label" +"\n")
    file_count = 0
    start_time = time.time()
    with torch.no_grad():
        model.eval()
        for file in images:
            image_path = os.path.join(source, file)

            img0 = default_loader(image_path)

            valid_transform = transforms.Compose([
                transforms.Resize(size=(args.image_size, args.image_size), interpolation=Image.LANCZOS),
                transforms.ToTensor(),
                transforms.Normalize([0.7823419,  0.59793377, 0.7674322 ], [0.1811313,  0.2713135,  0.15672927])
            ])

            img = valid_transform(img0)
            img = img.unsqueeze(0)
            model.to(device)
            img = img.to(device)
            out = model(img)
            preds = F.softmax(out, dim=1)
            prod, index = torch.max(preds, 1)

            print(f"{file} -> out: {out}")
            print(f"{file} -> preds: {preds}")
            print(f"{file} -> index: {index}-> index.item={index.item()}")
            print(f"{file} -> prod: {prod}")
            print("\n")
            print("\n")
            fileout.write(file+","+str(index.item())+"\n")
