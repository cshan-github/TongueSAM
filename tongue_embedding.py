#%% import packages
import numpy as np
import os
join = os.path.join 
from skimage import transform, io, segmentation
from tqdm import tqdm
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import argparse
import torch.nn as nn
import cv2
import pandas as pd 
from PIL import Image
from tqdm import tqdm
# from segment.nets.deeplabv3_plus import DeepLab
from segment.deeplab import DeeplabV3
import random
import csv
# set up the parser
parser = argparse.ArgumentParser(description='preprocess grey and RGB images')
parser.add_argument('-i', '--img_path', type=str, default='/home/cs/project/medsam_tongue/tongue_embed/', help='path to the images')
csv_path='/home/cs/project/medsam_tongue/all_output_idx.csv'
parser.add_argument('-o', '--npz_path', type=str, default='/home/cs/project/medsam_tongue/tongue_embed_npz', help='path to save the npz files')
parser.add_argument('--data_name', type=str, default='tongue', help='dataset name; used to name the final npz file, e.g., demo2d.npz')
parser.add_argument('--image_size', type=int, default=400, help='image size')
parser.add_argument('--img_name_suffix', type=str, default='.png', help='image name suffix')
parser.add_argument('--label_id', type=int, default=1, help='label id')
parser.add_argument('--model_type', type=str, default='vit_b', help='model type')
parser.add_argument('--checkpoint', type=str, default='./pretrained_model/sam.pth', help='checkpoint')
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--seed', type=int, default=2023, help='random seed')
train_rate=1
args = parser.parse_args()
df=pd.read_csv(csv_path, header=None)
df.fillna(0, inplace=True)
#%% set up the model
#%% convert 2d grey or rgb images to npz file
def deal(img_path,num):         
    names = sorted(os.listdir(img_path))    
    save_path = args.npz_path 
    os.makedirs(save_path, exist_ok=True)
    print('image number:', len(names))    
    train_img_embeddings = [] 
    train_csvs=[]   
    train_gts=[]
    test_img_embeddings = [] 
    test_csvs=[]   
    test_gts=[]
    for path in tqdm(names):                                                                  
        sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)
        image_name = path.split('.')[0] + args.img_name_suffix         
        image_data = io.imread(join(img_path, image_name))        
        if image_data.shape[-1]>3 and len(image_data.shape)==3:
                image_data = image_data[:,:,:3]
        if len(image_data.shape)==2:
            image_data = np.repeat(image_data[:,:,None], 3, axis=-1)

        lower_bound, upper_bound = np.percentile(image_data, 0.5), np.percentile(image_data, 99.5)
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        image_data_pre = (image_data_pre - np.min(image_data_pre))/(np.max(image_data_pre)-np.min(image_data_pre))*255.0
        image_data_pre[image_data==0] = 0
        image_data_pre = transform.resize(image_data_pre, (args.image_size,args.image_size), order=3, preserve_range=True, mode='constant', anti_aliasing=True)
        image_data_pre = np.uint8(image_data_pre)            
        H, W, _ = image_data_pre.shape
        sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        resize_img = sam_transform.apply_image(image_data_pre)
        resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(args.device)
        input_image = sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
        assert input_image.shape == (1, 3, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'                                                                    
        embedding = sam_model.image_encoder(input_image)
        embedding=nn.MaxPool2d(64)(embedding)
        embedding=embedding.squeeze().cpu().detach().numpy()
        c=[]
        n=random.random()      
        if n<train_rate:
            for i in range(1778):            
                if int(df.iloc[i,0])==int(path.split('.')[0]):                 
                    train_gts.append(df.iloc[i,-1])
                    train_csvs.append(df.iloc[i,1:-1].to_numpy())  
                    break         
            # train_img_embeddings.append(embedding.cpu().detach().numpy()[0])
            c=df.iloc[i,1:-1].to_list()+list(embedding)
            c.append(df.iloc[i,-1])
            with open('./all_img_out.csv','a',newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(c)
        else:
            for i in range(1778):            
                if int(df.iloc[i,0])==int(path.split('.')[0]):                 
                    test_gts.append(df.iloc[i,-1])
                    test_csvs.append(df.iloc[i,1:-1].to_numpy())  
                    break         
            test_img_embeddings.append(embedding.cpu().detach().numpy()[0])
        del sam_model              
    # #################################
    # csvs = np.stack(train_csvs, axis=0) 
    # gts = np.stack(train_gts, axis=0) 
    # img_embeddings = np.stack(train_img_embeddings, axis=0) 
    # #################################
    # np.savez_compressed(join(save_path, args.data_name+'_train.npz'), img_embeddings=img_embeddings,csvs=csvs,gts=gts)                                    
    # csvs = np.stack(test_csvs, axis=0) 
    # gts = np.stack(test_gts, axis=0) 
    # img_embeddings = np.stack(test_img_embeddings, axis=0) 
    # np.savez_compressed(join(save_path, args.data_name+'_test.npz'), img_embeddings=img_embeddings,csvs=csvs,gts=gts)                                    
        
num=0
# for f in os.listdir(args.img_path):    
deal(args.img_path,num)
# num+=1
