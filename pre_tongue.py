#%% import packages
import numpy as np
import os
join = os.path.join 
from skimage import transform, io, segmentation,util,exposure
from tqdm import tqdm
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import argparse
import torch.nn as nn
import cv2

from PIL import Image
from tqdm import tqdm
# from segment.nets.deeplabv3_plus import DeepLab
from segment.deeplab import DeeplabV3

# set up the parser
parser = argparse.ArgumentParser(description='preprocess grey and RGB images')
parser.add_argument('-i', '--img_path', type=str, default='/home/disk/cs/project/dataset/segmentation/tongueset3_split/img/', help='path to the images')
parser.add_argument('-gt', '--gt_path', type=str, default='/home/disk/cs/project/dataset/segmentation/tongueset3_split/gt/', help='path to the ground truth (gt)')
parser.add_argument('-o', '--npz_path', type=str, default='/home/disk/cs/project/dataset/segmentation/tongueset3_npz2/', help='path to save the npz files')
parser.add_argument('--data_name', type=str, default='tongue', help='dataset name; used to name the final npz file, e.g., demo2d.npz')
parser.add_argument('--image_size', type=int, default=400, help='image size')
parser.add_argument('--img_name_suffix', type=str, default='.jpg', help='image name suffix')
parser.add_argument('--label_id', type=int, default=1, help='label id')
parser.add_argument('--model_type', type=str, default='vit_b', help='model type')
parser.add_argument('--checkpoint', type=str, default='./pretrained_model/sam.pth', help='checkpoint')
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--seed', type=int, default=2023, help='random seed')
args = parser.parse_args()


def semantic_segmentation_augmentation(image, label, rotation_range=(-90, 90)):
    stretch_range=(0.8, 1.2)
    # 随机裁剪
    h, w, _ = image.shape
    top = np.random.randint(0, h - 300)
    left = np.random.randint(0, w - 300)
    image = image[top:top+300, left:left+300]
    label = label[top:top+300, left:left+300]

    # 随机水平翻转
    if np.random.rand() > 0.5:
        image = np.fliplr(image)
        label = np.fliplr(label)

    # 随机旋转
    rotation_angle = np.random.uniform(rotation_range[0], rotation_range[1])
    image = rotate_image(image, rotation_angle)
    label = rotate_image(label, rotation_angle)

    # 随机颜色抖动（可选）
    image = augment_colors(image)
    # 随机拉伸
    # stretch_factor_x = np.random.uniform(stretch_range[0], stretch_range[1])
    # stretch_factor_y = np.random.uniform(stretch_range[0], stretch_range[1])
    # image = stretch_image(image, stretch_factor_x, stretch_factor_y)
    # label = stretch_image(label, stretch_factor_x, stretch_factor_y)

    return image, label

def augment_colors(image):
    # 随机生成对比度和亮度的增益
    contrast_factor = np.random.uniform(0.5, 1.5)  # 可以调整范围以获得所需的效果
    brightness_factor = np.random.randint(-50, 51)  # 亮度增益的范围

    # 修改对比度和亮度
    image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=brightness_factor)


    return image

def rotate_image(image, angle):
    # 旋转图像
    image = transform.rotate(image, angle, resize=False, mode='reflect')
    return util.img_as_ubyte(image)


def deal(img_path,gt_path,num):         
    names = sorted(os.listdir(gt_path))    
    save_path = args.npz_path 
    os.makedirs(save_path, exist_ok=True)
    print('image number:', len(names))
    imgs = []
    gts =  []
    boxes=[]
    img_embeddings = []    
    for gt_name in tqdm(names):        
        sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)
        image_name = gt_name.split('.')[0] + args.img_name_suffix
        gt_data = io.imread(join(gt_path, gt_name))    
        image_data = io.imread(join(img_path, image_name))  
        image_data,gt_data=semantic_segmentation_augmentation(image_data, gt_data)        
        # cv2.imwrite(gt_name,image_data)
        gt_data=cv2.resize(gt_data,(args.image_size,args.image_size))
        if len(gt_data.shape)==3:
            gt_data = gt_data[:,:,0]
        assert len(gt_data.shape)==2, 'ground truth should be 2D'
        gt_data = transform.resize(gt_data==args.label_id, (args.image_size, args.image_size), order=0, preserve_range=True, mode='constant')    
        gt_data = np.uint8(gt_data)  
              
        if image_data.shape[-1]>3 and len(image_data.shape)==3:
                image_data = image_data[:,:,:3]
        if len(image_data.shape)==2:
            image_data = np.repeat(image_data[:,:,None], 3, axis=-1)              
        if gt_data.shape[-1]==3:
            gt=gt_data    
            z=np.zeros([gt.shape[0],gt.shape[1]])
            for i in range(gt.shape[0]):
                for j in range(gt.shape[1]): 
                    if gt[i][j][0]==1:
                        z[i][j]=1
            gt=z               
            gt_data=gt

        lower_bound, upper_bound = np.percentile(image_data, 0.5), np.percentile(image_data, 99.5)
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        image_data_pre = (image_data_pre - np.min(image_data_pre))/(np.max(image_data_pre)-np.min(image_data_pre))*255.0
        image_data_pre[image_data==0] = 0
        image_data_pre = transform.resize(image_data_pre, (args.image_size,args.image_size), order=3, preserve_range=True, mode='constant', anti_aliasing=True)
        image_data_pre = np.uint8(image_data_pre)
        imgs.append(image_data_pre)
        gts.append(gt_data)
        H, W, _ = image_data_pre.shape
        sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        resize_img = sam_transform.apply_image(image_data_pre)
        resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(args.device)
        input_image = sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
        assert input_image.shape == (1, 3, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'                                                                    
        embedding = sam_model.image_encoder(input_image) 
        img_embeddings.append(embedding.cpu().detach().numpy()[0])        
        ##########################################################################################         
        # pic=Image.open(join(img_path, image_name))
        # pic=Image.fromarray(image_data_pre)
        # pic= model.get_miou_png(pic)        
        y_indices, x_indices = np.where(gt_data > 0)
        xmin, xmax = np.min(x_indices), np.max(x_indices)
        ymin, ymax = np.min(y_indices), np.max(y_indices)        
        box=np.array([xmin,ymin,xmax,ymax])                                              
        boxes.append(box)
        ##########################################################################################
        del sam_model
    print('Num. of images:', len(imgs))
    if len(imgs)>0:
        imgs = np.stack(imgs, axis=0) # (n, 256, 256, 3)
        gts = np.stack(gts, axis=0) # (n, 256, 256)
        img_embeddings = np.stack(img_embeddings, axis=0) # (n, 1, 256, 64, 64)        
        np.savez_compressed(join(save_path, args.data_name + str(num)+'.npz'), imgs=imgs, boxes=boxes,gts=gts, img_embeddings=img_embeddings)
        # save an example image for sanity check
        idx = np.random.randint(imgs.shape[0])
        img_idx = imgs[idx,:,:,:]
        gt_idx = gts[idx,:,:]
        bd = segmentation.find_boundaries(gt_idx, mode='inner')
        img_idx[bd, :] = [args.image_size-1, 0, 0]
        # io.imsave(save_path + '.png', img_idx, check_contrast=False)
    else:
        print('Do not find image and ground-truth pairs. Please check your dataset and argument settings')
        
num=0
# for i in range(10):
for f in os.listdir(args.img_path):    
    print(f)
    # deal(args.img_path+'/'+f,args.gt_path+'/'+f,num)
    num+=1
