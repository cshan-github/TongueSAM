from PIL import ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io
join = os.path.join
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from utils.SurfaceDice import compute_dice_coefficient
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, jaccard_score
# set seeds
torch.manual_seed(2023)
np.random.seed(2023)
from skimage import io
from  utils_metrics import *
from skimage import transform, io, segmentation
from segment.yolox import YOLOX
import random
import warnings

# 永久性地忽略指定类型的警告
warnings.filterwarnings("ignore", category=UserWarning)
#########################################################################################################
ts_img_path = './data/test_in/'
model_type = 'vit_b'
checkpoint = './pretrained_model/tonguesam.pth'
device = 'cuda:1'
path_out='./data/test_out/'
segment=YOLOX()
##############################################################################################################
def get_bbox_from_mask(mask):
    '''Returns a bounding box from a mask'''    
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = mask.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))

    return np.array([x_min, y_min, x_max, y_max])
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))
best_iou=0
test_names = sorted(os.listdir(ts_img_path))

sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
#################################
# prune_threshold =0.005
# for param in sam_model.image_encoder.parameters():    
#     param.data[torch.abs(param.data) < prune_threshold] = 0
sam_model.eval()
#################################
val_gts=[]
val_preds=[]
for f in os.listdir(ts_img_path):   
    with torch.no_grad():             
        image_data = io.imread(join(ts_img_path, f))
    
        if image_data.shape[-1] > 3 and len(image_data.shape) == 3:
            image_data = image_data[:, :, :3]
        if len(image_data.shape) == 2:
            image_data = np.repeat(image_data[:, :, None], 3, axis=-1)
        
        lower_bound, upper_bound = np.percentile(image_data, 0.5), np.percentile(image_data, 99.5)
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        image_data_pre = (image_data_pre - np.min(image_data_pre)) / (np.max(image_data_pre) - np.min(image_data_pre)) * 255.0
        image_data_pre[image_data == 0] = 0
        image_data_pre = transform.resize(image_data_pre, (400, 400), order=3, preserve_range=True, mode='constant', anti_aliasing=True)
        image_data_pre = np.uint8(image_data_pre)
        
        H, W, _ = image_data_pre.shape
        sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        resize_img = sam_transform.apply_image(image_data_pre)
        resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
        input_image = sam_model.preprocess(resize_img_tensor[None, :, :, :])        
        ts_img_embedding = sam_model.image_encoder(input_image)      

        img = image_data_pre
        boxes = segment.get_prompt(img)
                
        if boxes is not None:
            sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)                   
            box = sam_trans.apply_boxes(boxes, (400,400))                                                
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)            
        else:            
            box_torch = None                
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        
        # 使用Mask_Decoder生成分割结果
        medsam_seg_prob, _ = sam_model.mask_decoder(
            image_embeddings=ts_img_embedding.to(device),
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )                        
        medsam_seg_prob =medsam_seg_prob.cpu().detach().numpy().squeeze()        
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
        
        medsam_seg=cv2.resize(medsam_seg,(400,400))       
        
        
        pred = cv2.Canny(cv2.resize((medsam_seg != 0).astype(np.uint8) * 255, (400, 400)), 100, 200)
        
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                if pred[i, j] != 0:
                    img[max(i - 1, 0):min(i + 2, 400), max(j - 1, 0):min(j + 2, 400), :] = [0, 0, 255]

        image1 = Image.fromarray(medsam_seg)
        image2 = Image.fromarray(img)

        image1 = image1.resize(image2.size).convert("RGBA")
        image2 = image2.convert("RGBA")
        data1 = image1.getdata()

        new_image = Image.new("RGBA", image2.size)
        new_data = [(0, 0, 128, 96) if pixel1[0] != 0 else (0, 0, 0, 0) for pixel1 in data1]

        new_image.putdata(new_data)
        if boxes is not None:              
            draw = ImageDraw.Draw(image2)
            draw.rectangle([boxes[0],boxes[1],boxes[2],boxes[3]],fill=None, outline=(0, 255, 0), width=5)  # 用红色绘制方框的边框，线宽为2
        image2.paste(new_image, (0, 0), mask=new_image)
        image2.save(path_out + f.split('.')[0] + '.png')
        print(f)
        