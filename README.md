# TongueSAM: An Universal Tongue Segmentation Model Based on SAM with Zero-Shot
This is the public project of paper:"TongueSAM: An Universal Tongue Segmentation Model Based on SAM with Zero-Shot", this paper can be get:https://arxiv.org/abs/2308.06444.

## Abstract

Tongue segmentation serves as the primary step in automated TCM tongue diagnosis, which plays a significant role in the di- agnostic results. Currently, numerous deep learning based methods have achieved promising results. However, most of these methods exhibit mediocre performance on tongues different from the training set. To address this issue, this paper proposes a universal tongue segmentation model named TongueSAM based on SAM (Segment Anything Model). SAM is a large-scale pretrained interactive segmentation model known for its powerful zero-shot generalization capability. Applying SAM to tongue segmentation enables the segmentation of various types of tongue images with zero-shot. In this study, a Prompt Generator based on object detection
is integrated into SAM to enable an end-to-end automated tongue segmentation method. Experiments demonstrate that TongueSAM achieves exceptional performance across various of tongue segmentation datasets, particularly under zero-shot. TongueSAM can be directly applied to other datasets without fine-tuning. As far as we know, this is the first application of large-scale pretrained model for tongue segmentation. 

## Method

TongueSAM consists primarily of two components: SAM and the Prompt Generator. For a given tongue image, TongueSAM first utilizes the pretrained Image Encoder in SAM for encoding. Meanwhile, the Prompt Generator generates bounding box prompt based on the tongue image. Finally, the image embedding and prompts are jointly fed into the Mask Decoder to generate the segmentation result. The entire segmentation process is end-to-end and does not require any additional manual prompts. The following sections will introduce different components of TongueSAM.

<p align="center">
    <img src="https://github.com/cshan-github/TongueSAM/blob/main/1.jpg" alt="The model structure of TonguSAM." width="600" height="300">


## Result

<p align="center">
    <img src="https://github.com/cshan-github/TongueSAM/blob/main/4.jpg" alt="The model structure of TonguSAM."width="1000" height="1000">

## Project Description

**1.Zero-Shot Segmentation**

The most crucial capability of TongueSAM lies in its Zero-Shot segmentation. To facilitate user adoption, we employed the three datasets mentioned in the paper for fine-tuning TongueSAM and openly released the pre-trained model. Users can perform tongue image segmentation directly using TongueSAM with just a few straightforward steps.

Download the pre-trained weights:[TongueSAM](https://pan.baidu.com/s/1gaVQRipZUgg5WcZE6fDfyw?pwd=1209)

Place the ```tonguesam.pth``` into the ```./pretrained_model/``` folder.

Place the tongue image files that need to be segmented into the ```./data/test_in/``` folder.

Run ```./python.py```

The segmented tongue images will be located in the ```./data/test_out/``` folder.

**2.Fine-tune**

If you wish to further fine-tune the model, please follow these steps:

To train the Prompt Generator based on YOLOX, please refer to the following guidelines:[YOLOX](https://github.com/bubbliiiing/yolox-pytorch)

Replace the pre-trained model in the ```./segment/yolox.pth``` file with your trained model.

Run ```./train.py```,please refer to the following guidelines:[MedSAM](https://github.com/bowang-lab/MedSAM)

## Acknowledge

The project is based on [YOLOX](https://github.com/bubbliiiing/yolox-pytorch) and [MedSAM](https://github.com/bowang-lab/MedSAM), and we appreciate their contributions.

## Future Work

Although TongueSAM performs well in zero-shot and challenging environments, its model architecture is overly large, which can impact inference times. To address this issue, we will be releasing a more lightweight version in the near future, making TongueSAM smaller in size and faster in inference speed. Thanks for your interest in our study.
