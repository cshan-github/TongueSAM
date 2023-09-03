# TongueSAM: An Universal Tongue Segmentation Model Based on SAM with Zero-Shot
This is the public project of paper:"TongueSAM: An Universal Tongue Segmentation Model Based on SAM with Zero-Shot", this paper can be get:https://arxiv.org/abs/2308.06444.

**Abstract**

Tongue segmentation serves as the primary step in automated TCM tongue diagnosis, which plays a significant role in the di- agnostic results. Currently, numerous deep learning based methods have achieved promising results. However, most of these methods exhibit mediocre performance on tongues different from the training set. To address this issue, this paper proposes a universal tongue segmentation model named TongueSAM based on SAM (Segment Anything Model). SAM is a large-scale pretrained interactive segmentation model known for its powerful zero-shot generalization capability. Applying SAM to tongue segmentation enables the segmentation of various types of tongue images with zero-shot. In this study, a Prompt Generator based on object detection
is integrated into SAM to enable an end-to-end automated tongue segmentation method. Experiments demonstrate that TongueSAM achieves exceptional performance across various of tongue segmentation datasets, particularly under zero-shot. TongueSAM can be directly applied to other datasets without fine-tuning. As far as we know, this is the first application of large-scale pretrained model for tongue segmentation. 

**Method**

TongueSAM consists primarily of two components: SAM and the Prompt Generator. For a given tongue image, TongueSAM first utilizes the pretrained Image Encoder in SAM for encoding. Meanwhile, the Prompt Generator generates bounding box prompt based on the tongue image. Finally, the image embedding and prompts are jointly fed into the Mask Decoder to generate the segmentation result. The entire segmentation process is end-to-end and does not require any additional manual prompts. The following sections will introduce different components of TongueSAM.

<div style="display: flex; justify-content: center;">
    <img src="https://github.com/cshan-github/TongueSAM/blob/main/1.jpg" alt="The model structure of TonguSAM." width="600" height="300">
</div>

<div style="display: flex; justify-content: center;">
    <img src="https://github.com/cshan-github/TongueSAM/blob/main/4.jpg" alt="The model structure of TonguSAM.">
</div>



**Project Description**

This project encompasses the model architecture of TongueSAM. Additionally, we have provided pretrained model on three tongue image datasets mentioned in the paper to facilitate zero-shot inference for users.


