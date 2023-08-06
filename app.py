import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2 #np.array -> torch.tensor
import os
from PIL import Image, ImageOps
import torchvision
from torchvision import transforms


#1. model UNet
def unet_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        nn.ReLU()
    )
class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.block_down1 = unet_block(3, 64)
        self.block_down2 = unet_block(64, 128)
        self.block_down3 = unet_block(128, 256)
        self.block_down4 = unet_block(256, 512)
        self.block_neck = unet_block(512, 1024)
        self.block_up1 = unet_block(1024+512, 512)
        self.block_up2 = unet_block(256+512, 256)
        self.block_up3 = unet_block(128+256, 128)
        self.block_up4 = unet_block(128+64, 64)
        self.conv_cls = nn.Conv2d(64, self.n_classes, 1) # -> (B, n_class, H, W)

    def forward(self, x):
        # (B, C, H, W)
        x1 = self.block_down1(x)
        x = self.downsample(x1)
        x2 = self.block_down2(x)
        x = self.downsample(x2)
        x3 = self.block_down3(x)
        x = self.downsample(x3)
        x4 = self.block_down4(x)
        x = self.downsample(x4)

        x = self.block_neck(x)

        x = torch.cat([x4, self.upsample(x)], dim=1)
        x = self.block_up1(x)
        x = torch.cat([x3, self.upsample(x)], dim=1)
        x = self.block_up2(x)
        x = torch.cat([x2, self.upsample(x)], dim=1)
        x = self.block_up3(x)
        x = torch.cat([x1, self.upsample(x)], dim=1)
        x = self.block_up4(x)

        x = self.conv_cls(x)
        return x
#4. Load model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = UNet(1).to(device)
model.load_state_dict(torch.load('model_ep_28.pth', map_location=device))
model.to(device)
model.eval()
  
#2. Normalization
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

#3. Streamlit
st.title('Dog, Cat Segmentation')
st.header('Please upload your image')
file = st.file_uploader('', type = ['jpeg', 'jpg', 'png'])

if file is not None:
    img = Image.open(file)
    transform = torchvision.transforms.Resize(size=(384, 384))
    img = transform(img)
    img_tensor = torchvision.transforms.ToTensor()(img)
    normalized_img_tensor = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(img_tensor)
    normalized_img_tensor = normalized_img_tensor.to(device).float().unsqueeze(0)
    y_hat = model(normalized_img_tensor).squeeze()
    y_hat_mask = y_hat.sigmoid().round().long()
    fig1 = plt.figure(figsize = (3,3))
    plt.subplot(1, 2, 1)
    plt.imshow(unorm(normalized_img_tensor.squeeze().cpu()).permute(1, 2, 0)) # x (GPU) -> x(CPU)
    st.pyplot(fig1)
    fig2 = plt.figure(figsize = (3,3))
    plt.subplot(1, 2, 2)
    plt.imshow(y_hat_mask.cpu())
    st.pyplot(fig2)
else:
    st.text("Please upload your image file")

    


    



