import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Brain Tumor Dashboard", layout="wide")
st.title("Brain Tumor Segmentation Comparison")

# ==================================================
# BASE MODEL
# ==================================================
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        c1 = self.enc1(x)
        c2 = self.enc2(self.pool1(c1))
        c3 = self.enc3(self.pool2(c2))
        bn = self.bottleneck(self.pool3(c3))

        d3 = self.up3(bn)
        d3 = self.dec3(torch.cat([d3, c3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, c2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, c1], dim=1))

        return torch.sigmoid(self.final(d1))

# ==================================================
# IMPROVED MODEL
# ==================================================
class ConvBlock(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,3,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x)

class AttentionBlock(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g,F_int,1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l,F_int,1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int,1,1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        psi = self.relu(self.W_g(g)+self.W_x(x))
        psi = self.psi(psi)
        return x*psi

class PyramidPooling(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

    def forward(self,x):
        size = x.size()[2:]
        p1 = F.interpolate(self.pool1(x), size=size, mode="bilinear", align_corners=False)
        p2 = F.interpolate(self.pool2(x), size=size, mode="bilinear", align_corners=False)
        p3 = F.interpolate(self.pool3(x), size=size, mode="bilinear", align_corners=False)
        p4 = F.interpolate(self.pool4(x), size=size, mode="bilinear", align_corners=False)
        return torch.cat([x,p1,p2,p3,p4], dim=1)

class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = ConvBlock(1,64)
        self.e2 = ConvBlock(64,128)
        self.e3 = ConvBlock(128,256)
        self.e4 = ConvBlock(256,512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(512,1024)
        self.ppm = PyramidPooling(1024)

        self.up4 = nn.ConvTranspose2d(5120,512,2,2)
        self.att4 = AttentionBlock(512,512,256)
        self.d4 = ConvBlock(1024,512)

        self.up3 = nn.ConvTranspose2d(512,256,2,2)
        self.att3 = AttentionBlock(256,256,128)
        self.d3 = ConvBlock(512,256)

        self.up2 = nn.ConvTranspose2d(256,128,2,2)
        self.att2 = AttentionBlock(128,128,64)
        self.d2 = ConvBlock(256,128)

        self.up1 = nn.ConvTranspose2d(128,64,2,2)
        self.att1 = AttentionBlock(64,64,32)
        self.d1 = ConvBlock(128,64)

        self.out = nn.Conv2d(64,1,1)

    def forward(self,x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        b = self.ppm(b)

        d4 = self.up4(b)
        d4 = self.d4(torch.cat([d4,self.att4(d4,e4)],1))

        d3 = self.up3(d4)
        d3 = self.d3(torch.cat([d3,self.att3(d3,e3)],1))

        d2 = self.up2(d3)
        d2 = self.d2(torch.cat([d2,self.att2(d2,e2)],1))

        d1 = self.up1(d2)
        d1 = self.d1(torch.cat([d1,self.att1(d1,e1)],1))

        return torch.sigmoid(self.out(d1))

# ==================================================
# LOAD MODELS
# ==================================================
# @st.cache_resource
# def load_models():
#     # Base model = weights only
#     base = UNet()
#     base.load_state_dict(
#         torch.load("base_model.pth", map_location="cpu", weights_only=False)
#     )
#     base.eval()

#     # Improved model = full saved model
#     improved = torch.load(
#         "brain_tumor_model_full.pth",
#         map_location="cpu",
#         weights_only=False
#     )
#     improved.eval()

#     return base, improved

@st.cache_resource
def load_models():
    base = UNet()
    improved = AttentionUNet()

    try:
        base.load_state_dict(torch.load("base_model.pth", map_location="cpu", weights_only=False))
        st.success("Base model loaded")
    except Exception as e:
        st.error(f"Base failed: {e}")

    try:
        improved.load_state_dict(torch.load("attention_unet_brain_tumor.pth", map_location="cpu", weights_only=False))
        st.success("Improved model loaded")
    except Exception as e:
        st.error(f"Improved failed: {e}")

    base.eval()
    improved.eval()
    return base, improved

base_model, your_model = load_models()

# ==================================================
# PREPROCESS
# ==================================================
def preprocess(image):
    img = np.array(image.resize((256,256)))

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = gray / 255.0
    gray_tensor = torch.tensor(gray, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    rgb = img / 255.0
    rgb = np.transpose(rgb, (2,0,1))
    rgb_tensor = torch.tensor(rgb, dtype=torch.float32).unsqueeze(0)

    return img, rgb_tensor, gray_tensor

# def predict(model, tensor):
#     with torch.no_grad():
#         out = model(tensor)
#     mask = (out[0,0].numpy() > 0.5).astype(np.uint8)
#     return mask
def predict(model, tensor, threshold=0.1, debug=False):
    with torch.no_grad():
        out = model(tensor)

    arr = out[0,0].cpu().numpy()

    if debug:
        st.write("Min:", arr.min(), "Max:", arr.max(), "Mean:", arr.mean())
        st.image(arr, caption="Raw Output Heatmap", clamp=True)

    mask = (arr > threshold).astype(np.uint8)
    return mask

# ==================================================
# UI
# ==================================================
uploaded = st.file_uploader("Upload MRI", type=["png","jpg","jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    raw, rgb_tensor, gray_tensor = preprocess(image)

    # base_mask = predict(base_model, rgb_tensor)
    # your_mask = predict(your_model, gray_tensor)

    base_mask = predict(base_model, rgb_tensor, 0.5)
    your_mask = predict(your_model, gray_tensor, 0.05, debug=True)

    c1,c2,c3 = st.columns(3)
    c1.image(raw, caption="Original MRI", use_container_width=True)
    c2.image(base_mask*255, caption="Base Model", use_container_width=True)
    c3.image(your_mask*255, caption="Improved Model", use_container_width=True)