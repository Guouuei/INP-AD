import argparse
from functools import partial
import gradio as gr
from torch.nn import functional as F
from torch import nn
from dataset import get_data_transforms
from PIL import Image
import os

from utils import get_gaussian_kernel

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import os
import torch
import cv2
import numpy as np

# # Model-Related Modules
from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block


# Configurations
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
parser = argparse.ArgumentParser(description='')
# model info
parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14')
parser.add_argument('--input_size', type=int, default=448)
parser.add_argument('--crop_size', type=int, default=392)
parser.add_argument('--INP_num', type=int, default=6)

args = parser.parse_args()


############ Init Model
ckt_path1 = 'saved_results/INP-Former-Multi-Class_dataset=Real-IAD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/model.pth'
ckt_path2 = "saved_results/INP-Former-Multi-Class_dataset=VisA_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/model.pth"

#
data_transform, _ = get_data_transforms(args.input_size, args.crop_size)

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Adopting a grouping-based reconstruction strategy similar to Dinomaly
target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

# Encoder info
encoder = vit_encoder.load(args.encoder)
if 'small' in args.encoder:
    embed_dim, num_heads = 384, 6
elif 'base' in args.encoder:
    embed_dim, num_heads = 768, 12
elif 'large' in args.encoder:
    embed_dim, num_heads = 1024, 16
    target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
else:
    raise "Architecture not in small, base, large."

# Model Preparation
Bottleneck = []
INP_Guided_Decoder = []
INP_Extractor = []

# bottleneck
Bottleneck.append(Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.))
Bottleneck = nn.ModuleList(Bottleneck)

# INP
INP = nn.ParameterList(
    [nn.Parameter(torch.randn(args.INP_num, embed_dim))
     for _ in range(1)])

# INP Extractor
for i in range(1):
    blk = Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
    INP_Extractor.append(blk)
INP_Extractor = nn.ModuleList(INP_Extractor)

# INP_Guided_Decoder
for i in range(8):
    blk = Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                          qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
    INP_Guided_Decoder.append(blk)
INP_Guided_Decoder = nn.ModuleList(INP_Guided_Decoder)

model = INP_Former(encoder=encoder, bottleneck=Bottleneck, aggregation=INP_Extractor, decoder=INP_Guided_Decoder,
                   target_layers=target_layers, remove_class_token=True, fuse_layer_encoder=fuse_layer_encoder,
                   fuse_layer_decoder=fuse_layer_decoder, prototype_token=INP)
model = model.to(device)

gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)


def resize_and_center_crop(image, resize_size=448, crop_size=392):
    # Resize to 448x448
    image_resized = cv2.resize(image, (resize_size, resize_size), interpolation=cv2.INTER_LINEAR)

    # Compute crop coordinates
    start = (resize_size - crop_size) // 2
    end = start + crop_size

    # Center crop to 392x392
    image_cropped = image_resized[start:end, start:end, :]

    return image_cropped

def process_image(image, options):
    # Load the model based on selected options
    if 'Real-IAD' in options:
        model.load_state_dict(torch.load(ckt_path1), strict=True)
    elif 'VisA' in options:
        model.load_state_dict(torch.load(ckt_path2), strict=True)
    else:
        # Default to 'All' if no valid option is provided
        model.load_state_dict(torch.load(ckt_path1), strict=True)
        print('Invalid option. Defaulting to All.')

    # Ensure image is in RGB mode
    image = image.convert('RGB')



    # Convert PIL image to NumPy array
    np_image = np.array(image)
    image_shape = np_image.shape[0]

    # Convert RGB to BGR for OpenCV
    np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    np_image = resize_and_center_crop(np_image, resize_size=args.input_size, crop_size=args.crop_size)

    # Preprocess the image and run the model
    input_image = data_transform(image)
    input_image = input_image.to(device)

    with torch.no_grad():
        _ = model(input_image.unsqueeze(0))
        anomaly_map = model.distance
        side = int(model.distance.shape[1] ** 0.5)
        anomaly_map = anomaly_map.reshape([anomaly_map.shape[0], side, side]).contiguous()
        anomaly_map = torch.unsqueeze(anomaly_map, dim=1)
        anomaly_map = F.interpolate(anomaly_map, size=input_image.shape[-1], mode='bilinear', align_corners=True)
        anomaly_map = gaussian_kernel(anomaly_map)

    # Process anomaly map
    anomaly_map = anomaly_map.squeeze().cpu().numpy()
    anomaly_map = (anomaly_map * 255).astype(np.uint8)

    # Apply color map and blend with original image
    heat_map = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
    vis_map = cv2.addWeighted(heat_map, 0.5, np_image, 0.5, 0)

    # Convert OpenCV image back to PIL image for Gradio
    vis_map_pil = Image.fromarray(cv2.resize(cv2.cvtColor(vis_map, cv2.COLOR_BGR2RGB), (image_shape, image_shape)))

    return vis_map_pil

# Define examples
examples = [
    ["assets/img2.png", "Real-IAD"],
    ["assets/img.png", "VisA"]
]

# Gradio interface layout
demo = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Radio(["Real-IAD",
                  "VisA"],
        label="Pre-trained Datasets")
    ],
    outputs=[
        gr.Image(type="pil", label="Output Image")
    ],
    examples=examples,
    title="INP-Former -- Zero-shot Anomaly Detection",
    description="Upload an image and select pre-trained datasets to do zero-shot anomaly detection"
)

# Launch the demo
demo.launch()
# demo.launch(server_name="0.0.0.0", server_port=10002)