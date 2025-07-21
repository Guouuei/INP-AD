import torch
import torch.nn as nn
import numpy as np
import os
from functools import partial
import warnings
import argparse
from utils import setup_seed, get_logger, evaluation_batch_vis_ZS

# Dataset-Related Modules
from dataset import MVTecDataset, RealIADDataset
from dataset import get_data_transforms
from torch.utils.data import DataLoader

# Model-Related Modules
from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block


warnings.filterwarnings("ignore")
def main(args):
    # Fixing the Random Seed
    setup_seed(1)

    # Data Preparation
    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)

    # Only the test set is needed
    if args.dataset == 'MVTec-AD' or args.dataset == 'VisA':
        test_data_list = []
        for i, item in enumerate(args.item_list):
            test_path = os.path.join(args.data_path, item)
            test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
            test_data_list.append(test_data)
    elif args.dataset == 'Real-IAD' :
        test_data_list = []
        for i, item in enumerate(args.item_list):
            test_data = RealIADDataset(root=args.data_path, category=item, transform=data_transform,
                                       gt_transform=gt_transform,
                                       phase="test")
            test_data_list.append(test_data)

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
                             target_layers=target_layers,  remove_class_token=True, fuse_layer_encoder=fuse_layer_encoder,
                             fuse_layer_decoder=fuse_layer_decoder, prototype_token=INP)
    model = model.to(device)



    # Zero-Shot Test
    model.load_state_dict(torch.load(os.path.join(args.save_dir, args.source_save_name, 'model.pth')), strict=True) # Load weights pre-trained on the source dataset
    auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
    auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []
    model.eval()
    for item, test_data in zip(args.item_list, test_data_list):
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
        results = evaluation_batch_vis_ZS(model, test_dataloader, device, max_ratio=0.01, resize_mask=256, save_root=os.path.join(args.save_dir, args.save_name, 'imgs'))
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
        auroc_sp_list.append(auroc_sp)
        ap_sp_list.append(ap_sp)
        f1_sp_list.append(f1_sp)
        auroc_px_list.append(auroc_px)
        ap_px_list.append(ap_px)
        f1_px_list.append(f1_px)
        aupro_px_list.append(aupro_px)
        print_fn(
            '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))

    print_fn(
        'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
            np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
            np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = argparse.ArgumentParser(description='')

    # source dataset info
    parser.add_argument('--source_dataset', type=str, default=r'Real-IAD') # 'VisA' or 'Real-IAD'

    # target dataset info
    parser.add_argument('--dataset', type=str, default=r'MVTec-AD') # 'MVTec-AD'
    parser.add_argument('--data_path', type=str, default=r'E:\IMSN-LW\dataset\mvtec_anomaly_detection') # Replace it with your path.

    # save info
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--source_save_name', type=str, default='INP-Former-Multi-Class') # For loading pre-trained weights.
    parser.add_argument('--save_name', type=str, default='INP-Former-Zero-Shot')

    # model info
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14') # 'dinov2reg_vit_small_14' or 'dinov2reg_vit_base_14' or 'dinov2reg_vit_large_14'
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size', type=int, default=392)
    parser.add_argument('--INP_num', type=int, default=6)

    # training info
    parser.add_argument('--batch_size', type=int, default=16)

    args = parser.parse_args()
    args.source_save_name = args.source_save_name + f'_dataset={args.source_dataset}_Encoder={args.encoder}_Resize={args.input_size}_Crop={args.crop_size}_INP_num={args.INP_num}'
    args.save_name = args.save_name + f'_Source={args.source_dataset}_Target={args.dataset}'
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # target category info
    if args.dataset == 'MVTec-AD':
        # args.data_path = 'E:\IMSN-LW\dataset\mvtec_anomaly_detection' # '/path/to/dataset/MVTec-AD/'
        args.item_list = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
                 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    elif args.dataset == 'VisA':
        # args.data_path = r'E:\IMSN-LW\dataset\VisA_pytorch\1cls'  # '/path/to/dataset/VisA/'
        args.item_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    elif args.dataset == 'Real-IAD':
        # args.data_path = 'E:\IMSN-LW\dataset\Real-IAD'  # '/path/to/dataset/Real-IAD/'
        args.item_list = ['audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
                 'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
                 'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
                 'terminalblock', 'toothbrush', 'toy', 'toy_brick', 'transistor1', 'usb',
                 'usb_adaptor', 'u_block', 'vcpill', 'wooden_beads', 'woodstick', 'zipper']
    main(args)
