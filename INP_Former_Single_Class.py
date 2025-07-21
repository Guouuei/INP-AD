import torch
import torch.nn as nn
import numpy as np
import os
from functools import partial
import warnings
from tqdm import tqdm
from torch.nn.init import trunc_normal_
import argparse
from optimizers import StableAdamW # 假设这个优化器定义在 optimizers.py
from utils import evaluation_batch, WarmCosineScheduler, global_cosine_hm_adaptive, setup_seed, get_logger # 假设这些工具函数定义在 utils.py
import json # 用于保存/加载优化器和调度器状态（如果需要更复杂的保存）

# 数据集相关模块
from dataset import MVTecDataset, RealIADDataset # 假设这些数据集类定义在 dataset.py
from dataset import get_data_transforms # 假设这个函数定义在 dataset.py
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# 模型相关模块
from models import vit_encoder # 假设这个模块定义在 models/vit_encoder.py
from models.uad import INP_Former # 假设这个模型定义在 models/uad.py
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block # 假设这些模块定义在 models/vision_transformer.py


warnings.filterwarnings("ignore") # 忽略警告信息

def main(args):
    # 固定随机种子
    setup_seed(1)
    # 数据准备
    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)

    if args.dataset == 'MVTec-AD' or args.dataset == 'VisA':
        train_path = os.path.join(args.data_path, args.item, 'train')
        test_path = os.path.join(args.data_path, args.item)

        train_data = ImageFolder(root=train_path, transform=data_transform)
        test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                                       drop_last=True)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    elif args.dataset == 'Real-IAD' :
        train_data = RealIADDataset(root=args.data_path, category=args.item, transform=data_transform, gt_transform=gt_transform,
                                    phase='train')
        test_data = RealIADDataset(root=args.data_path, category=args.item, transform=data_transform, gt_transform=gt_transform,
                                   phase="test")
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                                       drop_last=True)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 采用类似于 Dinomaly 的基于分组的重建策略
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    # Encoder 信息
    encoder = vit_encoder.load(args.encoder)
    if 'small' in args.encoder:
        embed_dim, num_heads = 384, 6
    elif 'base' in args.encoder:
        embed_dim, num_heads = 768, 12
    elif 'large' in args.encoder:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18] # 对于 large 模型，目标层可能不同
    else:
        raise ValueError("模型架构必须是 small, base, large 中的一种。")

    # 模型准备
    Bottleneck = []
    INP_Guided_Decoder = []
    INP_Extractor = []

    # bottleneck
    Bottleneck.append(Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.))
    Bottleneck = nn.ModuleList(Bottleneck)

    # INP (Implicit Neural Prototypes)
    INP = nn.ParameterList(
                    [nn.Parameter(torch.randn(args.INP_num, embed_dim))
                     for _ in range(1)]) # 假设只有一个INP组

    # INP Extractor
    for i in range(1): # 对应INP组的数量
        blk = Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        INP_Extractor.append(blk)
    INP_Extractor = nn.ModuleList(INP_Extractor)

    # INP_Guided_Decoder
    for i in range(8): # 对应 target_layers 的数量
        blk = Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        INP_Guided_Decoder.append(blk)
    INP_Guided_Decoder = nn.ModuleList(INP_Guided_Decoder)

    model = INP_Former(encoder=encoder, bottleneck=Bottleneck, aggregation=INP_Extractor, decoder=INP_Guided_Decoder,
                             target_layers=target_layers,  remove_class_token=True, fuse_layer_encoder=fuse_layer_encoder,
                             fuse_layer_decoder=fuse_layer_decoder, prototype_token=INP)
    model = model.to(device)

    if args.phase == 'train':
        # 定义可训练的模块
        trainable = nn.ModuleList([Bottleneck, INP_Guided_Decoder, INP_Extractor, INP])

        # 定义优化器
        optimizer = StableAdamW([{'params': trainable.parameters()}], # 只优化指定的可训练参数
                                lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
        # 定义学习率调度器
        # 注意：如果从中间恢复训练，total_iters 应该是整个训练过程的总迭代次数
        lr_scheduler = WarmCosineScheduler(optimizer, base_value=1e-3, final_value=1e-4,
                                           total_iters=args.total_epochs * len(train_dataloader),
                                           warmup_iters=100) # 预热迭代次数

        start_epoch = 0 # 初始化起始 epoch

        # --- 继续训练代码 ---
        if args.resume_checkpoint:
            checkpoint_path = os.path.join(args.save_dir, args.save_name, args.item, args.resume_checkpoint)
            if os.path.isfile(checkpoint_path):
                print_fn(f"=> 正在加载检查点 '{checkpoint_path}'")
                checkpoint_data = torch.load(checkpoint_path, map_location=device)  # Load the file

                # Check if the loaded data is a dictionary and contains 'model_state_dict'
                if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
                    model.load_state_dict(checkpoint_data['model_state_dict'])
                    print_fn("=> 已从检查点字典加载模型状态")
                    if 'optimizer_state_dict' in checkpoint_data:
                        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                        print_fn("=> 已加载优化器状态")
                    if 'scheduler_state_dict' in checkpoint_data:
                        lr_scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                        print_fn("=> 已加载调度器状态")
                    if 'epoch' in checkpoint_data:
                        start_epoch = checkpoint_data['epoch'] + 1
                        print_fn(f"=> 已加载 epoch {checkpoint_data['epoch']}, 将从 epoch {start_epoch} 继续训练")
                    print_fn(f"=> 成功加载完整检查点 '{checkpoint_path}' (epoch {checkpoint_data.get('epoch', 'N/A')})")
                elif isinstance(checkpoint_data, dict):  # It's a state_dict itself
                    model.load_state_dict(checkpoint_data)
                    print_fn("=> 已直接加载模型状态字典 (可能不包含优化器/epoch信息)")
                    # Note: If you load a raw state_dict, optimizer, scheduler, and epoch are not resumed from this file.
                    # The training will start from epoch 0 or the previously set start_epoch if this branch is hit
                    # unless you have a separate mechanism or assume fresh start for these components.
                else:
                    print_fn(f"错误：检查点文件 '{checkpoint_path}' 的格式无法识别。")
                    # Decide how to handle this: start from scratch or raise an error
                    # For now, let's proceed to initialize fresh if format is unknown
                    print_fn("将从头开始训练。")
                    start_epoch = 0  # Reset start_epoch
                    # Initialize fresh model if checkpoint format is unknown or only model state was loaded without epoch
                    for m in trainable.modules():
                        if isinstance(m, nn.Linear):
                            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                            if isinstance(m, nn.Linear) and m.bias is not None:
                                nn.init.constant_(m.bias, 0)
                        elif isinstance(m, nn.LayerNorm):
                            nn.init.constant_(m.bias, 0)
                            nn.init.constant_(m.weight, 1.0)

            else:
                print_fn(f"=> 未找到检查点 '{checkpoint_path}'，将从头开始训练。")
                # Initialize fresh model if no checkpoint
                for m in trainable.modules():
                    if isinstance(m, nn.Linear):
                        trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                        if isinstance(m, nn.Linear) and m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.LayerNorm):
                        nn.init.constant_(m.bias, 0)
                        nn.init.constant_(m.weight, 1.0)
        else:
            # Original initialization if not resuming
            print_fn("=> 不加载检查点，将从头开始训练并初始化权重。")
            for m in trainable.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
        # --- 继续训练代码结束 ---

        print_fn('训练图片数量:{}'.format(len(train_data)))

        # 训练过程
        for epoch in range(start_epoch, args.total_epochs): # 从 start_epoch 开始
            model.train() # 设置模型为训练模式
            loss_list = []
            # 使用 tqdm 显示进度条
            loop = tqdm(train_dataloader, ncols=80, desc=f"Epoch [{epoch+1}/{args.total_epochs}]")
            for img, _ in loop:
                img = img.to(device)
                en, de, g_loss = model(img) # 前向传播
                loss = global_cosine_hm_adaptive(en, de, y=3) # 计算主损失
                loss = loss + 0.2 * g_loss # 加入辅助损失

                optimizer.zero_grad() # 清空梯度
                loss.backward() # 反向传播
                # 对可训练参数进行梯度裁剪，防止梯度爆炸
                nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
                optimizer.step() # 更新参数

                loss_list.append(loss.item())
                lr_scheduler.step() # 每个 iteration 更新学习率
                # 更新tqdm的后缀显示
                loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

            avg_loss = np.mean(loss_list) if loss_list else 0
            print_fn('epoch [{}/{}], loss:{:.4f}, lr:{:.6f}'.format(epoch + 1, args.total_epochs, avg_loss, optimizer.param_groups[0]['lr']))

            # --- 保存检查点 ---
            # 在每个保存间隔或最后一个epoch保存检查点
            if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.total_epochs:
                checkpoint_save_dir = os.path.join(args.save_dir, args.save_name, args.item)
                os.makedirs(checkpoint_save_dir, exist_ok=True) # 确保目录存在
                checkpoint_save_path = os.path.join(checkpoint_save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(), # 保存整个模型的状态
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'args': args # 可选：保存当时的命令行参数
                }, checkpoint_save_path)
                print_fn(f"检查点已保存至 {checkpoint_save_path}")
            # --- 保存检查点结束 ---

        # 训练结束后保存最终模型
        final_model_save_dir = os.path.join(args.save_dir, args.save_name, args.item)
        os.makedirs(final_model_save_dir, exist_ok=True) # 确保目录存在
        final_model_save_path = os.path.join(final_model_save_dir, 'model_final.pth')
        torch.save(model.state_dict(), final_model_save_path) # 只保存模型权重
        print_fn(f"最终模型已保存至 {final_model_save_path}")

        # 评估模型
        results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
        print_fn(
            '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                args.item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))
        return results

    elif args.phase == 'test':
        # 测试阶段
        # 允许指定用于测试的模型文件
        model_path_to_load = os.path.join(args.save_dir, args.save_name, args.item, args.test_model_file)
        if not os.path.isfile(model_path_to_load):
            print_fn(f"错误: 模型文件未找到于 {model_path_to_load}")
            return [0.0] * 7 # 返回虚拟结果或抛出错误

        print_fn(f"正在从以下路径加载模型进行测试: {model_path_to_load}")
        # 加载模型权重，需要区分是检查点字典还是单纯的模型状态字典
        checkpoint_or_state_dict = torch.load(model_path_to_load, map_location=device)
        if isinstance(checkpoint_or_state_dict, dict) and 'model_state_dict' in checkpoint_or_state_dict:
            model.load_state_dict(checkpoint_or_state_dict['model_state_dict']) # 从检查点字典中加载
        else:
            model.load_state_dict(checkpoint_or_state_dict) # 直接加载状态字典

        model.eval() # 设置模型为评估模式
        results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
        return results


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # 设置CUDA环境变量，便于调试
    parser = argparse.ArgumentParser(description='INP-Former 训练与评估脚本')

    # 数据集信息
    parser.add_argument('--dataset', type=str, default=r'MVTec-AD', choices=['MVTec-AD', 'VisA', 'Real-IAD'], help="选择数据集")
    parser.add_argument('--data_path', type=str, default=r'F:\INP-Former-main\Mydatasets\dianrong', help="数据集路径，请替换为你的路径") # 示例路径，请修改

    # 保存信息
    parser.add_argument('--save_dir', type=str, default='./saved_results', help="结果保存目录")
    parser.add_argument('--save_name', type=str, default='INP-Former-Single-Class', help="保存名称前缀")
    parser.add_argument('--save_interval', type=int, default=50, help="每 N 个 epoch 保存一次检查点")

    # 模型信息
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14',
                        choices=['dinov2reg_vit_small_14', 'dinov2reg_vit_base_14', 'dinov2reg_vit_large_14'], help="选择Encoder类型")
    parser.add_argument('--input_size', type=int, default=392, help="输入图像尺寸")
    parser.add_argument('--crop_size', type=int, default=392, help="裁剪图像尺寸")
    parser.add_argument('--INP_num', type=int, default=6, help="INP (Implicit Neural Prototypes) 的数量")

    # 训练信息
    parser.add_argument('--total_epochs', type=int, default=200, help="总训练轮数")
    parser.add_argument('--batch_size', type=int, default=16, help="批量大小")
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test'], help="选择 'train' 或 'test' 模式")
    parser.add_argument('--resume_checkpoint', type=str, default=r'model.pth', help="要从中恢复训练的检查点文件名 (例如: checkpoint_epoch_50.pth)，应位于 item 对应的保存目录下")
    parser.add_argument('--test_model_file', type=str, default='model_final.pth', help="测试时使用的模型文件名 (例如: model_final.pth 或 checkpoint_epoch_200.pth)")


    args = parser.parse_args()

    # 根据参数动态生成完整的保存名称
    args.save_name = args.save_name + f'_dataset={args.dataset}_Encoder={args.encoder}_Resize={args.input_size}_Crop={args.crop_size}_INP_num={args.INP_num}'

    # 设置日志记录器
    # 日志文件会保存在 args.save_dir / args.save_name / logfile.log
    log_dir = os.path.join(args.save_dir, args.save_name)
    os.makedirs(log_dir, exist_ok=True) # 确保日志目录存在
    logger = get_logger(args.save_name, log_dir)
    print_fn = logger.info # 使用logger.info替代print

    # 设置设备
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print_fn(f"使用设备: {device}")
    print_fn(f"命令行参数: {args}")


    # 类别信息
    if args.dataset == 'MVTec-AD':
        # 示例：这里列出MVTec-AD所有类别，你可以根据需要修改或只选一个
        #args.item_list = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
        args.item_list = ['front'] # 如果只想训练 'front' 这个类别
    elif args.dataset == 'VisA':
        args.item_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    elif args.dataset == 'Real-IAD':
        args.item_list = ['audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
                 'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
                 'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
                 'terminalblock', 'toothbrush', 'toy', 'toy_brick', 'transistor1', 'usb',
                 'usb_adaptor', 'u_block', 'vcpill', 'wooden_beads', 'woodstick', 'zipper']
    else:
        args.item_list = None # 或者你可以提供一个默认的 item，或者让用户必须通过命令行指定

    # 检查是否有有效的 item_list
    if not hasattr(args, 'item_list') or not args.item_list:
        if hasattr(args, 'item') and args.item: # 如果通过 --item 指定了单个类别
             args.item_list = [args.item]
        else:
            parser.error("请为指定的数据集提供 item_list，或通过 --item 参数指定单个类别。")


    result_list = [] # 存储每个类别的结果

    # 遍历所有指定的类别进行训练或测试
    for item_name in args.item_list:
        args.item = item_name # 设置当前处理的类别
        print_fn(f"\n正在处理数据集: {args.dataset}中的类别: {args.item}")

        # 为每个item创建特定的保存目录，用于存放该item的检查点和最终模型
        item_specific_save_dir = os.path.join(args.save_dir, args.save_name, args.item)
        os.makedirs(item_specific_save_dir, exist_ok=True)

        results = main(args) # 调用主函数执行训练或测试
        if results: # main 函数在测试失败或某些情况下可能返回 None
            auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
            result_list.append([args.item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px])
            # 打印当前类别的结果
            print_fn(
            f'{args.item} 的结果: I-Auroc:{auroc_sp:.4f}, I-AP:{ap_sp:.4f}, I-F1:{f1_sp:.4f}, P-AUROC:{auroc_px:.4f}, P-AP:{ap_px:.4f}, P-F1:{f1_px:.4f}, P-AUPRO:{aupro_px:.4f}')
        else:
            print_fn(f"类别 {args.item} 未返回结果。")


    # 计算并打印所有类别的平均结果
    if result_list: # 确保 result_list 不为空
        mean_auroc_sp = np.mean([result[1] for result in result_list])
        mean_ap_sp = np.mean([result[2] for result in result_list])
        mean_f1_sp = np.mean([result[3] for result in result_list])

        mean_auroc_px = np.mean([result[4] for result in result_list])
        mean_ap_px = np.mean([result[5] for result in result_list])
        mean_f1_px = np.mean([result[6] for result in result_list])
        mean_aupro_px = np.mean([result[7] for result in result_list])

        print_fn("\n--- 所有类别结果汇总 ---")
        for res_idx, res_val in enumerate(result_list): # 修正索引变量名
            print_fn(f'{res_val[0]}: I-Auroc:{res_val[1]:.4f}, I-AP:{res_val[2]:.4f}, I-F1:{res_val[3]:.4f}, P-AUROC:{res_val[4]:.4f}, P-AP:{res_val[5]:.4f}, P-F1:{res_val[6]:.4f}, P-AUPRO:{res_val[7]:.4f}')

        print_fn(
            '\n平均结果: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                mean_auroc_sp, mean_ap_sp, mean_f1_sp,
                mean_auroc_px, mean_ap_px, mean_f1_px, mean_aupro_px))
    else:
        print_fn("没有结果可以汇总。")