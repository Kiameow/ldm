from torchvision import datasets, transforms
import json
import argparse
import os
import re
import torch.nn.functional as F

class ResizeWithPadding:
    def __init__(self, target_size=(256, 256)):
        # 确保target_size是tuple
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            self.target_size = target_size

    def __call__(self, image):
        width, height = image.size
        target_width, target_height = self.target_size
        
        # 计算缩放比例
        width_ratio = target_width / width
        height_ratio = target_height / height
        ratio = min(width_ratio, height_ratio)
        
        # 计算新尺寸
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        # 缩放图像
        resize_transform = transforms.Resize(
            (new_height, new_width), 
            transforms.InterpolationMode.BILINEAR
        )
        resized_image = resize_transform(image)
        
        # 计算填充尺寸
        pad_left = (target_width - new_width) // 2
        pad_top = (target_height - new_height) // 2
        pad_right = target_width - new_width - pad_left
        pad_bottom = target_height - new_height - pad_top
        
        # 填充图像
        pad_transform = transforms.Pad((pad_left, pad_top, pad_right, pad_bottom), fill=0)
        padded_image = pad_transform(resized_image)
        
        return padded_image

def load_config(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='args/01.json', help='Path to the config file')
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--initial_lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--output_interval", type=int, default=10, help="debug info show")
    parser.add_argument("--save_interval", type=int, default=10, help="Save model every N epochs")
    parser.add_argument("--save_dir", type=str, default="ckpts", help="Directory to save checkpoints")
    parser.add_argument("--save_ae_dir", type=str, default="ckpts_ae", help="Directory to save ae checkpoints")
    parser.add_argument("--ae_sample_dir", type=str, default="results_ae", help="Directory to save ae samples")
    parser.add_argument("--sample_dir", type=str, default="results_ldm", help="Directory to save ldm samples")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="accumulation step")
    parser.add_argument("--ldm_sample_t", type=int, default=100, help="ldm add noise and denoise step")
    parser.add_argument("--sample_batch_size", type=int, default=1, help="sample batch size")
    parser.add_argument("--no_clip", type=bool, default=True, help="whether to use clip")
    args = parser.parse_args()

    # 读取 JSON 配置
    config = load_config(args.config)

    # 解析参数
    for key, value in config.items():
        setattr(args, key, value)

    return args

def get_latest_ckpt_path(dataset_name, save_dir):
    if not os.path.exists(save_dir):
        return None
    # 假设所有的ckpt文件夹都在当前目录下
    ckpt_folders = [f for f in os.listdir(save_dir) if f.startswith('ckpt-')]
    
    # 过滤出与dataset_name相关的文件夹
    related_folders = [f for f in ckpt_folders if f.split('-')[1] == dataset_name]
    
    if not related_folders:
        return None  # 如果没有找到相关的文件夹，返回None
    
    # 提取步骤数并找到最新的文件夹
    latest_folder = max(related_folders, key=lambda x: int(re.search(r'\d+', x).group()))
    
    # 返回完整的路径
    return os.path.join(save_dir, latest_folder)

def visualize_latents(tensor, target_size):
    """
    将输入的 latent tensor 处理成目标大小的单通道图像。
    
    参数:
        tensor (torch.Tensor): 输入张量，形状为 [B, C, H, W]。
        target_size (tuple): 目标大小，格式为 (height, width)。
    
    返回:
        torch.Tensor: 处理后的张量，形状为 [B, 1, target_height, target_width]。
    """
    B, C, H, W = tensor.shape
    
    # 自动判断通道数并拼接
    if C == 4:
        # 将 4 个通道拼接成 2x2 的网格
        combined = tensor.view(B, 2, 2, H, W)  # [B, 2, 2, H, W]
        combined = combined.permute(0, 1, 3, 2, 4).contiguous()  # [B, 2, H, 2, W]
        combined = combined.view(B, 1, 2 * H, 2 * W)  # [B, 1, 2H, 2W]
    elif C == 1:
        # 如果已经是单通道，直接使用
        combined = tensor
    else:
        raise ValueError(f"Unsupported number of channels: {C}. Expected 1 or 4.")
    
    # 将图像放缩到目标大小
    resized = F.interpolate(combined, size=target_size, mode='bilinear', align_corners=False)
    
    return resized
    

if __name__ == "__main__":
    args = parse_args()
    print(args.batch_size, args.learning_rate) 


