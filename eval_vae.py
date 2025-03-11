import os
from dataloader import OPMEDDataset, Sample
import torch
from labml_nn.diffusion.stable_diffusion.model.autoencoder import Autoencoder, Encoder, Decoder
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from utils import get_latest_ckpt_path, parse_args
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.utils as vutils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_sample_images(autoencoder: Autoencoder, images: torch.Tensor, batch_idx: int, args: dict):
    """
    保存解码后的图像和原始图像的对比图。
    
    :param autoencoder: Autoencoder 模型
    :param images: 原始图像张量，形状为 [batch_size, channels, height, width]
    :param batch_idx: 当前 batch
    :param save_dir: 保存图像的目录
    """
    autoencoder.eval()  # 切换到评估模式
    with torch.no_grad():
        # 编码并解码图像
        encoded = autoencoder.encode(images)
        decoded = autoencoder.decode(encoded.sample())
        
        # 确保不会超出 batch_size 的范围
        num_images = min(8, images.size(0))
        
        # 将解码后的图像和原始图像拼接在一起
        comparison = torch.cat([images[:num_images], decoded[:num_images]])  # 取前 num_images 张图像进行对比
        comparison = vutils.make_grid(comparison, nrow=num_images, normalize=True, scale_each=True)
        
        # 将张量转换为图像并保存
        plt.figure(figsize=(12, 6))
        plt.imshow(comparison.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        plt.title(f"Epoch {batch_idx}: Original (Top) vs Decoded (Bottom)")
        
        # 保存图像
        sample_dir = os.path.join(args.ae_sample_dir, args.dataset_name)
        os.makedirs(sample_dir, exist_ok=True)
        save_path = os.path.join(sample_dir, f"sample_batch_{batch_idx}.png")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        print(f"Sample images saved at {save_path}")
    
    
def load_model(args):
    # 定义Autoencoder模型
    encoder = Encoder(channels=128, channel_multipliers=[1, 2, 4, 8], n_resnet_blocks=2, in_channels=1, z_channels=4)
    decoder = Decoder(channels=128, channel_multipliers=[1, 2, 4, 8], n_resnet_blocks=2, out_channels=1, z_channels=4)
    autoencoder = Autoencoder(encoder, decoder, emb_channels=4, z_channels=4).to(device)
 
    ckpt_folder_path, _ = get_latest_ckpt_path(args.dataset_name, args.save_ae_dir)
    if ckpt_folder_path:
        encoder_path = os.path.join(ckpt_folder_path, "encoder.pt")
        decoder_path = os.path.join(ckpt_folder_path, "decoder.pt")

        if os.path.exists(encoder_path):
            encoder.load_state_dict(torch.load(encoder_path, map_location=device))
            print(f"Loaded encoder from {encoder_path}")
        else:
            print(f"No encoder checkpoint found at {encoder_path}")
            
        if os.path.exists(decoder_path):
            decoder.load_state_dict(torch.load(decoder_path, map_location=device))
            print(f"Loaded decoder from {decoder_path}")
        else:
            print(f"No decoder checkpoint found at {decoder_path}")
    else:
        print("No checkpoint folder found. Starting training from scratch.")

    return autoencoder

def eval(args):
    # 1. 准备数据
    if args.dataset_name == "opmed":
        dataset_test = OPMEDDataset(
            root_dir=args.dataset_root,
            modality="FLAIR",
            train=False,
            img_size=(256, 256)
        )
    else:
        raise RuntimeError(f"dataset {args.dataset_name} is not implemented yet.")
    
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    # 2. 定义模型, loss function and optimizer
    autoencoder: Autoencoder
    autoencoder = load_model(args)
    
    # 3. 训练循环
    for batch_idx, sample in enumerate(dataloader_test):
        print(f"Batch {batch_idx}")
        sample = Sample(**sample)
        images = sample.img.to(device)
        save_sample_images(autoencoder, images, batch_idx, args)

        

if __name__ == "__main__":
    args = parse_args()
    print(f"Start vae evaluation")
    eval(args=args)
