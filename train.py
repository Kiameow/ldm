from sched import scheduler
from types import new_class
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import parse_args, get_latest_ckpt_path
from dataloader import OPMEDDataset, Sample
from ldm import LatentDiffusion
from labml_nn.diffusion.stable_diffusion.model.unet import UNetModel
from labml_nn.diffusion.stable_diffusion.model.autoencoder import Autoencoder, Encoder, Decoder
from labml_nn.diffusion.stable_diffusion.model.clip_embedder import CLIPTextEmbedder
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(args):
    unet = UNetModel(
        in_channels=4,          # 输入通道数（潜在空间的通道数）
        out_channels=4,         # 输出通道数（与输入相同）
        channels=128,           # 基础通道数
        n_res_blocks=2,         # 每个分辨率级别的残差块数量
        attention_levels=[1, 2],# 在第 1 和第 2 个分辨率级别上应用注意力
        channel_multipliers=[1, 2, 4, 8],  # 通道数倍增因子
        n_heads=8,              # 多头注意力机制的头数
        tf_layers=1,            # Transformer 层数
        d_cond=768              # 条件嵌入维度（与 CLIP 一致）
    )
    
    clip = CLIPTextEmbedder().to(device)
    if next(clip.parameters()).device is None:
        raise RuntimeError("No CLIP ckpt found")
    
    ae_ckpt_folder = get_latest_ckpt_path(args.dataset_name, args.save_ae_dir)
    if ae_ckpt_folder and os.path.exists(ae_ckpt_folder):
        encoder = Encoder(channels=128, channel_multipliers=[1, 2, 4, 8], n_resnet_blocks=2, in_channels=1, z_channels=4)
        decoder = Decoder(channels=128, channel_multipliers=[1, 2, 4, 8], n_resnet_blocks=2, out_channels=1, z_channels=4)
        autoencoder = Autoencoder(encoder, decoder, emb_channels=4, z_channels=4).to(device)
        
        encoder_path = os.path.join(ae_ckpt_folder, "encoder.pt")
        decoder_path = os.path.join(ae_ckpt_folder, "decoder.pt")
        
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    else:
        raise RuntimeError("No autoencoder ckpt found")
        
    
    clip.eval()
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad_(False)
    for param in clip.parameters():
        param.requires_grad_(False)
    
    if args.resume:
        ckpt_folder_path = get_latest_ckpt_path(args.dataset_name, args.save_dir)
        unet.load_state_dict(torch.load(os.path.join(ckpt_folder_path, "unet.pt"), map_location=device))
        
    ldm = LatentDiffusion(
        unet_model=unet,
        autoencoder=autoencoder,
        clip_embedder=clip,  # 你需要定义或导入 CLIPTextEmbedder
        latent_scaling_factor=0.18215,  # 示例值
        n_steps=1000,  # 示例值
        linear_start=0.00085,  # 示例值
        linear_end=0.012  # 示例值
    ).to(device)
    
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(ldm.unet_model.parameters(), lr=args.initial_lr)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    return ldm, loss_fn, optimizer, lr_scheduler

def save_model(unet: UNetModel, optimizer: optim.Adam, epoch, args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    new_ckpt_path = os.path.join(args.save_dir, f"ckpt-{args.dataset_name}-{epoch}")
    os.makedirs(new_ckpt_path, exist_ok=True)
    
    torch.save(unet.state_dict(), os.path.join(new_ckpt_path, f"unet.pt"))
    torch.save(optimizer.state_dict(), os.path.join(new_ckpt_path, f"opt.pt"))


def train(args):
    # 1. 准备数据
    if args.dataset_name == "opmed":
        dataset = OPMEDDataset(
            root_dir=args.dataset_root,
            modality="FLAIR",
            train=True,
            img_size=(256, 256)
        )
    else:
        raise RuntimeError(f"dataset {args.dataset_name} is not implemented yet.")
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    # 2. 定义模型, loss function and optimizer
    ldm: LatentDiffusion
    lr_scheduler: ReduceLROnPlateau
    ldm, loss_fn, optimizer, lr_scheduler = load_model(args)
    
    
    # 4. 训练循环
    num_epochs = args.epochs
    for epoch in range(num_epochs):
        ldm.train()
        for batch_idx, sample in enumerate(dataloader):
            sample = Sample(**sample)
            images = sample.img.to(ldm.device)
            
            # 获取文本条件（假设你有文本数据）
            prompts = ["example prompt"] * images.size(0)  # 你需要提供实际的文本提示
            context = ldm.get_text_conditioning(prompts)
            
            # 编码图像到潜在空间
            latents = ldm.autoencoder_encode(images)
            
            # 随机生成时间步
            t = torch.randint(0, ldm.n_steps, (images.size(0),), device=ldm.device).long()
            
            # 前向传播
            noise = torch.randn_like(latents, device=ldm.device)
            noisy_latents = ldm.alpha_bar[t].sqrt() * latents + (1 - ldm.alpha_bar[t]).sqrt() * noise
            predicted_noise = ldm(noisy_latents, t, context)
            
            # 计算损失
            loss = loss_fn(predicted_noise, noise)
            loss = loss / args.accumulation_steps
            loss.backward()
            
            # 反向传播和优化
            if (batch_idx + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            if batch_idx % args.output_interval == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")
                
        lr_scheduler.step(loss)
        print(f"Learning rate: {lr_scheduler.get_last_lr()[0]}")

        if (epoch + 1) % args.save_interval == 0:
            save_model(ldm.unet_model, optimizer, epoch + 1, args)

if __name__ == "__main__":
    args = parse_args()
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    train(args=args)
