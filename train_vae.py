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

def save_sample_images(autoencoder: Autoencoder, images: torch.Tensor, epoch: int, args: dict):
    """
    保存解码后的图像和原始图像的对比图。
    
    :param autoencoder: Autoencoder 模型
    :param images: 原始图像张量，形状为 [batch_size, channels, height, width]
    :param epoch: 当前 epoch
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
        plt.title(f"Epoch {epoch}: Original (Top) vs Decoded (Bottom)")
        
        # 保存图像
        sample_dir = os.path.join(args.ae_sample_dir, args.dataset_name)
        os.makedirs(sample_dir, exist_ok=True)
        save_path = os.path.join(sample_dir, f"sample_epoch_{epoch}.png")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        print(f"Sample images saved at {save_path}")
    
    autoencoder.train()  # 设回训练模式

def save_model(autoencoder: Autoencoder, optimizer: optim.Adam, lr_scheduler: ReduceLROnPlateau, epoch, args):
    if not os.path.exists(args.save_ae_dir):
        os.makedirs(args.save_ae_dir)
        
    new_ckpt_path = os.path.join(args.save_ae_dir, f"ckpt-{args.dataset_name}-{epoch}")
    os.makedirs(new_ckpt_path, exist_ok=True)
    
    # 保存模型权重
    torch.save(autoencoder.encoder.state_dict(), os.path.join(new_ckpt_path, f"encoder.pt"))
    torch.save(autoencoder.decoder.state_dict(), os.path.join(new_ckpt_path, f"decoder.pt"))
    
    # 保存优化器状态（包括学习率）
    torch.save(optimizer.state_dict(), os.path.join(new_ckpt_path, f"optimizer.pt"))
    
    # 保存学习率调度器状态
    torch.save(lr_scheduler.state_dict(), os.path.join(new_ckpt_path, f"lr_scheduler.pt"))
    
    print(f"Model, optimizer, and lr_scheduler saved at {new_ckpt_path}")
    
    
def load_model(args):
    # 定义Autoencoder模型
    encoder = Encoder(channels=128, channel_multipliers=[1, 2, 4, 8], n_resnet_blocks=2, in_channels=1, z_channels=4)
    decoder = Decoder(channels=128, channel_multipliers=[1, 2, 4, 8], n_resnet_blocks=2, out_channels=1, z_channels=4)
    autoencoder = Autoencoder(encoder, decoder, emb_channels=4, z_channels=4).to(device)

    # 定义损失函数
    loss_fn = nn.MSELoss()

    # 定义优化器
    optimizer = optim.Adam(autoencoder.parameters(), lr=args.initial_lr)

    # 定义学习率调度器
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # 如果设置了继续训练，则加载模型、优化器和学习率调度器状态
    if args.resume:
        ckpt_folder_path, runned_epoch = get_latest_ckpt_path(args.dataset_name, args.save_ae_dir)
        if ckpt_folder_path:
            encoder_path = os.path.join(ckpt_folder_path, "encoder.pt")
            decoder_path = os.path.join(ckpt_folder_path, "decoder.pt")
            optimizer_path = os.path.join(ckpt_folder_path, "optimizer.pt")
            lr_scheduler_path = os.path.join(ckpt_folder_path, "lr_scheduler.pt")

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

            if os.path.exists(optimizer_path):
                optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
                print(f"Loaded optimizer state from {optimizer_path}")
            else:
                print(f"No optimizer checkpoint found at {optimizer_path}")

            if os.path.exists(lr_scheduler_path):
                lr_scheduler.load_state_dict(torch.load(lr_scheduler_path, map_location=device))
                print(f"Loaded lr_scheduler state from {lr_scheduler_path}")
            else:
                print(f"No lr_scheduler checkpoint found at {lr_scheduler_path}")
        else:
            print("No checkpoint folder found. Starting training from scratch.")

    return autoencoder, loss_fn, optimizer, lr_scheduler, runned_epoch

def train(args):
    # 1. 准备数据
    if args.dataset_name == "opmed":
        dataset = OPMEDDataset(
            root_dir=args.dataset_root,
            modality="FLAIR",
            train=True,
            img_size=(256, 256)
        )
        dataset_test = OPMEDDataset(
            root_dir=args.dataset_root,
            modality="FLAIR",
            train=False,
            img_size=(256, 256)
        )
    else:
        raise RuntimeError(f"dataset {args.dataset_name} is not implemented yet.")
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    dataloader_test_iter = iter(dataloader_test)
    
    # 2. 定义模型, loss function and optimizer
    autoencoder: Autoencoder
    lr_scheduler: ReduceLROnPlateau
    autoencoder, loss_fn, optimizer, lr_scheduler, runned_epoch = load_model(args)
    
    # 3. 训练循环
    num_epochs = args.epochs
    for epoch in range(runned_epoch, num_epochs):
        autoencoder.train()
        for batch_idx, sample in enumerate(dataloader):
            sample = Sample(**sample)
            images = sample.img.to(device)
            
            # 前向传播
            encoded = autoencoder.encode(images)
            decoded = autoencoder.decode(encoded.sample())
            
            # 计算损失 backward
            loss = loss_fn(decoded, images)
            loss = loss / args.accumulation_steps
            loss.backward()
            # 反向传播和优化
            if (batch_idx + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            if batch_idx % args.output_interval == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")
                
        # 更新学习率
        lr_scheduler.step(loss)
        print(f"Learning rate: {lr_scheduler.get_last_lr()[0]}")

        if (epoch + 1) % args.sample_interval == 0:
            try:
                test_sample = next(dataloader_test_iter)
            except:
                dataloader_test_iter = iter(dataloader_test)
                test_sample = next(dataloader_test_iter)

            test_sample = Sample(**test_sample)
            test_images = test_sample.img.to(device)
            save_sample_images(autoencoder, test_images, epoch + 1, args)
        # 每隔一定epoch保存模型
        if (epoch + 1) % args.save_interval == 0:
            save_model(autoencoder, optimizer, lr_scheduler, epoch + 1, args)

if __name__ == "__main__":
    args = parse_args()
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.initial_lr}")
    train(args=args)
