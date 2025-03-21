import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import parse_args, get_latest_ckpt_path, visualize_latents
from dataloader import OPMEDDataset, Sample
from ldm import LatentDiffusion
from unet import UNetModel
from labml_nn.diffusion.stable_diffusion.model.autoencoder import Autoencoder, Encoder, Decoder
from labml_nn.diffusion.stable_diffusion.model.clip_embedder import CLIPTextEmbedder
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from dummy_text_embedder import DummyTextEmbedder
import statistics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def draw_training_process(train_losses, test_losses, save_path, sample_interval):
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(12, 6))

    # Adjust x-axis for test loss since it's recorded every sample_interval epochs
    test_epochs = list(range(sample_interval, sample_interval * len(test_losses) + 1, sample_interval))

    # Plot Training and Test Loss
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(test_epochs, test_losses, label='Test Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()

    plt.tight_layout()
    save_path = os.path.join(save_path, "metrics_plot.png")
    plt.savefig(save_path)

def save_sample_images(ldm: LatentDiffusion, images: torch.Tensor, prompt: str, epoch: int, args: dict):
    """
    save the original, original_feature_map, recon_feature_map, recon
    
    :param ldm: Autoencoder 模型
    :param images: 原始图像张量，形状为 [batch_size, channels, height, width]
    :param epoch: 当前 epoch
    :param args
    """
    ldm.eval()  # switch to evaluation mode
    with torch.no_grad():
        # Duplicate prompt for each image and get text conditioning
        prompts = [prompt] * images.size(0)
        if args.no_text:
            context = None
        else:
            context = ldm.get_text_conditioning(prompts)

        # Encode the original images into latent space
        latents = ldm.autoencoder_encode(images)

        # Choose the starting timestep T (e.g., provided by args.ldm_sample_t)
        T = args.ldm_sample_t

        # Forward process: add noise to obtain x_T
        noise = torch.randn_like(latents, device=ldm.device)
        alpha_bar_T = ldm.alpha_bar[T].view(-1, 1, 1, 1)
        x_t = alpha_bar_T.sqrt() * latents + (1 - alpha_bar_T).sqrt() * noise

        # Set eta to control stochasticity: eta=0 yields a deterministic reverse process
        eta = 0.0

        # Reverse diffusion: iteratively denoise from t = T down to 1
        for t in reversed(range(1, T+1)):
            # Create a tensor filled with the current timestep for the batch
            t_tensor = torch.full((x_t.size(0),), t, device=ldm.device, dtype=torch.long)
            
            # Predict the noise component using the trained diffusion model
            predicted_noise = ldm(x_t, t_tensor, context)
            
            # Get alpha_bar for the current timestep and for t-1
            alpha_bar_t = ldm.alpha_bar[t].view(-1, 1, 1, 1)
            if t > 1:
                alpha_bar_t_minus_1 = ldm.alpha_bar[t-1].view(-1, 1, 1, 1)
            else:
                alpha_bar_t_minus_1 = torch.ones_like(alpha_bar_t)
            
            # Compute the deterministic estimate of x0 from x_t
            x0_est = (x_t - (1 - alpha_bar_t).sqrt() * predicted_noise) / alpha_bar_t.sqrt()
            
            # Compute the mean for the reverse step (following DDPM update)
            mean_x = (
                alpha_bar_t_minus_1.sqrt() * x0_est +
                ((1 - alpha_bar_t_minus_1).sqrt() - eta * ( (1 - alpha_bar_t_minus_1) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_minus_1) )**0.5) * predicted_noise
            )
            
            # Compute sigma_t: if eta==0, then sigma_t becomes 0 (deterministic)
            if t > 1:
                beta_t = 1 - ldm.alpha_bar[t] / ldm.alpha_bar[t-1]
                sigma_t = eta * (beta_t * (1 - alpha_bar_t_minus_1) / (1 - alpha_bar_t))**0.5
            else:
                sigma_t = 0.0
            
            # Sample x_{t-1}: if sigma_t is 0, this is a deterministic update
            noise_term = sigma_t * torch.randn_like(x_t, device=ldm.device) if t > 1 else 0.0
            x_t = mean_x + noise_term

        # Decode the final latent (which approximates x0) back to image space
        recon_latents = x_t
        recons = ldm.autoencoder_decode(recon_latents)
        ori_decoded = ldm.autoencoder_decode(latents)
        
        loss_fn = nn.MSELoss()
        test_loss = loss_fn(recons, images).item()
        
        v_images = images.to("cpu")
        v_latents = visualize_latents(latents.to("cpu"), (256, 256))
        v_recon_latents = visualize_latents(recon_latents.to("cpu"), (256, 256))
        v_recons = recons.to("cpu")
        v_ori_decoded = ori_decoded.to("cpu")
        # print(f"original: {v_images.shape}")
        # print(f"ori_latents: {v_latents.shape}")
        # print(f"recon_latents: {v_recon_latents.shape}")
        # print(f"recon: {v_recons.shape}")
        
        # 确保不会超出 batch_size 的范围
        num_images = min(8, images.size(0))
        
        # 将解码后的图像和原始图像拼接在一起
        comparison = torch.cat([
            v_images[:num_images], 
            v_latents[:num_images], 
            v_recon_latents[:num_images],
            v_recons[:num_images],
            v_ori_decoded[:num_images]
        ])  # 取前 num_images 张图像进行对比
        comparison = vutils.make_grid(comparison, nrow=num_images, normalize=True, scale_each=True)
        
        # 将张量转换为图像并保存
        plt.figure(figsize=(12, 6))
        plt.imshow(comparison.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        plt.title(f"Epoch {epoch}: Original, Latents, ReconLatents, Decoded, OriDecoded")
        
        # 保存图像
        sample_dir = os.path.join(args.sample_dir, args.dataset_name)
        os.makedirs(sample_dir, exist_ok=True)
        save_path = os.path.join(sample_dir, f"sample_epoch_{epoch}.png")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        print(f"Sample images saved at {save_path}")
    
    ldm.train()  # 设回训练模式
    return test_loss

def load_model(args):
    unet = UNetModel(
        in_channels=args.ae_dim,          # 输入通道数（潜在空间的通道数）
        out_channels=args.ae_dim,         # 输出通道数（与输入相同）
        channels=128,           # 基础通道数
        n_res_blocks=2,         # 每个分辨率级别的残差块数量
        attention_levels=[1, 2],    # No attention applied
        channel_multipliers=[1, 2, 4, 8],  # 通道数倍增因子
        n_heads=8,              # 多头注意力机制的头数
        tf_layers=1,            # Transformer 层数
        d_cond=768              # 条件嵌入维度（与 CLIP 一致）
    ).to(device)
    
    if args.no_clip:
        clip = DummyTextEmbedder(d_cond=768).to(device)
        print("run without clip")
    else:
        clip = CLIPTextEmbedder().to(device)
        if next(clip.parameters()).device is None:
            raise RuntimeError("No CLIP ckpt found")
        else:
            print("CLIP ready")
    
    ae_ckpt_folder, _ = get_latest_ckpt_path(args.dataset_name, args.save_ae_dir)
    if ae_ckpt_folder and os.path.exists(ae_ckpt_folder):
        encoder = Encoder(channels=128, channel_multipliers=[1, 2, 4, 8], n_resnet_blocks=2, in_channels=1, z_channels=args.ae_dim)
        decoder = Decoder(channels=128, channel_multipliers=[1, 2, 4, 8], n_resnet_blocks=2, out_channels=1, z_channels=args.ae_dim)
        autoencoder = Autoencoder(encoder, decoder, emb_channels=args.ae_dim, z_channels=args.ae_dim).to(device)
        
        ae_path = os.path.join(ae_ckpt_folder, "ae.pt")
        
        autoencoder.load_state_dict(torch.load(ae_path, map_location=device))
        print("Autoencoder ready")
    else:
        raise RuntimeError("No autoencoder ckpt found")
        
    
    clip.eval()
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad_(False)
    for param in clip.parameters():
        param.requires_grad_(False)
    
    runned_epoch = 0
    if args.resume:
        ckpt_folder_path, runned_epoch = get_latest_ckpt_path(args.dataset_name, args.save_dir)
        if ckpt_folder_path is None:
            runned_epoch = 0
            print("start new training for ldm/unet")
        else:
            # Load the entire wrapper checkpoint
            wrapper_state = torch.load(os.path.join(ckpt_folder_path, "unet.pt"), map_location=device)
            # Extract the unet state dict by stripping the prefix
            unet_state_dict = {
                key.replace("diffusion_model.", ""): value
                for key, value in wrapper_state.items()
                if key.startswith("diffusion_model.")
            }
            # Load into your unet
            unet.load_state_dict(unet_state_dict)
            print(f"loaded unet from {os.path.join(ckpt_folder_path, 'unet.pt')}")
        
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
    optimizer = optim.Adam(ldm.model.parameters(), lr=args.initial_lr)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    return ldm, loss_fn, optimizer, lr_scheduler, runned_epoch

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
            type="healthy",
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
    dataloader_test = DataLoader(dataset_test, batch_size=args.sample_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    dataloader_test_iter = iter(dataloader_test)
    
    print(f"Dataloader for {args.dataset_name} ready")
    print(f"Train dataset num of batch: {len(dataloader)}")
    print(f"Test dataset num of batch: {len(dataloader_test)} (Infinite Loop)")
    
    # 2. 定义模型, loss function and optimizer
    ldm: LatentDiffusion
    lr_scheduler: ReduceLROnPlateau
    ldm, loss_fn, optimizer, lr_scheduler, runned_epoch = load_model(args)
    
    
    # 4. 训练循环
    num_epochs = args.epochs
    train_losses = []
    test_losses = []
    for epoch in range(runned_epoch if runned_epoch else 0, num_epochs):
        ldm.train()
        epoch_loss = []
        for batch_idx, sample in enumerate(dataloader):
            sample = Sample(**sample)
            images = sample.img.to(ldm.device)
            
            # 获取文本条件（假设你有文本数据）
            prompts = sample.prompt
            if args.no_text:
                context = None
            else:
                context = ldm.get_text_conditioning(prompts)
            
            # 编码图像到潜在空间
            latents = ldm.autoencoder_encode(images)
            
            # 随机生成时间步
            t = torch.randint(0, ldm.n_steps, (images.size(0),), device=ldm.device).long()
            
            # 前向传播
            noise = torch.randn_like(latents, device=ldm.device)
            alpha_bar_t = ldm.alpha_bar[t].view(-1, 1, 1, 1)
            # print(f"noise: {noise.shape}")
            # print(f"t shape: {alpha_bar_t.sqrt().shape}")
            # print(f"latents: {(latents).shape}")

            noisy_latents = alpha_bar_t.sqrt() * latents + (1 - alpha_bar_t).sqrt() * noise
            # print(f"noisy latents: {(noisy_latents).shape}")
            predicted_noise = ldm(noisy_latents, t, context)
            
            # 计算损失
            loss = loss_fn(predicted_noise, noise)
            loss = loss / args.accumulation_steps
            loss.backward()
            
            epoch_loss.append(loss.item())
            
            # 反向传播和优化
            if (batch_idx + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            if batch_idx % args.output_interval == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")
        train_losses.append(statistics.mean(epoch_loss))
        lr_scheduler.step(statistics.mean(epoch_loss))
        print(f"Learning rate: {lr_scheduler.get_last_lr()[0]}")
        


        if (epoch + 1) % args.sample_interval == 0:
            try:
                test_sample = next(dataloader_test_iter)
            except:
                dataloader_test_iter = iter(dataloader_test)
                test_sample = next(dataloader_test_iter)

            test_sample = Sample(**test_sample)
            test_images = test_sample.img.to(device)
            test_loss = save_sample_images(ldm, test_images, "healthy", epoch + 1, args)
            test_losses.append(test_loss)
        if (epoch + 1) % args.save_interval == 0:
            save_model(ldm.model, optimizer, epoch + 1, args)
            
        draw_training_process(train_losses, test_losses, os.path.join(args.sample_dir, args.dataset_name), args.sample_interval)
            

if __name__ == "__main__":
    args = parse_args()
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.initial_lr}")
    train(args=args)
