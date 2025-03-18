from dataclasses import asdict, dataclass
from torch.utils.data import Dataset
from torchvision import transforms
from utils import ResizeWithPadding
import os
import torch
from PIL import Image

@dataclass
class Sample:
    img: torch.Tensor = None
    img_path: str = ""
    mask: torch.Tensor = None
    mask_path: str = ""
    prompt: str = ""
    postive: bool = False
        

class OPMEDDataset(Dataset):
    def __init__(self, root_dir, modality='FLAIR', train=True, type='both', img_size=(256, 256)):
        self.root_dir = root_dir
        self.modality = modality
        self.train = train
        self.type = type
        self.img_size = img_size
        self.transform = transforms.Compose([
            ResizeWithPadding(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        
        # Set paths based on train/test
        if train:
            self.hc_image_dir = os.path.join(root_dir, 'train', f'hc_{modality}')
            self.un_image_dir = os.path.join(root_dir, 'train', f'un_{modality}')
            self.mask_dir = None  # No masks for healthy controls
        else:
            self.hc_image_dir = os.path.join(root_dir, 'test', f'hc_{modality}')
            self.un_image_dir = os.path.join(root_dir, 'test', f'un_{modality}')
            self.mask_dir = os.path.join(root_dir, 'test', 'mask')
            
        self.hc_filenames = sorted([f for f in os.listdir(self.hc_image_dir) if f.endswith('.png')])
        self.un_filenames = sorted([f for f in os.listdir(self.un_image_dir) if f.endswith('.png')])
        
    def __len__(self):
        if self.type == 'healthy':
            return len(self.hc_filenames)
        elif self.type == 'unhealthy':
            return len(self.un_filenames)
        else:
            return len(self.hc_filenames) + len(self.un_filenames)
        
    def __getitem__(self, idx):
        sample = Sample()
        
        if self.type == 'healthy':
            image_dir = self.hc_image_dir
            filenames = self.hc_filenames
            is_positive = False
        elif self.type == 'unhealthy':
            image_dir = self.un_image_dir
            filenames = self.un_filenames
            is_positive = True
        else:  # both
            if idx < len(self.hc_filenames):
                image_dir = self.hc_image_dir
                filenames = self.hc_filenames
                is_positive = False
            else:
                image_dir = self.un_image_dir
                filenames = self.un_filenames
                idx -= len(self.hc_filenames)  # Adjust index for unhealthy samples
                is_positive = True

        img_path = os.path.join(image_dir, filenames[idx])
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("L")
            sample.img_path = img_path
        else:
            raise RuntimeError(f"Sample {img_path} not found")

        # Load mask for unhealthy samples in test mode
        if (not self.train) and is_positive:
            mask_filename = filenames[idx].replace(self.modality, f"{self.modality}_roi")
            mask_path = os.path.join(self.mask_dir, mask_filename)
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")
                sample.mask_path = mask_path
            else:
                mask = Image.new("L", (img.width, img.height), 0)  # Empty mask if not found
        else:
            mask = Image.new("L", (img.width, img.height), 0)

        sample.prompt = "diseased" if is_positive else "healthy"

        # Apply transforms
        sample.img = self.transform(img)
        trans_mask = self.transform(mask)
        sample.mask = (trans_mask > 0.).float()

        return asdict(sample)

    

